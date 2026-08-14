#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Objective-to-ground-truth concordance analysis for scBOA.

This script evaluates whether scBOA internal objective scores
(CAS, MCS, MPS/F1, final objective score) correlate with independent
FACS-defined PBMC ground-truth labels across all Stage-1 Bayesian
optimization trials.

Outputs:
  1. trial_objective_facs_concordance.csv
  2. spearman_internal_vs_facs.csv
  3. top_vs_bottom_decile_objective.csv
  4. best_vs_default_facs_concordance.csv
  5. scatter plots and heatmaps
"""

import os
import re
import warnings
import numpy as np
import pandas as pd
import scanpy as sc
import celltypist
from celltypist import models

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from scipy.stats import spearmanr
from sklearn.metrics import (
    adjusted_rand_score,
    normalized_mutual_info_score,
    homogeneity_completeness_v_measure,
    accuracy_score,
    f1_score
)

warnings.filterwarnings("ignore")

# =============================================================================
# User configuration — parsed from CLI
# =============================================================================

import argparse


def parse_cli():
    p = argparse.ArgumentParser(
        description=(
            "Objective-to-ground-truth concordance analysis for scBOA. "
            "Re-runs each Stage-1 trial and compares internal objectives "
            "to external ground-truth metrics."
        )
    )

    # --- Required I/O ---
    p.add_argument("--data_dir", required=True,
                   help="10x matrix directory (contains matrix.mtx.gz, "
                        "features.tsv.gz, barcodes.tsv.gz).")
    p.add_argument("--ground_truth", required=True,
                   help="CSV with cell_id + ground_truth cell type columns.")
    p.add_argument("--model_path", required=True,
                   help="CellTypist .pkl model file.")
    p.add_argument("--yield_csv", required=True,
                   help="Stage-1 yield_scores_report.csv from scBOA.")
    p.add_argument("--output_dir", required=True,
                   help="Directory for all output CSVs and figures.")

    # --- QC filters (mirror scBOA defaults) ---
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--min_genes", type=int, default=200)
    p.add_argument("--max_genes", type=int, default=7000)
    p.add_argument("--max_pct_mt", type=float, default=10.0)
    p.add_argument("--min_cells", type=int, default=3)

    # --- HVG selection (dataset-specific) ---
    p.add_argument("--hvg_min_mean", type=float, default=0.0125)
    p.add_argument("--hvg_max_mean", type=float, default=3.0)
    p.add_argument("--hvg_min_disp", type=float, default=0.3)

    # --- PCA / clustering ---
    p.add_argument("--n_pcs_compute", type=int, default=105)
    p.add_argument("--cas_aggregation_method", default="leiden")

    # --- Default Scanpy baseline (for best-vs-default comparison) ---
    p.add_argument("--default_n_hvg", type=int, default=3000)
    p.add_argument("--default_n_pcs", type=int, default=50)
    p.add_argument("--default_n_neighbors", type=int, default=15)
    p.add_argument("--default_resolution", type=float, default=1.0)

    # --- Mitochondrial gene regex ---
    p.add_argument("--mito_regex", default=r"^(MT|Mt|mt)[-._:]")

    return p.parse_args()


_ARGS = parse_cli()

# Bind to module-level names so the rest of the script works unchanged.
DATA_DIR              = _ARGS.data_dir
GROUND_TRUTH_PATH     = _ARGS.ground_truth
MODEL_PATH            = _ARGS.model_path
YIELD_CSV             = _ARGS.yield_csv
OUTPUT_DIR            = _ARGS.output_dir

os.makedirs(OUTPUT_DIR, exist_ok=True)
FIGURE_DIR = os.path.join(OUTPUT_DIR, "figures")
os.makedirs(FIGURE_DIR, exist_ok=True)

SEED                  = _ARGS.seed
MIN_GENES             = _ARGS.min_genes
MAX_GENES             = _ARGS.max_genes
MAX_PCT_MT            = _ARGS.max_pct_mt
MIN_CELLS             = _ARGS.min_cells

HVG_MIN_MEAN          = _ARGS.hvg_min_mean
HVG_MAX_MEAN          = _ARGS.hvg_max_mean
HVG_MIN_DISP          = _ARGS.hvg_min_disp

N_PCS_COMPUTE         = _ARGS.n_pcs_compute
CAS_AGGREGATION_METHOD = _ARGS.cas_aggregation_method

DEFAULT_PARAMS = {
    "n_hvg":       _ARGS.default_n_hvg,
    "n_pcs":       _ARGS.default_n_pcs,
    "n_neighbors": _ARGS.default_n_neighbors,
    "resolution":  _ARGS.default_resolution,
}

MITO_REGEX_PATTERN = _ARGS.mito_regex

EXTERNAL_METRICS = [
    "external_leiden_ari",
    "external_leiden_nmi",
    "external_homogeneity",
    "external_completeness",
    "external_celltypist_broad_accuracy",
    "external_celltypist_broad_macro_f1",
    "external_mean_confidence",
]

TRIAL_META_KEYS = [
    "matched_cells",
    "n_facs_labels",
    "n_leiden_clusters",
    "effective_n_hvg",
    "effective_n_pcs",
]

# =============================================================================
# Helper functions
# =============================================================================

def normalize_cell_id(x):
    """
    Normalize 10x barcode / metadata cell IDs.
    Keeps original form but also strips common '-1' suffix if needed.
    """
    x = str(x)
    return x.strip()


def strip_10x_suffix(x):
    x = str(x).strip()
    return re.sub(r"-\d+$", "", x)


def find_ground_truth_columns(df):
    """
    Robustly infer cell ID and ground-truth label columns.
    """
    lower_map = {c.lower(): c for c in df.columns}

    possible_id_cols = [
        "cell_id", "barcode", "barcodes", "cell", "cellid", "obs_names"
    ]
    possible_label_cols = [
        "ground_truth_cell_type", "ground_truth", "facs_label",
        "cell_type", "celltype", "label", "annotation"
    ]

    id_col = None
    label_col = None

    for c in possible_id_cols:
        if c in lower_map:
            id_col = lower_map[c]
            break

    for c in possible_label_cols:
        if c in lower_map:
            label_col = lower_map[c]
            break

    if id_col is None:
        id_col = df.columns[0]

    if label_col is None:
        if len(df.columns) < 2:
            raise ValueError("Could not infer ground-truth label column.")
        label_col = df.columns[1]

    return id_col, label_col


def map_to_pbmc_broad(label):
    """
    Map CellTypist Healthy_COVID19_PBMC predictions AND 10x FACS-sorted PBMC
    ground-truth labels to a shared broad vocabulary:

        CD4_T, CD8_T, B_cell, Monocyte, NK, Other

    Ground-truth (10x FACS-sorted purified populations):
        CD4_T_Helper -> CD4_T
        CD8_Cytotoxic_T -> CD8_T
        B_cells -> B_cell
        Monocytes -> Monocyte
        NK_cells -> NK

    CellTypist subtypes collapsed to their major class. Populations with
    no counterpart in the 10x FACS vocabulary (pDC, cDC1/2/3, ILC*,
    Platelets, HSC/progenitors, gdT, MAIT) are collapsed to "Other" so
    they contribute to accuracy denominators without inflating any single
    class. MAIT and gdT could be argued into CD8_T, but the FACS purified
    'Cytotoxic T' bead-sort excludes them, so keeping them in Other
    matches the ground-truth definition more honestly.
    """
    if pd.isna(label):
        return "Other"

    raw = str(label).strip()
    s = raw.lower()

    # --- FACS ground-truth fast path ---
    gt_exact = {
        "cd4_t_helper":     "CD4_T",
        "cd8_cytotoxic_t":  "CD8_T",
        "b_cells":          "B_cell",
        "monocytes":        "Monocyte",
        "nk_cells":         "NK",
    }
    if s in gt_exact:
        return gt_exact[s]

    # --- CellTypist Healthy_COVID19_PBMC prediction handling ---

    # T cell subtypes: match on CD4./CD8. prefix regardless of what follows
    if s.startswith("cd4.") or s.startswith("cd4_t") or s.startswith("cd4 t"):
        return "CD4_T"
    if s.startswith("cd8.") or s.startswith("cd8_t") or s.startswith("cd8 t"):
        return "CD8_T"

    # Innate-like / unconventional T (not in FACS vocabulary)
    if s == "mait" or s == "gdt" or s.startswith("gdt") or "nkt" in s:
        return "Other"

    # NK
    if s.startswith("nk_") or s.startswith("nk ") or s == "nk" \
            or "natural killer" in s:
        return "NK"

    # B lineage: covers B_naive, B_switched_memory, B_non-switched_memory,
    # B_immature, B_exhausted, Plasmablast, Plasma_cell_Ig*
    if s.startswith("b_") or s.startswith("b ") or s == "b" \
            or "plasma" in s or "plasmablast" in s:
        return "B_cell"

    # Monocytes: CD14_mono, C1_CD16_mono, CD83_CD14_mono, ...
    if "mono" in s:
        return "Monocyte"

    # Everything else with no FACS counterpart:
    #   pDC, DC1/2/3, ILC1_3/ILC2, Platelets, HSC_CD38pos/neg
    return "Other"

def map_to_pancreas_broad(label):
    """
    Map Baron ground-truth labels AND CellTypist Adult_Human_PancreaticIslet
    predictions to a shared broad pancreas vocabulary.

    CellTypist islet subtypes such as `alpha_immature`, `beta_stress I`,
    `beta_MHC/autoantigen`, `PP`, ... are collapsed to their major cell type.
    Baron labels such as `activated_stellate`, `quiescent_stellate`, `t_cell`,
    ... are preserved.
    """
    if pd.isna(label):
        return None

    s = str(label).strip().lower()

    # --- Endocrine ---
    if s.startswith("alpha"):
        return "alpha"
    if s.startswith("beta"):
        return "beta"
    if s.startswith("delta"):
        return "delta"
    # PP cells == gamma cells (pancreatic polypeptide)
    if s == "pp" or s.startswith("pp_") or s.startswith("pp ") \
            or s.startswith("gamma"):
        return "gamma"
    if s.startswith("epsilon"):
        return "epsilon"

    # --- Exocrine ---
    if "ductal" in s:
        return "ductal"
    if "acinar" in s:
        return "acinar"

    # --- Stromal ---
    if "activated_stellate" in s or "activated stellate" in s:
        return "activated_stellate"
    if "quiescent_stellate" in s or "quiescent stellate" in s:
        return "quiescent_stellate"
    if s == "stellate" or "stellate" in s:
        return "stellate"

    # --- Vascular / neural ---
    if "endothelial" in s:
        return "endothelial"
    if "schwann" in s:
        return "schwann"

    # --- Immune ---
    if "macrophage" in s:
        return "macrophage"
    if "mast" in s:
        return "mast"
    if s in ("t_cell", "t cell", "tcell") or s.startswith("t_cell"):
        return "t_cell"

    return "other"

def map_to_cbmc_broad(label):
    """
    Map CBMC ground-truth labels AND CellTypist immune-model predictions
    to a shared broad vocabulary defined by the CBMC ground truth:

        CD4_T, CD8_T, T_other, NK, B, CD14_Mono, CD16_Mono, pDC,
        Progenitor, Doublet, Other

    - CD4_T:      CD4 T ground truth; helper/regulatory/Th17 predictions
    - CD8_T:      CD8 T ground truth; cytotoxic T / Trm predictions; CD8a/a
    - T_other:    thymocytes / ETP / early lymphoid (not in GT vocabulary)
    - NK:         NK, CD16+/- NK, transitional NK
    - B:          B cells ground truth; naive/memory/transitional/AAB
    - CD14_Mono:  CD14+ Mono ground truth; classical monocytes; mono-mac
    - CD16_Mono:  CD16+ Mono ground truth; non-classical monocytes
    - pDC:        pDCs ground truth; pDC predictions
    - Progenitor: CD34+ ground truth; HSC/MPP, MEMP, GMP, CMP, MK precursor,
                  monocyte precursor, myelocytes, promyelocytes,
                  neutrophil-myeloid progenitor, early lymphoid/T lymphoid
    - Doublet:    CBMC T/Mono doublets ground truth
    - Other:      erythroid lineage, megakaryocytes/platelets, macrophages,
                  cDC (DC/DC1/DC2/DC3, MNP) - not in CBMC GT vocabulary
    """
    if pd.isna(label):
        return None

    raw = str(label).strip()
    s = raw.lower()

    # --- CBMC ground-truth exact matches first (they are short and clean) ---
    if raw == "CD4 T":
        return "CD4_T"
    if raw == "CD8 T":
        return "CD8_T"
    if raw == "NK":
        return "NK"
    if raw == "B":
        return "B"
    if raw == "CD14+ Mono":
        return "CD14_Mono"
    if raw == "CD16+ Mono":
        return "CD16_Mono"
    if raw == "pDCs":
        return "pDC"
    if raw == "CD34+":
        return "Progenitor"
    if raw == "T/Mono doublets":
        return "Doublet"

    # --- CellTypist immune-model prediction handling ---

    # Doublets
    if "doublet" in s:
        return "Doublet"

    # pDC (must come before generic "dc" check)
    if s == "pdc" or "pdc" in s:
        return "pDC"

    # NK
    if "nk" in s or "natural killer" in s:
        return "NK"

    # B cells
    if s == "b cells" or s.startswith("b cell") or "b_cell" in s \
            or "naive b" in s or "memory b" in s \
            or "transitional b" in s or "age-associated b" in s \
            or "plasma" in s:
        return "B"

    # T cells — order matters (CD4/CD8 first, then thymocyte/ETP → T_other)
    if "cd8" in s or "cytotoxic t" in s or "cd8a/a" in s or "trm cytotoxic" in s:
        return "CD8_T"
    if "cd4" in s or "helper t" in s or "regulatory t" in s or "type 17 helper" in s:
        return "CD4_T"
    if "thymocyte" in s or s == "etp" or "early lymphoid" in s \
            or "t lymphoid" in s:
        return "T_other"

    # Monocytes
    if "non-classical monocyte" in s or "cd16+ mono" in s:
        return "CD16_Mono"
    if "classical monocyte" in s or "cd14+ mono" in s or s == "monocytes" \
            or s == "mono-mac":
        return "CD14_Mono"
    if "monocyte precursor" in s:
        return "Progenitor"

    # Progenitors / myeloid development
    if "hsc" in s or "mpp" in s or "memp" in s or "gmp" in s or "cmp" in s \
            or "megakaryocyte precursor" in s \
            or "neutrophil-myeloid progenitor" in s \
            or "myelocytes" in s or "promyelocytes" in s or s == "early mk":
        return "Progenitor"

    # cDC / mononuclear phagocytes (not in GT vocabulary)
    if s == "dc" or s.startswith("dc") or "mnp" in s or "macrophage" in s:
        return "Other"

    # Erythroid / megakaryocyte (not in GT vocabulary)
    if "eryth" in s or "megakaryocyte" in s or "platelet" in s:
        return "Other"

    return "Other"
def map_to_heart_broad(label):
    """
    Map Litvinukova ground-truth labels AND CellTypist Healthy_Adult_Heart
    subtype predictions to a shared 11-class broad heart vocabulary defined
    by the Litvinukova ground truth:

        Ventricular_Cardiomyocyte, Atrial_Cardiomyocyte, Endothelial,
        Pericytes, Fibroblast, Smooth_muscle_cells, Adipocytes, Neuronal,
        Mesothelial, Myeloid, Lymphoid

    Prediction subtypes are collapsed to their major class:
      vCM1..vCM5, vCM3_stressed          -> Ventricular_Cardiomyocyte
      aCM1..aCM5                         -> Atrial_Cardiomyocyte
      FB1..FB6, FB4_activated            -> Fibroblast
      PC1_vent, PC2_atria, PC3_str,
          PC4_CMC-like                   -> Pericytes
      EC1_cap..EC10_CMC-like,
          EC9_FB-like, EC7_endocardial   -> Endothelial
      SMC1_basic, SMC2_art               -> Smooth_muscle_cells
      Adip1..Adip4                       -> Adipocytes
      NC1_glial, NC2_glial_NGF+,
          NC4_glial, NC6_schwann         -> Neuronal
      Meso                               -> Mesothelial
      Mast, LYVE1+IGF1+MP, LYVE1+TIMD4+MP,
          LYVE1+MP_cycling, MoMP,
          CD14+Mo, CD16+Mo, DC           -> Myeloid
      CD4+T_*, CD8+T_*, NK_CD56hi,
          NK_CD16hi, B, B_plasma,
          MAIT-like, ILC, T/NK_cycling   -> Lymphoid
    """
    if pd.isna(label):
        return None

    raw = str(label).strip()
    s = raw.lower()

    # --- Ground-truth exact matches (fast path, avoid regex confusion) ---
    gt_exact = {
        "Ventricular_Cardiomyocyte": "Ventricular_Cardiomyocyte",
        "Atrial_Cardiomyocyte":      "Atrial_Cardiomyocyte",
        "Endothelial":               "Endothelial",
        "Pericytes":                 "Pericytes",
        "Fibroblast":                "Fibroblast",
        "Smooth_muscle_cells":       "Smooth_muscle_cells",
        "Adipocytes":                "Adipocytes",
        "Neuronal":                  "Neuronal",
        "Mesothelial":               "Mesothelial",
        "Myeloid":                   "Myeloid",
        "Lymphoid":                  "Lymphoid",
    }
    if raw in gt_exact:
        return gt_exact[raw]

    # --- CellTypist heart-model prediction handling ---
    # Order matters: check longer/more specific prefixes first.

    # Cardiomyocytes
    if s.startswith("vcm"):
        return "Ventricular_Cardiomyocyte"
    if s.startswith("acm"):
        return "Atrial_Cardiomyocyte"

    # Fibroblast
    if s.startswith("fb"):
        return "Fibroblast"

    # Pericytes (must precede any generic "pc" check)
    if s.startswith("pc1") or s.startswith("pc2") \
            or s.startswith("pc3") or s.startswith("pc4") \
            or s == "pc" or s.startswith("pc_"):
        return "Pericytes"

    # Endothelial
    if s.startswith("ec"):
        return "Endothelial"

    # Smooth muscle
    if s.startswith("smc"):
        return "Smooth_muscle_cells"

    # Adipocytes
    if s.startswith("adip"):
        return "Adipocytes"

    # Neural crest / Schwann
    if s.startswith("nc") or "schwann" in s or "glial" in s:
        return "Neuronal"

    # Mesothelial
    if s == "meso" or "mesothel" in s:
        return "Mesothelial"

    # Myeloid: macrophages, monocytes, mast, DC
    if "mast" in s:
        return "Myeloid"
    if "lyve1" in s or "momp" in s or "macrophage" in s or s.endswith("mp") \
            or "mp_" in s:
        return "Myeloid"
    if "cd14+mo" in s or "cd16+mo" in s or s.endswith("mo") \
            or s == "monocyte" or "monocyte" in s:
        return "Myeloid"
    if s == "dc" or s.startswith("dc") or "dendritic" in s:
        return "Myeloid"

    # Lymphoid: T, NK, B, ILC, MAIT
    if s.startswith("cd4+t") or s.startswith("cd8+t") \
            or s.startswith("cd4+ t") or s.startswith("cd8+ t") \
            or "t/nk" in s:
        return "Lymphoid"
    if s.startswith("nk"):
        return "Lymphoid"
    if s == "b" or s.startswith("b_") or "b_plasma" in s or "plasma" in s:
        return "Lymphoid"
    if "mait" in s or s == "ilc" or "innate lymphoid" in s:
        return "Lymphoid"

    return "Other"

def map_to_he_heart_broad(label):
    """
    Map He Organ Atlas heart ground-truth labels AND CellTypist
    Healthy_Adult_Heart subtype predictions to a shared broad vocabulary
    defined by the He heart ground truth:

        Fibroblast, Endothelial, SmoothMuscle, Macrophage, Monocyte,
        T_NK, Schwann, Other

    Note: the He Organ Atlas heart subset is stromal/immune only. It
    contains no cardiomyocytes, pericytes, adipocytes, mast cells,
    B cells, DCs, or neural-crest cells in the ground truth. Predicted
    labels for those subtypes (vCM*, aCM*, PC*, Adip*, Mast, B, DC,
    Meso, ...) are collapsed to "Other" — they are legitimate CellTypist
    outputs but have no matching ground-truth class in this dataset.
    """
    if pd.isna(label):
        return None

    raw = str(label).strip()
    s = raw.lower()

    # --- He heart ground-truth handling ---
    # Examples:
    #   "Fibroblast APOE Heart", "Fibroblast TNFAIP6 Heart",
    #   "Endothelial cell SLC9A3R2 Heart", "Endothelial cell ACKR1 Heart",
    #   "Smooth muscle cell MYH11 Heart", "Smooth muscle cell AGT Heart",
    #   "Macrophage C1QA_high Heart" (or C1QA_hjgh — source typo),
    #   "Macrophage SPP1 Heart", "T cell Heart", "Monocyte Heart",
    #   "Schwann Heart"
    if s.startswith("fibroblast"):
        return "Fibroblast"
    if s.startswith("endothelial"):
        return "Endothelial"
    if s.startswith("smooth muscle"):
        return "SmoothMuscle"
    if s.startswith("macrophage"):
        return "Macrophage"
    if s.startswith("monocyte"):
        return "Monocyte"
    if s.startswith("t cell") or s == "t_cell" or s == "t/nk":
        return "T_NK"
    if s.startswith("schwann"):
        return "Schwann"

    # --- CellTypist Healthy_Adult_Heart prediction handling ---
    # Fibroblast subtypes: FB1..FB6, FB4_activated
    if s.startswith("fb"):
        return "Fibroblast"

    # Endothelial subtypes: EC1_cap..EC10_CMC-like, EC7_endocardial, ...
    if s.startswith("ec"):
        return "Endothelial"

    # Smooth muscle: SMC1_basic, SMC2_art
    if s.startswith("smc"):
        return "SmoothMuscle"

    # Tissue-resident macrophages: LYVE1+*MP, MoMP, Macrophage_cycling
    if "lyve1" in s or s == "momp" or "macrophage" in s \
            or s.endswith("_mp") or "_mp_" in s:
        return "Macrophage"

    # Monocytes: CD14+Mo, CD16+Mo
    if "cd14+mo" in s or "cd16+mo" in s or s.endswith("+mo") \
            or "monocyte" in s:
        return "Monocyte"

    # T / NK lymphoid: CD4+T_*, CD8+T_*, NK_CD56hi, NK_CD16hi, MAIT, ILC
    if s.startswith("cd4+t") or s.startswith("cd8+t") \
            or s.startswith("cd4+ t") or s.startswith("cd8+ t") \
            or s.startswith("nk") or "t/nk" in s \
            or "mait" in s or s == "ilc" or "innate lymphoid" in s:
        return "T_NK"

    # Neural crest / glial / schwann
    if s.startswith("nc") or "schwann" in s or "glial" in s:
        return "Schwann"

    # Everything else has no matching class in He heart GT:
    #   vCM1..5, aCM1..5, PC1..4, Adip1..4, Meso, Mast, B, B_plasma, DC
    return "Other"
def map_to_lung_broad(label):
    """
    Map HLCA `ann_level_3` ground-truth labels AND CellTypist
    Human_Lung_Atlas (`ann_level_4`) predictions to the shared broad
    vocabulary defined by the ground truth present in this dataset:

        Basal, Secretory, Multiciliated lineage, AT2,
        Macrophages, T cell lineage, EC capillary, Other

    The two label spaces are hierarchically nested (GT = HLCA level 3,
    predictions = HLCA level 4), so predictions are rolled UP to the GT
    granularity. Never push GT down to level 4 -- that invents information.

    Roll-up:
      Basal resting, Suprabasal, Hillock-like          -> Basal
      Club (nasal/non-nasal), Goblet (nasal/bronchial/
          subsegmental), pre-TB secretory              -> Secretory
      Multiciliated (nasal/non-nasal), Deuterosomal    -> Multiciliated lineage
      AT2, AT2 proliferating, AT0                      -> AT2
      Alveolar macrophages, Alveolar Mph CCL3+,
          Alveolar Mph MT-positive, Alveolar Mph
          proliferating, Monocyte-derived Mph,
          Interstitial Mph perivascular                -> Macrophages
      CD4 T cells, CD8 T cells, T cells proliferating,
          Tregs                                        -> T cell lineage
      EC general capillary, EC aerocyte capillary      -> EC capillary

    Collapsed to "Other" -- legitimate CellTypist outputs whose level-3
    parent is absent from this ground truth: AT1, Ionocyte, Tuft,
    Neuroendocrine, SMG serous/mucous/duct, Classical and Non-classical
    monocytes, DC1/DC2/Migratory/Plasmacytoid DCs, Mast cells,
    Neutrophils, B cells, Plasma cells, NK cells, all fibroblast
    subtypes, smooth muscle, pericytes, mesothelium, lymphatic EC,
    EC arterial/venous.

    ORDER MATTERS. Three traps specific to the HLCA label space:
      1. "Monocyte-derived Mph" contains "monocyte", but its level-3
         parent is Macrophages, not Monocytes. The macrophage check must
         precede the monocyte check.
      2. "NK cells" rolls up to level-3 "Innate lymphoid cell NK", NOT
         "T cell lineage". NK/ILC must be excluded before any generic
         lymphoid check.
      3. "EC general/aerocyte capillary" must match before the generic
         "ec " prefix, which routes arterial/venous EC to Other.

    AT0 caveat: AT0 sits under level-3 AT2 in the HLCA hierarchy, so it
    maps to AT2 here. Flag this if you rerun on a dataset where AT0 is
    a distinct ground-truth class.
    """
    if pd.isna(label):
        return None

    raw = str(label).strip()
    s = raw.lower()

    # --- HLCA ann_level_3 ground-truth fast path ---
    gt_exact = {
        "basal":                 "Basal",
        "secretory":             "Secretory",
        "multiciliated lineage": "Multiciliated lineage",
        "at2":                   "AT2",
        "macrophages":           "Macrophages",
        "t cell lineage":        "T cell lineage",
        "ec capillary":          "EC capillary",
    }
    if s in gt_exact:
        return gt_exact[s]

    # --- CellTypist Human_Lung_Atlas (ann_level_4) prediction handling ---

    # Macrophages FIRST (trap 1: catches "Monocyte-derived Mph")
    if "mph" in s or "macrophage" in s:
        return "Macrophages"

    # Monocytes are their own level-3 class -> no GT counterpart here
    if "monocyte" in s:
        return "Other"

    # Basal
    if s.startswith("basal") or s.startswith("suprabasal") or "hillock" in s:
        return "Basal"

    # Submucosal gland secretory is level-3 "Submucosal Secretory" -> Other.
    # Must precede the Secretory block.
    if s.startswith("smg"):
        return "Other"

    # Secretory
    if s.startswith("club") or s.startswith("goblet") \
            or "pre-tb secretory" in s:
        return "Secretory"

    # Multiciliated lineage
    if s.startswith("multiciliated") or s.startswith("deuterosomal"):
        return "Multiciliated lineage"

    # Alveolar epithelium
    if s.startswith("at2") or s == "at0":
        return "AT2"
    if s.startswith("at1"):
        return "Other"

    # Trap 2: NK / ILC are NOT T cell lineage
    if s.startswith("nk") or "innate lymphoid" in s or s.startswith("ilc"):
        return "Other"

    # T cell lineage
    if s.startswith("cd4 t") or s.startswith("cd8 t") \
            or s.startswith("t cells") or s.startswith("treg"):
        return "T cell lineage"

    # Trap 3: capillary EC before generic EC prefix
    if s.startswith("ec") and "capillary" in s:
        return "EC capillary"
    if s.startswith("ec ") or "lymphatic ec" in s:
        return "Other"

    return "Other"

def get_broad_mapper(model_path, data_dir=None):
    """
    Choose the correct broad-label mapper based on model and dataset.
    Route order (specific -> general):
      1. Pancreas islet models         -> pancreas mapper
      2. He Organ Atlas heart          -> He heart mapper
      3. Any other heart (Litvinukova) -> Litvinukova heart mapper
      4. CBMC data directory           -> CBMC mapper
      5. Everything else immune        -> PBMC mapper

    IMPORTANT: the He heart check must come BEFORE the generic heart
    check because "he_heart_ground_truth" also matches "heart".
    """
    m = str(model_path).lower()
    d = str(data_dir).lower() if data_dir else ""

    if "pancrea" in m or "islet" in m:
        print("[INFO] Using pancreas broad-label mapper.")
        return map_to_pancreas_broad
    if "lung" in m or "hlca" in m or "lung" in d or "hlca" in d:
        print("[INFO] Using HLCA human lung broad-label mapper.")
        return map_to_lung_broad

    if "he_heart" in d or "he_organ" in d:
        print("[INFO] Using He Organ Atlas heart broad-label mapper.")
        return map_to_he_heart_broad

    if "heart" in m or "cardiac" in m or "litvinukova" in d or "heart" in d:
        print("[INFO] Using Litvinukova heart broad-label mapper.")
        return map_to_heart_broad

    if "cbmc" in d:
        print("[INFO] Using CBMC broad-label mapper.")
        return map_to_cbmc_broad

    print("[INFO] Using PBMC broad-label mapper.")
    return map_to_pbmc_broad


BROAD_MAPPER = get_broad_mapper(MODEL_PATH, DATA_DIR)

def prepare_base_adata():
    print("Loading 10x data...")
    adata = sc.read_10x_mtx(DATA_DIR, var_names="gene_symbols", cache=True)
    adata.var_names_make_unique()

    print(f"Raw data: {adata.n_obs} cells x {adata.n_vars} genes")

    print("QC filtering...")
    adata.var["mt"] = adata.var_names.str.contains(MITO_REGEX_PATTERN, regex=True)
    sc.pp.calculate_qc_metrics(adata, qc_vars=["mt"], percent_top=None, log1p=False, inplace=True)

    sc.pp.filter_cells(adata, min_genes=MIN_GENES)
    sc.pp.filter_cells(adata, max_genes=MAX_GENES)
    adata = adata[adata.obs["pct_counts_mt"] < MAX_PCT_MT, :].copy()
    sc.pp.filter_genes(adata, min_cells=MIN_CELLS)

    print(f"After QC: {adata.n_obs} cells x {adata.n_vars} genes")

    print("Normalization...")
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    adata.raw = adata.copy()

    return adata


def load_ground_truth(adata):
    print("Loading FACS ground truth...")
    gt = pd.read_csv(GROUND_TRUTH_PATH)

    id_col, label_col = find_ground_truth_columns(gt)
    print(f"Using ground-truth ID column: {id_col}")
    print(f"Using ground-truth label column: {label_col}")

    gt = gt[[id_col, label_col]].copy()
    gt.columns = ["cell_id", "facs_label"]
    gt["cell_id"] = gt["cell_id"].astype(str).map(normalize_cell_id)
    gt["cell_id_stripped"] = gt["cell_id"].map(strip_10x_suffix)
    gt["facs_broad"] = gt["facs_label"].map(BROAD_MAPPER)

    # Try exact matching first
    obs = pd.DataFrame(index=adata.obs_names)
    obs["cell_id"] = obs.index.astype(str).map(normalize_cell_id)
    obs["cell_id_stripped"] = obs["cell_id"].map(strip_10x_suffix)

    exact = obs.merge(gt[["cell_id", "facs_label", "facs_broad"]], on="cell_id", how="left")
    exact.index = obs.index

    n_exact = exact["facs_broad"].notna().sum()

    # If exact matching is poor, try stripped barcode matching
    stripped = obs.merge(
        gt[["cell_id_stripped", "facs_label", "facs_broad"]],
        on="cell_id_stripped",
        how="left"
    )
    stripped.index = obs.index
    n_stripped = stripped["facs_broad"].notna().sum()

    if n_stripped > n_exact:
        print(f"Using stripped-barcode matching: {n_stripped} matched cells")
        adata.obs["facs_label"] = stripped["facs_label"].values
        adata.obs["facs_broad"] = stripped["facs_broad"].values
    else:
        print(f"Using exact barcode matching: {n_exact} matched cells")
        adata.obs["facs_label"] = exact["facs_label"].values
        adata.obs["facs_broad"] = exact["facs_broad"].values

    matched = adata.obs["facs_broad"].notna().sum()
    print(f"Matched FACS cells: {matched} / {adata.n_obs}")

    adata = adata[adata.obs["facs_broad"].notna(), :].copy()
    print(f"Using matched subset: {adata.n_obs} cells")
    print("Ground-truth broad-label composition:")
    print(adata.obs["facs_broad"].value_counts())

    # Optional: restrict evaluation to CBMC GT classes that CellTypist can
    # actually predict. Doublets and (predicted-only) classes have no
    # symmetric counterpart and inflate error.
    valid_gt = {"CD4_T", "CD8_T", "NK", "B",
                "CD14_Mono", "CD16_Mono", "pDC", "Progenitor"}

    if "cbmc" in str(DATA_DIR).lower():
        before = adata.n_obs
        adata = adata[adata.obs["facs_broad"].isin(valid_gt), :].copy()
        print(f"[INFO] Filtered doublets/other-only classes: "
              f"{before} -> {adata.n_obs} cells")
        print("Ground-truth broad-label composition after filtering:")
        print(adata.obs["facs_broad"].value_counts())

    return adata


def annotate_celltypist_once(adata):
    print("Running CellTypist annotation once on full log-normalized data...")
    model = models.Model.load(MODEL_PATH)

    adata_for_annot = adata.raw.to_adata() if adata.raw is not None else adata
    predictions = celltypist.annotate(
        adata_for_annot,
        model=model,
        majority_voting=False
    )

    adata.obs["ctpt_individual_prediction"] = (
        predictions.predicted_labels["predicted_labels"]
        .astype(str)
        .values
    )
    adata.obs["ctpt_confidence"] = predictions.probability_matrix.max(axis=1).values
    adata.obs["ctpt_individual_broad"] = adata.obs["ctpt_individual_prediction"].map(BROAD_MAPPER)

    print("CellTypist individual broad-label composition:")
    print(adata.obs["ctpt_individual_broad"].value_counts())

    return adata


def run_params_and_score_external(adata_base, params, trial_name="trial"):
    """
    Re-run HVG/PCA/neighbors/Leiden for one hyperparameter configuration
    and compute external FACS concordance metrics.
    """
    n_hvg = int(params["n_hvg"])
    n_pcs = int(params["n_pcs"])
    n_neighbors = int(params["n_neighbors"])
    resolution = float(params["resolution"])

    adata = adata_base.copy()

    # Two-step HVG selection, matching your command
    sc.pp.highly_variable_genes(
        adata,
        min_mean=HVG_MIN_MEAN,
        max_mean=HVG_MAX_MEAN,
        min_disp=HVG_MIN_DISP
    )

    hvg_df = adata.var[adata.var["highly_variable"]].copy()
    hvg_df = hvg_df.sort_values("dispersions_norm", ascending=False)

    if hvg_df.shape[0] == 0:
        raise ValueError(f"No HVGs found for {trial_name}")

    n_hvg_eff = min(n_hvg, hvg_df.shape[0])
    top_genes = hvg_df.index[:n_hvg_eff]

    adata.var["highly_variable"] = False
    adata.var.loc[top_genes, "highly_variable"] = True

    adata = adata[:, adata.var["highly_variable"]].copy()

    sc.pp.scale(adata, max_value=10)

    n_pcs_to_compute = min(N_PCS_COMPUTE, adata.n_obs - 1, adata.n_vars - 1)
    n_pcs_eff = min(n_pcs, n_pcs_to_compute)

    if n_pcs_eff < 2:
        raise ValueError(f"Too few PCs for {trial_name}")

    sc.tl.pca(
        adata,
        svd_solver="arpack",
        n_comps=n_pcs_to_compute,
        random_state=SEED
    )

    sc.pp.neighbors(
        adata,
        n_neighbors=n_neighbors,
        n_pcs=n_pcs_eff,
        random_state=SEED
    )

    sc.tl.leiden(
        adata,
        resolution=resolution,
        random_state=SEED
    )

    # Cluster consensus CellTypist label
    cluster2label = (
        adata.obs
        .groupby("leiden")["ctpt_individual_prediction"]
        .agg(lambda x: x.value_counts().idxmax())
        .to_dict()
    )

    adata.obs["ctpt_consensus_prediction"] = adata.obs["leiden"].map(cluster2label)
    adata.obs["ctpt_consensus_broad"] = adata.obs["ctpt_consensus_prediction"].map(BROAD_MAPPER)

    # External FACS metrics
    y_true = adata.obs["facs_broad"].astype(str).values
    leiden_labels = adata.obs["leiden"].astype(str).values
    y_pred_broad = adata.obs["ctpt_consensus_broad"].astype(str).values

    leiden_ari = adjusted_rand_score(y_true, leiden_labels)
    leiden_nmi = normalized_mutual_info_score(y_true, leiden_labels)

    homogeneity, completeness, v_measure = homogeneity_completeness_v_measure(
        y_true,
        leiden_labels
    )

    broad_acc = accuracy_score(y_true, y_pred_broad)

    facs_classes = sorted(pd.Series(y_true).unique().tolist())
    broad_macro_f1 = f1_score(
        y_true,
        y_pred_broad,
        labels=facs_classes,
        average="macro",
        zero_division=0
    )

    mean_confidence = float(adata.obs["ctpt_confidence"].mean())

    return {
        "matched_cells": adata.n_obs,
        "n_facs_labels": len(np.unique(y_true)),
        "n_leiden_clusters": adata.obs["leiden"].nunique(),
        "external_leiden_ari": leiden_ari,
        "external_leiden_nmi": leiden_nmi,
        "external_homogeneity": homogeneity,
        "external_completeness": completeness,          # <-- restore
        # "external_v_measure": v_measure,              # <-- drop (== NMI arithmetic)
        "external_celltypist_broad_accuracy": broad_acc,
        "external_celltypist_broad_macro_f1": broad_macro_f1,
        "external_mean_confidence": mean_confidence,
        "effective_n_hvg": n_hvg_eff,
        "effective_n_pcs": n_pcs_eff
    }


def calculate_spearman(df, internal_cols, external_cols):
    rows = []

    for i_col in internal_cols:
        for e_col in external_cols:
            tmp = df[[i_col, e_col]].dropna()
            if tmp.shape[0] < 3:
                rho, pval = np.nan, np.nan
            else:
                rho, pval = spearmanr(tmp[i_col], tmp[e_col])

            rows.append({
                "internal_metric": i_col,
                "external_facs_metric": e_col,
                "spearman_rho": rho,
                "p_value": pval,
                "n_trials": tmp.shape[0]
            })

    return pd.DataFrame(rows)


def summarize_top_bottom_decile(df, objective_col="yield_score_target"):
    df2 = df.dropna(subset=[objective_col]).copy()
    n = len(df2)

    if n < 10:
        raise ValueError("Need at least 10 trials for top/bottom decile analysis.")

    decile_n = max(1, int(np.ceil(n * 0.10)))

    df_sorted = df2.sort_values(objective_col, ascending=True)
    bottom = df_sorted.head(decile_n).copy()
    top = df_sorted.tail(decile_n).copy()

    metrics = [
        "external_leiden_ari",
        "external_leiden_nmi",
        "external_homogeneity",
        "external_completeness",                # already in the list, good
        # "external_v_measure",                 # ensure this stays commented
        "external_celltypist_broad_accuracy",
        "external_celltypist_broad_macro_f1",
        "external_mean_confidence"
    ]

    rows = []
    for m in metrics:
        rows.append({
            "metric": m,
            "bottom_decile_mean": bottom[m].mean(),
            "bottom_decile_sd": bottom[m].std(),
            "top_decile_mean": top[m].mean(),
            "top_decile_sd": top[m].std(),
            "difference_top_minus_bottom": top[m].mean() - bottom[m].mean(),
            "n_bottom": len(bottom),
            "n_top": len(top)
        })

    return pd.DataFrame(rows)


def plot_correlation_heatmap(corr_df):
    pivot = corr_df.pivot(
        index="internal_metric",
        columns="external_facs_metric",
        values="spearman_rho"
    )

    plt.figure(figsize=(14, 6))
    sns.heatmap(
        pivot,
        annot=True,
        fmt=".2f",
        cmap="vlag",
        center=0,
        linewidths=0.5,
        cbar_kws={"label": "Spearman rho"}
    )
    plt.title("Correlation between scBOA internal objectives and FACS external metrics")
    plt.xlabel("External FACS metric")
    plt.ylabel("Internal scBOA metric")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()

    out = os.path.join(FIGURE_DIR, "internal_vs_facs_spearman_heatmap.png")
    plt.savefig(out, dpi=300)
    plt.savefig(out.replace(".png", ".pdf"))
    plt.close()
    print(f"Saved: {out}")

def plot_raw_partial_correlation_heatmaps(
    raw_df,
    partial_cluster_df,
    partial_cluster_resolution_df,
    internal_metric_order=None,
    external_metric_order=None,
):
    """
    Plot side-by-side heatmaps comparing raw Spearman correlations with
    partial Spearman correlations controlling for clustering granularity.

    Panels:
      1. Raw Spearman rho
      2. Partial rho controlling for n_leiden_clusters
      3. Partial rho controlling for n_leiden_clusters + resolution
    """
    if internal_metric_order is None:
        internal_metric_order = [
            "yield_score_target",
            "balanced_score_gmean",
            "weighted_mean_cas",
            "simple_mean_cas",
            "mean_mcs",
            "mean_f1",
            "mean_confidence",
            "silhouette_score",
        ]

    if external_metric_order is None:
        external_metric_order = [
            "external_leiden_ari",
            "external_leiden_nmi",
            "external_homogeneity",
            "external_completeness",
            "external_celltypist_broad_accuracy",
            "external_celltypist_broad_macro_f1",
            "external_mean_confidence",
        ]

    def make_pivot(df, value_col):
        pivot = df.pivot(
            index="internal_metric",
            columns="external_facs_metric",
            values=value_col,
        )

        rows = [r for r in internal_metric_order if r in pivot.index]
        cols = [c for c in external_metric_order if c in pivot.columns]
        return pivot.loc[rows, cols]

    raw_pivot = make_pivot(raw_df, "spearman_rho")
    partial_cluster_pivot = make_pivot(
        partial_cluster_df,
        "partial_spearman_rho",
    )
    partial_cluster_resolution_pivot = make_pivot(
        partial_cluster_resolution_df,
        "partial_spearman_rho",
    )

    pivots = [
        raw_pivot,
        partial_cluster_pivot,
        partial_cluster_resolution_pivot,
    ]

    titles = [
        "Raw Spearman rho",
        "Partial rho controlling for cluster count",
        "Partial rho controlling for cluster count + resolution",
    ]

    fig, axes = plt.subplots(
        1,
        3,
        figsize=(24, 7),
        sharey=True,
        constrained_layout=True,
    )

    for ax, pivot, title in zip(axes, pivots, titles):
        sns.heatmap(
            pivot,
            annot=True,
            fmt=".2f",
            cmap="vlag",
            center=0,
            vmin=-1,
            vmax=1,
            linewidths=0.5,
            cbar=ax is axes[-1],
            cbar_kws={"label": "Correlation coefficient"} if ax is axes[-1] else None,
            ax=ax,
        )
        ax.set_title(title)
        ax.set_xlabel("External ground-truth metric")
        ax.set_ylabel("Internal scBOA metric" if ax is axes[0] else "")
        ax.tick_params(axis="x", rotation=45)
        for label in ax.get_xticklabels():
            label.set_horizontalalignment("right")

    out = os.path.join(
        FIGURE_DIR,
        "internal_vs_facs_raw_partial_spearman_heatmaps.png",
    )
    plt.savefig(out, dpi=300)
    plt.savefig(out.replace(".png", ".pdf"))
    plt.close()
    print(f"Saved: {out}")

def plot_scatter_panels(df):
    pairs = [
        ("yield_score_target", "external_celltypist_broad_accuracy"),
        ("yield_score_target", "external_celltypist_broad_macro_f1"),
        ("yield_score_target", "external_leiden_ari"),
        ("yield_score_target", "external_leiden_nmi"),
        ("weighted_mean_cas", "external_celltypist_broad_accuracy"),
        ("simple_mean_cas", "external_leiden_ari"),
        ("mean_mcs", "external_celltypist_broad_macro_f1"),
        ("mean_f1", "external_celltypist_broad_macro_f1"),
    ]

    ncols = 4
    nrows = int(np.ceil(len(pairs) / ncols))

    fig, axes = plt.subplots(nrows, ncols, figsize=(20, 5 * nrows))
    axes = axes.flatten()

    # >>> INSERT: collect per-point and per-panel summaries for CSV export
    points_rows = []
    stats_rows = []
    # <<<

    for ax, (x, y) in zip(axes, pairs):
        if x not in df.columns or y not in df.columns:
            ax.axis("off")
            # >>> INSERT: still record that this panel was requested but skipped
            stats_rows.append({
                "panel_x": x,
                "panel_y": y,
                "n_trials": 0,
                "spearman_rho": np.nan,
                "p_value": np.nan,
                "status": "skipped_missing_column"
            })
            # <<<
            continue

        sns.regplot(
            data=df,
            x=x,
            y=y,
            ax=ax,
            scatter_kws={"s": 35, "alpha": 0.75},
            line_kws={"color": "red"}
        )

        tmp = df[[x, y]].dropna()
        if tmp.shape[0] >= 3:
            rho, p = spearmanr(tmp[x], tmp[y])
            ax.set_title(f"{x} vs {y}\nSpearman rho={rho:.2f}, p={p:.2g}")
        else:
            ax.set_title(f"{x} vs {y}")
            rho, p = np.nan, np.nan

        ax.set_xlabel(x)
        ax.set_ylabel(y)

        # >>> INSERT: per-panel stats row
        stats_rows.append({
            "panel_x": x,
            "panel_y": y,
            "n_trials": int(tmp.shape[0]),
            "spearman_rho": rho,
            "p_value": p,
            "status": "ok" if tmp.shape[0] >= 3 else "insufficient_points"
        })

        # >>> INSERT: per-point rows (long format, one row per trial per panel)
        # Preserve trial identifiers if present so the CSV can be joined back
        # to the trial-level table.
        id_cols = [
            c for c in ["strategy", "call_number", "trial_id"]
            if c in df.columns
        ]
        panel_points = df[id_cols + [x, y]].copy()
        panel_points = panel_points.rename(columns={x: "x_value", y: "y_value"})
        panel_points["panel_x"] = x
        panel_points["panel_y"] = y
        # Keep only rows where both x and y are finite, matching what was plotted
        panel_points = panel_points.dropna(subset=["x_value", "y_value"])
        points_rows.append(panel_points)
        # <<<

    for j in range(len(pairs), len(axes)):
        axes[j].axis("off")

    plt.tight_layout()
    out = os.path.join(FIGURE_DIR, "internal_vs_facs_scatter_panels.png")
    plt.savefig(out, dpi=300)
    plt.savefig(out.replace(".png", ".pdf"))
    plt.close()
    print(f"Saved: {out}")

    # >>> INSERT: write the two companion CSVs alongside the figure
    stats_df = pd.DataFrame(stats_rows)
    stats_out = os.path.join(
        OUTPUT_DIR, "internal_vs_facs_scatter_panels_stats.csv"
    )
    stats_df.to_csv(stats_out, index=False)
    print(f"Saved: {stats_out}")

    if points_rows:
        points_df = pd.concat(points_rows, ignore_index=True)
        # Reorder for readability
        lead = [c for c in ["panel_x", "panel_y",
                            "strategy", "call_number", "trial_id"]
                if c in points_df.columns]
        rest = [c for c in points_df.columns if c not in lead]
        points_df = points_df[lead + rest]
        points_out = os.path.join(
            OUTPUT_DIR, "internal_vs_facs_scatter_panels_points.csv"
        )
        points_df.to_csv(points_out, index=False)
        print(f"Saved: {points_out}")
    # <<<

def plot_x_vs_external(df, x_col, x_label=None, tag=None):
    """
    Generic scatter of a per-trial variable (x_col) vs each external
    ground-truth metric. Produces both a PNG/PDF figure and a stats CSV.

    Parameters
    ----------
    x_col : str
        Column name in `df` to plot on the x-axis, e.g. "n_leiden_clusters"
        or "resolution".
    x_label : str or None
        Axis label; defaults to x_col.
    tag : str or None
        Filename tag; defaults to x_col.
    """
    if x_col not in df.columns:
        print(f"[WARN] Column {x_col} not found; skipping x-vs-external plot.")
        return

    x_label = x_label or x_col
    tag = tag or x_col

    external_metrics = [
        "external_leiden_ari",
        "external_leiden_nmi",
        "external_homogeneity",
        "external_completeness",
        "external_celltypist_broad_accuracy",
        "external_celltypist_broad_macro_f1",
    ]
    external_metrics = [m for m in external_metrics if m in df.columns]

    ncols = 3
    nrows = int(np.ceil(len(external_metrics) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4.5 * nrows))
    axes = np.atleast_1d(axes).flatten()

    stats_rows = []
    for ax, y in zip(axes, external_metrics):
        tmp = df[[x_col, y]].dropna()

        if tmp.shape[0] >= 3:
            sns.regplot(
                data=tmp, x=x_col, y=y, ax=ax,
                scatter_kws={"s": 30, "alpha": 0.7},
                line_kws={"color": "red"},
            )
            rho, p = spearmanr(tmp[x_col], tmp[y])
            ax.set_title(
                f"{x_label} vs {y}\nSpearman rho={rho:.2f}, p={p:.2g}"
            )
            stats_rows.append({
                "external_metric": y,
                f"spearman_rho_vs_{tag}": rho,
                "p_value": p,
                "n_trials": int(tmp.shape[0]),
            })
        else:
            ax.set_title(f"{x_label} vs {y}")
            stats_rows.append({
                "external_metric": y,
                f"spearman_rho_vs_{tag}": np.nan,
                "p_value": np.nan,
                "n_trials": int(tmp.shape[0]),
            })

        ax.set_xlabel(x_label)
        ax.set_ylabel(y)

    for j in range(len(external_metrics), len(axes)):
        axes[j].axis("off")

    plt.tight_layout()
    out = os.path.join(FIGURE_DIR, f"{tag}_vs_external_metrics.png")
    plt.savefig(out, dpi=300)
    plt.savefig(out.replace(".png", ".pdf"))
    plt.close()
    print(f"Saved: {out}")

    stats_df = pd.DataFrame(stats_rows)
    stats_out = os.path.join(OUTPUT_DIR, f"{tag}_vs_external_metrics_stats.csv")
    stats_df.to_csv(stats_out, index=False)
    print(f"Saved: {stats_out}")


# Backward-compatible alias so existing call sites still work.
def plot_cluster_number_vs_external(df):
    plot_x_vs_external(
        df,
        x_col="n_leiden_clusters",
        x_label="Number of Leiden clusters",
        tag="cluster_number",
    )


def compute_partial_spearman(df, internal_cols, external_cols,
                             control_cols="n_leiden_clusters"):
    """
    Partial Spearman correlation between each internal objective and each
    external ground-truth metric, controlling for one or more variables.

    Parameters
    ----------
    control_cols : str or list of str
        Column name(s) to control for. If a list is passed, all listed
        variables are residualized out simultaneously via multivariable
        rank-regression on the design matrix [1, rank(z1), rank(z2), ...].

    Method
    ------
    1. Rank-transform the internal metric, external metric, and each
       control variable.
    2. Fit ordinary least squares on the ranks to regress internal and
       external independently against the control ranks (with intercept).
    3. Correlate the OLS residuals via Pearson correlation.

    Rationale for adding `resolution` as a second control alongside
    `n_leiden_clusters`: `resolution` is the Leiden knob the optimizer
    tunes; `n_leiden_clusters` is a downstream consequence of that knob
    combined with the neighbor graph. Controlling only for the cluster
    count leaves an open confounding path
        objective <- resolution -> external_metric
    that survives even when trials with the same n_leiden_clusters are
    compared. Adding `resolution` closes that path.

    A large drop from the raw Spearman rho to the partial rho indicates
    the raw correlation was mediated by the control variable(s).
    """
    if isinstance(control_cols, str):
        control_cols = [control_cols]
    control_cols = list(control_cols)

    controlling_label = "+".join(control_cols)

    rows = []
    for i_col in internal_cols:
        for e_col in external_cols:
            needed = [i_col, e_col] + control_cols
            tmp = df[needed].dropna()
            n = tmp.shape[0]

            # Need at least (k+2) trials with variation in each control
            k = len(control_cols)
            if n < max(5, k + 3) or any(
                tmp[c].nunique() < 2 for c in control_cols
            ):
                rows.append({
                    "internal_metric": i_col,
                    "external_facs_metric": e_col,
                    "controlling_for": controlling_label,
                    "raw_spearman_rho": np.nan,
                    "partial_spearman_rho": np.nan,
                    "n_trials": n,
                })
                continue

            rx = pd.Series(tmp[i_col]).rank().values.astype(float)
            ry = pd.Series(tmp[e_col]).rank().values.astype(float)

            # Design matrix of ranked controls with intercept
            Z = np.column_stack([
                np.ones(n),
                *[pd.Series(tmp[c]).rank().values.astype(float)
                  for c in control_cols],
            ])

            # OLS residuals for internal and external against Z
            try:
                beta_x, *_ = np.linalg.lstsq(Z, rx, rcond=None)
                beta_y, *_ = np.linalg.lstsq(Z, ry, rcond=None)
            except np.linalg.LinAlgError:
                rows.append({
                    "internal_metric": i_col,
                    "external_facs_metric": e_col,
                    "controlling_for": controlling_label,
                    "raw_spearman_rho": np.nan,
                    "partial_spearman_rho": np.nan,
                    "n_trials": n,
                })
                continue

            ex = rx - Z @ beta_x
            ey = ry - Z @ beta_y

            num = float((ex * ey).sum())
            den = float(np.sqrt((ex ** 2).sum() * (ey ** 2).sum()))
            partial_rho = num / den if den > 0 else np.nan

            raw_rho, _ = spearmanr(tmp[i_col], tmp[e_col])

            rows.append({
                "internal_metric": i_col,
                "external_facs_metric": e_col,
                "controlling_for": controlling_label,
                "raw_spearman_rho": raw_rho,
                "partial_spearman_rho": partial_rho,
                "n_trials": n,
            })

    return pd.DataFrame(rows)

def plot_top_bottom_bar(top_bottom_df):
    plot_metrics = [
        "external_leiden_ari",
        "external_leiden_nmi",
        "external_celltypist_broad_accuracy",
        "external_celltypist_broad_macro_f1",
        "external_mean_confidence"
    ]

    df = top_bottom_df[top_bottom_df["metric"].isin(plot_metrics)].copy()

    long_rows = []
    for _, row in df.iterrows():
        long_rows.append({
            "metric": row["metric"],
            "group": "Bottom objective decile",
            "mean": row["bottom_decile_mean"],
            "sd": row["bottom_decile_sd"]
        })
        long_rows.append({
            "metric": row["metric"],
            "group": "Top objective decile",
            "mean": row["top_decile_mean"],
            "sd": row["top_decile_sd"]
        })

    long_df = pd.DataFrame(long_rows)

    plt.figure(figsize=(14, 6))
    sns.barplot(
        data=long_df,
        x="metric",
        y="mean",
        hue="group"
    )
    plt.xticks(rotation=35, ha="right")
    plt.ylabel("Mean metric value")
    plt.xlabel("")
    plt.title("External FACS concordance in top vs bottom scBOA objective deciles")
    plt.tight_layout()

    out = os.path.join(FIGURE_DIR, "top_vs_bottom_objective_decile_facs_metrics.png")
    plt.savefig(out, dpi=300)
    plt.savefig(out.replace(".png", ".pdf"))
    plt.close()
    print(f"Saved: {out}")

def plot_partial_correlation_bars(
    corr_df,
    partial_cc_df,
    partial_cc_res_df,
    internal_metric="yield_score_target",
    external_metrics_order=None,
):
    """
    Panel D: grouped bar chart comparing Raw Spearman rho vs
    partial rho (controlling for cluster count) vs partial rho
    (controlling for cluster count + resolution), for a single
    internal objective across the six external metrics.

    Inputs are the three DataFrames already computed in main():
      corr_df           : spearman_internal_vs_facs.csv contents
      partial_cc_df     : ..._partial_controlling_cluster_count.csv
      partial_cc_res_df : ..._partial_controlling_cluster_count_and_resolution.csv
    """
    if external_metrics_order is None:
        external_metrics_order = [
            "external_celltypist_broad_accuracy",
            "external_celltypist_broad_macro_f1",
            "external_homogeneity",
            "external_leiden_ari",
            "external_leiden_nmi",
            "external_completeness",
        ]

    pretty = {
        "external_celltypist_broad_accuracy": "CellTypist\naccuracy",
        "external_celltypist_broad_macro_f1": "CellTypist\nmacro-F1",
        "external_homogeneity":               "Homogeneity",
        "external_leiden_ari":                "Leiden ARI",
        "external_leiden_nmi":                "Leiden NMI",
        "external_completeness":              "Completeness",
    }

    def _lookup_raw(metric):
        sub = corr_df[
            (corr_df["internal_metric"] == internal_metric)
            & (corr_df["external_facs_metric"] == metric)
        ]
        return float(sub["spearman_rho"].iloc[0]) if not sub.empty else np.nan

    def _lookup_partial(df, metric):
        sub = df[
            (df["internal_metric"] == internal_metric)
            & (df["external_facs_metric"] == metric)
        ]
        return float(sub["partial_spearman_rho"].iloc[0]) if not sub.empty else np.nan

    raw_vals  = [_lookup_raw(m)               for m in external_metrics_order]
    part_cc   = [_lookup_partial(partial_cc_df, m)     for m in external_metrics_order]
    part_ccr  = [_lookup_partial(partial_cc_res_df, m) for m in external_metrics_order]

    x = np.arange(len(external_metrics_order))
    w = 0.27

    fig, ax = plt.subplots(figsize=(10, 5.2))
    b1 = ax.bar(x - w, raw_vals, w, color="#9e9e9e", label="Raw ρ")
    b2 = ax.bar(x,     part_cc,  w, color="#4a90d9",
                label="Partial ρ | cluster count")
    b3 = ax.bar(x + w, part_ccr, w, color="#1b4f8b",
                label="Partial ρ | cluster count + resolution")

    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels([pretty.get(m, m) for m in external_metrics_order],
                       rotation=0, ha="center", fontsize=9)
    ax.set_ylabel(f"Spearman ρ  (internal = {internal_metric})")
    ax.set_title(
        "Panel D: raw vs partial Spearman correlations between the scBOA "
        "objective\nand external ground-truth metrics"
    )
    ax.legend(loc="best", fontsize=9, frameon=False)

    for bars in (b1, b2, b3):
        for rect in bars:
            h = rect.get_height()
            if np.isnan(h):
                continue
            va = "bottom" if h >= 0 else "top"
            ax.text(rect.get_x() + rect.get_width() / 2,
                    h + (0.015 if h >= 0 else -0.015),
                    f"{h:.2f}", ha="center", va=va, fontsize=8)

    ax.set_ylim(min(-1.0, min(raw_vals + part_cc + part_ccr) - 0.1),
                max( 1.0, max(raw_vals + part_cc + part_ccr) + 0.15))

    plt.tight_layout()
    out = os.path.join(FIGURE_DIR, "partial_correlation_bars_panelD.png")
    plt.savefig(out, dpi=300)
    plt.savefig(out.replace(".png", ".pdf"))
    plt.close()
    print(f"Saved: {out}")

    # Companion CSV so the figure numbers are auditable
    csv_rows = []
    for m, r, pc, pcr in zip(external_metrics_order, raw_vals, part_cc, part_ccr):
        csv_rows.append({
            "internal_metric": internal_metric,
            "external_metric": m,
            "raw_spearman_rho": r,
            "partial_rho_ctrl_cluster_count": pc,
            "partial_rho_ctrl_cluster_count_and_resolution": pcr,
        })
    csv_out = os.path.join(OUTPUT_DIR, "partial_correlation_bars_panelD.csv")
    pd.DataFrame(csv_rows).to_csv(csv_out, index=False)
    print(f"Saved: {csv_out}")
def plot_positive_control_baselines(
    corr_df,
    partial_cc_df,
    partial_cc_res_df,
    target_internal="yield_score_target",
    baseline_internals=("silhouette_score", "mean_f1", "mean_mcs"),
    external_metric="external_leiden_ari",
):
    """
    Panel E: positive-control comparison.

    For a single external metric (default: Leiden ARI, the partition-based
    metric most susceptible to the cluster-count artifact), plot raw
    Spearman rho vs partial rho (cluster count) vs partial rho (cluster
    count + resolution) for the scBOA target objective alongside several
    baseline internal metrics.

    The point of the figure is falsification: partial-correlation control
    should collapse or invert any baseline whose apparent concordance
    with the external metric is mediated by Leiden granularity. A target
    objective that *does not* collapse under the same control is direct
    evidence that its concordance is not a granularity artifact.

    Companion CSV
    -------------
    Written for the same external_metric, so every bar in the figure is
    auditable. A second, longer CSV is also written covering all six
    external metrics x all listed internal metrics, for readers who want
    to see the full table.
    """
    internals = [target_internal] + [
        m for m in baseline_internals if m in corr_df["internal_metric"].unique()
    ]

    if not internals:
        print("[WARN] No matching internal metrics found for Panel E; skipping.")
        return

    def _raw(i_col, e_col):
        sub = corr_df[
            (corr_df["internal_metric"] == i_col)
            & (corr_df["external_facs_metric"] == e_col)
        ]
        return float(sub["spearman_rho"].iloc[0]) if not sub.empty else np.nan

    def _partial(df, i_col, e_col):
        sub = df[
            (df["internal_metric"] == i_col)
            & (df["external_facs_metric"] == e_col)
        ]
        return float(sub["partial_spearman_rho"].iloc[0]) if not sub.empty else np.nan

    raw_vals = [_raw(m, external_metric) for m in internals]
    part_cc = [_partial(partial_cc_df, m, external_metric) for m in internals]
    part_ccr = [_partial(partial_cc_res_df, m, external_metric) for m in internals]

    # --- plot ---
    x = np.arange(len(internals))
    w = 0.27
    fig, ax = plt.subplots(figsize=(1.9 * len(internals) + 3.0, 5.2))

    b1 = ax.bar(x - w, raw_vals, w, color="#9e9e9e", label="Raw ρ")
    b2 = ax.bar(x,     part_cc,  w, color="#4a90d9",
                label="Partial ρ | cluster count")
    b3 = ax.bar(x + w, part_ccr, w, color="#1b4f8b",
                label="Partial ρ | cluster count + resolution")

    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(internals, rotation=15, ha="right", fontsize=9)
    ax.set_ylabel(f"Spearman ρ  vs  {external_metric}")
    ax.set_title(
        f"Panel E: positive-control falsification of the granularity-artifact\n"
        f"hypothesis (external metric = {external_metric})"
    )
    ax.legend(loc="best", fontsize=9, frameon=False)

    # highlight the target internal metric
    ax.axvspan(-0.5, 0.5, color="#fff3b0", alpha=0.35, zorder=0)

    for bars in (b1, b2, b3):
        for rect in bars:
            h = rect.get_height()
            if np.isnan(h):
                continue
            va = "bottom" if h >= 0 else "top"
            ax.text(rect.get_x() + rect.get_width() / 2,
                    h + (0.015 if h >= 0 else -0.015),
                    f"{h:.2f}", ha="center", va=va, fontsize=8)

    all_vals = [v for v in raw_vals + part_cc + part_ccr if not np.isnan(v)]
    if all_vals:
        ax.set_ylim(min(-1.0, min(all_vals) - 0.1),
                    max( 1.0, max(all_vals) + 0.15))

    plt.tight_layout()
    out = os.path.join(FIGURE_DIR, "positive_control_baselines_panelE.png")
    plt.savefig(out, dpi=300)
    plt.savefig(out.replace(".png", ".pdf"))
    plt.close()
    print(f"Saved: {out}")

    # --- CSV shown in the figure ---
    csv_rows = []
    for m, r, pc, pcr in zip(internals, raw_vals, part_cc, part_ccr):
        csv_rows.append({
            "internal_metric": m,
            "external_metric": external_metric,
            "raw_spearman_rho": r,
            "partial_rho_ctrl_cluster_count": pc,
            "partial_rho_ctrl_cluster_count_and_resolution": pcr,
            "role": "target" if m == target_internal else "positive_control",
        })
    csv_out = os.path.join(
        OUTPUT_DIR, "positive_control_baselines_panelE.csv"
    )
    pd.DataFrame(csv_rows).to_csv(csv_out, index=False)
    print(f"Saved: {csv_out}")

    # --- full table across all external metrics (for the appendix / audit) ---
    all_external = [
        "external_leiden_ari",
        "external_leiden_nmi",
        "external_homogeneity",
        "external_completeness",
        "external_celltypist_broad_accuracy",
        "external_celltypist_broad_macro_f1",
    ]
    full_rows = []
    for m in internals:
        for e in all_external:
            full_rows.append({
                "internal_metric": m,
                "external_metric": e,
                "raw_spearman_rho": _raw(m, e),
                "partial_rho_ctrl_cluster_count": _partial(partial_cc_df, m, e),
                "partial_rho_ctrl_cluster_count_and_resolution":
                    _partial(partial_cc_res_df, m, e),
                "role": "target" if m == target_internal else "positive_control",
            })
    full_out = os.path.join(
        OUTPUT_DIR, "positive_control_baselines_full_table.csv"
    )
    pd.DataFrame(full_rows).to_csv(full_out, index=False)
    print(f"Saved: {full_out}")
# =============================================================================
# Main
# =============================================================================

def main():
    print("=" * 80)
    print("Objective-to-FACS concordance analysis")
    print("=" * 80)

    yield_df = pd.read_csv(YIELD_CSV)
    print(f"Loaded yield CSV: {YIELD_CSV}")
    print(f"Number of trials in yield CSV: {yield_df.shape[0]}")

    required_param_cols = ["n_hvg", "n_pcs", "n_neighbors", "resolution"]
    for col in required_param_cols:
        if col not in yield_df.columns:
            raise ValueError(f"Required column missing from yield CSV: {col}")

    adata = prepare_base_adata()
    adata = load_ground_truth(adata)
    adata = annotate_celltypist_once(adata)
    print("\n=== Sanity check: crosstab of GT vs individual CellTypist broad ===")
    print(pd.crosstab(
        adata.obs["facs_broad"],
        adata.obs["ctpt_individual_broad"],
        dropna=False
    ))
    print("=== end sanity check ===\n")

    rows = []

    print("\nRe-running each scBOA trial and computing FACS concordance...")
    for idx, row in yield_df.iterrows():
        params = {
            "n_hvg": int(row["n_hvg"]),
            "n_pcs": int(row["n_pcs"]),
            "n_neighbors": int(row["n_neighbors"]),
            "resolution": float(row["resolution"])
        }

        trial_id = row.get("call_number", idx + 1)
        strategy = row.get("strategy", "NA")

        print(
            f"[{idx + 1}/{yield_df.shape[0]}] "
            f"strategy={strategy}, call={trial_id}, "
            f"HVG={params['n_hvg']}, PCs={params['n_pcs']}, "
            f"neighbors={params['n_neighbors']}, res={params['resolution']:.3f}"
        )

        try:
            external = run_params_and_score_external(
                adata,
                params,
                trial_name=f"{strategy}_{trial_id}"
            )
        except Exception as e:
            print(f"  [WARNING] Trial failed: {e}")
            external = {
                "matched_cells": np.nan,
                "n_facs_labels": np.nan,
                "n_leiden_clusters": np.nan,
                "external_leiden_ari": np.nan,
                "external_leiden_nmi": np.nan,
                "external_homogeneity": np.nan,
                "external_completeness": np.nan,        # <-- restore
                # "external_v_measure": np.nan,         # <-- drop
                "external_celltypist_broad_accuracy": np.nan,
                "external_celltypist_broad_macro_f1": np.nan,
                "external_mean_confidence": np.nan,
                "effective_n_hvg": np.nan,
                "effective_n_pcs": np.nan
            }

        combined = row.to_dict()
        combined.update(external)
        rows.append(combined)

    trial_df = pd.DataFrame(rows)

    trial_out = os.path.join(OUTPUT_DIR, "trial_objective_facs_concordance.csv")
    trial_df.to_csv(trial_out, index=False)
    print(f"\nSaved trial-level concordance table: {trial_out}")

    print("\n--- Spearman(objective, n_leiden_clusters) ---")
    obj_vs_cc_rows = []
    for obj_col in [
        "yield_score_target",
        "balanced_score_gmean",
        "weighted_mean_cas",
        "simple_mean_cas",
        "mean_mcs",
        "mean_f1",
    ]:
        if obj_col not in trial_df.columns:
            continue
        tmp = trial_df[[obj_col, "n_leiden_clusters"]].dropna()
        if tmp.shape[0] < 3:
            r, p = np.nan, np.nan
        else:
            r, p = spearmanr(tmp[obj_col], tmp["n_leiden_clusters"])
        print(f"  {obj_col:25s}  rho={r:+.3f}   p={p:.2e}   n={tmp.shape[0]}")
        obj_vs_cc_rows.append({
            "internal_metric": obj_col,
            "spearman_rho_vs_cluster_count": r,
            "p_value": p,
            "n_trials": int(tmp.shape[0]),
        })
    obj_vs_cc_df = pd.DataFrame(obj_vs_cc_rows)
    obj_vs_cc_out = os.path.join(
        OUTPUT_DIR, "objective_vs_cluster_count_stats.csv"
    )
    obj_vs_cc_df.to_csv(obj_vs_cc_out, index=False)
    print(f"Saved: {obj_vs_cc_out}")

    print("\n--- Spearman(objective, resolution) ---")
    obj_vs_res_rows = []
    for obj_col in [
        "yield_score_target",
        "balanced_score_gmean",
        "weighted_mean_cas",
        "simple_mean_cas",
        "mean_mcs",
        "mean_f1",
    ]:
        if obj_col not in trial_df.columns:
            continue
        tmp = trial_df[[obj_col, "resolution"]].dropna()
        if tmp.shape[0] < 3:
            r, p = np.nan, np.nan
        else:
            r, p = spearmanr(tmp[obj_col], tmp["resolution"])
        print(f"  {obj_col:25s}  rho={r:+.3f}   p={p:.2e}   n={tmp.shape[0]}")
        obj_vs_res_rows.append({
            "internal_metric": obj_col,
            "spearman_rho_vs_resolution": r,
            "p_value": p,
            "n_trials": int(tmp.shape[0]),
        })
    obj_vs_res_df = pd.DataFrame(obj_vs_res_rows)
    obj_vs_res_out = os.path.join(
        OUTPUT_DIR, "objective_vs_resolution_stats.csv"
    )
    obj_vs_res_df.to_csv(obj_vs_res_out, index=False)
    print(f"Saved: {obj_vs_res_out}")

    # -------------------------------------------------------------------------
    # Spearman correlations
    # -------------------------------------------------------------------------
    internal_cols = [
        c for c in [
            "yield_score_target",
            "balanced_score_gmean",
            "weighted_mean_cas",
            "simple_mean_cas",
            "mean_mcs",
            "mean_f1",
            "mean_confidence",
            "silhouette_score"
        ] if c in trial_df.columns
    ]

    external_cols = [
        "external_leiden_ari",
        "external_leiden_nmi",
        "external_homogeneity",
        "external_completeness",                        # kept
        # "external_v_measure",                         # dropped
        "external_celltypist_broad_accuracy",
        "external_celltypist_broad_macro_f1",
        "external_mean_confidence"
    ]

    corr_df = calculate_spearman(trial_df, internal_cols, external_cols)

    corr_out = os.path.join(OUTPUT_DIR, "spearman_internal_vs_facs.csv")
    corr_df.to_csv(corr_out, index=False)
    print(f"Saved Spearman correlation table: {corr_out}")

    # -------------------------------------------------------------------------
    # Partial Spearman correlations
    #
    # Two control sets are reported so the reader can distinguish an artifact
    # driven by cluster count alone from one driven by the resolution knob
    # itself:
    #
    #   (A) Controlling for n_leiden_clusters only.
    #       Answers: "Among trials that produced the same number of clusters,
    #                does the internal objective still track external biology?"
    #
    #   (B) Controlling for both n_leiden_clusters AND resolution.
    #       Answers: "Among trials with the same cluster count AND the same
    #                Leiden resolution knob, does the correlation survive?"
    #       This closes the residual path
    #           objective <- resolution -> external_metric
    #       that (A) alone cannot rule out.
    # -------------------------------------------------------------------------
    if "n_leiden_clusters" in trial_df.columns:
        partial_df_cc = compute_partial_spearman(
            trial_df,
            internal_cols=internal_cols,
            external_cols=external_cols,
            control_cols=["n_leiden_clusters"],
        )
        partial_out_cc = os.path.join(
            OUTPUT_DIR,
            "spearman_internal_vs_facs_partial_controlling_cluster_count.csv"
        )
        partial_df_cc.to_csv(partial_out_cc, index=False)
        print(f"Saved partial Spearman table (cluster count only): "
              f"{partial_out_cc}")

    if "n_leiden_clusters" in trial_df.columns and "resolution" in trial_df.columns:
        partial_df_cc_res = compute_partial_spearman(
            trial_df,
            internal_cols=internal_cols,
            external_cols=external_cols,
            control_cols=["n_leiden_clusters", "resolution"],
        )
        partial_out_cc_res = os.path.join(
            OUTPUT_DIR,
            "spearman_internal_vs_facs_partial_"
            "controlling_cluster_count_and_resolution.csv"
        )
        partial_df_cc_res.to_csv(partial_out_cc_res, index=False)
        print(f"Saved partial Spearman table (cluster count + resolution): "
              f"{partial_out_cc_res}")

    # -------------------------------------------------------------------------
    # Top vs bottom objective decile
    # -------------------------------------------------------------------------
    top_bottom_df = summarize_top_bottom_decile(
        trial_df,
        objective_col="yield_score_target"
    )

    top_bottom_out = os.path.join(OUTPUT_DIR, "top_vs_bottom_decile_objective.csv")
    top_bottom_df.to_csv(top_bottom_out, index=False)
    print(f"Saved top-vs-bottom decile table: {top_bottom_out}")

    # -------------------------------------------------------------------------
    # Best scBOA-selected trial vs default baseline
    # -------------------------------------------------------------------------
    print("\nComputing default baseline external FACS metrics...")
    default_external = run_params_and_score_external(
        adata,
        DEFAULT_PARAMS,
        trial_name="default_baseline"
    )

    best_idx = trial_df["yield_score_target"].idxmax()
    best_row = trial_df.loc[best_idx].copy()

    best_vs_default_rows = []

    best_vs_default_rows.append({
        "condition": "Default baseline",
        "n_hvg": DEFAULT_PARAMS["n_hvg"],
        "n_pcs": DEFAULT_PARAMS["n_pcs"],
        "n_neighbors": DEFAULT_PARAMS["n_neighbors"],
        "resolution": DEFAULT_PARAMS["resolution"],
        "yield_score_target": np.nan,
        **default_external
    })

    best_vs_default_rows.append({
        "condition": "Best scBOA objective trial",
        "n_hvg": int(best_row["n_hvg"]),
        "n_pcs": int(best_row["n_pcs"]),
        "n_neighbors": int(best_row["n_neighbors"]),
        "resolution": float(best_row["resolution"]),
        "yield_score_target": best_row["yield_score_target"],
        "matched_cells": best_row["matched_cells"],
        "n_facs_labels": best_row["n_facs_labels"],
        "n_leiden_clusters": best_row["n_leiden_clusters"],
        "external_leiden_ari": best_row["external_leiden_ari"],
        "external_leiden_nmi": best_row["external_leiden_nmi"],
        "external_homogeneity": best_row["external_homogeneity"],
        "external_completeness": best_row["external_completeness"],
        # "external_v_measure": best_row["external_v_measure"],
        "external_celltypist_broad_accuracy": best_row["external_celltypist_broad_accuracy"],
        "external_celltypist_broad_macro_f1": best_row["external_celltypist_broad_macro_f1"],
        "external_mean_confidence": best_row["external_mean_confidence"],
        "effective_n_hvg": best_row["effective_n_hvg"],
        "effective_n_pcs": best_row["effective_n_pcs"]
    })

    best_default_df = pd.DataFrame(best_vs_default_rows)
    best_default_out = os.path.join(OUTPUT_DIR, "best_vs_default_facs_concordance.csv")
    best_default_df.to_csv(best_default_out, index=False)
    print(f"Saved best-vs-default table: {best_default_out}")

    # -------------------------------------------------------------------------
    # Figures
    # -------------------------------------------------------------------------
    plot_correlation_heatmap(corr_df)

    if (
        "partial_df_cc" in locals()
        and "partial_df_cc_res" in locals()
    ):
        plot_raw_partial_correlation_heatmaps(
            raw_df=corr_df,
            partial_cluster_df=partial_df_cc,
            partial_cluster_resolution_df=partial_df_cc_res,
        )

    plot_scatter_panels(trial_df)
    plot_top_bottom_bar(top_bottom_df)
    plot_cluster_number_vs_external(trial_df)
    plot_x_vs_external(
        trial_df,
        x_col="resolution",
        x_label="Leiden resolution",
        tag="resolution",
    )

    if "n_leiden_clusters" in trial_df.columns and "resolution" in trial_df.columns:
        plot_partial_correlation_bars(
            corr_df=corr_df,
            partial_cc_df=partial_df_cc,
            partial_cc_res_df=partial_df_cc_res,
            internal_metric="yield_score_target",
        )

        # Panel E: positive-control falsification of the artifact hypothesis.
        # Default external metric is Leiden ARI (the partition-based metric
        # most susceptible to a cluster-count artifact); baselines are
        # silhouette_score, mean_f1, mean_mcs. The target objective sits in
        # column 1 with a yellow highlight for direct visual comparison.
        plot_positive_control_baselines(
            corr_df=corr_df,
            partial_cc_df=partial_df_cc,
            partial_cc_res_df=partial_df_cc_res,
            target_internal="yield_score_target",
            baseline_internals=("silhouette_score", "mean_f1", "mean_mcs"),
            external_metric="external_leiden_ari",
        )

    print("\n" + "=" * 80)
    print("Analysis complete.")
    print(f"All outputs saved to: {OUTPUT_DIR}")
    print("=" * 80)


if __name__ == "__main__":
    main()