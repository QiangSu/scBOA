"""
Create a CellTypist model with all B-lineage labels removed.
"""
import celltypist
import numpy as np
import pickle
import os

MODEL_IN = "/home/data/qs/.celltypist/data/models/Healthy_COVID19_PBMC.pkl"
MODEL_OUT = "/home/data/qs/.celltypist/data/models/Healthy_COVID19_PBMC_noB.pkl"

LABELS_TO_DROP = [
    "B_exhausted", "B_immature", "B_malignant",
    "B_naive", "B_non-switched_memory", "B_switched_memory",
    "Plasma_cell_IgA", "Plasma_cell_IgG", "Plasma_cell_IgM",
    "Plasmablast",
]

# Load original
model = celltypist.models.Model.load(MODEL_IN)
clf = model.classifier

print(f"Original: {len(clf.classes_)} classes")
for lb in LABELS_TO_DROP:
    assert lb in clf.classes_, f"'{lb}' not found"

# Filter classifier arrays
keep_mask = ~np.isin(clf.classes_, LABELS_TO_DROP)
clf.classes_ = clf.classes_[keep_mask]
clf.coef_ = clf.coef_[keep_mask, :]
clf.intercept_ = clf.intercept_[keep_mask]

print(f"Hold-out: {len(clf.classes_)} classes")

# CellTypist expects a dict on load — save in the exact same format
# Inspect original pickle to confirm structure
with open(MODEL_IN, "rb") as f:
    orig_pkl = pickle.load(f)

print(f"\nOriginal pickle type: {type(orig_pkl)}")
if isinstance(orig_pkl, dict):
    print(f"Keys: {list(orig_pkl.keys())}")
    # Rebuild dict with modified classifier
    new_pkl = {
        "Model": clf,
        "Scaler_": orig_pkl["Scaler_"],
        "description": orig_pkl.get("description", {}),
    }
else:
    # Some versions save the Model object directly; Model.load handles both
    # but the error says yours expects dict — so force dict
    new_pkl = {
        "Model": clf,
        "Scaler_": model.scaler,
        "description": getattr(model, "description", {}),
    }

with open(MODEL_OUT, "wb") as f:
    pickle.dump(new_pkl, f)

print(f"\nSaved: {MODEL_OUT}")
print(f"Size: {os.path.getsize(MODEL_OUT) / 1024 / 1024:.2f} MB")

# Verify by reloading
print("\n=== Verifying reload ===")
reloaded = celltypist.models.Model.load(MODEL_OUT)
print(f"Reloaded classes: {len(reloaded.classifier.classes_)}")
b_labels_remaining = [x for x in reloaded.classifier.classes_
                      if x.startswith("B_") or "Plasma" in x]
if b_labels_remaining:
    print(f"ERROR: B labels still present: {b_labels_remaining}")
else:
    print("✓ No B-lineage labels in reloaded model")