#!/usr/bin/env python3
import gzip
import random
import os
import sys

print("Starting Step 3: Convert FASTA/FASTQ, Add REAL Barcodes/UMIs/PolyT to R1, Create Combined Cell Ranger Structure")

# --- 3.1: Define Paths ---
base_dir = "/home/data/qs/scRNA_simulation_data/simulated_data_15000c_10g_standard_2"
# Input directory with Polyester FASTQ files (sample_X_1.fastq, sample_X_2.fastq)
input_dir = os.path.join(base_dir, "simulated_fastq_15000c_10g_standard/")
# Output directory for the final combined Cell Ranger FASTQ files
output_dir = os.path.join(base_dir, "simulated_data_15000c_10g_standard_cellranger_whitelist_polyT") # New distinct name
os.makedirs(output_dir, exist_ok=True)

# --- User Defined ---
# Cell Ranger sample name (MUST MATCH --sample in cellranger count command)
cr_sample_name = "simulated"
# Path to the 10x barcode whitelist file (gzipped)
barcode_whitelist_path = "/home/data/qs/cellranger-8.0.1/lib/python/cellranger/barcodes/3M-february-2018.txt.gz" # <-- USER PROVIDED PATH

# --- 3.2: Define Simulation Parameters ---
n_cells = 15000  # Number of cells simulated in the R script

# Cell Ranger Parameters (check consistency with chosen chemistry/whitelist)
barcode_length = 16 # Matches 3M-february-2018 (often used with v2/v3 whitelists)
umi_length = 12     # Using 12bp (common for v3, though the v2 whitelist usually paired with 10bp UMI)
poly_t_length = 20  # Length of the poly-T tail to add
cr_read1_length = barcode_length + umi_length + poly_t_length # 16 + 12 + 20 = 48 bp
cr_read2_length = 150 # Matches Polyester R1 sequence length from R script

bases = 'ATCG'
placeholder_quality_char = 'I' # Phred score 40 placeholder

# --- Read Barcode Whitelist ---
print(f"Reading barcode whitelist from: {barcode_whitelist_path}")
valid_barcodes = []
try:
    # Use gzip.open for reading compressed file ('rt' for read text mode)
    with gzip.open(barcode_whitelist_path, 'rt') as f:
        for line in f:
            barcode = line.strip()
            if barcode: # Ensure non-empty lines
                # Basic validation (optional but good)
                if len(barcode) == barcode_length and all(c in bases for c in barcode):
                     valid_barcodes.append(barcode)
                else:
                     print(f"Warning: Skipping invalid barcode format in whitelist: {barcode}")

    print(f"Read {len(valid_barcodes)} valid barcodes from whitelist.")
    if not valid_barcodes:
        raise ValueError("Whitelist file is empty or contains no valid barcodes.")
    if len(valid_barcodes) < n_cells:
        raise ValueError(f"Whitelist contains fewer barcodes ({len(valid_barcodes)}) than requested cells ({n_cells}). Cannot proceed.")

except FileNotFoundError:
    print(f"ERROR: Barcode whitelist file not found at {barcode_whitelist_path}")
    sys.exit(1)
except Exception as e:
    print(f"ERROR: Failed to read or process barcode whitelist: {e}")
    sys.exit(1)

# --- Sample unique barcodes from the whitelist for our cells ---
print(f"Sampling {n_cells} unique barcodes from the whitelist...")
try:
    cell_barcodes_assigned = random.sample(valid_barcodes, n_cells)
    print(f"Successfully sampled {len(cell_barcodes_assigned)} unique barcodes.")
except ValueError as e:
     # This should be caught by the check above, but as a safeguard
     print(f"ERROR: Could not sample barcodes. {e}")
     sys.exit(1)

# Map cell input index (0 to n_cells-1) to its assigned barcode
# This determines which barcode goes with reads from sample_1, sample_2 etc.
cell_barcode_map = {i: cell_barcodes_assigned[i] for i in range(n_cells)}

# Save barcode mapping (optional but highly recommended for traceability)
barcode_map_file = os.path.join(output_dir, f"{cr_sample_name}_barcode_mapping.txt")
try:
    with open(barcode_map_file, 'w') as f:
        f.write("InputCellIndex\tAssignedBarcode\n")
        for i in range(n_cells):
             # Map 1-based input cell number (like sample_1) to barcode
             f.write(f"{i+1}\t{cell_barcode_map[i]}\n")
    print(f"Saved barcode mapping to {barcode_map_file}")
except IOError as e:
    print(f"Warning: Could not save barcode map file: {e}")


# --- 3.3: Function to Generate Random UMI ---
def generate_umi(length=umi_length):
    """Generates a random UMI sequence."""
    return ''.join(random.choice(bases) for _ in range(length))

# --- 3.4: Process Input Files and Create Combined Cell Ranger Output ---
print(f"\nProcessing {n_cells} input cell FASTQ files...")
print(f"Writing combined output to directory: {output_dir}")

# Define the *single pair* of output filenames using the Cell Ranger sample name
cr_r1_outfile_path = os.path.join(output_dir, f"{cr_sample_name}_S1_R1_001.fastq.gz")
cr_r2_outfile_path = os.path.join(output_dir, f"{cr_sample_name}_S1_R2_001.fastq.gz")

processed_input_files = 0
total_reads_written = 0
error_count_files = 0

# Pre-calculate the Poly-T tail string
poly_t_tail_seq = 'T' * poly_t_length

# Open the *single* output file pair ONCE before the loop
try:
    # Use gzip.open for writing compressed FASTQ files ('wt' for write text mode)
    with gzip.open(cr_r1_outfile_path, 'wt', compresslevel=6) as cr_r1_out_gz, \
         gzip.open(cr_r2_outfile_path, 'wt', compresslevel=6) as cr_r2_out_gz:

        # Iterate through the input cell files (corresponding to sample_1, sample_2, ...)
        for i in range(n_cells):
            cell_input_index = i # 0-based index for mapping
            sample_number = i + 1 # 1-based index for filenames
            input_sample_id = f"sample_{sample_number}" # Name from R script output

            # Define input filenames from R script output
            # Polyester R1 contains the sequence that becomes Cell Ranger R2 (cDNA)
            # Note: Polyester output is FASTA by default, not FASTQ. Read accordingly.
            input_r1_polyester_path = os.path.join(input_dir, f"{input_sample_id}_1.fasta")
            # Polyester R2 is not typically used for GEX simulation structure, but check existence
            input_r2_polyester_path = os.path.join(input_dir, f"{input_sample_id}_2.fasta")

            # Get the *assigned valid barcode* for this input cell file
            current_cell_barcode = cell_barcode_map[cell_input_index] # The barcode sampled from the whitelist

            # Check if the primary input file exists
            # Modified to handle FASTA format (pairs of lines)
            if not os.path.exists(input_r1_polyester_path):
                print(f"Warning: Input file not found: {input_r1_polyester_path}. Skipping input {input_sample_id}.")
                error_count_files += 1
                continue # Skip to the next input cell file

            # Optional check for the paired file
            if not os.path.exists(input_r2_polyester_path):
                 print(f"Warning: Paired input file not found: {input_r2_polyester_path}. Proceeding using only {input_r1_polyester_path}.")

            if sample_number % 500 == 0: # Print progress
                 print(f"  Processing input file {sample_number}/{n_cells} ({input_sample_id})...")

            # Process the reads from this input cell file (FASTA format)
            try:
                # Open the *individual* input Polyester R1 FASTA file for this cell
                with open(input_r1_polyester_path, 'r') as poly_r1_in:
                    read_counter_this_file = 0
                    # Process reads 2 lines at a time (one FASTA record)
                    while True:
                        poly_r1_l1_header = poly_r1_in.readline().strip() # Header line starting with >
                        if not poly_r1_l1_header: break # End of this input file
                        poly_r1_l2_seq = poly_r1_in.readline().strip() # Sequence line

                        # --- Create Cell Ranger R1 Record (VALID Barcode + UMI + PolyT) ---
                        umi = generate_umi(umi_length)
                        # Concatenate Barcode, UMI, and Poly-T tail
                        cr_r1_seq = current_cell_barcode + umi + poly_t_tail_seq
                        # Use placeholder quality for R1 (length matches cr_r1_seq)
                        cr_r1_qual = placeholder_quality_char * len(cr_r1_seq)

                        # --- Create Cell Ranger R2 Record (cDNA) ---
                        cr_r2_seq = poly_r1_l2_seq
                        # Ensure R2 sequence length matches target (trim if necessary)
                        if len(cr_r2_seq) > cr_read2_length:
                            cr_r2_seq = cr_r2_seq[:cr_read2_length]
                        # Use placeholder quality for R2 (length matches cr_r2_seq)
                        cr_r2_qual = placeholder_quality_char * len(cr_r2_seq)


                        # --- Construct FASTQ Headers ---
                        # Make header unique across *all* reads in the output files
                        # Include input sample ID and read counter within that file
                        # The original Polyester header might be useful too, embed it?
                        # The Cell Ranger header format is specific. Let's use a standard one.
                        # @<read_name> <read_number>:<is_filtered>:<control_bits>:<index_sequence>
                        # Use original header info to make read_name unique and trace back
                        # Example: @<Polyester_ID> 1:N:0:BARCODE
                        # Let's extract the ID from the Polyester header (after '>')
                        poly_read_id = poly_r1_l1_header.lstrip('>').split(' ')[0] # Get ID before first space
                        cr_header_base = f"@{input_sample_id}:{poly_read_id}" # Use sample and original read ID
                        cr_r1_header = f"{cr_header_base} 1:N:0:{current_cell_barcode}" # Read 1, No filter, Control 0, Barcode
                        cr_r2_header = f"{cr_header_base} 2:N:0:{current_cell_barcode}" # Read 2, No filter, Control 0, Barcode


                        # --- Write to the SINGLE combined output files ---
                        cr_r1_out_gz.write(f"{cr_r1_header}\n{cr_r1_seq}\n+\n{cr_r1_qual}\n")
                        cr_r2_out_gz.write(f"{cr_r2_header}\n{cr_r2_seq}\n+\n{cr_r2_qual}\n")

                        read_counter_this_file += 1
                        total_reads_written += 1

                processed_input_files += 1

            except Exception as e:
                print(f"Error processing reads from input file {input_sample_id}: {e}")
                # Optionally add to a separate error counter for read processing errors
                continue # Move to the next input file

except Exception as e:
    print(f"FATAL ERROR: Could not open or write to output files in {output_dir}. Check permissions and path.")
    print(f"Error details: {e}")
    # Clean up potentially partially written files if error occurs during opening/writing
    if os.path.exists(cr_r1_outfile_path): os.remove(cr_r1_outfile_path)
    if os.path.exists(cr_r2_outfile_path): os.remove(cr_r2_outfile_path)
    sys.exit(1)


# --- 3.5: Final Summary ---
print("\n--- Processing Summary ---")
print(f"Total input cell FASTA files targeted: {n_cells}")
print(f"Successfully processed input files: {processed_input_files}")
print(f"Input files skipped due to errors/missing: {error_count_files}")
print(f"Total reads written to combined output files: {total_reads_written}")
if error_count_files > 0:
    print("Warning: Some input files were skipped. Check logs above.")

if processed_input_files > 0:
    print(f"Step 3: Created combined Cell Ranger compatible FASTQ files using whitelist and added Poly-T tail.")
    print(f"  Output R1: {cr_r1_outfile_path} (Length: {cr_read1_length} bp = {barcode_length}B + {umi_length}U + {poly_t_length}T)")
    print(f"  Output R2: {cr_r2_outfile_path} (Length: max {cr_read2_length} bp)")
    print(f"  Barcode whitelist used: {barcode_whitelist_path}")
    print(f"  Barcode-to-Input mapping saved: {barcode_map_file}")
else:
    print("\nERROR: No input files were successfully processed. Check input directory and naming.")
    sys.exit(1)
