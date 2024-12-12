import os
import numpy as np

# processed_dir="../human_features/processed/"
processed_dir="./human_features/processed/"
# npy_file = "../human_features/npy_file_new(human_dataset).npy"
npy_file = "./human_features/npy_file_new(human_dataset).npy"
npy_ar = np.load(npy_file)

protein_1 = npy_ar[:,2]
unique_protein_1 = set(np.unique(protein_1))
print(f"npy_ar column 2 is {protein_1}, length = {len(protein_1)}, num_unique = {len(unique_protein_1)}")
protein_2 = npy_ar[:,5]
unique_protein_2 = set(np.unique(protein_2))
print(f"npy_ar column 5 is {protein_2}, length = {len(protein_2)}, num_unique = {len(unique_protein_2)}")

processed_files = set(os.path.splitext(file)[0] for file in os.listdir(processed_dir) if os.path.isfile(os.path.join(processed_dir, file)))
missing_proteins_1 = unique_protein_1 - processed_files
missing_proteins_2 = unique_protein_2 - processed_files
# extra_files = processed_files - unique_proteins

print("Proteins missing in the processed folder:")
print("\n".join(missing_proteins_1) if missing_proteins_1 else "None")
print("\n".join(missing_proteins_2) if missing_proteins_2 else "None")

print(np.where(protein_1 == "3CKI"))



