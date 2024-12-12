import numpy as np
import os

processed_dir="./Human_features/processed/"
npy_file = "./Human_features/npy_file_new(human_dataset).npy"
new_npy_file = "./Human_features/npy_file_new(human_dataset)_v2.npy"

# Load the .npy file
data = np.load(npy_file)
print(data.shape)

valid_labels = []

for example in data:
    if os.path.isfile(os.path.join(processed_dir, example[2])+'.pt') and os.path.isfile(os.path.join(processed_dir, example[5])+'.pt'):
        valid_labels.append(example)

print(len(valid_labels))
new_labels = np.array(valid_labels)
print(new_labels.shape)

print(data.shape[0]-new_labels.shape[0])

np.save(new_npy_file, new_labels)