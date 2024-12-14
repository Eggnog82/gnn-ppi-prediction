# New Graph Neural Network Approaches for Prediction of Protein-Protein Interaction

## Reproducibility

### Reproducing GCNN, GAT, or Multi-Hop GAT
In order to replicate the results on GCNN, GAT, or Multi-Hop Attention GNNs, do the following steps
  1. Download the conda environment as follows. In order to maintain support for the 2022 paper that we compare our models to, many of the packages are old and hard to source correctly. As a result, the install may differ based on your computer OS. It is also why this is not the cleanest download process. However, it worked on my Windows PC and on the Grace cluster at Yale. If you do try to replicate this on the Grace Cluster, you will need a GPU compatible with our code. Try to run with partition "gpu_devel" and additional job option "--constraint=rtx2080ti". If you have any issues, email brennan.lagasse@yale.edu and eric.w.li@yale.edu.

```bash
conda env create -f ppi_env3.yml
conda install anaconda::scikit-learn
conda install conda-forge::tqdm
conda install conda-forge::torch-optimizer
conda install anaconda::networkx
pip install --no-index torch-scatter -f https://data.pyg.org/whl/torch-1.8.0+cu102.html
pip install --no-index torch-sparse -f https://data.pyg.org/whl/torch-1.8.0+cu102.html
pip install --no-index torch-cluster -f https://data.pyg.org/whl/torch-1.8.0+cu102.html
pip install --no-index torch-spline-conv -f https://data.pyg.org/whl/torch-1.8.0+cu102.html
pip install torch-geometric==1.7.0
```

  2. Create directory ```Human_features/``` in the main directory, and add subdirectory ```processed/```

  3. Download the graph dataset in the paper from Jha et al found here https://drive.google.com/file/d/1mpMB2Gu6zH6W8fZv-vGwTj_mmeitIV2-/view?usp=sharing and store the files in ```./Human_features/processed/```.
  
  4. Run the following to train the model. MODEL is replaced with GCNN, GAT, or MultiHopAttGNN depending on the model you want to run

```bash
python train.py MODEL
```

  5. Run the following to test the model

```bash
python test.py MODEL
```

### Reproducing Weighted GAT (or any of the above using the weighted dataset)
To replicate the results of WeightedAttGNN, there are a few additional steps since the datset must be generated from scratch. To replicate the results on this model, do the following steps. This process is somewhat complicated, but it is the result of significant effort to ensure that the weighted dataset generated is as similar as possible to the unweighted dataset from the Jha et al. paper, up to weights of course. I tried my best to consolidate the process, but I am already 40 hours into data processing. All of the code below should be run from base directory of this repository.

1. Same as above.

2. Same as above.

3. Download the graph dataset I generated found here https://drive.google.com/drive/folders/1nwtpS1Yu7Pmx9kLSDZJtbbyhCf91f32P?usp=drive_link and store the files in ```./Human_features/processed/```. If there is an issue, see the steps below for how to generate the graphs from scratch, or contact brennan.lagasse@yale.edu.

4. Take the ```npy_file_new(human_dataset)_v2.npy``` file in the main directory and put it in the ```Human_features/``` directory. Edit ```data_prepare.py``` line 32 to read ```npy_file = "./Human_features/npy_file_new(human_dataset)_v2.npy"```

5. Run the following to train the model. MODEL is replaced with WeightedAttGNN, GCNN, GAT, or MultiHopAttGNN depending on the model you want to run

```bash
python train.py MODEL
```

6. Run the following to test the model

```bash
python test.py MODEL
``` 

### How to generate the Weighted Graphs from scratch (if you really want to)

1. Follow steps one and two above.

2. Create directory ```Human_features/``` in the main directory, and add subdirectories ```processed/```, ```processed_old```, and ```raw```. These will store the new weighted graphs that will be generates, the old unweighted graphs from Jha et al, and the raw protein files respectively.

3. Download the graph dataset in the paper from Jha et al found here https://drive.google.com/file/d/1mpMB2Gu6zH6W8fZv-vGwTj_mmeitIV2-/view?usp=sharing and put the results in ```./Human_features/processed_old/```

4. Run the following script to download all of the proteins from the original processed folder (that are still available) to directory ```./Human_features/raw```. The first line downloads all the files from the RCSB website in compressed form, and the second line decompresses all of the files.

```bash
./batch_download.sh -f protein_list.txt -o ./Human_features/raw -c
python decompress_raw.py
```

5. Convert the raw proteins into graphs using the following. Note that by default these files will be placed in the ```processed``` directory. Since there are still a few more steps to go with processing, move them to subdirectory ```./Human_features/processed_temp```

```bash
python proteins_to_graphs.py
mv -v ./Human_features/processed/* ./Human_features/processed_temp
```

6. Since the library used to make the original embeddings for the graph does not seem to be accessible, the embeddings from the original graphs are used instead (the result would be equivalent to recomputing the embeddings assuming the autoencoder is stable). This can be done as follows:

```bash
python get_embeddings.py
```

7. The next problem is that the npy file used to generate the dataset includes protein pairings with proteins that are not included in the updated dataset, either because they were removed from RCSB website or lack information about atomic positions. To fix this, run the following script to generate a new npy file that only includes proteins that have a weighted graph representation. This generates a file ```npy_file_new(human_dataset)_v2.npy``` in the ```Human_features/``` directory. After generating this file, edit ```data_prepare.py``` line 32 to read ```npy_file = "./Human_features/npy_file_new(human_dataset)_v2.npy"```

```bash
python update_nby.py
```

