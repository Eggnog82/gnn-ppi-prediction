# New Graph Neural Network Approaches for Prediction of Protein-Protein Interaction

## Reproducibility

In order to replicate the results on GCNN, GAT, or Multi-Hop Attention GNNs, do the following steps
  1. Download the conda environment as follows

```bash
conda env create -f ppi_env.yml
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

To replicate the results of WeightedAttGNN, there are a few additional steps since the datset must be generated from scratch. To replicate the results on this model, do the following steps. This process is somewhat complicated, but it is the result of significant effort to ensure that the weighted dataset generated is as similar as possible to the unweighted dataset from the Jha et al. paper, up to weights of course. I tried my best to consolidate the process, but I am already 40 hours into data processing. All of the code below should be run from base directory of this repository.

1. Download the conda environment as follows

```bash
conda env create -f ppi_env.yml
```

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

8. Run the following to train the model. MODEL is replaced with GCNN, GAT, or MultiHopAttGNN depending on the model you want to run

```bash
python train.py WeightedAttGNN
```

9. Run the following to test the model

```bash
python test.py WeightedAttGNN
``` 

