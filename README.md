# New Graph Neural Network Approaches for Prediction of Protein-Protein Interaction

## Reproducibility

In order to replicate the results on GCNN, GAT, or Multi-Hop Attention GNNs, do the following steps
  1. Download the conda environment as follows

```bash
conda env create -f ppi_env.yml
```

  2. Download the graph dataset in the paper from Jha et al found here https://drive.google.com/file/d/1mpMB2Gu6zH6W8fZv-vGwTj_mmeitIV2-/view?usp=sharing and put the results in ```./Human_features/processed/```.
  
  3. Run the following to train the model. MODEL is replaced with GCNN, GAT, or MultiHopAttGNN depending on the model you want to run

```bash
python train.py MODEL
```

  4. Run the following to test the model

```bash
python test.py MODEL
```

To replicate the results of WeightedAttGNN, there are a few additional steps since the datset must be generated from scratch. To replicate the results on this model, do the following steps
1. Download the conda environment as follows

```bash
conda env create -f ppi_env.yml
```

2. Download the graph dataset in the paper from Jha et al found here https://drive.google.com/file/d/1mpMB2Gu6zH6W8fZv-vGwTj_mmeitIV2-/view?usp=sharing and put the results in ```./Human_features/processed_old/```

3. 