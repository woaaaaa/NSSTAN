# Benchmark input data

Download the raw files from the original sources listed below and place them under `data/raw/<DATASET>/`, for example `data/raw/PEMS08/PEMS08.npz`, `data/raw/PEMS08/PEMS08.csv`, `data/raw/METR-LA/metr-la.h5` and `data/raw/METR-LA/adj_mx.pkl`.

| Dataset | Target | Time steps | Sensors | Raw source |
|---|---|---:|---:|---|
| PEMS03 | Traffic flow | 26208 | 358 | ASTGNN benchmark distribution |
| PEMS04 | Traffic flow | 16992 | 307 | ASTGNN benchmark distribution |
| PEMS07 | Traffic flow | 28224 | 883 | ASTGNN benchmark distribution |
| PEMS08 | Traffic flow | 17856 | 170 | ASTGNN benchmark distribution |
| METR-LA | Traffic speed | 34272 | 207 | DCRNN benchmark distribution |
| PEMS-BAY | Traffic speed | 52116 | 325 | DCRNN benchmark distribution |

Original access and documentation:

* Flow data and road-distance CSVs: https://github.com/guoshnBJTU/ASTGNN/tree/main/data
* Speed HDF5 measurements and sensor graphs: https://github.com/liyaguang/DCRNN#data-preparation
* Original data provider: Caltrans Performance Measurement System (PeMS).

The authors do not claim ownership of these benchmark datasets. Original provider rights and access conditions remain applicable; software licenses included in this repository do not relicense the data.

## Preparing windows

`prepare_data.py` requires an explicit split convention and graph construction choice. For example, using one traffic-flow channel with time partitions:

```bash
python prepare_data.py \
  --raw data/raw/PEMS08/PEMS08.npz \
  --output data/processed/PEMS08 \
  --channels 0 --split 0.6 0.2 0.2 --split-mode time \
  --adj-csv data/raw/PEMS08/PEMS08.csv --graph-weighting binary
```

An example for the speed benchmark, partitioning chronologically after forming windows:

```bash
python prepare_data.py \
  --raw data/raw/METR-LA/metr-la.h5 \
  --output data/processed/METR-LA \
  --channels 0 --split 0.7 0.1 0.2 --split-mode window \
  --adj-pickle data/raw/METR-LA/adj_mx.pkl
```

For PEMS03 CSV graphs, pass `--sensor-ids data/raw/PEMS03/PEMS03.txt` to align detector IDs with the node axis. For other channel orders or weighted/directed graphs, provide the corresponding arguments. These commands are explicit examples, not blanket settings for every manuscript comparison. Preprocessing differences can change evaluation results even when the dataset name is unchanged.
