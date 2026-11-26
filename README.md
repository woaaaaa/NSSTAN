# NSSTAN

PyTorch implementation of **NSSTAN**.

## Repository contents

The NSSTAN source code is stored directly in this repository as ordinary Python files. Open the files below to inspect the implementation; there is no code archive to extract. The benchmark datasets are publicly available from their original sources; see [data/README.md](data/README.md) for download links and the expected file layout.

| File | Purpose |
| --- | --- |
| [model.py](model.py) | NSSTAN forecasting architecture and constituent modules |
| [train.py](train.py) | Training, validation-based checkpoint selection and final evaluation |
| [engine.py](engine.py) | Optimization and metric wrapper |
| [util.py](util.py) | Data loading, scaling, graph supports and metrics |
| [prepare_data.py](prepare_data.py) | Explicitly configured input-window and graph preparation |
| [requirements.txt](requirements.txt) | Python dependencies |
| [data/README.md](data/README.md) | Dataset access, file layout and preprocessing examples |

## Architecture

The implemented forecasting path is:

1. Selected Fourier reconstruction and complementary residual of the observed input window.
2. Two independently parameterized temporal-convolution pathways, each with node aggregation and redistribution (NAR).
3. Fusion of the terminal pathway features, followed by a shared graph-propagation stage.
4. Temporal and graph skip projections combined by the prediction head.

The input has shape `[batch, channels, nodes, 12]`; the output has shape `[batch, nodes, 12]`. The learned adjacency is fixed for a trained model; NAR uses the current node features. A one-hour Fourier input window is not an estimate of daily or weekly periodicity.

## Installation

Use Python 3.9 or later, with a PyTorch build appropriate for your CUDA environment:

```bash
pip install -r requirements.txt
```

The full training entry point expects CUDA. The model itself can be imported independently on CPU or GPU.

## Data

The six benchmark datasets are PEMS03, PEMS04, PEMS07, PEMS08, METR-LA and PEMS-BAY. See [data/README.md](data/README.md) for the original sources and data layout. Dataset rights remain with their original providers.

The training loader accepts a directory containing `train.npz`, `val.npz` and `test.npz`, each with `x` and `y` arrays of shape `[samples, 12, nodes, features]`. It standardizes the selected input target channel using training-set statistics only. Targets remain in their original scale.

## Training

### Reproducing the reported results

Input channels used in the paper:

| Dataset | Window channels (in order) | `--task` | `--in-dim` | Target |
|---|---|---|---|---|
| PEMS04, PEMS08 | speed (raw), flow | B | 2 | flow |
| PEMS03, PEMS07 | flow | A | 1 | flow |
| METR-LA, PEMS-BAY | speed, time of day | A | 2 | speed |

The loader standardizes the target channel with training-set statistics; the other channel is left unchanged.

Prepare the windows for a flow dataset (raw PEMS04/PEMS08 arrays are ordered flow, occupancy, speed, so `--channels 2,0` gives speed then flow; use `--channels 0` for PEMS03 and PEMS07, and add `--sensor-ids data/raw/PEMS03/PEMS03.txt` for PEMS03):

```bash
python prepare_data.py \
  --raw data/raw/PEMS08/PEMS08.npz --output data/processed/PEMS08 \
  --channels 2,0 --split 0.6 0.2 0.2 --split-mode time \
  --adj-csv data/raw/PEMS08/PEMS08.csv --graph-weighting binary
```

For METR-LA and PEMS-BAY, use the standard DCRNN windows (speed plus time of day, 70/10/20 split) produced by `generate_training_data.py` in the DCRNN repository, together with its `adj_mx.pkl`.

Train with the three seeds used in the paper (20261, 20262, 20263). PEMS04 and PEMS08 are trained for a fixed 600 epochs:

```bash
for seed in 20261 20262 20263; do
  python train.py \
    --data data/processed/PEMS08 --adjdata data/processed/PEMS08/adj_mx.pkl \
    --num-nodes 170 --task B --in-dim 2 \
    --gcn-bool --addaptadj --randomadj \
    --device cuda:0 --epochs 600 --seed $seed \
    --output-dir outputs/PEMS08_$seed
done
```

The other datasets are trained for at most 600 epochs with early stopping on validation MAE (patience 50, at least 200 epochs), for example:

```bash
python train.py \
  --data data/processed/METR-LA --adjdata data/processed/METR-LA/adj_mx.pkl \
  --num-nodes 207 --task A --in-dim 2 \
  --gcn-bool --addaptadj --randomadj \
  --device cuda:0 --epochs 600 --patience 50 --min-epochs 200 --seed 20261 \
  --output-dir outputs/METR-LA_20261
```

Node counts: PEMS03 358, PEMS04 307, PEMS07 883, PEMS08 170, METR-LA 207, PEMS-BAY 325. In all runs the checkpoint with the lowest validation MAE is evaluated on the test set. `--output-dir` must not already exist.

The number of retained Fourier components is k = 1 and the NAR core width is 256 by default.
