# NSSTAN

PyTorch implementation of **Node-aware Spectral Spatio-Temporal Aggregation Network for Traffic Flow Forecasting**.

## Repository contents

The NSSTAN source code is stored directly in this repository as ordinary Python files. Open the files below to inspect the implementation; there is no code archive to extract. The benchmark datasets are distributed separately as a ZIP asset in [Releases](https://github.com/woaaaaa/NSSTAN/releases/tag/benchmark-inputs-v1).

| File | Purpose |
| --- | --- |
| [model.py](model.py) | NSSTAN forecasting architecture and constituent modules |
| [train.py](train.py) | Training, validation-based checkpoint selection and final evaluation |
| [engine.py](engine.py) | Optimization and metric wrapper |
| [util.py](util.py) | Data loading, scaling, graph supports and metrics |
| [prepare_data.py](prepare_data.py) | Explicitly configured input-window and graph preparation |
| [requirements.txt](requirements.txt) | Python dependencies |
| [data/README.md](data/README.md) | Dataset access, file layout and preprocessing examples |

The public repository does not include standalone baseline implementations, comparison experiments, experimental results, training logs or trained checkpoints. Training and evaluation utilities generate outputs locally; those outputs are not part of the published files.

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

The six benchmark datasets are PEMS03, PEMS04, PEMS07, PEMS08, METR-LA and PEMS-BAY. See [data/README.md](data/README.md) for the raw input archive, original sources and data layout. Dataset rights remain with their original providers.

The training loader accepts a directory containing `train.npz`, `val.npz` and `test.npz`, each with `x` and `y` arrays of shape `[samples, 12, nodes, features]`. It standardizes the selected input target channel using training-set statistics only. Targets remain in their original scale.

**Channel selection must match the preprocessing configuration.** `--task A` selects target channel 0; `--task B` selects target channel 1. `--in-dim` controls the model's number of input features. With a single input channel, the archived loader/engine uses the first channel for task A and the last channel for task B. Do not use task B with a one-feature target array. Window boundaries, raw feature order and graph construction also need to match the intended evaluation protocol.

The raw data archive is not a certification that newly generated windows are byte-identical to every historical evaluation array. Preserve the exact preprocessing and model configuration when reproducing a particular experiment.

## Training

For already-prepared one-channel traffic-flow windows, an example is:

```bash
python train.py \
  --data data/processed/PEMS08 \
  --adjdata data/processed/PEMS08/adj_mx.pkl \
  --num-nodes 170 --task A --in-dim 1 \
  --gcn-bool --addaptadj --randomadj \
  --device cuda:0 --epochs 600 --seed 0 \
  --output-dir outputs/PEMS08_seed0
```

This is an invocation example, not a statement that seed 0 or this channel configuration was used for every manuscript experiment. Supply the recorded seed and task configuration for the run to reproduce. `--output-dir` must not already exist. Optional early stopping is controlled by `--patience` and `--min-epochs`; leave patience at 0 for fixed-epoch training.

Each run saves its best-validation-MAE model locally, then computes the test metrics with that checkpoint. Local output files are excluded from this repository. MAPE returned by the utilities is a ratio; multiply by 100 to report a percentage. Evaluation excludes zero targets under the implemented benchmark convention, which also excludes genuine zero traffic-flow values.

## Model API

The public model entry point is `NSSTAN` in [model.py](model.py):

```python
from model import NSSTAN
```

See [model.py](model.py) for the constructor signature and [engine.py](engine.py) for the training wrapper. Use the dataset's actual road supports and match the input-channel configuration to the prepared data.


Authors: Xin Liu, Yi Xu, Tongyu Zhu, Liangzhe Han, Mingzhe Liu and Leilei Sun.
