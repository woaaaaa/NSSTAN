# NSSTAN

PyTorch implementation of **Node-aware Spectral Spatio-Temporal Aggregation Network for Traffic Flow Forecasting**.

## Scope

This repository contains the NSSTAN model and its necessary training, data-loading and evaluation utilities. It does not distribute baseline implementations, comparison experiments, experimental results, training logs or trained checkpoints.

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

```python
import torch
from model import NSSTAN

# Replace these demonstration supports with the dataset's actual road supports.
device = torch.device("cpu")
nodes = 170
supports = [torch.eye(nodes), torch.eye(nodes)]
model = NSSTAN(
    device=device, num_nodes=nodes, dropout=0.3, supports=supports,
    gcn_bool=True, addaptadj=True, aptinit=None,
    in_dim=1, seq_length=12, nhid=32,
    fan_freq_topk=1, fan_rfft=True,
).to(device)
model.eval()
```

`NSSTAN` is an alias of the preserved `fan_gwnet_SOFTS` class. The computational definitions and parameter initialization order are retained. Some constructor members unused by the forecast remain for compatibility with existing state dictionaries and initialization order. They should not be removed when reproducing existing runs without accounting for the resulting change in random-number consumption.

## Attribution

NSSTAN adapts building blocks from Graph WaveNet, FAN and SOFTS. See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) and the upstream licenses in `licenses/`. These are necessary components of NSSTAN, not independently distributed baseline models.

Authors: Xin Liu, Yi Xu, Tongyu Zhu, Liangzhe Han, Mingzhe Liu and Leilei Sun.
