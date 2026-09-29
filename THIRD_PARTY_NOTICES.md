# Component attribution

This repository distributes the NSSTAN implementation only, not separate
baseline implementations, comparison runs, trained checkpoints, or results.
It retains building blocks adapted from the following projects:

* Graph WaveNet, Zonghan Wu and contributors (MIT): graph propagation,
  temporal convolution conventions, training/data utilities.
  https://github.com/nnzhan/Graph-WaveNet
* SOFTS, Xu-Yang Chen and contributors (MIT): the STAR aggregate-redistribute
  mechanism, adapted here to sensor features inside temporal pathways.
  https://github.com/Secilia-Cxy/SOFTS
* FAN, Weiwei Ye and contributors (Apache-2.0): frequency-adaptive
  decomposition/normalization and compatibility modules.
  https://github.com/wwy155/FAN

NSSTAN's adaptation uses selected spectral inputs and a residual, two separate
TCN-NAR pathways, fusion before a shared graph stage, and combined skip outputs.
Unused compatibility members are retained in constructors to preserve the
initialization order and existing checkpoint keys; they are not executed in
the forecast. This is not a claim that the original components are new.

The corresponding upstream license texts are included in `licenses/`.
Dataset access and attribution are described separately in `data/README.md`;
software licenses do not relicense third-party datasets.
