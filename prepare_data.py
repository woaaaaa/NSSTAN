#!/usr/bin/env python3
"""Create explicitly configured NSSTAN windows from public raw measurements.

This utility does not infer historical experiment configurations. Specify the
feature order, split convention and graph settings required for your run.
"""
from pathlib import Path
import argparse
import csv
import json
import pickle
import shutil

import numpy as np


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--raw', type=Path, required=True, help='NPZ data array or benchmark HDF5 dataframe')
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--channels', default='0', help='Raw NPZ channels in the required output order')
    p.add_argument('--split', type=float, nargs=3, required=True, metavar=('TRAIN', 'VAL', 'TEST'))
    p.add_argument('--split-mode', choices=['time', 'window'], required=True,
                   help='Split raw time first, or construct all windows before chronological partition')
    p.add_argument('--history', type=int, default=12)
    p.add_argument('--horizon', type=int, default=12)
    p.add_argument('--adj-pickle', type=Path, help='Copy a trusted, existing benchmark graph unchanged')
    p.add_argument('--adj-csv', type=Path, help='Build a graph from a source,target,distance CSV')
    p.add_argument('--sensor-ids', type=Path, help='Sensor IDs in array order, one ID per line')
    p.add_argument('--graph-weighting', choices=['binary', 'gaussian'])
    p.add_argument('--directed', action='store_true')
    p.add_argument('--threshold', type=float, default=0.1)
    return p.parse_args()


def windows(values, history, horizon):
    count = len(values) - history - horizon + 1
    if count <= 0:
        raise ValueError('Partition is too short for the requested windows')
    x = np.stack([values[t:t+history] for t in range(count)])
    y = np.stack([values[t+history:t+history+horizon] for t in range(count)])
    return x, y


def graph_from_csv(args, nodes):
    if args.graph_weighting is None:
        raise ValueError('--graph-weighting is required for CSV graphs; it must match the run configuration')
    ids = args.sensor_ids.read_text().split() if args.sensor_ids else [str(i) for i in range(nodes)]
    if len(ids) != nodes or len(set(ids)) != nodes:
        raise ValueError('Sensor-ID order does not match the measurement node axis')
    mapping = {str(sensor): i for i, sensor in enumerate(ids)}
    edges = []
    with args.adj_csv.open(newline='') as f:
        for row in csv.reader(f):
            if len(row) < 3:
                continue
            try:
                a, b, distance = row[0].strip(), row[1].strip(), float(row[2])
            except ValueError:
                continue  # header
            if a not in mapping or b not in mapping:
                raise ValueError(f'Unknown sensor ID in graph: {a}, {b}')
            edges.append((mapping[a], mapping[b], distance))
    if not edges:
        raise ValueError('No graph edges were found')
    scale = float(np.std([e[2] for e in edges]))
    if args.graph_weighting == 'gaussian' and scale <= 0:
        raise ValueError('Gaussian distance scale must be positive')
    adjacency = np.zeros((nodes, nodes), dtype=np.float32)
    for a, b, distance in edges:
        weight = 1.0 if args.graph_weighting == 'binary' else np.exp(-(distance / scale)**2)
        if weight < args.threshold:
            continue
        adjacency[a, b] = weight
        if not args.directed:
            adjacency[b, a] = weight
    return ids, mapping, adjacency


def main():
    args = parse_args()
    if not np.isclose(sum(args.split), 1.0) or min(args.split) <= 0:
        raise ValueError('Split fractions must be positive and sum to one')
    if args.output.exists():
        raise FileExistsError(f'Refusing to overwrite {args.output}')
    if bool(args.adj_pickle) == bool(args.adj_csv):
        raise ValueError('Provide exactly one of --adj-pickle or --adj-csv')
    if args.raw.suffix.lower() in ['.h5', '.hdf5']:
        import pandas as pd
        frame = pd.read_hdf(args.raw)
        values = frame.to_numpy()[..., None]
    else:
        with np.load(args.raw, allow_pickle=False) as z:
            values = z['data']
        if values.ndim == 2:
            values = values[..., None]
    channels = [int(x) for x in args.channels.split(',')]
    values = values[..., channels].astype(np.float32)
    if values.ndim != 3 or not np.isfinite(values).all():
        raise ValueError('Expected finite [time, nodes, features] measurements')
    graph = None if args.adj_pickle else graph_from_csv(args, values.shape[1])
    args.output.mkdir(parents=True)
    if args.split_mode == 'time':
        a = int(len(values) * args.split[0]); b = int(len(values) * sum(args.split[:2]))
        partitions = [values[:a], values[a:b], values[b:]]
        batches = ((name, *windows(v, args.history, args.horizon))
                   for name, v in zip(['train', 'val', 'test'], partitions))
    else:
        x, y = windows(values, args.history, args.horizon)
        # DCRNN-style rounding for counts; verify this against your target protocol.
        n = len(x); n_train = round(n * args.split[0]); n_test = round(n * args.split[2])
        cuts = [slice(0, n_train), slice(n_train, n-n_test), slice(n-n_test, n)]
        batches = ((name, x[cut], y[cut]) for name, cut in zip(['train', 'val', 'test'], cuts))
    for name, x, y in batches:
        np.savez_compressed(args.output/(name+'.npz'), x=x, y=y)
        print(name, x.shape, y.shape)
    if args.adj_pickle:
        shutil.copy2(args.adj_pickle, args.output/'adj_mx.pkl')
    else:
        with (args.output/'adj_mx.pkl').open('wb') as f:
            pickle.dump(graph, f, protocol=4)
    settings = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    (args.output/'preprocessing.json').write_text(json.dumps(settings, indent=2)+'\n')


if __name__ == '__main__':
    main()
