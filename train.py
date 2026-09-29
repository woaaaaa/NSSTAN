#!/usr/bin/env python3
"""Train and evaluate NSSTAN only; outputs are written locally."""

import os

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
os.environ.setdefault("MPLBACKEND", "Agg")

import argparse
import hashlib
import json
import platform
import random
import socket
import sys
import time
from pathlib import Path

import numpy as np
import torch

import util
from engine import trainer


def parse_args():
    parser = argparse.ArgumentParser(description="Seeded NSSTAN reproduction runner")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--data", required=True)
    parser.add_argument("--adjdata", required=True)
    parser.add_argument("--model", default="NSSTAN", choices=["NSSTAN", "fan_gwnet_SOFTS"])
    parser.add_argument("--task", choices=["A", "B"], required=True)
    parser.add_argument("--adjtype", default="doubletransition")
    parser.add_argument("--num-nodes", type=int, required=True)
    parser.add_argument("--in-dim", type=int, default=None)
    parser.add_argument("--seq-length", type=int, default=12)
    parser.add_argument("--layer-num", type=int, default=8)
    parser.add_argument("--nhid", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--weight-decay", type=float, default=0.0001)
    parser.add_argument("--epochs", type=int, default=600)
    parser.add_argument("--patience", type=int, default=0,
                        help="0 disables early stopping; otherwise stop after this many non-improving epochs")
    parser.add_argument("--min-epochs", type=int, default=0)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--print-every", type=int, default=1)
    parser.add_argument("--gcn-bool", action="store_true")
    parser.add_argument("--addaptadj", action="store_true")
    parser.add_argument("--aptonly", action="store_true")
    parser.add_argument("--randomadj", action="store_true")
    parser.add_argument("--deterministic", action="store_true",
                        help="Use deterministic cuDNN kernels while keeping other CUDA kernels unchanged")
    parser.add_argument("--strict-deterministic", action="store_true",
                        help="Also require deterministic algorithms globally; may be much slower on older PyTorch")
    return parser.parse_args()


def set_seed(seed, deterministic, strict_deterministic):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = bool(deterministic or strict_deterministic)
    if hasattr(torch, "use_deterministic_algorithms"):
        torch.use_deterministic_algorithms(bool(strict_deterministic), warn_only=True)


def target_tensor(y, task, device):
    value = torch.as_tensor(y, dtype=torch.float32, device=device).transpose(1, 3)
    return value[:, 1, :, :] if task == "B" else value[:, 0, :, :]


def input_tensor(x, device):
    return torch.as_tensor(x, dtype=torch.float32, device=device).transpose(1, 3)


def infer_in_dim(data_dir, requested):
    if requested is not None:
        return requested
    with np.load(Path(data_dir) / "train.npz") as sample:
        return int(sample["x"].shape[-1])


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def code_manifest():
    names = [
        "train.py", "engine.py", "model.py", "util.py",
        
    ]
    return {name: sha256(name) for name in names if Path(name).is_file()}


def save_json(path, value):
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(path)


def mean_metrics(values):
    array = np.asarray(values, dtype=np.float64)
    return {
        "mae": float(array[:, 0].mean()),
        "mape": float(array[:, 1].mean()),
        "rmse": float(array[:, 2].mean()),
    }


def main():
    args = parse_args()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=False)
    checkpoint_path = output_dir / "best.pt"
    history_path = output_dir / "history.jsonl"
    result_path = output_dir / "result.json"

    set_seed(args.seed, args.deterministic, args.strict_deterministic)
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError(f"CUDA device unavailable: {args.device}")
    torch.cuda.set_device(device)

    in_dim = infer_in_dim(args.data, args.in_dim)
    _, _, adj_mx = util.load_adj(args.adjdata, args.adjtype)
    dataloader = util.load_dataset(
        args.task, args.data, args.batch_size, args.batch_size, args.batch_size
    )
    scaler = dataloader["scaler"]
    supports = [torch.as_tensor(item, dtype=torch.float32, device=device) for item in adj_mx]
    adjinit = None if args.randomadj else supports[0]
    if args.aptonly:
        supports = None

    engine = trainer(
        scaler, args.task, args.model, in_dim, args.layer_num, args.seq_length,
        args.num_nodes, args.batch_size, args.nhid, args.dropout,
        args.learning_rate, args.weight_decay, device, supports,
        args.gcn_bool, args.addaptadj, adjinit,
    )

    # The archived engine enables anomaly detection on every batch. It is a
    # debugging aid rather than part of the optimization objective and greatly
    # increases runtime, so the reproduction runner disables that call.
    torch.autograd.set_detect_anomaly = lambda *unused_args, **unused_kwargs: None

    best_loss = float("inf")
    best_epoch = 0
    epochs_without_improvement = 0
    start_time = time.time()

    with history_path.open("w", encoding="utf-8") as history_file:
        for epoch in range(1, args.epochs + 1):
            epoch_start = time.time()
            dataloader["train_loader"].shuffle()
            train_values = []

            for x, y in dataloader["train_loader"].get_iterator():
                metrics = engine.train(
                    input_tensor(x, device),
                    target_tensor(y, args.task, device),
                    None, None, epoch, save_interval=0,
                )
                train_values.append(metrics)

            valid_values = []
            with torch.no_grad():
                for x, y in dataloader["val_loader"].get_iterator():
                    metrics = engine.eval(
                        input_tensor(x, device),
                        target_tensor(y, args.task, device),
                        None, None, epoch, save_interval=0,
                    )
                    valid_values.append(metrics)

            train_summary = mean_metrics(train_values)
            valid_summary = mean_metrics(valid_values)
            improved = valid_summary["mae"] < best_loss
            if improved:
                best_loss = valid_summary["mae"]
                best_epoch = epoch
                epochs_without_improvement = 0
                torch.save(engine.model.state_dict(), checkpoint_path)
            else:
                epochs_without_improvement += 1

            record = {
                "epoch": epoch,
                "seconds": time.time() - epoch_start,
                "train": train_summary,
                "valid": valid_summary,
                "improved": improved,
            }
            history_file.write(json.dumps(record, sort_keys=True) + "\n")
            history_file.flush()

            if epoch == 1 or epoch % args.print_every == 0 or improved:
                print(
                    f"epoch={epoch:04d} train_mae={train_summary['mae']:.4f} "
                    f"valid_mae={valid_summary['mae']:.4f} "
                    f"best={best_loss:.4f}@{best_epoch} "
                    f"seconds={record['seconds']:.2f}",
                    flush=True,
                )

            if (
                args.patience > 0
                and epoch >= args.min_epochs
                and epochs_without_improvement >= args.patience
            ):
                print(f"early_stop epoch={epoch} best_epoch={best_epoch}", flush=True)
                break

    try:
        state = torch.load(checkpoint_path, map_location=device, weights_only=True)
    except TypeError:
        state = torch.load(checkpoint_path, map_location=device)
    engine.model.load_state_dict(state)
    engine.model.eval()

    predictions = []
    with torch.no_grad():
        for x, _ in dataloader["test_loader"].get_iterator():
            batch_input = input_tensor(x, device)
            if in_dim == 1:
                batch_input = batch_input[:, :1, :, :] if args.task == "A" else batch_input[:, -1:, :, :]
            output = engine.model(batch_input)
            if output.ndim >= 4:
                output = output.transpose(1, 3)
            predictions.append(torch.squeeze(output))

    real = target_tensor(dataloader["y_test"], args.task, device)
    prediction = torch.cat(predictions, dim=0)[: real.shape[0]]
    if prediction.ndim == 4:
        prediction = prediction[:, 1, :, :]

    horizons = []
    for horizon in range(args.seq_length):
        pred_h = scaler.inverse_transform(prediction[:, :, horizon])
        real_h = real[:, :, horizon]
        mae, mape, rmse = util.metric(pred_h, real_h)
        horizons.append({
            "horizon": horizon + 1,
            "mae": float(mae),
            "mape": float(mape),
            "rmse": float(rmse),
        })

    averages = {
        metric: float(np.mean([row[metric] for row in horizons]))
        for metric in ("mae", "mape", "rmse")
    }
    result = {
        "schema_version": 1,
        "status": "completed",
        "dataset": Path(args.data).name,
        "task": args.task,
        "model": args.model,
        "seed": args.seed,
        "best_epoch": best_epoch,
        "best_valid_mae": best_loss,
        "epochs_completed": epoch,
        "test_average": averages,
        "test_horizons": horizons,
        "runtime_seconds": time.time() - start_time,
        "arguments": vars(args) | {"in_dim_effective": in_dim},
        "environment": {
            "hostname": socket.gethostname(),
            "platform": platform.platform(),
            "python": sys.version,
            "torch": torch.__version__,
            "cuda_runtime": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(device),
            "numpy": np.__version__,
        },
        "code_sha256": code_manifest(),
    }
    save_json(result_path, result)
    print("RESULT " + json.dumps(averages, sort_keys=True), flush=True)
    print(f"RESULT_FILE {result_path}", flush=True)


if __name__ == "__main__":
    main()
