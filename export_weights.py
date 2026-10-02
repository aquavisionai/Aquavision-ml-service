"""Export the trusted training checkpoint without optimizer/scheduler tensors."""
import argparse
import hashlib
from pathlib import Path

import torch


def export_weights(source: Path, destination: Path):
    if source.resolve() == destination.resolve():
        raise ValueError("Export to a different path; preserve the training checkpoint.")
    checkpoint = torch.load(source, map_location="cpu", weights_only=True)
    with source.open("rb") as stream:
        digest = checkpoint.get("source_sha256") or hashlib.file_digest(stream, "sha256").hexdigest()
    artifact = {"model_state_dict": checkpoint["model_state_dict"], "epoch": checkpoint.get("epoch"),
                "source_sha256": digest}
    torch.save(artifact, destination)
    print(f"Exported {source.stat().st_size:,} -> {destination.stat().st_size:,} bytes (epoch={artifact['epoch']})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path, nargs="?", default=Path(__file__).with_name("best.pt"))
    parser.add_argument("destination", type=Path, nargs="?", default=Path(__file__).with_name("inference.pt"))
    args = parser.parse_args()
    export_weights(args.source, args.destination)
