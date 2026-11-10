"""Plot benchmark_bridge.py JSON; needs matplotlib (pip install ".[benchmark]")."""
import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("report", type=Path)
    p.add_argument("--out", type=Path, default=Path("bridge_benchmark.svg"))
    args = p.parse_args()
    data = json.loads(args.report.read_text())
    fig, ax = plt.subplots(figsize=(6.4, 4))
    for arm in ("base", "plumed"):
        rows = sorted((r for r in data["results"] if r["arm"] == arm), key=lambda r: r["systems"])
        ax.errorbar([r["systems"] for r in rows], [r["ms_per_system_call_mean"] for r in rows],
                    yerr=[r["ms_per_batch_call_stdev"] / r["systems"] for r in rows], marker="o", capsize=3, label=arm)
    ax.set(xlabel="Independent systems per batch", ylabel="ms per system force call")
    ax.set_xticks(sorted({r["systems"] for r in data["results"]}))
    ax.legend()
    ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out)
    plt.close(fig)


if __name__ == "__main__":
    main()
