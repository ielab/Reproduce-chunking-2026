#!/usr/bin/env python3
"""
Produce publication-ready figures aligned with the paper's research questions.

Outputs (saved under analysis/figures by default):
  - rq1_multi_document.png
  - rq2_cross_paradigm.png
  - rq3_contextualized_gain.png
  - rq4_late_chunking_sweep.png (generated only when enough metadata is available)

Each figure summarizes evaluation scores stored under src/chunked_output/.
The script is resilient to missing runs—plots are skipped with a warning
when the required data is unavailable.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import pandas as pd

try:
    import matplotlib.pyplot as plt
except ImportError as exc:  # pragma: no cover - optional dependency
    raise SystemExit(
        "matplotlib is required for plotting. Install it with `pip install matplotlib`."
    ) from exc

# ----------------------------
# Configuration
# ----------------------------

DEFAULT_BASE_PATH = Path("src/chunked_output")
DEFAULT_OUTPUT_DIR = Path("analysis/figures")
DEFAULT_MODELS = [
    "jina-embeddings-v2-small-en",
    "jina-embeddings-v3",
    "nomic-embed-text-v1",
    "multilingual-e5-large-instruct",
]
DEFAULT_ENCODERS = ["RegularEncoder", "LateEncoder"]

DATASET_GROUP = {
    "GutenQA": "document",
}

CHUNKER_DISPLAY_NAMES = {
    "ParagraphChunker": "Paragraph",
    "SentenceChunker": "Sentence",
    "FixedSizeChunker": "Fixed-Size",
    "SemanticChunker": "Semantic Boundary",
    "LumberChunker": "Lumber",
    "Proposition": "Proposition",
}

CHUNKER_ORDER = [
    "Paragraph",
    "Sentence",
    "Fixed-Size",
    "Semantic Boundary",
    "Lumber (GPT)",
    "Lumber (Gemini)",
    "Lumber (LLM)",
]

METRIC_FILE_BY_DATASET = defaultdict(lambda: "nDCG@10.eval", {"GutenQA": "DCG@10.eval"})

NUMERIC_CONFIG_KEYS = (
    "fixed_size",
    "chunk_size",
    "window_size",
    "target_chunk_size",
    "max_tokens",
)


# ----------------------------
# IO helpers
# ----------------------------

def parse_eval_file(path: Path) -> Optional[float]:
    """Return the averaged score from an *.eval file."""
    if not path.exists():
        return None

    try:
        with path.open("r", encoding="utf-8") as fh:
            for line in reversed(fh.readlines()):
                parts = line.strip().split()
                if len(parts) == 2 and parts[0] == "average":
                    return float(parts[1])
    except ValueError:
        return None
    return None


def infer_chunker_name(run_id: str) -> str:
    """Best-effort guess of the chunker class from the run directory name."""
    return run_id.split("-")[0]


def load_chunk_metadata(base_path: Path) -> Dict[Tuple[str, str], Dict[str, Any]]:
    """
    Load chunker metadata from manifests when available.

    Returns a dict keyed by (dataset, chunk_run_id) containing:
      - chunker_name
      - config (dict with chunker kwargs)
    """
    metadata: Dict[Tuple[str, str], Dict[str, Any]] = {}

    for dataset_dir in base_path.iterdir():
        if not dataset_dir.is_dir():
            continue
        chunk_dir = dataset_dir / "chunks"
        if not chunk_dir.exists():
            continue
        for run_dir in chunk_dir.iterdir():
            if not run_dir.is_dir():
                continue

            chunk_run_id = run_dir.name
            manifest_path = run_dir / "manifest.json"
            chunker_name: Optional[str] = None
            config: Dict[str, Any] = {}

            if manifest_path.exists():
                try:
                    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                    chunker_data = manifest.get("chunker", {})
                    chunker_name = chunker_data.get("chunker_name")
                    config = {
                        k: v for k, v in chunker_data.items() if k != "chunker_name"
                    }
                except json.JSONDecodeError:
                    pass

            if chunker_name is None:
                chunker_name = infer_chunker_name(chunk_run_id)

            metadata[(dataset_dir.name, chunk_run_id)] = {
                "chunker_name": chunker_name,
                "config": config,
            }

    # Ensure we also capture result-only runs (no manifest available)
    for dataset_dir in base_path.iterdir():
        if not dataset_dir.is_dir():
            continue
        results_dir = dataset_dir / "results"
        if not results_dir.exists():
            continue
        for run_dir in results_dir.iterdir():
            if not run_dir.is_dir():
                continue
            key = (dataset_dir.name, run_dir.name)
            if key not in metadata:
                metadata[key] = {
                    "chunker_name": infer_chunker_name(run_dir.name),
                    "config": {},
                }

    return metadata


def extract_first_integer(text: str) -> Optional[int]:
    match = re.search(r"(\d+)", text)
    if match:
        value = int(match.group(1))
        # Ignore values that are clearly timestamps or run ids
        if value > 10000:
            return None
        return value
    return None


def compute_chunk_labels(
    chunker_name: str,
    chunk_run_id: str,
    config: Dict[str, Any],
) -> Tuple[str, str]:
    """
    Return (base_label, variant_label) for plotting.
    base_label groups chunkers of the same family.
    variant_label keeps configuration-specific details.
    """
    base_label = CHUNKER_DISPLAY_NAMES.get(chunker_name, chunker_name)
    variant_label = base_label

    chunk_run_lower = chunk_run_id.lower()
    generator = (config.get("generator_name") or "").lower()

    if chunker_name == "LumberChunker":
        if "gemini" in chunk_run_lower or "gemini" in generator:
            base_label = "Lumber (Gemini)"
        elif "gpt" in chunk_run_lower or "gpt" in generator:
            base_label = "Lumber (GPT)"
        else:
            base_label = "Lumber (LLM)"
        variant_label = base_label
    elif chunker_name == "FixedSizeChunker":
        chunk_size = None
        for key in NUMERIC_CONFIG_KEYS:
            value = config.get(key)
            if isinstance(value, (int, float)):
                chunk_size = int(value)
                break
        if chunk_size is None:
            chunk_size = extract_first_integer(chunk_run_id)
        if chunk_size is not None:
            variant_label = f"{base_label} ({chunk_size})"
        else:
            variant_label = base_label
    else:
        variant_label = base_label

    return base_label, variant_label


def collect_results(
    base_path: Path,
    models: Iterable[str],
    encoders: Iterable[str],
) -> pd.DataFrame:
    """
    Assemble a tidy dataframe with evaluation scores and chunker metadata.
    """
    chunk_metadata = load_chunk_metadata(base_path)
    rows: List[Dict[str, Any]] = []

    for (dataset, chunk_run_id), meta in chunk_metadata.items():
        dataset_dir = base_path / dataset
        result_dir = dataset_dir / "results" / chunk_run_id
        if not result_dir.exists():
            continue

        chunker_name = meta["chunker_name"]
        config = meta.get("config", {}) or {}
        base_label, variant_label = compute_chunk_labels(
            chunker_name, chunk_run_id, config
        )

        dataset_group = DATASET_GROUP.get(dataset, "corpus")
        metric_file = METRIC_FILE_BY_DATASET[dataset]

        for encoder in encoders:
            for model in models:
                score_path = result_dir / f"{encoder}-{model}" / metric_file
                score = parse_eval_file(score_path)
                if score is None:
                    continue

                numeric_config = {key: config.get(key) for key in NUMERIC_CONFIG_KEYS}

                chunk_size = None
                for value in numeric_config.values():
                    if isinstance(value, (int, float)):
                        chunk_size = int(value)
                        break
                    if isinstance(value, str) and value.isdigit():
                        chunk_size = int(value)
                        break
                if chunk_size is None and chunker_name == "FixedSizeChunker":
                    chunk_size = extract_first_integer(chunk_run_id)

                rows.append(
                    {
                        "dataset": dataset,
                        "dataset_group": dataset_group,
                        "chunk_run_id": chunk_run_id,
                        "chunker_name": chunker_name,
                        "chunker_base": base_label,
                        "chunker_variant": variant_label,
                        "model": model,
                        "encoder": encoder,
                        "score": score,
                        "chunk_size": chunk_size,
                        "chunker_config": config,
                    }
                )

    return pd.DataFrame(rows)


# ----------------------------
# Plotting helpers
# ----------------------------

def ensure_output_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def format_dataset_name(name: str) -> str:
    return name.replace("-", "\u2011")  # use non-breaking hyphen for prettier labels


def plot_rq1(df: pd.DataFrame, output_dir: Path) -> None:
    """
    RQ1 – Compare segmentation methods on BEIR (multi-document) retrieval.
    """
    subset = df[
        (df["dataset_group"] == "corpus")
        & (df["encoder"] == "RegularEncoder")
    ]
    if subset.empty:
        print("RQ1: No RegularEncoder corpus results found; skipping figure.")
        return

    agg = (
        subset.groupby(["dataset", "chunker_base"])["score"]
        .mean()
        .reset_index()
    )

    pivot = agg.pivot(index="chunker_base", columns="dataset", values="score")
    order = [label for label in CHUNKER_ORDER if label in pivot.index]
    if order:
        pivot = pivot.reindex(order)
    else:
        pivot = pivot.sort_index()
    pivot.columns = [format_dataset_name(col) for col in pivot.columns]

    fig, ax = plt.subplots(figsize=(10, 6))
    pivot.plot(kind="bar", ax=ax)
    ax.set_ylabel("nDCG@10")
    ax.set_xlabel("Segmentation strategy")
    ax.set_title("RQ1: Multi-document retrieval performance (RegularEncoder)")
    ax.legend(title="Dataset", bbox_to_anchor=(1.02, 1), loc="upper left")
    fig.tight_layout()

    fig_path = output_dir / "rq1_multi_document.png"
    fig.savefig(fig_path, dpi=300)
    plt.close(fig)
    print(f"RQ1 figure saved to {fig_path}")


def plot_rq2(df: pd.DataFrame, output_dir: Path) -> None:
    """
    RQ2 – Assess whether segmentation rankings transfer between corpus and document settings.
    """
    subset = df[df["encoder"] == "RegularEncoder"]
    if subset.empty:
        print("RQ2: No RegularEncoder results found; skipping figure.")
        return

    corpus_scores = (
        subset[subset["dataset_group"] == "corpus"]
        .groupby("chunker_base")["score"]
        .mean()
    )
    document_scores = (
        subset[subset["dataset_group"] == "document"]
        .groupby("chunker_base")["score"]
        .mean()
    )

    combined = pd.DataFrame(
        {
            "Corpus avg (nDCG@10)": corpus_scores,
            "Document avg (DCG@10)": document_scores,
        }
    ).dropna()

    if combined.empty:
        print("RQ2: Insufficient overlap between corpus and document chunkers; skipping.")
        return

    order = [label for label in CHUNKER_ORDER if label in combined.index]
    if order:
        combined = combined.reindex(order)
    else:
        combined = combined.sort_index()

    fig, ax = plt.subplots(figsize=(8, 5))
    combined.plot(kind="bar", ax=ax)
    ax.set_ylabel("Score")
    ax.set_xlabel("Segmentation strategy")
    ax.set_title("RQ2: Cross-paradigm robustness of segmentation methods")
    ax.legend(loc="upper left")
    fig.tight_layout()

    fig_path = output_dir / "rq2_cross_paradigm.png"
    fig.savefig(fig_path, dpi=300)
    plt.close(fig)
    print(f"RQ2 figure saved to {fig_path}")


def plot_rq3(df: pd.DataFrame, output_dir: Path) -> None:
    """
    RQ3 – Quantify improvements from contextualized chunking (LateEncoder vs RegularEncoder).
    """
    if not {"RegularEncoder", "LateEncoder"}.issubset(set(df["encoder"].unique())):
        print("RQ3: Need both RegularEncoder and LateEncoder results; skipping.")
        return

    regular = df[df["encoder"] == "RegularEncoder"]
    late = df[df["encoder"] == "LateEncoder"]

    merged = pd.merge(
        regular,
        late,
        on=["dataset", "dataset_group", "chunk_run_id", "chunker_base", "model"],
        suffixes=("_regular", "_late"),
    )
    if merged.empty:
        print("RQ3: No overlapping runs between Regular and Late encoders; skipping.")
        return

    merged["delta_pct"] = (
        (merged["score_late"] - merged["score_regular"])
        / merged["score_regular"]
        * 100
    )

    agg = (
        merged.groupby(["chunker_base", "dataset_group"])["delta_pct"]
        .mean()
        .reset_index()
    )
    pivot = agg.pivot(index="chunker_base", columns="dataset_group", values="delta_pct")
    order = [label for label in CHUNKER_ORDER if label in pivot.index]
    if order:
        pivot = pivot.reindex(order)
    else:
        pivot = pivot.sort_index()

    fig, ax = plt.subplots(figsize=(8, 5))
    pivot.plot(kind="bar", ax=ax)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_ylabel("Relative gain (%)")
    ax.set_xlabel("Segmentation strategy")
    ax.set_title("RQ3: Late chunking gains over pre-embedding baseline")
    ax.legend(title="Retrieval scope", loc="upper left")
    fig.tight_layout()

    fig_path = output_dir / "rq3_contextualized_gain.png"
    fig.savefig(fig_path, dpi=300)
    plt.close(fig)
    print(f"RQ3 figure saved to {fig_path}")


def plot_rq4(df: pd.DataFrame, output_dir: Path) -> None:
    """
    RQ4 – Explore when contextualized chunking pays off, focusing on fixed-size sweeps.
    """
    if not {"RegularEncoder", "LateEncoder"}.issubset(set(df["encoder"].unique())):
        print("RQ4: Need both RegularEncoder and LateEncoder results; skipping.")
        return

    focus_chunkers = df["chunker_name"].unique()
    if "FixedSizeChunker" not in focus_chunkers:
        print("RQ4: No FixedSizeChunker runs detected; skipping.")
        return

    regular = df[(df["encoder"] == "RegularEncoder") & (df["chunker_name"] == "FixedSizeChunker")]
    late = df[(df["encoder"] == "LateEncoder") & (df["chunker_name"] == "FixedSizeChunker")]

    merged = pd.merge(
        regular,
        late,
        on=["dataset", "dataset_group", "chunk_run_id", "model"],
        suffixes=("_regular", "_late"),
    )
    if merged.empty:
        print("RQ4: No overlapping FixedSizeChunker runs between encoders; skipping.")
        return

    merged["chunk_size"] = merged["chunk_size_regular"].combine_first(
        merged["chunk_size_late"]
    )
    merged = merged.dropna(subset=["chunk_size"])
    if merged.empty:
        print("RQ4: Chunk size metadata missing; skipping.")
        return

    merged["delta_pct"] = (
        (merged["score_late"] - merged["score_regular"])
        / merged["score_regular"]
        * 100
    )

    agg = (
        merged.groupby(["chunk_size", "dataset_group"])["delta_pct"]
        .mean()
        .reset_index()
    )

    fig, ax = plt.subplots(figsize=(8, 5))
    for scope, scoped_df in agg.groupby("dataset_group"):
        ax.plot(
            scoped_df["chunk_size"],
            scoped_df["delta_pct"],
            marker="o",
            label=f"{scope.capitalize()} retrieval",
        )

    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Fixed-size window (tokens)")
    ax.set_ylabel("Relative gain (%)")
    ax.set_title("RQ4: Late chunking benefit across window sizes")
    ax.legend()
    fig.tight_layout()

    fig_path = output_dir / "rq4_late_chunking_sweep.png"
    fig.savefig(fig_path, dpi=300)
    plt.close(fig)
    print(f"RQ4 figure saved to {fig_path}")


# ----------------------------
# CLI
# ----------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create plots for RQ1-RQ4 based on evaluation artifacts."
    )
    parser.add_argument(
        "--base_path",
        type=Path,
        default=DEFAULT_BASE_PATH,
        help="Directory containing dataset outputs (default: src/chunked_output)",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory to store generated figures (default: analysis/figures)",
    )
    parser.add_argument(
        "--models",
        nargs="*",
        default=DEFAULT_MODELS,
        help="List of embedding model identifiers (folder names).",
    )
    parser.add_argument(
        "--encoders",
        nargs="*",
        default=DEFAULT_ENCODERS,
        help="Encoder names to load (must match result folder prefixes).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    base_path = args.base_path
    output_dir = ensure_output_dir(args.output_dir)

    if not base_path.exists():
        raise SystemExit(f"Base path '{base_path}' does not exist. Nothing to plot.")

    df = collect_results(base_path, args.models, args.encoders)
    if df.empty:
        raise SystemExit(
            "No evaluation results found. Ensure run_evaluator has produced *.eval files."
        )

    plot_rq1(df, output_dir)
    plot_rq2(df, output_dir)
    plot_rq3(df, output_dir)
    plot_rq4(df, output_dir)


if __name__ == "__main__":
    main()
