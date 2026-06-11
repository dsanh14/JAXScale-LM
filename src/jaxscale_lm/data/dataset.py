"""Text sources: deterministic synthetic, local plain text, and TinyStories.

Every source returns a list of *documents* (strings). The train/validation
split happens at the document level, **before** packing, so no token of any
validation document can leak into a training sequence.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from jaxscale_lm.config import DataConfig
from jaxscale_lm.utils.logging import get_logger, log_event
from jaxscale_lm.utils.seed import host_rng

_logger = get_logger("data")

# Small word inventory for the synthetic corpus: enough structure that a
# tiny model's loss visibly decreases, fully deterministic under the seed.
_SYNTH_WORDS = [
    "the",
    "cat",
    "sat",
    "on",
    "a",
    "mat",
    "and",
    "saw",
    "the",
    "dog",
    "run",
    "to",
    "the",
    "red",
    "house",
    "near",
    "a",
    "tall",
    "tree",
    "while",
    "birds",
    "sang",
    "over",
    "green",
    "hills",
    "under",
    "one",
    "bright",
    "sun",
]


@dataclass(frozen=True)
class DocumentSplits:
    """Documents partitioned into disjoint train and validation sets."""

    train: tuple[str, ...]
    validation: tuple[str, ...]


def synthetic_documents(num_documents: int, document_length: int, seed: int) -> list[str]:
    """Generate word-salad documents deterministically from the seed."""
    rng = host_rng(seed, "synthetic")
    docs: list[str] = []
    for _ in range(num_documents):
        words: list[str] = []
        size = 0
        while size < document_length:
            word = _SYNTH_WORDS[int(rng.integers(0, len(_SYNTH_WORDS)))]
            words.append(word)
            size += len(word) + 1
        docs.append(" ".join(words))
    return docs


def local_text_documents(path: Path) -> list[str]:
    """Load documents from a .txt file (one doc) or a directory of .txt files."""
    if path.is_file():
        files = [path]
    elif path.is_dir():
        files = sorted(path.glob("*.txt"))
        if not files:
            raise FileNotFoundError(f"No .txt files found in directory {path}")
    else:
        raise FileNotFoundError(f"data.local_path does not exist: {path}")
    docs = [f.read_text(encoding="utf-8") for f in files]
    return [d for d in docs if d.strip()]


def tinystories_documents(cache_dir: Path, max_documents: int | None) -> list[str]:
    """Load TinyStories via Hugging Face Datasets (downloads on first use)."""
    try:
        from datasets import load_dataset
    except ImportError as exc:  # pragma: no cover - environment issue
        raise ImportError(
            "The 'datasets' package is required for data.source=tinystories; run `uv sync`."
        ) from exc

    split = "train" if max_documents is None else f"train[:{max_documents}]"
    ds = load_dataset("roneneldan/TinyStories", split=split, cache_dir=str(cache_dir))
    log_event(_logger, "loaded tinystories", documents=len(ds), cache_dir=str(cache_dir))
    texts: list[str] = list(ds["text"])  # pyright: ignore[reportIndexIssue, reportArgumentType]
    return texts


def load_documents(config: DataConfig, seed: int) -> list[str]:
    """Load raw documents for the configured source."""
    if config.source == "synthetic":
        return synthetic_documents(
            config.synthetic_num_documents, config.synthetic_document_length, seed
        )
    if config.source == "local_text":
        assert config.local_path is not None  # validated by DataConfig
        docs = local_text_documents(config.local_path)
    else:  # tinystories
        docs = tinystories_documents(config.cache_dir, config.max_train_documents)
    if config.max_train_documents is not None:
        docs = docs[: config.max_train_documents]
    return docs


def split_documents(docs: list[str], validation_fraction: float, seed: int) -> DocumentSplits:
    """Shuffle and split documents into disjoint train/validation sets.

    At least one document lands on each side; fewer than two documents is an
    error because a leak-free split would be impossible.
    """
    if len(docs) < 2:
        raise ValueError(
            f"Need at least 2 documents to build disjoint train/validation splits, "
            f"got {len(docs)}. Provide more data or raise synthetic_num_documents."
        )
    order = host_rng(seed, "split").permutation(len(docs))
    n_val = min(max(1, round(len(docs) * validation_fraction)), len(docs) - 1)
    val_idx = set(order[:n_val].tolist())
    train = tuple(docs[i] for i in range(len(docs)) if i not in val_idx)
    validation = tuple(docs[i] for i in range(len(docs)) if i in val_idx)
    return DocumentSplits(train=train, validation=validation)
