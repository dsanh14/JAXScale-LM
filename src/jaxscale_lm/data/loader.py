"""Host-side batching.

The loader lives entirely on the host (NumPy): it shuffles packed blocks and
yields :class:`~jaxscale_lm.types.Batch` tuples of NumPy arrays. Device
placement/sharding is the trainer's job — keeping the loader out of jitted
code is deliberate (host I/O must never run inside a traced function).
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass

import numpy as np

from jaxscale_lm.config import DataConfig
from jaxscale_lm.data.dataset import load_documents, split_documents
from jaxscale_lm.data.packing import pack_documents, split_blocks
from jaxscale_lm.data.tokenizer import Tokenizer
from jaxscale_lm.types import Batch
from jaxscale_lm.utils.logging import get_logger, log_event
from jaxscale_lm.utils.seed import host_rng

_logger = get_logger("loader")


@dataclass(frozen=True)
class PackedDataset:
    """Packed blocks for one split, ready to batch."""

    blocks: np.ndarray  # [num_blocks, sequence_length + 1] int32
    pad_id: int

    @property
    def num_blocks(self) -> int:
        return int(self.blocks.shape[0])

    def batch_at(self, indices: np.ndarray) -> Batch:
        inputs, targets, mask = split_blocks(self.blocks[indices], self.pad_id)
        return Batch(input_ids=inputs, target_ids=targets, loss_mask=mask)


@dataclass(frozen=True)
class DataBundle:
    """Train and validation packed datasets plus the tokenizer that built them."""

    train: PackedDataset
    validation: PackedDataset
    tokenizer: Tokenizer


def build_data(config: DataConfig, tokenizer: Tokenizer, seed: int) -> DataBundle:
    """Load documents, split (document-level), tokenize, and pack both splits."""
    docs = load_documents(config, seed)
    splits = split_documents(docs, config.validation_fraction, seed)
    train_blocks = pack_documents(
        splits.train, tokenizer, config.sequence_length, drop_remainder=config.drop_remainder
    )
    val_blocks = pack_documents(
        splits.validation, tokenizer, config.sequence_length, drop_remainder=False
    )
    log_event(
        _logger,
        "packed dataset",
        source=config.source,
        train_documents=len(splits.train),
        validation_documents=len(splits.validation),
        train_blocks=int(train_blocks.shape[0]),
        validation_blocks=int(val_blocks.shape[0]),
        sequence_length=config.sequence_length,
    )
    return DataBundle(
        train=PackedDataset(train_blocks, tokenizer.pad_id),
        validation=PackedDataset(val_blocks, tokenizer.pad_id),
        tokenizer=tokenizer,
    )


def train_batches(
    dataset: PackedDataset,
    batch_size: int,
    seed: int,
    *,
    shuffle: bool = True,
    start_step: int = 0,
) -> Iterator[Batch]:
    """Infinite deterministic batch stream for training.

    Each epoch reshuffles with a generator derived from ``(seed, epoch)``, so
    the stream for a given seed is reproducible and — crucially for exact
    checkpoint resumption — ``start_step`` fast-forwards to the same batch
    sequence an uninterrupted run would have seen.
    """
    if dataset.num_blocks < batch_size:
        raise ValueError(
            f"Training split has {dataset.num_blocks} blocks but batch_size is "
            f"{batch_size}; reduce data.batch_size or provide more data."
        )
    batches_per_epoch = dataset.num_blocks // batch_size
    step = 0
    epoch = 0
    while True:
        if shuffle:
            order = host_rng(seed, f"epoch_{epoch}").permutation(dataset.num_blocks)
        else:
            order = np.arange(dataset.num_blocks)
        for i in range(batches_per_epoch):
            if step >= start_step:
                yield dataset.batch_at(order[i * batch_size : (i + 1) * batch_size])
            step += 1
        epoch += 1


def eval_batches(dataset: PackedDataset, batch_size: int, num_batches: int) -> Iterator[Batch]:
    """Finite, deterministic, unshuffled batch stream for evaluation.

    The final batch may be smaller than ``batch_size``; metric aggregation is
    token-weighted so this does not bias results. Stops after covering the
    split once even if ``num_batches`` asks for more.
    """
    available = (dataset.num_blocks + batch_size - 1) // batch_size
    for i in range(min(num_batches, available)):
        idx = np.arange(i * batch_size, min((i + 1) * batch_size, dataset.num_blocks))
        yield dataset.batch_at(idx)
