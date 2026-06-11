"""Sequence construction: concatenate-and-chunk packing.

The primary training path tokenizes each document, joins them with EOS
separators into one long token stream, and slices the stream into
fixed-length blocks of ``sequence_length + 1`` ids. A block yields
``input_ids = block[:-1]`` and ``target_ids = block[1:]`` — the standard
next-token shift. Because blocks are full by construction, no padding or
attention mask is needed in training; the loss mask only zeroes targets that
are PAD (which occurs solely in the optional padded final block).
"""

from __future__ import annotations

import numpy as np

from jaxscale_lm.data.tokenizer import Tokenizer


def pack_documents(
    documents: tuple[str, ...] | list[str],
    tokenizer: Tokenizer,
    sequence_length: int,
    *,
    drop_remainder: bool = True,
) -> np.ndarray:
    """Tokenize, concatenate with EOS separators, and chunk.

    Returns:
        int32 array of shape ``[num_blocks, sequence_length + 1]``.

    Raises:
        ValueError: if the corpus is too short to produce a single block
            (with ``drop_remainder=True``) or is empty.
    """
    if sequence_length <= 0:
        raise ValueError(f"sequence_length must be positive, got {sequence_length}")

    stream: list[int] = []
    for doc in documents:
        ids = tokenizer.encode(doc)
        if ids:
            stream.extend(ids)
            stream.append(tokenizer.eos_id)
    if not stream:
        raise ValueError("No tokens produced from the given documents; corpus is empty.")

    block = sequence_length + 1
    num_full = len(stream) // block
    remainder = len(stream) % block

    if num_full == 0 and drop_remainder:
        raise ValueError(
            f"Corpus has only {len(stream)} tokens but one packed block needs "
            f"{block} (sequence_length + 1). Provide more text, lower "
            f"data.sequence_length, or set data.drop_remainder=false to pad."
        )

    blocks = np.asarray(stream[: num_full * block], dtype=np.int32).reshape(num_full, block)
    if remainder and not drop_remainder:
        tail = np.full((1, block), tokenizer.pad_id, dtype=np.int32)
        tail[0, :remainder] = stream[num_full * block :]
        blocks = np.concatenate([blocks, tail], axis=0) if num_full else tail
    return blocks


def split_blocks(blocks: np.ndarray, pad_id: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Turn packed blocks into (input_ids, target_ids, loss_mask).

    ``loss_mask`` is 1.0 where the *target* is a real token, 0.0 where it is
    padding; with concatenate-and-chunk packing only an optional final padded
    block has zeros.
    """
    if blocks.ndim != 2 or blocks.shape[1] < 2:
        raise ValueError(
            f"Expected blocks of shape [N, sequence_length + 1] with at least 2 "
            f"columns, got {blocks.shape}."
        )
    input_ids = blocks[:, :-1].astype(np.int32)
    target_ids = blocks[:, 1:].astype(np.int32)
    loss_mask = (target_ids != pad_id).astype(np.float32)
    return input_ids, target_ids, loss_mask
