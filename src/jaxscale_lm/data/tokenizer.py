"""Tokenizers behind one small interface.

Two implementations:

- :class:`ByteTokenizer` — ids 0..255 are raw UTF-8 bytes, plus PAD/BOS/EOS
  (256/257/258). Zero training, fully deterministic, used by the tiny preset
  and every test.
- :class:`BpeTokenizer` — a Hugging Face ``tokenizers`` byte-level BPE model
  trained by :func:`train_bpe_tokenizer`, used by the small preset.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Protocol, runtime_checkable

from jaxscale_lm.config import TokenizerConfig

BYTE_VOCAB_SIZE = 259
PAD_ID = 256
BOS_ID = 257
EOS_ID = 258

_SPECIAL_TOKENS = ("<pad>", "<bos>", "<eos>")


@runtime_checkable
class Tokenizer(Protocol):
    """Minimal tokenizer interface used throughout the project."""

    @property
    def vocab_size(self) -> int: ...
    @property
    def pad_id(self) -> int: ...
    @property
    def bos_id(self) -> int: ...
    @property
    def eos_id(self) -> int: ...

    def encode(self, text: str) -> list[int]:
        """Encode text to token ids (no special tokens added)."""
        ...

    def decode(self, ids: Sequence[int]) -> str:
        """Decode token ids to text, skipping special tokens."""
        ...


class ByteTokenizer:
    """UTF-8 byte tokenizer with PAD/BOS/EOS appended after the byte range."""

    vocab_size = BYTE_VOCAB_SIZE
    pad_id = PAD_ID
    bos_id = BOS_ID
    eos_id = EOS_ID

    def encode(self, text: str) -> list[int]:
        return list(text.encode("utf-8"))

    def decode(self, ids: Sequence[int]) -> str:
        data = bytes(i for i in ids if 0 <= i < 256)
        return data.decode("utf-8", errors="replace")


class BpeTokenizer:
    """Byte-level BPE tokenizer loaded from a trained ``tokenizers`` file."""

    def __init__(self, path: Path) -> None:
        if not path.exists():
            raise FileNotFoundError(
                f"BPE tokenizer file not found: {path}. "
                f"Train one with: uv run python scripts/train_tokenizer.py --config <config>"
            )
        from tokenizers import Tokenizer as HFTokenizer

        self._tok = HFTokenizer.from_file(str(path))
        pad = self._tok.token_to_id("<pad>")
        bos = self._tok.token_to_id("<bos>")
        eos = self._tok.token_to_id("<eos>")
        if pad is None or bos is None or eos is None:
            raise ValueError(
                f"Tokenizer at {path} is missing one of the required special tokens "
                f"{_SPECIAL_TOKENS}; retrain it with scripts/train_tokenizer.py."
            )
        self._pad: int = pad
        self._bos: int = bos
        self._eos: int = eos

    @property
    def vocab_size(self) -> int:
        return self._tok.get_vocab_size()

    @property
    def pad_id(self) -> int:
        return self._pad

    @property
    def bos_id(self) -> int:
        return self._bos

    @property
    def eos_id(self) -> int:
        return self._eos

    def encode(self, text: str) -> list[int]:
        return self._tok.encode(text, add_special_tokens=False).ids

    def decode(self, ids: Sequence[int]) -> str:
        return self._tok.decode(list(ids), skip_special_tokens=True)


def train_bpe_tokenizer(
    texts: Iterable[str],
    vocab_size: int,
    output_path: Path,
) -> BpeTokenizer:
    """Train a byte-level BPE tokenizer and save it to ``output_path``.

    Byte-level pre-tokenization guarantees full coverage (no UNK token is
    needed) regardless of the training corpus.
    """
    if vocab_size <= len(_SPECIAL_TOKENS) + 256:
        raise ValueError(
            f"vocab_size must exceed {len(_SPECIAL_TOKENS) + 256} "
            f"(256 base bytes + {len(_SPECIAL_TOKENS)} specials); got {vocab_size}."
        )
    from tokenizers import Tokenizer as HFTokenizer
    from tokenizers import decoders, models, pre_tokenizers, trainers

    tok = HFTokenizer(models.BPE())
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=list(_SPECIAL_TOKENS),
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        show_progress=False,
    )
    tok.train_from_iterator(texts, trainer=trainer)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tok.save(str(output_path))
    return BpeTokenizer(output_path)


def build_tokenizer(config: TokenizerConfig) -> Tokenizer:
    """Construct the tokenizer described by the config."""
    if config.kind == "byte":
        return ByteTokenizer()
    if config.path is None:
        raise ValueError(
            "tokenizer.kind is 'bpe' but tokenizer.path is not set; "
            "point it at a file produced by scripts/train_tokenizer.py."
        )
    tokenizer = BpeTokenizer(config.path)
    if tokenizer.vocab_size != config.vocab_size:
        raise ValueError(
            f"Trained tokenizer at {config.path} has vocab_size={tokenizer.vocab_size} "
            f"but the config declares {config.vocab_size}; align them before training."
        )
    return tokenizer
