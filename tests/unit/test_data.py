"""Data pipeline tests: tokenizers, packing, splitting, loading."""

from __future__ import annotations

import numpy as np
import pytest

from jaxscale_lm.config import DataConfig
from jaxscale_lm.data.dataset import split_documents, synthetic_documents
from jaxscale_lm.data.loader import build_data, eval_batches, train_batches
from jaxscale_lm.data.packing import pack_documents, split_blocks
from jaxscale_lm.data.tokenizer import (
    BOS_ID,
    EOS_ID,
    PAD_ID,
    ByteTokenizer,
    train_bpe_tokenizer,
)

pytestmark = pytest.mark.unit


class TestByteTokenizer:
    def test_round_trip_ascii(self):
        tok = ByteTokenizer()
        text = "Once upon a time."
        assert tok.decode(tok.encode(text)) == text

    def test_round_trip_unicode(self):
        tok = ByteTokenizer()
        text = "héllo wörld — 日本語"
        assert tok.decode(tok.encode(text)) == text

    def test_specials_skipped_on_decode(self):
        tok = ByteTokenizer()
        ids = [*tok.encode("hi"), EOS_ID, PAD_ID, BOS_ID]
        assert tok.decode(ids) == "hi"

    def test_vocab_layout(self):
        tok = ByteTokenizer()
        assert tok.vocab_size == 259
        assert (tok.pad_id, tok.bos_id, tok.eos_id) == (256, 257, 258)


class TestBpeTokenizer:
    def test_train_and_round_trip(self, tmp_path):
        corpus = ["the cat sat on the mat"] * 50 + ["a dog ran to the house"] * 50
        tok = train_bpe_tokenizer(corpus, vocab_size=300, output_path=tmp_path / "bpe.json")
        text = "the cat ran to the house"
        assert tok.decode(tok.encode(text)) == text
        assert tok.vocab_size <= 300

    def test_too_small_vocab_rejected(self, tmp_path):
        with pytest.raises(ValueError, match="vocab_size"):
            train_bpe_tokenizer(["x"], vocab_size=100, output_path=tmp_path / "bpe.json")

    def test_missing_file_actionable(self, tmp_path):
        from jaxscale_lm.data.tokenizer import BpeTokenizer

        with pytest.raises(FileNotFoundError, match="train_tokenizer"):
            BpeTokenizer(tmp_path / "missing.json")


class TestPacking:
    def test_shapes_and_shift(self):
        tok = ByteTokenizer()
        blocks = pack_documents(["abcdefghij" * 10], tok, sequence_length=8)
        assert blocks.shape[1] == 9
        inputs, targets, mask = split_blocks(blocks, tok.pad_id)
        assert inputs.shape == targets.shape == mask.shape == (blocks.shape[0], 8)
        # next-token shift: targets are inputs moved one position left
        np.testing.assert_array_equal(inputs[:, 1:], targets[:, :-1])

    def test_eos_separates_documents(self):
        tok = ByteTokenizer()
        blocks = pack_documents(["ab", "cd"], tok, sequence_length=5, drop_remainder=False)
        stream = blocks.reshape(-1)
        expected = [*tok.encode("ab"), EOS_ID, *tok.encode("cd"), EOS_ID]
        np.testing.assert_array_equal(stream[: len(expected)], expected)

    def test_short_text_errors_with_drop_remainder(self):
        tok = ByteTokenizer()
        with pytest.raises(ValueError, match="sequence_length"):
            pack_documents(["hi"], tok, sequence_length=64, drop_remainder=True)

    def test_short_text_pads_without_drop_remainder(self):
        tok = ByteTokenizer()
        blocks = pack_documents(["hi"], tok, sequence_length=64, drop_remainder=False)
        assert blocks.shape == (1, 65)
        _, targets, mask = split_blocks(blocks, tok.pad_id)
        # 'hi' + EOS = 3 tokens -> 2 valid targets; the rest is masked padding
        assert mask.sum() == 2
        assert (targets[mask == 0] == tok.pad_id).all()

    def test_empty_corpus_rejected(self):
        with pytest.raises(ValueError, match="empty"):
            pack_documents([""], ByteTokenizer(), sequence_length=8)

    def test_deterministic(self):
        tok = ByteTokenizer()
        docs = synthetic_documents(8, 128, seed=3)
        a = pack_documents(docs, tok, sequence_length=32)
        b = pack_documents(docs, tok, sequence_length=32)
        np.testing.assert_array_equal(a, b)


class TestSplitting:
    def test_disjoint_and_complete(self):
        docs = [f"document number {i} " * 5 for i in range(20)]
        splits = split_documents(docs, validation_fraction=0.2, seed=0)
        assert len(splits.train) + len(splits.validation) == 20
        assert set(splits.train).isdisjoint(splits.validation)

    def test_deterministic_under_seed(self):
        docs = [f"doc {i}" for i in range(10)]
        a = split_documents(docs, 0.3, seed=1)
        b = split_documents(docs, 0.3, seed=1)
        assert a == b
        c = split_documents(docs, 0.3, seed=2)
        assert a != c

    def test_too_few_documents_rejected(self):
        with pytest.raises(ValueError, match="at least 2"):
            split_documents(["only one"], 0.5, seed=0)


class TestLoader:
    @pytest.fixture()
    def bundle(self):
        config = DataConfig(
            source="synthetic",
            sequence_length=32,
            batch_size=4,
            validation_fraction=0.25,
            synthetic_num_documents=32,
            synthetic_document_length=128,
        )
        return build_data(config, ByteTokenizer(), seed=0)

    def test_batch_shapes(self, bundle):
        batch = next(train_batches(bundle.train, batch_size=4, seed=0))
        assert batch.input_ids.shape == (4, 32)
        assert batch.target_ids.shape == (4, 32)
        assert batch.loss_mask.shape == (4, 32)
        assert batch.input_ids.dtype == np.int32
        assert batch.loss_mask.dtype == np.float32

    def test_train_stream_deterministic(self, bundle):
        a = [
            b.input_ids
            for _, b in zip(range(5), train_batches(bundle.train, 4, seed=0), strict=False)
        ]
        b = [
            b.input_ids
            for _, b in zip(range(5), train_batches(bundle.train, 4, seed=0), strict=False)
        ]
        for x, y in zip(a, b, strict=True):
            np.testing.assert_array_equal(x, y)

    def test_start_step_fast_forward_matches(self, bundle):
        full = [
            b.input_ids
            for _, b in zip(range(6), train_batches(bundle.train, 4, seed=0), strict=False)
        ]
        resumed = [
            b.input_ids
            for _, b in zip(
                range(3), train_batches(bundle.train, 4, seed=0, start_step=3), strict=False
            )
        ]
        for x, y in zip(full[3:], resumed, strict=True):
            np.testing.assert_array_equal(x, y)

    def test_no_train_validation_overlap(self, bundle):
        # With byte tokenization any shared 33-token window would indicate
        # leakage; compare raw packed rows for an exact check.
        train_rows = {r.tobytes() for r in bundle.train.blocks}
        val_rows = {r.tobytes() for r in bundle.validation.blocks}
        assert train_rows.isdisjoint(val_rows)

    def test_eval_batches_finite_and_cover_once(self, bundle):
        batches = list(eval_batches(bundle.validation, batch_size=4, num_batches=10_000))
        seen = sum(b.input_ids.shape[0] for b in batches)
        assert seen == bundle.validation.num_blocks  # covered exactly once

    def test_batch_size_larger_than_data_rejected(self, bundle):
        with pytest.raises(ValueError, match="batch_size"):
            next(train_batches(bundle.validation, batch_size=10_000, seed=0))
