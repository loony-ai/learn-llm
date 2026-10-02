"""Tests for llmfp.data and the bigram model (Chapter 11)."""

from __future__ import annotations

import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader

from llmfp.data import IGNORE_INDEX, TokenWindowDataset, pack_documents, pad_batch
from llmfp.model.bigram import BigramModel
from solutions.ch11_bucketing import bucketed_batches, padding_waste, random_batches

IDS = list(range(100, 120))   # distinct values make every position identifiable


def test_targets_are_inputs_shifted_by_one():
    """The test that catches every off-by-one variant (Chapter 7.9, 11.3)."""
    dataset = TokenWindowDataset(IDS, context_length=5)
    for index in range(len(dataset)):
        inputs, targets = dataset[index]
        assert torch.equal(targets[:-1], inputs[1:])               # same tokens, shifted
        start = IDS.index(inputs[0].item())
        assert targets[-1].item() == IDS[start + 5]                # last target is the token after the window


@pytest.mark.parametrize(("context", "stride", "count"), [(5, 5, 3), (5, 1, 15), (5, 2, 8), (19, 19, 1), (20, 20, 0)])
def test_window_counts(context, stride, count):
    assert len(TokenWindowDataset(IDS, context, stride)) == count


def test_default_stride_covers_every_transition_once():
    dataset = TokenWindowDataset(IDS, context_length=4)
    targets = torch.cat([dataset[i][1] for i in range(len(dataset))]).tolist()
    assert targets == IDS[1 : 1 + len(targets)]                    # contiguous, no gaps, no repeats


def test_window_index_out_of_range():
    with pytest.raises(IndexError):
        TokenWindowDataset(IDS, 5)[3]


def test_dataloader_batches_and_reproducible_shuffle():
    dataset = TokenWindowDataset(IDS, context_length=4)
    def order(seed):
        loader = DataLoader(dataset, batch_size=2, shuffle=True, generator=torch.Generator().manual_seed(seed))
        return [x[:, 0].tolist() for x, _ in loader]
    assert order(0) == order(0)
    inputs, targets = next(iter(DataLoader(dataset, batch_size=2)))
    assert inputs.shape == targets.shape == (2, 4)


def test_pad_batch_right_and_left():
    right = pad_batch([[1, 2, 3], [4, 5]], pad_id=0)
    assert right.input_ids.tolist() == [[1, 2], [4, 0]]
    assert right.targets.tolist() == [[2, 3], [5, IGNORE_INDEX]]
    assert right.attention_mask.tolist() == [[True, True], [True, False]]
    left = pad_batch([[1, 2, 3], [4, 5]], pad_id=0, side="left")
    assert left.input_ids.tolist() == [[1, 2], [0, 4]]
    assert left.targets.tolist() == [[2, 3], [IGNORE_INDEX, 5]]


def test_pad_batch_rejects_too_short_sequences():
    with pytest.raises(ValueError):
        pad_batch([[1, 2], [3]], pad_id=0)


def test_ignored_padding_does_not_change_the_loss():
    torch.manual_seed(0)
    batch = pad_batch([[1, 2, 3, 4, 5], [6, 7]], pad_id=0)
    logits = torch.randn(2, 4, 10)
    with_padding = nn.functional.cross_entropy(logits.reshape(-1, 10), batch.targets.reshape(-1))
    real = batch.attention_mask.reshape(-1)
    only_real = nn.functional.cross_entropy(logits.reshape(-1, 10)[real], batch.targets.reshape(-1)[real])
    assert with_padding.item() == pytest.approx(only_real.item())


def test_pack_documents_keeps_order_and_shares_one_token():
    windows, owners = pack_documents([[1, 2, 3], [4, 5], [6, 7, 8, 9]], context_length=3, separator_id=99)
    assert windows.tolist() == [[1, 2, 3, 99], [99, 4, 5, 99], [99, 6, 7, 8]]
    assert owners.tolist() == [[0, 0, 0, 0], [0, 1, 1, 1], [1, 2, 2, 2]]


def test_pack_documents_reconstructs_the_stream():
    documents = [[i] * (i % 5 + 1) for i in range(1, 30)]
    windows, _ = pack_documents(documents, context_length=6, separator_id=0)
    stream = [token for document in documents for token in document + [0]]
    rebuilt = windows[0].tolist() + [t for row in windows[1:].tolist() for t in row[1:]]
    assert rebuilt == stream[: len(rebuilt)]


def test_pack_documents_too_little_data():
    with pytest.raises(ValueError):
        pack_documents([[1, 2]], context_length=8, separator_id=0)


def test_bigram_model_shapes_and_meaning():
    model = BigramModel(vocab_size=7)
    logits = model(torch.tensor([[1, 2, 3]]))
    assert logits.shape == (1, 3, 7)
    assert torch.equal(logits[0, 0], model.table.weight[1])        # row for token 1 = its next-token logits


def test_bucketing_reduces_padding():
    documents = [[1] * n for n in [2, 9, 3, 8, 2, 9, 3, 8] * 10]
    assert padding_waste(bucketed_batches(documents, 4, 0)) < padding_waste(random_batches(documents, 4, 0))
