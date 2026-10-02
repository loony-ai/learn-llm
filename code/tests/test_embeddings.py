"""Tests for llmfp.model.embeddings and llmfp.model.embedding_mlp (Chapter 10)."""

from __future__ import annotations

import pytest
import torch
from torch.nn import functional as F

from llmfp.model.embedding_mlp import MODES, EmbeddingMLP, EmbeddingMLPConfig
from llmfp.model.embeddings import TokenAndPositionEmbedding, cosine_similarity_matrix, nearest_neighbors
from solutions.ch10_average_first import AverageFirstModel


def test_embedding_equals_one_hot_times_table():
    table = torch.nn.Embedding(7, 4)
    ids = torch.tensor([[1, 6, 1], [0, 2, 3]])
    assert torch.allclose(table(ids), F.one_hot(ids, 7).float() @ table.weight)


def test_token_and_position_shapes_and_position_effect():
    torch.manual_seed(0)
    embed = TokenAndPositionEmbedding(vocab_size=10, context_length=5, d_model=3)
    out = embed(torch.tensor([[2, 2, 2]]))
    assert out.shape == (1, 3, 3)
    assert not torch.allclose(out[0, 0], out[0, 1])   # same token, different positions
    assert torch.allclose(out[0, 0], embed.token.weight[2] + embed.position.weight[0])


def test_too_long_sequence_rejected():
    embed = TokenAndPositionEmbedding(10, context_length=4, d_model=3)
    with pytest.raises(ValueError, match="longer than the 4 positions"):
        embed(torch.zeros(1, 5, dtype=torch.int64))


def test_cosine_similarity_matrix_properties():
    vectors = torch.tensor([[1.0, 0.0], [10.0, 0.0], [0.0, 3.0], [-1.0, 0.0]])
    similarity = cosine_similarity_matrix(vectors)
    assert similarity[0, 1].item() == pytest.approx(1.0)
    assert similarity[0, 2].item() == pytest.approx(0.0)
    assert similarity[0, 3].item() == pytest.approx(-1.0)
    assert torch.allclose(similarity, similarity.T)


def test_nearest_neighbors_excludes_itself():
    vectors = torch.tensor([[1.0, 0.0], [0.9, 0.1], [0.0, 1.0]])
    neighbors = nearest_neighbors(vectors, 0, k=2)
    assert [index for index, _ in neighbors] == [1, 2]


@pytest.mark.parametrize("mode", MODES)
def test_embedding_mlp_shapes(mode):
    model = EmbeddingMLP(EmbeddingMLPConfig(vocab_size=50, context_size=6, d_model=4, hidden=8, mode=mode))
    assert model(torch.randint(0, 50, (3, 6))).shape == (3, 50)
    assert model.token_table.shape == (50, 4)


def test_unknown_mode_rejected():
    with pytest.raises(ValueError):
        EmbeddingMLPConfig(vocab_size=5, mode="sum")


@pytest.mark.parametrize(("mode", "order_sensitive"), [("bag", False), ("bag_position", True), ("concat", True)])
def test_which_modes_see_word_order(mode, order_sensitive):
    torch.manual_seed(0)
    model = EmbeddingMLP(EmbeddingMLPConfig(vocab_size=20, context_size=4, d_model=4, hidden=8, mode=mode))
    original = torch.tensor([[1, 2, 3, 4]])
    reordered = torch.tensor([[4, 3, 2, 1]])
    assert torch.allclose(model(original), model(reordered), atol=1e-6) is not order_sensitive


def test_averaging_first_cancels_position_information():
    torch.manual_seed(0)
    model = AverageFirstModel(EmbeddingMLPConfig(vocab_size=20, context_size=4, d_model=4, hidden=8, mode="bag_position"))
    assert torch.allclose(model(torch.tensor([[1, 2, 3, 4]])), model(torch.tensor([[4, 3, 2, 1]])), atol=1e-6)
