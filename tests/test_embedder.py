"""
Tests for the ProteinEmbedder class — model registry, ProtT5 preprocessing,
and the unified interface. These tests run without downloading model weights.
"""

import pytest
import torch
from unittest.mock import MagicMock, patch

from protembedder.embedder import (
    ProteinEmbedder,
    ALL_MODELS,
    ESM2_MODELS,
    PROT_T5_MODELS,
    _is_esm2,
    _is_prot_t5,
    _ProtT5Backend,
)


# ---------------------------------------------------------------------------
# Model registry tests
# ---------------------------------------------------------------------------

def test_all_models_non_empty():
    assert len(ALL_MODELS) > 0


def test_esm2_models_present():
    for name in ESM2_MODELS:
        assert name in ALL_MODELS


def test_prot_t5_models_present():
    for name in PROT_T5_MODELS:
        assert name in ALL_MODELS


def test_prot_t5_xl_registered():
    assert "prot_t5_xl" in PROT_T5_MODELS
    repo, dim = PROT_T5_MODELS["prot_t5_xl"]
    assert repo == "Rostlab/prot_t5_xl_half_uniref50-enc"
    assert dim == 1024


def test_is_esm2():
    assert _is_esm2("esm2_t33_650M")
    assert not _is_esm2("prot_t5_xl")


def test_is_prot_t5():
    assert _is_prot_t5("prot_t5_xl")
    assert not _is_prot_t5("esm2_t33_650M")


def test_unknown_model_raises():
    with pytest.raises(ValueError, match="Unknown model"):
        ProteinEmbedder("nonexistent_model_xyz")


def test_list_models():
    models = ProteinEmbedder.list_models()
    assert "prot_t5_xl" in models
    assert "esm2_t33_650M" in models
    assert models == sorted(models)


# ---------------------------------------------------------------------------
# ProtT5 preprocessing tests (no weights needed)
# ---------------------------------------------------------------------------

def test_prot_t5_preprocess_spaces():
    result = _ProtT5Backend._preprocess("MKTAY")
    assert result == "M K T A Y"


def test_prot_t5_preprocess_nonstandard():
    # U, Z, O, B all -> X per the ProtT5 preprocessing convention
    result = _ProtT5Backend._preprocess("ACUZOB")
    assert result == "A C X X X X"
    result2 = _ProtT5Backend._preprocess("UZOB")
    assert result2 == "X X X X"


def test_prot_t5_preprocess_lowercase():
    result = _ProtT5Backend._preprocess("mktay")
    assert result == "M K T A Y"


# ---------------------------------------------------------------------------
# Embedder interface tests (mocked backends)
# ---------------------------------------------------------------------------

def _make_mock_backend(embed_dim: int, per_residue: bool, seq_len: int = 5):
    """Return a mock backend whose embed_batch mimics real output shapes."""
    backend = MagicMock()
    backend.embed_dim = embed_dim

    def fake_embed_batch(sequences, per_residue):
        result = {}
        for header, seq in sequences:
            L = len(seq)
            if per_residue:
                result[header] = torch.zeros(L, embed_dim)
            else:
                result[header] = torch.zeros(embed_dim)
        return result

    backend.embed_batch.side_effect = fake_embed_batch
    return backend


@pytest.fixture
def sequences():
    return [("prot_a", "MKTAY"), ("prot_b", "ACDEF")]


def test_embed_sequences_per_protein(sequences):
    with patch("protembedder.embedder._ESM2Backend") as MockESM:
        instance = _make_mock_backend(embed_dim=1280, per_residue=False)
        MockESM.return_value = instance

        embedder = ProteinEmbedder.__new__(ProteinEmbedder)
        embedder.model_name = "esm2_t33_650M"
        embedder.device = torch.device("cpu")
        embedder._backend = instance
        embedder.embed_dim = 1280

        result = embedder.embed_sequences(sequences, per_residue=False, batch_size=8)
        assert set(result.keys()) == {"prot_a", "prot_b"}
        assert result["prot_a"].shape == (1280,)
        assert result["prot_b"].shape == (1280,)


def test_embed_sequences_per_residue(sequences):
    with patch("protembedder.embedder._ProtT5Backend") as MockT5:
        instance = _make_mock_backend(embed_dim=1024, per_residue=True)
        MockT5.return_value = instance

        embedder = ProteinEmbedder.__new__(ProteinEmbedder)
        embedder.model_name = "prot_t5_xl"
        embedder.device = torch.device("cpu")
        embedder._backend = instance
        embedder.embed_dim = 1024

        result = embedder.embed_sequences(sequences, per_residue=True, batch_size=8)
        assert result["prot_a"].shape == (5, 1024)   # len("MKTAY") == 5
        assert result["prot_b"].shape == (5, 1024)   # len("ACDEF") == 5
