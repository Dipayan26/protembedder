"""
Tests for protembedder.io — multi-format save/load round-trips.
"""

import pytest
import torch
import numpy as np
from pathlib import Path

from protembedder.io import save_embeddings, load_embeddings, SUPPORTED_FORMATS, _ensure_extension


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def per_protein_embeddings():
    return {
        "seq_A": torch.randn(1280),
        "seq_B": torch.randn(1280),
        "seq/C": torch.randn(1280),  # header with slash — tests h5 sanitisation
    }


@pytest.fixture
def per_residue_embeddings():
    return {
        "seq_A": torch.randn(10, 1024),
        "seq_B": torch.randn(7, 1024),
    }


# ---------------------------------------------------------------------------
# Extension helper
# ---------------------------------------------------------------------------

def test_ensure_extension_appends():
    p = _ensure_extension("/tmp/out", "h5")
    assert p.suffix == ".h5"


def test_ensure_extension_corrects():
    p = _ensure_extension("/tmp/out.pt", "npz")
    assert p.suffix == ".npz"


def test_ensure_extension_keeps():
    p = _ensure_extension("/tmp/out.csv", "csv")
    assert p.suffix == ".csv"


# ---------------------------------------------------------------------------
# .pt round-trip
# ---------------------------------------------------------------------------

def test_pt_per_protein_roundtrip(per_protein_embeddings, tmp_path):
    out = save_embeddings(per_protein_embeddings, str(tmp_path / "emb.pt"), fmt="pt")
    loaded = load_embeddings(str(out))
    assert set(loaded.keys()) == set(per_protein_embeddings.keys())
    for k in per_protein_embeddings:
        assert torch.allclose(loaded[k], per_protein_embeddings[k])


def test_pt_per_residue_roundtrip(per_residue_embeddings, tmp_path):
    out = save_embeddings(per_residue_embeddings, str(tmp_path / "emb.pt"), fmt="pt")
    loaded = load_embeddings(str(out))
    for k in per_residue_embeddings:
        assert torch.allclose(loaded[k], per_residue_embeddings[k])


# ---------------------------------------------------------------------------
# .h5 round-trip
# ---------------------------------------------------------------------------

def test_h5_per_protein_roundtrip(per_protein_embeddings, tmp_path):
    out = save_embeddings(per_protein_embeddings, str(tmp_path / "emb.h5"), fmt="h5")
    loaded = load_embeddings(str(out))
    assert set(loaded.keys()) == set(per_protein_embeddings.keys())
    for k in per_protein_embeddings:
        assert torch.allclose(loaded[k].float(), per_protein_embeddings[k].float(), atol=1e-5)


def test_h5_per_residue_roundtrip(per_residue_embeddings, tmp_path):
    out = save_embeddings(per_residue_embeddings, str(tmp_path / "emb.h5"), fmt="h5")
    loaded = load_embeddings(str(out))
    for k in per_residue_embeddings:
        assert loaded[k].shape == per_residue_embeddings[k].shape


def test_h5_slash_in_header(per_protein_embeddings, tmp_path):
    # 'seq/C' has a slash — must be stored and retrieved correctly
    out = save_embeddings(per_protein_embeddings, str(tmp_path / "emb.h5"), fmt="h5")
    loaded = load_embeddings(str(out))
    assert "seq/C" in loaded


# ---------------------------------------------------------------------------
# .npz round-trip
# ---------------------------------------------------------------------------

def test_npz_per_protein_roundtrip(per_protein_embeddings, tmp_path):
    out = save_embeddings(per_protein_embeddings, str(tmp_path / "emb.npz"), fmt="npz")
    loaded = load_embeddings(str(out))
    assert set(loaded.keys()) == set(per_protein_embeddings.keys())
    for k in per_protein_embeddings:
        assert torch.allclose(loaded[k].float(), per_protein_embeddings[k].float(), atol=1e-5)


def test_npz_per_residue_roundtrip(per_residue_embeddings, tmp_path):
    out = save_embeddings(per_residue_embeddings, str(tmp_path / "emb.npz"), fmt="npz")
    loaded = load_embeddings(str(out))
    for k in per_residue_embeddings:
        assert loaded[k].shape == per_residue_embeddings[k].shape


# ---------------------------------------------------------------------------
# .csv output (write-only, no loader)
# ---------------------------------------------------------------------------

def test_csv_per_protein_written(per_protein_embeddings, tmp_path):
    out = save_embeddings(per_protein_embeddings, str(tmp_path / "emb.csv"), fmt="csv")
    lines = out.read_text().splitlines()
    # Header row + one row per sequence
    assert len(lines) == len(per_protein_embeddings) + 1
    assert lines[0].startswith("header,emb_0")


def test_csv_per_residue_written(per_residue_embeddings, tmp_path):
    out = save_embeddings(per_residue_embeddings, str(tmp_path / "emb.csv"), fmt="csv")
    lines = out.read_text().splitlines()
    # Header row + one row per residue across all sequences
    total_residues = sum(t.shape[0] for t in per_residue_embeddings.values())
    assert len(lines) == total_residues + 1
    assert lines[0].startswith("header,residue_idx")


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------

def test_unknown_format_raises(per_protein_embeddings, tmp_path):
    with pytest.raises(ValueError, match="Unknown format"):
        save_embeddings(per_protein_embeddings, str(tmp_path / "emb.xyz"), fmt="xyz")


def test_load_missing_file_raises():
    with pytest.raises(FileNotFoundError):
        load_embeddings("/nonexistent/path.pt")
