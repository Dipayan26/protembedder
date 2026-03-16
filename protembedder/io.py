"""
Save and load protein embeddings in multiple formats.

Supported formats:
    pt   — PyTorch dict  (.pt)         torch.save / torch.load
    h5   — HDF5          (.h5)         h5py, one dataset per sequence
    npz  — NumPy archive (.npz)        np.savez / np.load
    csv  — CSV           (.csv)        pandas-free; one row per residue or sequence
"""

import csv
import json
import logging
from pathlib import Path
from typing import Dict

import numpy as np
import torch

logger = logging.getLogger(__name__)

SUPPORTED_FORMATS = ("pt", "h5", "npz", "csv")
FORMAT_EXTENSIONS = {"pt": ".pt", "h5": ".h5", "npz": ".npz", "csv": ".csv"}


def _ensure_extension(path: str, fmt: str) -> Path:
    """Append the correct extension if the user didn't include it."""
    p = Path(path)
    expected = FORMAT_EXTENSIONS[fmt]
    if p.suffix != expected:
        p = p.with_suffix(expected)
    return p


def save_embeddings(
    embeddings: Dict[str, torch.Tensor],
    output_path: str,
    fmt: str = "pt",
) -> Path:
    """
    Save a dict of embedding tensors to disk.

    Parameters
    ----------
    embeddings : Dict[str, torch.Tensor]
        Mapping from sequence header to tensor.
        Per-protein: shape (embed_dim,)
        Per-residue: shape (seq_len, embed_dim)
    output_path : str
        Destination file path. Extension is auto-corrected if needed.
    fmt : str
        One of 'pt', 'h5', 'npz', 'csv'.

    Returns
    -------
    Path
        The final path the file was written to.
    """
    if fmt not in SUPPORTED_FORMATS:
        raise ValueError(f"Unknown format '{fmt}'. Choose from: {SUPPORTED_FORMATS}")

    out = _ensure_extension(output_path, fmt)
    out.parent.mkdir(parents=True, exist_ok=True)

    if fmt == "pt":
        _save_pt(embeddings, out)
    elif fmt == "h5":
        _save_h5(embeddings, out)
    elif fmt == "npz":
        _save_npz(embeddings, out)
    elif fmt == "csv":
        _save_csv(embeddings, out)

    logger.info(f"Saved {len(embeddings)} embeddings → {out} (format={fmt})")
    return out


# ---------------------------------------------------------------------------
# Format writers
# ---------------------------------------------------------------------------

def _save_pt(embeddings: Dict[str, torch.Tensor], path: Path) -> None:
    torch.save(embeddings, str(path))


def _save_h5(embeddings: Dict[str, torch.Tensor], path: Path) -> None:
    try:
        import h5py
    except ImportError:
        raise ImportError(
            "h5py is required for HDF5 output. Install with: pip install h5py"
        )

    with h5py.File(str(path), "w") as f:
        # Store metadata: per_residue, embed_dim, number of sequences
        sample = next(iter(embeddings.values()))
        f.attrs["per_residue"] = sample.ndim == 2
        f.attrs["embed_dim"] = sample.shape[-1]
        f.attrs["n_sequences"] = len(embeddings)

        emb_grp = f.create_group("embeddings")
        for header, tensor in embeddings.items():
            # h5py dataset names cannot contain '/' — replace with '|'
            safe_key = header.replace("/", "|")
            emb_grp.create_dataset(safe_key, data=tensor.numpy(), compression="gzip")

        # Store original headers as a separate dataset for round-trip fidelity
        dt = h5py.special_dtype(vlen=str)
        headers_ds = f.create_dataset("headers", (len(embeddings),), dtype=dt)
        for i, header in enumerate(embeddings.keys()):
            headers_ds[i] = header


def _save_npz(embeddings: Dict[str, torch.Tensor], path: Path) -> None:
    headers = list(embeddings.keys())
    arrays = [v.numpy() for v in embeddings.values()]
    sample = arrays[0]

    if sample.ndim == 1:
        # Per-protein: stack into (N, embed_dim)
        stacked = np.stack(arrays)
        np.savez_compressed(
            str(path),
            embeddings=stacked,
            headers=np.array(headers, dtype=object),
        )
    else:
        # Per-residue: variable lengths — store as object array
        obj_arr = np.empty(len(arrays), dtype=object)
        for i, arr in enumerate(arrays):
            obj_arr[i] = arr
        np.savez_compressed(
            str(path),
            embeddings=obj_arr,
            headers=np.array(headers, dtype=object),
        )


def _save_csv(embeddings: Dict[str, torch.Tensor], path: Path) -> None:
    sample = next(iter(embeddings.values()))
    per_residue = sample.ndim == 2
    embed_dim = sample.shape[-1]

    with open(str(path), "w", newline="") as f:
        writer = csv.writer(f)

        if per_residue:
            # Columns: header, residue_idx, emb_0, emb_1, ...
            header_row = ["header", "residue_idx"] + [f"emb_{i}" for i in range(embed_dim)]
            writer.writerow(header_row)
            for seq_header, tensor in embeddings.items():
                for pos, vec in enumerate(tensor.numpy()):
                    writer.writerow([seq_header, pos] + vec.tolist())
        else:
            # Columns: header, emb_0, emb_1, ...
            header_row = ["header"] + [f"emb_{i}" for i in range(embed_dim)]
            writer.writerow(header_row)
            for seq_header, tensor in embeddings.items():
                writer.writerow([seq_header] + tensor.numpy().tolist())


# ---------------------------------------------------------------------------
# Format loader (convenience)
# ---------------------------------------------------------------------------

def load_embeddings(path: str) -> Dict[str, torch.Tensor]:
    """
    Load embeddings saved by :func:`save_embeddings`.

    Supports .pt, .h5, and .npz files. CSV loading is not supported
    (use pandas directly for that case).

    Parameters
    ----------
    path : str
        Path to the saved embeddings file.

    Returns
    -------
    Dict[str, torch.Tensor]
        Mapping from sequence header to tensor.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"File not found: {p}")

    suffix = p.suffix.lower()
    if suffix == ".pt":
        return torch.load(str(p), weights_only=False)
    elif suffix in (".h5", ".hdf5"):
        return _load_h5(p)
    elif suffix == ".npz":
        return _load_npz(p)
    else:
        raise ValueError(f"Cannot load format '{suffix}'. Use .pt, .h5, or .npz.")


def _load_h5(path: Path) -> Dict[str, torch.Tensor]:
    import h5py

    embeddings = {}
    with h5py.File(str(path), "r") as f:
        headers = [h.decode() if isinstance(h, bytes) else h for h in f["headers"][:]]
        for header in headers:
            safe_key = header.replace("/", "|")
            embeddings[header] = torch.from_numpy(f["embeddings"][safe_key][:])
    return embeddings


def _load_npz(path: Path) -> Dict[str, torch.Tensor]:
    data = np.load(str(path), allow_pickle=True)
    headers = data["headers"].tolist()
    raw = data["embeddings"]

    embeddings = {}
    if raw.ndim == 2:
        # Per-protein stacked array (N, embed_dim)
        for header, arr in zip(headers, raw):
            embeddings[header] = torch.from_numpy(arr)
    else:
        # Per-residue object array
        for header, arr in zip(headers, raw):
            embeddings[header] = torch.from_numpy(arr)
    return embeddings
