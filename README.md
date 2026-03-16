# ProtEmbedder

[![PyPI version](https://img.shields.io/pypi/v/protembedder.svg)](https://pypi.org/project/protembedder/)
[![Python](https://img.shields.io/pypi/pyversions/protembedder.svg)](https://pypi.org/project/protembedder/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Extract protein embeddings from FASTA files using ESM-2, ProtT5, and ProtBert protein language models.

## Installation

```bash
pip install protembedder
```

**Requirements:** Python ≥ 3.8, PyTorch ≥ 1.12, [fair-esm](https://github.com/facebookresearch/esm) ≥ 2.0, [transformers](https://github.com/huggingface/transformers) ≥ 4.30

> For development / editable install: `pip install -e .`

## Quick Start

### List available models

```bash
protembedder list-models

# With details (family, embedding dim, repo)
protembedder list-models --verbose
```

### CLI Usage

```bash
# Per-protein embeddings — PyTorch output (default)
protembedder embed -m esm2_t33_650M -i proteins.fasta -o embeddings.pt

# HDF5 output
protembedder embed -m prot_t5_xl -i proteins.fasta -o embeddings.h5 --format h5

# NumPy .npz output
protembedder embed -m prot_bert -i proteins.fasta -o embeddings.npz --format npz

# CSV output
protembedder embed -m prot_bert -i proteins.fasta -o embeddings.csv --format csv

# Per-residue (per amino acid) embeddings
protembedder embed -m prot_bert -i proteins.fasta -o embeddings.pt --per-residue

# GPU with custom batch size, disable progress bar
protembedder embed -m esm2_t33_650M -i proteins.fasta -o embeddings.pt --device cuda --batch-size 16 --no-progress
```

### CLI Flags (`embed` subcommand)

| Flag | Short | Required | Default | Description |
|------|-------|----------|---------|-------------|
| `--model` | `-m` | Yes | — | Model name (see `list-models`) |
| `--input` | `-i` | Yes | — | Input FASTA file path |
| `--output` | `-o` | Yes | — | Output file path |
| `--format` | — | No | `pt` | Output format: `pt`, `h5`, `npz`, `csv` |
| `--per-residue` | — | No | `False` | Per amino acid embeddings |
| `--device` | — | No | auto | `cpu`, `cuda`, `cuda:0`, etc. |
| `--batch-size` | — | No | `8` | Sequences per batch |
| `--no-progress` | — | No | `False` | Disable tqdm progress bar |
| `--verbose` | `-v` | No | `False` | Verbose logging |

### Output Formats

| Flag | Extension | Description | Load with |
|------|-----------|-------------|-----------|
| `pt` | `.pt` | PyTorch dict `{header: tensor}` | `torch.load()` |
| `h5` | `.h5` | HDF5 — `embeddings/` group + `headers` dataset | `h5py` / `protembedder.io.load_embeddings()` |
| `npz` | `.npz` | NumPy archive — `embeddings` array + `headers` array | `np.load()` / `protembedder.io.load_embeddings()` |
| `csv` | `.csv` | CSV table — `header, emb_0, emb_1, ...` (per-protein) or `header, residue_idx, emb_0, ...` (per-residue) | `pandas.read_csv()` |

### Available Models

**ESM-2** (Meta AI)

| Model | Parameters | Embedding Dim | Layers |
|-------|-----------|---------------|--------|
| `esm2_t6_8M` | 8M | 320 | 6 |
| `esm2_t12_35M` | 35M | 480 | 12 |
| `esm2_t30_150M` | 150M | 640 | 30 |
| `esm2_t33_650M` | 650M | 1280 | 33 |
| `esm2_t36_3B` | 3B | 2560 | 36 |
| `esm2_t48_15B` | 15B | 5120 | 48 |

**ProtT5** (Rostlab)

| Model | Parameters | Embedding Dim | HuggingFace Repo |
|-------|-----------|---------------|------------------|
| `prot_t5_xl` | 3B | 1024 | [Rostlab/prot_t5_xl_half_uniref50-enc](https://huggingface.co/Rostlab/prot_t5_xl_half_uniref50-enc) |

**ProtBert** (Rostlab)

| Model | Parameters | Embedding Dim | HuggingFace Repo |
|-------|-----------|---------------|------------------|
| `prot_bert` | 420M | 1024 | [Rostlab/prot_bert](https://huggingface.co/Rostlab/prot_bert) |

### Python API

```python
import torch
from protembedder import ProteinEmbedder
from protembedder.io import save_embeddings, load_embeddings

# Load any supported model
embedder = ProteinEmbedder("esm2_t33_650M", device="cuda")
# embedder = ProteinEmbedder("prot_t5_xl")
# embedder = ProteinEmbedder("prot_bert")

# From FASTA file with progress bar
embeddings = embedder.embed_fasta("proteins.fasta", per_residue=False, show_progress=True)

# From a sequence list
sequences = [("prot_1", "MKTAYIAKQRQISFVKSH"), ("prot_2", "MDEVLQAELPAEG")]
embeddings = embedder.embed_sequences(sequences, per_residue=True, batch_size=4)

# Save in any format
save_embeddings(embeddings, "embeddings.h5",   fmt="h5")
save_embeddings(embeddings, "embeddings.npz",  fmt="npz")
save_embeddings(embeddings, "embeddings.csv",  fmt="csv")
save_embeddings(embeddings, "embeddings.pt",   fmt="pt")

# Load back (pt, h5, npz)
loaded = load_embeddings("embeddings.h5")

# List all models
print(ProteinEmbedder.list_models())
```

### Output Format Details

- **Per-protein** (default): `tensor` shape `(embed_dim,)`
- **Per-residue** (`--per-residue`): `tensor` shape `(seq_len, embed_dim)`

```python
emb = load_embeddings("embeddings.pt")
for name, tensor in emb.items():
    print(f"{name}: {tensor.shape}")
# prot_1: torch.Size([1280])        # ESM-2 650M, per-protein
# prot_1: torch.Size([18, 1024])    # ProtBert, per-residue
```

## OOM Handling

If a batch causes an out-of-memory error on GPU, the package automatically falls back to processing sequences one at a time. You can also reduce `--batch-size` manually.

## References

> Lin, Z., et al. "Evolutionary-scale prediction of atomic-level protein structure with a language model." *Science* 379.6637 (2023): 1123-1130. [https://doi.org/10.1126/science.ade2574](https://doi.org/10.1126/science.ade2574)

> Elnaggar, A., et al. "ProtTrans: Toward Understanding the Language of Life Through Self-Supervised Learning." *IEEE Transactions on Pattern Analysis and Machine Intelligence* 44.10 (2021): 7112-7127. [https://doi.org/10.1109/TPAMI.2021.3095381](https://doi.org/10.1109/TPAMI.2021.3095381)

## License

MIT
