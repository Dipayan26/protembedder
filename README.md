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

### CLI Usage

```bash
# Per-protein embeddings with ESM-2 650M (default)
protembedder -m esm2_t33_650M -i proteins.fasta -o embeddings.pt

# Per-protein embeddings with ProtT5-XL
protembedder -m prot_t5_xl -i proteins.fasta -o embeddings.pt

# Per-protein embeddings with ProtBert
protembedder -m prot_bert -i proteins.fasta -o embeddings.pt

# Per-residue (per amino acid) embeddings
protembedder -m prot_bert -i proteins.fasta -o embeddings.pt --per-residue

# GPU with custom batch size
protembedder -m esm2_t33_650M -i proteins.fasta -o embeddings.pt --device cuda --batch-size 16

# Small ESM-2 model for quick testing
protembedder -m esm2_t6_8M -i proteins.fasta -o embeddings.pt -v
```

### CLI Flags

| Flag | Short | Required | Default | Description |
|------|-------|----------|---------|-------------|
| `--model` | `-m` | Yes | — | Model name (see tables below) |
| `--input` | `-i` | Yes | — | Input FASTA file path |
| `--output` | `-o` | Yes | — | Output .pt file path |
| `--per-residue` | — | No | `False` | Per amino acid embeddings |
| `--device` | — | No | auto | `cpu`, `cuda`, `cuda:0`, etc. |
| `--batch-size` | — | No | `8` | Sequences per batch |
| `--verbose` | `-v` | No | `False` | Verbose logging |

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

# ESM-2
embedder = ProteinEmbedder("esm2_t33_650M", device="cuda")

# ProtT5-XL
embedder = ProteinEmbedder("prot_t5_xl", device="cuda")

# ProtBert
embedder = ProteinEmbedder("prot_bert", device="cuda")

# From FASTA file — per-protein embeddings (default)
embeddings = embedder.embed_fasta("proteins.fasta", per_residue=False)

# From sequence list — per-residue embeddings
sequences = [
    ("protein_1", "MKTAYIAKQRQISFVKSH"),
    ("protein_2", "MDEVLQAELPAEG"),
]
embeddings = embedder.embed_sequences(sequences, per_residue=True, batch_size=4)

# Save / Load
torch.save(embeddings, "embeddings.pt")
loaded = torch.load("embeddings.pt")
```

### Output Format

The output `.pt` file contains a Python dict: `{header: tensor}`.

- **Per-protein** (default): `tensor` shape is `(embed_dim,)`
- **Per-residue** (`--per-residue`): `tensor` shape is `(seq_len, embed_dim)`

```python
emb = torch.load("embeddings.pt")
for name, tensor in emb.items():
    print(f"{name}: {tensor.shape}")
# protein_1: torch.Size([1280])        # ESM-2 650M, per-protein
# protein_1: torch.Size([18, 1024])    # ProtBert, per-residue
```

## OOM Handling

If a batch causes an out-of-memory error on GPU, the package automatically falls back to processing sequences one at a time. You can also reduce `--batch-size` manually.

## References

> Lin, Z., et al. "Evolutionary-scale prediction of atomic-level protein structure with a language model." *Science* 379.6637 (2023): 1123-1130. [https://doi.org/10.1126/science.ade2574](https://doi.org/10.1126/science.ade2574)

> Elnaggar, A., et al. "ProtTrans: Toward Understanding the Language of Life Through Self-Supervised Learning." *IEEE Transactions on Pattern Analysis and Machine Intelligence* 44.10 (2021): 7112-7127. [https://doi.org/10.1109/TPAMI.2021.3095381](https://doi.org/10.1109/TPAMI.2021.3095381)

## License

MIT
