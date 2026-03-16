# ProtEmbedder

Extract protein embeddings from FASTA files using ESM-2 protein language models.

## Installation

```bash
pip install -e .
```

**Requirements:** Python ≥ 3.8, PyTorch ≥ 1.12, [fair-esm](https://github.com/facebookresearch/esm) ≥ 2.0

## Quick Start

### CLI Usage

```bash
# Per-protein embeddings (default) — one vector per sequence
protembedder -m esm2_t33_650M -i proteins.fasta -o embeddings.pt

# Per-residue embeddings — one vector per amino acid
protembedder -m esm2_t33_650M -i proteins.fasta -o embeddings.pt --per-residue

# GPU with custom batch size
protembedder -m esm2_t33_650M -i proteins.fasta -o embeddings.pt --device cuda --batch-size 16

# Small model for quick testing
protembedder -m esm2_t6_8M -i proteins.fasta -o embeddings.pt -v
```

### CLI Flags

| Flag | Short | Required | Default | Description |
|------|-------|----------|---------|-------------|
| `--model` | `-m` | Yes | — | ESM-2 model name (see table below) |
| `--input` | `-i` | Yes | — | Input FASTA file path |
| `--output` | `-o` | Yes | — | Output .pt file path |
| `--per-residue` | — | No | `False` | Per amino acid embeddings |
| `--device` | — | No | auto | `cpu`, `cuda`, `cuda:0`, etc. |
| `--batch-size` | — | No | `8` | Sequences per batch |
| `--verbose` | `-v` | No | `False` | Verbose logging |

### Available Models

| Model | Parameters | Embedding Dim | Layers |
|-------|-----------|---------------|--------|
| `esm2_t6_8M` | 8M | 320 | 6 |
| `esm2_t12_35M` | 35M | 480 | 12 |
| `esm2_t30_150M` | 150M | 640 | 30 |
| `esm2_t33_650M` | 650M | 1280 | 33 |
| `esm2_t36_3B` | 3B | 2560 | 36 |
| `esm2_t48_15B` | 15B | 5120 | 48 |

### Python API

```python
import torch
from protembedder import ProteinEmbedder

# Initialize
embedder = ProteinEmbedder("esm2_t33_650M", device="cuda")

# From FASTA file
embeddings = embedder.embed_fasta("proteins.fasta", per_residue=False)

# From sequence list
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
# protein_1: torch.Size([1280])        # per-protein
# protein_1: torch.Size([18, 1280])    # per-residue
```

## OOM Handling

If a batch causes an out-of-memory error on GPU, the package automatically falls back to processing sequences one at a time for that batch. You can also reduce `--batch-size` manually.

## Reference

> Lin, Z., et al. "Evolutionary-scale prediction of atomic-level protein structure with a language model." *Science* 379.6637 (2023): 1123-1130. [https://doi.org/10.1126/science.ade2574](https://doi.org/10.1126/science.ade2574)

## License

MIT
