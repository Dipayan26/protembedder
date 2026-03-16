"""
ProtEmbedder - Extract protein embeddings from FASTA files using protein language models.

Supported models: ESM-2 family (Meta AI)
References:
    Lin, Z., et al. "Evolutionary-scale prediction of atomic-level protein structure
    with a language model." Science 379.6637 (2023): 1123-1130.
    https://doi.org/10.1126/science.ade2574
"""

__version__ = "0.1.0"
__author__ = "ProtEmbedder Contributors"

from protembedder.embedder import ProteinEmbedder
from protembedder.fasta import read_fasta

__all__ = ["ProteinEmbedder", "read_fasta"]
