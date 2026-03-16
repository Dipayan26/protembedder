"""
ProtEmbedder - Extract protein embeddings from FASTA files using protein language models.

Supported models:
    ESM-2 family (Meta AI)
        Lin, Z., et al. Science 379.6637 (2023). https://doi.org/10.1126/science.ade2574
    ProtT5-XL / ProtBert (Rostlab)
        Elnaggar, A., et al. IEEE TPAMI 44.10 (2021). https://doi.org/10.1109/TPAMI.2021.3095381
"""

__version__ = "0.4.0"
__author__ = "Dipayan Sarkar"

from protembedder.embedder import ProteinEmbedder
from protembedder.fasta import read_fasta

__all__ = ["ProteinEmbedder", "read_fasta"]
