"""
Protein embedding extraction using ESM-2 models.

Reference:
    Lin, Z., et al. "Evolutionary-scale prediction of atomic-level protein structure
    with a language model." Science 379.6637 (2023): 1123-1130.
    https://doi.org/10.1126/science.ade2574
"""

import sys
import logging
from typing import Dict, List, Optional, Tuple

import torch
import esm

from protembedder.fasta import read_fasta, validate_protein_sequences

logger = logging.getLogger(__name__)

# ESM-2 model registry: name -> (hub_name, embedding_dim, num_layers)
ESM2_MODELS = {
    "esm2_t6_8M":   ("esm2_t6_8M_UR50D",   320,  6),
    "esm2_t12_35M":  ("esm2_t12_35M_UR50D",  480,  12),
    "esm2_t30_150M": ("esm2_t30_150M_UR50D", 640,  30),
    "esm2_t33_650M": ("esm2_t33_650M_UR50D", 1280, 33),
    "esm2_t36_3B":   ("esm2_t36_3B_UR50D",   2560, 36),
    "esm2_t48_15B":  ("esm2_t48_15B_UR50D",  5120, 48),
}


class ProteinEmbedder:
    """
    Extract protein embeddings from sequences using ESM-2 models.

    Parameters
    ----------
    model_name : str
        Name of the ESM-2 model. One of:
        'esm2_t6_8M', 'esm2_t12_35M', 'esm2_t30_150M',
        'esm2_t33_650M', 'esm2_t36_3B', 'esm2_t48_15B'.
    device : str, optional
        Device to use ('cpu', 'cuda', 'cuda:0', etc.).
        If None, auto-detects GPU availability.

    Examples
    --------
    >>> embedder = ProteinEmbedder("esm2_t33_650M")
    >>> results = embedder.embed_fasta("proteins.fasta", per_residue=False)
    >>> torch.save(results, "embeddings.pt")
    """

    def __init__(self, model_name: str, device: Optional[str] = None):
        if model_name not in ESM2_MODELS:
            available = ", ".join(sorted(ESM2_MODELS.keys()))
            raise ValueError(
                f"Unknown model '{model_name}'. Available models: {available}"
            )

        self.model_name = model_name
        self.hub_name, self.embed_dim, self.num_layers = ESM2_MODELS[model_name]

        # Set device
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        logger.info(f"Loading model {self.hub_name} on {self.device}...")
        self.model, self.alphabet = esm.pretrained.load_model_and_alphabet(self.hub_name)
        self.model = self.model.to(self.device)
        self.model.eval()
        self.batch_converter = self.alphabet.get_batch_converter()
        logger.info(f"Model loaded. Embedding dim: {self.embed_dim}")

    def _embed_batch(
        self,
        sequences: List[Tuple[str, str]],
        per_residue: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute embeddings for a batch of sequences.

        Parameters
        ----------
        sequences : List[Tuple[str, str]]
            List of (label, sequence) tuples.
        per_residue : bool
            If True, return per-residue (per amino acid) embeddings.
            If False, return mean-pooled per-protein embeddings.

        Returns
        -------
        Dict[str, torch.Tensor]
            Mapping from sequence header to embedding tensor.
            Per-protein: shape (embed_dim,)
            Per-residue: shape (seq_len, embed_dim)
        """
        batch_labels, batch_strs, batch_tokens = self.batch_converter(sequences)
        batch_tokens = batch_tokens.to(self.device)

        with torch.no_grad():
            results = self.model(
                batch_tokens,
                repr_layers=[self.num_layers],
                return_contacts=False,
            )

        # Extract representations from the last layer
        token_representations = results["representations"][self.num_layers]

        embeddings = {}
        for i, (header, seq) in enumerate(sequences):
            # token_representations includes BOS and EOS tokens
            # Actual residue tokens are at positions 1 to len(seq)
            seq_len = len(seq)
            residue_repr = token_representations[i, 1 : seq_len + 1]  # (seq_len, embed_dim)

            if per_residue:
                embeddings[header] = residue_repr.cpu()
            else:
                # Mean pooling over residue dimension
                embeddings[header] = residue_repr.mean(dim=0).cpu()

        return embeddings

    def embed_sequences(
        self,
        sequences: List[Tuple[str, str]],
        per_residue: bool = False,
        batch_size: int = 8,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute embeddings for a list of protein sequences.

        Parameters
        ----------
        sequences : List[Tuple[str, str]]
            List of (header, sequence) tuples.
        per_residue : bool
            If True, return per-residue embeddings (seq_len, embed_dim).
            If False (default), return per-protein embeddings (embed_dim,).
        batch_size : int
            Number of sequences per batch. Reduce if running out of memory.

        Returns
        -------
        Dict[str, torch.Tensor]
            Mapping from sequence header to embedding tensor.
        """
        # Validate sequences
        sequences = validate_protein_sequences(sequences)

        all_embeddings: Dict[str, torch.Tensor] = {}
        total = len(sequences)

        for start in range(0, total, batch_size):
            end = min(start + batch_size, total)
            batch = sequences[start:end]
            logger.info(f"Processing batch {start // batch_size + 1} "
                        f"({start + 1}-{end}/{total} sequences)")

            try:
                batch_embeddings = self._embed_batch(batch, per_residue=per_residue)
                all_embeddings.update(batch_embeddings)
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    logger.warning(
                        f"OOM on batch size {len(batch)}. "
                        f"Falling back to single-sequence processing."
                    )
                    torch.cuda.empty_cache()
                    for seq in batch:
                        single_emb = self._embed_batch([seq], per_residue=per_residue)
                        all_embeddings.update(single_emb)
                else:
                    raise

        return all_embeddings

    def embed_fasta(
        self,
        fasta_path: str,
        per_residue: bool = False,
        batch_size: int = 8,
    ) -> Dict[str, torch.Tensor]:
        """
        Read a FASTA file and compute embeddings for all sequences.

        Parameters
        ----------
        fasta_path : str
            Path to input FASTA file.
        per_residue : bool
            If True, return per-residue embeddings.
            If False (default), return per-protein embeddings.
        batch_size : int
            Number of sequences per batch.

        Returns
        -------
        Dict[str, torch.Tensor]
            Mapping from sequence header to embedding tensor.
        """
        logger.info(f"Reading sequences from {fasta_path}...")
        sequences = read_fasta(fasta_path)
        logger.info(f"Found {len(sequences)} sequences.")

        return self.embed_sequences(
            sequences,
            per_residue=per_residue,
            batch_size=batch_size,
        )

    @staticmethod
    def list_models() -> List[str]:
        """Return list of available model names."""
        return sorted(ESM2_MODELS.keys())
