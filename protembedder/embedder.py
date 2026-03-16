"""
Protein embedding extraction using ESM-2, ProtT5, and ProtBert models.

References:
    ESM-2:
        Lin, Z., et al. "Evolutionary-scale prediction of atomic-level protein structure
        with a language model." Science 379.6637 (2023): 1123-1130.
        https://doi.org/10.1126/science.ade2574

    ProtT5-XL / ProtBert:
        Elnaggar, A., et al. "ProtTrans: Toward Understanding the Language of Life
        Through Self-Supervised Learning." IEEE TPAMI 44.10 (2021): 7112-7127.
        https://doi.org/10.1109/TPAMI.2021.3095381
"""

import logging
import re
from typing import Dict, List, Optional, Tuple

import torch

from protembedder.fasta import read_fasta, validate_protein_sequences

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Model registries
# ---------------------------------------------------------------------------

# ESM-2: name -> (hub_name, embedding_dim, num_layers)
ESM2_MODELS = {
    "esm2_t6_8M":    ("esm2_t6_8M_UR50D",    320,  6),
    "esm2_t12_35M":  ("esm2_t12_35M_UR50D",   480,  12),
    "esm2_t30_150M": ("esm2_t30_150M_UR50D",  640,  30),
    "esm2_t33_650M": ("esm2_t33_650M_UR50D",  1280, 33),
    "esm2_t36_3B":   ("esm2_t36_3B_UR50D",    2560, 36),
    "esm2_t48_15B":  ("esm2_t48_15B_UR50D",   5120, 48),
}

# ProtT5: name -> (hf_repo, embedding_dim)
PROT_T5_MODELS = {
    "prot_t5_xl": ("Rostlab/prot_t5_xl_half_uniref50-enc", 1024),
}

# ProtBert: name -> (hf_repo, embedding_dim)
PROT_BERT_MODELS = {
    "prot_bert": ("Rostlab/prot_bert", 1024),
}

ALL_MODELS = (
    list(ESM2_MODELS.keys()) +
    list(PROT_T5_MODELS.keys()) +
    list(PROT_BERT_MODELS.keys())
)


def _is_esm2(model_name: str) -> bool:
    return model_name in ESM2_MODELS


def _is_prot_t5(model_name: str) -> bool:
    return model_name in PROT_T5_MODELS


def _is_prot_bert(model_name: str) -> bool:
    return model_name in PROT_BERT_MODELS


# ---------------------------------------------------------------------------
# ESM-2 backend
# ---------------------------------------------------------------------------

class _ESM2Backend:
    def __init__(self, model_name: str, device: torch.device):
        import esm as esm_lib

        hub_name, embed_dim, num_layers = ESM2_MODELS[model_name]
        self.embed_dim = embed_dim
        self.num_layers = num_layers
        self.device = device

        logger.info(f"Loading ESM-2 model {hub_name} on {device}...")
        self.model, self.alphabet = esm_lib.pretrained.load_model_and_alphabet(hub_name)
        self.model = self.model.to(device).eval()
        self.batch_converter = self.alphabet.get_batch_converter()
        logger.info(f"ESM-2 loaded. Embedding dim: {embed_dim}")

    def embed_batch(
        self, sequences: List[Tuple[str, str]], per_residue: bool
    ) -> Dict[str, torch.Tensor]:
        batch_labels, batch_strs, batch_tokens = self.batch_converter(sequences)
        batch_tokens = batch_tokens.to(self.device)

        with torch.no_grad():
            results = self.model(
                batch_tokens,
                repr_layers=[self.num_layers],
                return_contacts=False,
            )

        token_repr = results["representations"][self.num_layers]
        embeddings = {}
        for i, (header, seq) in enumerate(sequences):
            residue_repr = token_repr[i, 1 : len(seq) + 1]  # strip BOS/EOS
            embeddings[header] = residue_repr.cpu() if per_residue else residue_repr.mean(0).cpu()
        return embeddings


# ---------------------------------------------------------------------------
# ProtT5 backend
# ---------------------------------------------------------------------------

class _ProtT5Backend:
    def __init__(self, model_name: str, device: torch.device):
        from transformers import T5EncoderModel, T5Tokenizer

        hf_repo, embed_dim = PROT_T5_MODELS[model_name]
        self.embed_dim = embed_dim
        self.device = device

        logger.info(f"Loading ProtT5 model {hf_repo} on {device}...")
        self.tokenizer = T5Tokenizer.from_pretrained(hf_repo, do_lower_case=False)
        self.model = T5EncoderModel.from_pretrained(hf_repo)

        # Model ships as float16; cast to float32 when running on CPU
        if device.type == "cpu":
            self.model = self.model.float()

        self.model = self.model.to(device).eval()
        logger.info(f"ProtT5 loaded. Embedding dim: {embed_dim}")

    @staticmethod
    def _preprocess(seq: str) -> str:
        """
        ProtT5 requires space-separated amino acids.
        Non-standard residues U, Z, O, B are mapped to X.
        """
        seq = re.sub(r"[UZOB]", "X", seq.upper())
        return " ".join(seq)

    def embed_batch(
        self, sequences: List[Tuple[str, str]], per_residue: bool
    ) -> Dict[str, torch.Tensor]:
        headers = [h for h, _ in sequences]
        processed = [self._preprocess(s) for _, s in sequences]
        seq_lens = [len(s) for _, s in sequences]

        encoding = self.tokenizer(
            processed,
            add_special_tokens=True,
            padding="longest",
            return_tensors="pt",
        )
        input_ids = encoding["input_ids"].to(self.device)
        attention_mask = encoding["attention_mask"].to(self.device)

        with torch.no_grad():
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)

        # last_hidden_state: (batch, padded_len, embed_dim)
        hidden = outputs.last_hidden_state

        embeddings = {}
        for i, (header, seq_len) in enumerate(zip(headers, seq_lens)):
            # Each AA tokenises to exactly one token; EOS sits at position seq_len
            residue_repr = hidden[i, :seq_len]  # (seq_len, embed_dim)
            embeddings[header] = residue_repr.cpu() if per_residue else residue_repr.mean(0).cpu()
        return embeddings


# ---------------------------------------------------------------------------
# ProtBert backend
# ---------------------------------------------------------------------------

class _ProtBertBackend:
    def __init__(self, model_name: str, device: torch.device):
        from transformers import BertModel, BertTokenizer

        hf_repo, embed_dim = PROT_BERT_MODELS[model_name]
        self.embed_dim = embed_dim
        self.device = device

        logger.info(f"Loading ProtBert model {hf_repo} on {device}...")
        self.tokenizer = BertTokenizer.from_pretrained(hf_repo, do_lower_case=False)
        self.model = BertModel.from_pretrained(hf_repo)
        self.model = self.model.to(device).eval()
        logger.info(f"ProtBert loaded. Embedding dim: {embed_dim}")

    @staticmethod
    def _preprocess(seq: str) -> str:
        """
        ProtBert requires space-separated amino acids.
        Non-standard residues U, Z, O, B are mapped to X.
        """
        seq = re.sub(r"[UZOB]", "X", seq.upper())
        return " ".join(seq)

    def embed_batch(
        self, sequences: List[Tuple[str, str]], per_residue: bool
    ) -> Dict[str, torch.Tensor]:
        headers = [h for h, _ in sequences]
        processed = [self._preprocess(s) for _, s in sequences]
        seq_lens = [len(s) for _, s in sequences]

        encoding = self.tokenizer(
            processed,
            add_special_tokens=True,
            padding="longest",
            return_tensors="pt",
        )
        input_ids = encoding["input_ids"].to(self.device)
        attention_mask = encoding["attention_mask"].to(self.device)
        token_type_ids = encoding.get("token_type_ids")
        if token_type_ids is not None:
            token_type_ids = token_type_ids.to(self.device)

        with torch.no_grad():
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
            )

        # last_hidden_state: (batch, padded_len, embed_dim)
        # BERT adds [CLS] at position 0 and [SEP] at position seq_len+1
        hidden = outputs.last_hidden_state

        embeddings = {}
        for i, (header, seq_len) in enumerate(zip(headers, seq_lens)):
            # Skip [CLS] token at index 0; take only AA tokens
            residue_repr = hidden[i, 1 : seq_len + 1]  # (seq_len, embed_dim)
            embeddings[header] = residue_repr.cpu() if per_residue else residue_repr.mean(0).cpu()
        return embeddings


# ---------------------------------------------------------------------------
# Unified ProteinEmbedder
# ---------------------------------------------------------------------------

class ProteinEmbedder:
    """
    Extract protein embeddings from sequences using ESM-2, ProtT5, or ProtBert.

    Parameters
    ----------
    model_name : str
        One of the supported model names. Run ``ProteinEmbedder.list_models()``
        to see all options.
    device : str, optional
        Compute device ('cpu', 'cuda', 'cuda:0', ...). Auto-detects GPU if None.

    Examples
    --------
    >>> embedder = ProteinEmbedder("esm2_t33_650M")
    >>> embedder = ProteinEmbedder("prot_t5_xl")
    >>> embedder = ProteinEmbedder("prot_bert")
    >>> results = embedder.embed_fasta("proteins.fasta", per_residue=False)
    >>> torch.save(results, "embeddings.pt")
    """

    def __init__(self, model_name: str, device: Optional[str] = None):
        if model_name not in ALL_MODELS:
            available = ", ".join(sorted(ALL_MODELS))
            raise ValueError(
                f"Unknown model '{model_name}'. Available: {available}"
            )

        self.model_name = model_name
        self.device = torch.device(
            device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        )

        if _is_esm2(model_name):
            self._backend = _ESM2Backend(model_name, self.device)
        elif _is_prot_t5(model_name):
            self._backend = _ProtT5Backend(model_name, self.device)
        else:
            self._backend = _ProtBertBackend(model_name, self.device)

        self.embed_dim = self._backend.embed_dim

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
            If True, return per-residue embeddings of shape (seq_len, embed_dim).
            If False (default), return mean-pooled per-protein embeddings of shape (embed_dim,).
        batch_size : int
            Sequences per batch. Reduce if OOM.

        Returns
        -------
        Dict[str, torch.Tensor]
            Mapping from sequence header to embedding tensor.
        """
        sequences = validate_protein_sequences(sequences)
        all_embeddings: Dict[str, torch.Tensor] = {}
        total = len(sequences)

        for start in range(0, total, batch_size):
            end = min(start + batch_size, total)
            batch = sequences[start:end]
            logger.info(
                f"Processing batch {start // batch_size + 1} "
                f"({start + 1}-{end}/{total})"
            )
            try:
                all_embeddings.update(self._backend.embed_batch(batch, per_residue))
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    logger.warning("OOM — falling back to single-sequence processing.")
                    torch.cuda.empty_cache()
                    for seq in batch:
                        all_embeddings.update(self._backend.embed_batch([seq], per_residue))
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
            Sequences per batch.

        Returns
        -------
        Dict[str, torch.Tensor]
            Mapping from sequence header to embedding tensor.
        """
        logger.info(f"Reading sequences from {fasta_path}...")
        sequences = read_fasta(fasta_path)
        logger.info(f"Found {len(sequences)} sequences.")
        return self.embed_sequences(sequences, per_residue=per_residue, batch_size=batch_size)

    @staticmethod
    def list_models() -> List[str]:
        """Return list of all available model names."""
        return sorted(ALL_MODELS)
