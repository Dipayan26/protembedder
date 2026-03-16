"""
FASTA file parser for protein sequences.
"""

from typing import Dict, List, Tuple
from pathlib import Path


def read_fasta(fasta_path: str) -> List[Tuple[str, str]]:
    """
    Parse a FASTA file and return a list of (header, sequence) tuples.

    Parameters
    ----------
    fasta_path : str
        Path to the input FASTA file.

    Returns
    -------
    List[Tuple[str, str]]
        List of (header, sequence) tuples. Headers have the leading '>' stripped.

    Raises
    ------
    FileNotFoundError
        If the FASTA file does not exist.
    ValueError
        If the file is empty or contains no valid sequences.
    """
    fasta_path = Path(fasta_path)
    if not fasta_path.exists():
        raise FileNotFoundError(f"FASTA file not found: {fasta_path}")

    sequences: List[Tuple[str, str]] = []
    current_header = None
    current_seq_parts: List[str] = []

    with open(fasta_path, "r") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                # Save previous sequence if exists
                if current_header is not None:
                    seq = "".join(current_seq_parts)
                    if not seq:
                        raise ValueError(
                            f"Empty sequence for header '{current_header}'"
                        )
                    sequences.append((current_header, seq))
                current_header = line[1:].strip()
                current_seq_parts = []
            else:
                if current_header is None:
                    raise ValueError(
                        f"Line {line_num}: Sequence data found before any header"
                    )
                # Remove whitespace and validate characters
                seq_chunk = line.replace(" ", "").upper()
                current_seq_parts.append(seq_chunk)

    # Don't forget the last sequence
    if current_header is not None:
        seq = "".join(current_seq_parts)
        if not seq:
            raise ValueError(f"Empty sequence for header '{current_header}'")
        sequences.append((current_header, seq))

    if not sequences:
        raise ValueError(f"No valid sequences found in {fasta_path}")

    return sequences


def validate_protein_sequences(sequences: List[Tuple[str, str]]) -> List[Tuple[str, str]]:
    """
    Validate that sequences contain only standard amino acid characters.
    Non-standard residues are replaced with 'X' (unknown).

    Parameters
    ----------
    sequences : List[Tuple[str, str]]
        List of (header, sequence) tuples.

    Returns
    -------
    List[Tuple[str, str]]
        Validated sequences with non-standard residues replaced.
    """
    standard_aa = set("ACDEFGHIKLMNPQRSTVWY")
    # Also allow common non-standard: X (unknown), U (selenocysteine),
    # O (pyrrolysine), B, Z, J (ambiguous)
    allowed = standard_aa | set("XUBZJO")

    validated = []
    for header, seq in sequences:
        cleaned = []
        for aa in seq:
            if aa in allowed:
                cleaned.append(aa)
            else:
                cleaned.append("X")
        validated.append((header, "".join(cleaned)))

    return validated
