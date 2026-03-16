"""Tests for the FASTA parser module."""

import pytest
import tempfile
from pathlib import Path

from protembedder.fasta import read_fasta, validate_protein_sequences


@pytest.fixture
def sample_fasta(tmp_path):
    """Create a sample FASTA file."""
    content = """>sp|P0A7Y4|ATPA_ECOLI ATP synthase subunit alpha
MQLNSTEISELIKQRIAQFNVVSEAHNEGTIVSVSDGVIRIHLETDDHPILDRMDRFHIE
NQSGILGIDLPGVGERTYRGKIALTQYVDTKLTSLEDKLAHGHFDKVATEQSMTAYIPVN
>sp|P00698|LYSC_CHICK Lysozyme C
KVFGRCELAAAMKRHGLDNYRGYSLGNWVCAAKFESNFNTQATNRNTDGSTDYGILQINS
RWWCNDGRTPGSRNLCNIPCSALLSSDITASVNCAKKIVSDGNGMNAWVAWRNRCKGTDVQ
>short_protein
MKTAY
"""
    fasta_file = tmp_path / "test.fasta"
    fasta_file.write_text(content)
    return fasta_file


@pytest.fixture
def empty_fasta(tmp_path):
    fasta_file = tmp_path / "empty.fasta"
    fasta_file.write_text("")
    return fasta_file


def test_read_fasta_basic(sample_fasta):
    sequences = read_fasta(str(sample_fasta))
    assert len(sequences) == 3
    assert sequences[0][0] == "sp|P0A7Y4|ATPA_ECOLI ATP synthase subunit alpha"
    assert sequences[2][0] == "short_protein"
    assert sequences[2][1] == "MKTAY"


def test_read_fasta_multiline_sequences(sample_fasta):
    sequences = read_fasta(str(sample_fasta))
    # First sequence is split across 3 lines
    seq1 = sequences[0][1]
    assert seq1.startswith("MQLNSTEISELIKQ")
    assert len(seq1) > 60  # Must be multi-line concatenated


def test_read_fasta_file_not_found():
    with pytest.raises(FileNotFoundError):
        read_fasta("/nonexistent/path.fasta")


def test_read_fasta_empty(empty_fasta):
    with pytest.raises(ValueError, match="No valid sequences"):
        read_fasta(str(empty_fasta))


def test_validate_protein_sequences():
    sequences = [("test", "ACDEFGHIKLMNPQRSTVWY")]
    validated = validate_protein_sequences(sequences)
    assert validated[0][1] == "ACDEFGHIKLMNPQRSTVWY"


def test_validate_replaces_nonstandard():
    sequences = [("test", "ACDE123FGH")]
    validated = validate_protein_sequences(sequences)
    assert validated[0][1] == "ACDEXXXFGH"
