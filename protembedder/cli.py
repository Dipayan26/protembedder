"""
Command-line interface for ProtEmbedder.

Usage examples:
    # Per-protein embeddings (default) with ESM-2 650M
    protembedder --model esm2_t33_650M --input proteins.fasta --output embeddings.pt

    # Per-protein embeddings with ProtT5-XL
    protembedder --model prot_t5_xl --input proteins.fasta --output embeddings.pt

    # Per-protein embeddings with ProtBert
    protembedder --model prot_bert --input proteins.fasta --output embeddings.pt

    # Per-residue (per amino acid) embeddings
    protembedder --model prot_bert --input proteins.fasta --output embeddings.pt --per-residue

    # Use GPU with custom batch size
    protembedder --model esm2_t33_650M --input proteins.fasta --output embeddings.pt --device cuda --batch-size 16
"""

import argparse
import logging
import sys
import time
from pathlib import Path

import torch

from protembedder.embedder import ProteinEmbedder, ALL_MODELS


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="protembedder",
        description=(
            "Extract protein embeddings from FASTA files using ESM-2, ProtT5, or ProtBert models. "
            "Outputs a .pt file containing a dict mapping sequence headers to "
            "embedding tensors."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  # Per-protein embeddings with ESM-2 650M\n"
            "  protembedder -m esm2_t33_650M -i proteins.fasta -o embeddings.pt\n\n"
            "  # Per-protein embeddings with ProtT5-XL\n"
            "  protembedder -m prot_t5_xl -i proteins.fasta -o embeddings.pt\n\n"
            "  # Per-protein embeddings with ProtBert\n"
            "  protembedder -m prot_bert -i proteins.fasta -o embeddings.pt\n\n"
            "  # Per-residue embeddings on GPU\n"
            "  protembedder -m prot_bert -i proteins.fasta -o embeddings.pt "
            "--per-residue --device cuda\n\n"
            "  # Small ESM-2 model, large batch\n"
            "  protembedder -m esm2_t6_8M -i proteins.fasta -o embeddings.pt "
            "--batch-size 32\n\n"
            "References:\n"
            "  ESM-2:    Lin et al., Science 379.6637 (2023). https://doi.org/10.1126/science.ade2574\n"
            "  ProtT5/ProtBert: Elnaggar et al., IEEE TPAMI 44.10 (2021). https://doi.org/10.1109/TPAMI.2021.3095381"
        ),
    )

    parser.add_argument(
        "-m", "--model",
        type=str,
        required=True,
        choices=sorted(ALL_MODELS),
        help="Model to use for embedding extraction (ESM-2, ProtT5, or ProtBert).",
    )
    parser.add_argument(
        "-i", "--input",
        type=str,
        required=True,
        help="Path to input FASTA file containing protein sequences.",
    )
    parser.add_argument(
        "-o", "--output",
        type=str,
        required=True,
        help="Path for output .pt file containing embeddings dict.",
    )
    parser.add_argument(
        "--per-residue",
        action="store_true",
        default=False,
        help=(
            "Output per-residue (per amino acid) embeddings instead of "
            "per-protein embeddings. Per-protein: shape (embed_dim,). "
            "Per-residue: shape (seq_len, embed_dim)."
        ),
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help=(
            "Device for inference: 'cpu', 'cuda', 'cuda:0', etc. "
            "Auto-detects GPU if not specified."
        ),
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Number of sequences per batch (default: 8). Reduce if OOM.",
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        default=False,
        help="Enable verbose logging output.",
    )
    parser.add_argument(
        "--version",
        action="version",
        version="%(prog)s 0.3.0",
    )

    return parser


def main():
    parser = create_parser()
    args = parser.parse_args()

    # Setup logging
    log_level = logging.INFO if args.verbose else logging.WARNING
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
    )

    # Validate input file
    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Error: Input file not found: {input_path}", file=sys.stderr)
        sys.exit(1)

    # Validate output path
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not args.output.endswith(".pt"):
        print("Warning: Output file does not end with .pt extension.", file=sys.stderr)

    # Run embedding extraction
    embedding_type = "per-residue" if args.per_residue else "per-protein"
    print(f"ProtEmbedder v0.3.0")
    print(f"  Model:      {args.model}")
    print(f"  Input:      {input_path}")
    print(f"  Output:     {output_path}")
    print(f"  Embedding:  {embedding_type}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Device:     {args.device or 'auto'}")
    print()

    start_time = time.time()

    # Load model
    print("Loading model...")
    embedder = ProteinEmbedder(model_name=args.model, device=args.device)
    print(f"  Device: {embedder.device}")
    print(f"  Embedding dim: {embedder.embed_dim}")
    print()

    # Extract embeddings
    print("Extracting embeddings...")
    embeddings = embedder.embed_fasta(
        fasta_path=str(input_path),
        per_residue=args.per_residue,
        batch_size=args.batch_size,
    )

    # Save
    print(f"Saving {len(embeddings)} embeddings to {output_path}...")
    torch.save(embeddings, str(output_path))

    elapsed = time.time() - start_time
    print(f"\nDone! Processed {len(embeddings)} sequences in {elapsed:.1f}s.")

    # Print summary
    if embeddings:
        sample_key = next(iter(embeddings))
        sample_shape = embeddings[sample_key].shape
        print(f"  Embedding shape: {sample_shape}")
        print(f"  Output file: {output_path} ({output_path.stat().st_size / 1024:.1f} KB)")


if __name__ == "__main__":
    main()
