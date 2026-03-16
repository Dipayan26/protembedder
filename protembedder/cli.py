"""
Command-line interface for ProtEmbedder.

Subcommands
-----------
    protembedder embed        Extract embeddings from a FASTA file
    protembedder list-models  Print all available model names and exit

Usage examples:
    # Per-protein embeddings (default) with ESM-2 650M, PyTorch output
    protembedder embed -m esm2_t33_650M -i proteins.fasta -o embeddings.pt

    # HDF5 output
    protembedder embed -m prot_t5_xl -i proteins.fasta -o embeddings.h5 --format h5

    # NumPy .npz output
    protembedder embed -m prot_bert -i proteins.fasta -o embeddings.npz --format npz

    # CSV output, per-residue
    protembedder embed -m prot_bert -i proteins.fasta -o embeddings.csv --format csv --per-residue

    # List available models
    protembedder list-models
"""

import logging
import sys
import time
import argparse
from pathlib import Path

import torch

from protembedder.embedder import ProteinEmbedder, ALL_MODELS
from protembedder.io import save_embeddings, SUPPORTED_FORMATS


# ---------------------------------------------------------------------------
# Sub-parser: embed
# ---------------------------------------------------------------------------

def _build_embed_parser(subparsers):
    p = subparsers.add_parser(
        "embed",
        help="Extract embeddings from a FASTA file.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=(
            "Extract protein embeddings from a FASTA file using ESM-2, ProtT5, "
            "or ProtBert models. Output is a file mapping sequence headers to "
            "embedding tensors."
        ),
        epilog=(
            "Examples:\n"
            "  protembedder embed -m esm2_t33_650M -i proteins.fasta -o out.pt\n"
            "  protembedder embed -m prot_t5_xl    -i proteins.fasta -o out.h5  --format h5\n"
            "  protembedder embed -m prot_bert     -i proteins.fasta -o out.npz --format npz\n"
            "  protembedder embed -m prot_bert     -i proteins.fasta -o out.csv --format csv --per-residue\n\n"
            "References:\n"
            "  ESM-2:         Lin et al., Science 379.6637 (2023). https://doi.org/10.1126/science.ade2574\n"
            "  ProtT5/ProtBert: Elnaggar et al., IEEE TPAMI 44.10 (2021). https://doi.org/10.1109/TPAMI.2021.3095381"
        ),
    )
    p.add_argument(
        "-m", "--model",
        type=str,
        required=True,
        choices=sorted(ALL_MODELS),
        metavar="MODEL",
        help=f"Model name. Run 'protembedder list-models' to see all options.",
    )
    p.add_argument(
        "-i", "--input",
        type=str,
        required=True,
        help="Path to input FASTA file.",
    )
    p.add_argument(
        "-o", "--output",
        type=str,
        required=True,
        help="Output file path. Extension is auto-corrected to match --format.",
    )
    p.add_argument(
        "--format",
        type=str,
        default="pt",
        choices=SUPPORTED_FORMATS,
        dest="fmt",
        help=(
            "Output format (default: pt). "
            "pt=PyTorch dict, h5=HDF5, npz=NumPy archive, csv=CSV table."
        ),
    )
    p.add_argument(
        "--per-residue",
        action="store_true",
        default=False,
        help=(
            "Return per-residue (per amino acid) embeddings of shape (seq_len, embed_dim) "
            "instead of per-protein mean-pooled embeddings of shape (embed_dim,)."
        ),
    )
    p.add_argument(
        "--device",
        type=str,
        default=None,
        help="Compute device: 'cpu', 'cuda', 'cuda:0', etc. Auto-detects GPU if omitted.",
    )
    p.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Sequences per batch (default: 8). Reduce if OOM.",
    )
    p.add_argument(
        "--no-progress",
        action="store_true",
        default=False,
        help="Disable the tqdm progress bar.",
    )
    p.add_argument(
        "-v", "--verbose",
        action="store_true",
        default=False,
        help="Enable verbose logging output.",
    )
    return p


# ---------------------------------------------------------------------------
# Sub-parser: list-models
# ---------------------------------------------------------------------------

def _build_list_models_parser(subparsers):
    p = subparsers.add_parser(
        "list-models",
        help="Print all available model names and exit.",
        description="List every model supported by protembedder.",
    )
    p.add_argument(
        "--verbose",
        action="store_true",
        default=False,
        help="Show model details (family, embedding dim, HuggingFace repo).",
    )
    return p


# ---------------------------------------------------------------------------
# Root parser
# ---------------------------------------------------------------------------

def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="protembedder",
        description="Extract protein embeddings from FASTA files using protein language models.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--version",
        action="version",
        version="%(prog)s 0.4.0",
    )
    subparsers = parser.add_subparsers(dest="subcommand", metavar="SUBCOMMAND")
    subparsers.required = True
    _build_embed_parser(subparsers)
    _build_list_models_parser(subparsers)
    return parser


# ---------------------------------------------------------------------------
# Handlers
# ---------------------------------------------------------------------------

def _handle_list_models(args):
    from protembedder.embedder import ESM2_MODELS, PROT_T5_MODELS, PROT_BERT_MODELS

    if args.verbose:
        print(f"{'Model':<20} {'Family':<10} {'Embed Dim':>10}  Repo / Hub")
        print("-" * 72)
        for name in sorted(ESM2_MODELS):
            hub, dim, _ = ESM2_MODELS[name]
            print(f"{name:<20} {'ESM-2':<10} {dim:>10}  {hub}")
        for name in sorted(PROT_T5_MODELS):
            repo, dim = PROT_T5_MODELS[name]
            print(f"{name:<20} {'ProtT5':<10} {dim:>10}  {repo}")
        for name in sorted(PROT_BERT_MODELS):
            repo, dim = PROT_BERT_MODELS[name]
            print(f"{name:<20} {'ProtBert':<10} {dim:>10}  {repo}")
    else:
        for model in ProteinEmbedder.list_models():
            print(model)


def _handle_embed(args):
    log_level = logging.INFO if args.verbose else logging.WARNING
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
    )

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Error: Input file not found: {input_path}", file=sys.stderr)
        sys.exit(1)

    embedding_type = "per-residue" if args.per_residue else "per-protein"
    show_progress = not args.no_progress

    print(f"ProtEmbedder v0.4.0")
    print(f"  Model:      {args.model}")
    print(f"  Input:      {input_path}")
    print(f"  Output:     {args.output}")
    print(f"  Format:     {args.fmt}")
    print(f"  Embedding:  {embedding_type}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Device:     {args.device or 'auto'}")
    print()

    start_time = time.time()

    print("Loading model...")
    embedder = ProteinEmbedder(model_name=args.model, device=args.device)
    print(f"  Device:        {embedder.device}")
    print(f"  Embedding dim: {embedder.embed_dim}")
    print()

    print("Extracting embeddings...")
    embeddings = embedder.embed_fasta(
        fasta_path=str(input_path),
        per_residue=args.per_residue,
        batch_size=args.batch_size,
        show_progress=show_progress,
    )

    print(f"\nSaving {len(embeddings)} embeddings...")
    out_path = save_embeddings(embeddings, args.output, fmt=args.fmt)

    elapsed = time.time() - start_time
    print(f"Done! Processed {len(embeddings)} sequences in {elapsed:.1f}s.")

    sample_key = next(iter(embeddings))
    print(f"  Embedding shape: {embeddings[sample_key].shape}")
    print(f"  Output file:     {out_path} ({out_path.stat().st_size / 1024:.1f} KB)")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = create_parser()
    args = parser.parse_args()

    if args.subcommand == "list-models":
        _handle_list_models(args)
    elif args.subcommand == "embed":
        _handle_embed(args)


if __name__ == "__main__":
    main()
