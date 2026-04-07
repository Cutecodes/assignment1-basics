from cs336_basics.bpe_tokenizer import BPETokenizerTrainer

import os
import argparse
import json

def run_train_bpe(
    input_path: str | os.PathLike,
    vocab_size: int,
    special_tokens: list[str],
    processor_num = 4,
    chunk_size = 100 * 1024 * 1024
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    bpe = BPETokenizerTrainer(input_path, vocab_size, special_tokens, processor_num = processor_num, chunk_size = chunk_size)
    bpe.train()
    return bpe.vocab, bpe.merges


def save_tokenizer(
    vocab: dict[int, bytes],
    merges: list[tuple[bytes, bytes]],
    save_dir: str = "bpe_tokenizer"
) -> None:
    """Save the trained vocab and merges to a directory for later use."""
    os.makedirs(save_dir, exist_ok=True)

    # Save vocabulary as JSON (serializable format)
    vocab_serializable = {str(idx): token.hex() for idx, token in vocab.items()}
    with open(os.path.join(save_dir, "vocab.json"), "w", encoding="utf-8") as f:
        json.dump(vocab_serializable, f, indent=2)

    # Save merges
    with open(os.path.join(save_dir, "merges.txt"), "w", encoding="utf-8") as f:
        f.write("# BPE merge operations (ordered by creation)\n")
        for a, b in merges:
            f.write(f"{a.hex()} {b.hex()}\n")

    print(f"Tokenizer saved to: {save_dir}")


def main():
    parser = argparse.ArgumentParser(description="Train a BPE tokenizer from a text corpus.")
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to the input text corpus for training"
    )
    parser.add_argument(
        "--vocab-size",
        type=int,
        required=True,
        help="Total final vocabulary size (including special tokens)"
    )
    parser.add_argument(
        "--special-tokens",
        nargs="+",
        default=[],
        help="List of special tokens that will not be split"
    )
    parser.add_argument(
        "--save-dir",
        type=str,
        default="bpe_tokenizer",
        help="Directory to save the trained tokenizer"
    )

    parser.add_argument(
        "--num-processes",
        type=int,
        default=16,
        help="Number of processes to use for parallel processing (default: 4)"
    )

    parser.add_argument(
        "--chunk-size",
        type=int,
        default=100,
        help="Size of file chunks in MB to process per process (default: 100 MB)"
    )

    args = parser.parse_args()

    # Validation
    if not os.path.isfile(args.input):
        print(f"Error: Input file not found: {args.input}")
        return

    min_required_vocab = 256 + len(args.special_tokens)
    if args.vocab_size < min_required_vocab:
        print(f"Error: vocab_size must be at least {min_required_vocab}")
        return

    # Training
    print("Starting BPE training...")
    print(f"Input corpus: {args.input}")
    print(f"Target vocab size: {args.vocab_size}")
    print(f"Special tokens: {args.special_tokens}")
    print(f"Number of processes: {args.num_processes}")
    print(f"Size of file chunks: {args.chunk_size} MB")

    vocab, merges = run_train_bpe(
        input_path=args.input,
        vocab_size=args.vocab_size,
        special_tokens=args.special_tokens,
        processor_num=args.num_processes,
        chunk_size=args.chunk_size * 1024 * 1024
    )

    print("\nTraining complete!")
    print(f"Final vocab size: {len(vocab)}")
    print(f"Number of merges: {len(merges)}")
    longest_idx = max(vocab, key=lambda i: len(vocab[i]))
    print(f"Longest token: index {longest_idx}, value {vocab[longest_idx]} (length: {len(vocab[longest_idx])} bytes)")

    save_tokenizer(vocab, merges, args.save_dir)
    print("Done.")

if __name__ == "__main__":
    main()