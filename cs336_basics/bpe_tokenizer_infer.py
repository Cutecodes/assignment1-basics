import os
import argparse
import numpy as np
from tqdm import tqdm
from cs336_basics.bpe_tokenizer import BPETokenizer


def main():
    parser = argparse.ArgumentParser(description="BPE Tokenization Inference Script - Outputs int16 numpy array")

    # Core parameters
    parser.add_argument(
        "--vocab-dir",
        type=str,
        required=True,
        help="Directory of trained BPE vocabulary (contains vocab.json + merges.txt)"
    )
    parser.add_argument(
        "--input-file",
        type=str,
        required=True,
        help="Text file to tokenize (one article per line)"
    )

    parser.add_argument(
        "--special-tokens",
        nargs="+",
        default=["<|endoftext|>"],
        help="List of special tokens that will not be split"
    )

    parser.add_argument(
        "--save-path",
        type=str,
        default="tokenized_result.npy",
        help="Path to save tokenization result (.npy)"
    )

    args = parser.parse_args()

    # Validation
    if not os.path.isdir(args.vocab_dir):
        print(f"Error: Vocabulary directory does not exist: {args.vocab_dir}")
        return
    
    vocab_filepath: str = os.path.join(args.vocab_dir, "vocab.json")
    merges_filepath: str = os.path.join(args.vocab_dir, "merges.txt")
    special_tokens = args.special_tokens
    tokenizer = BPETokenizer.from_files(vocab_filepath, merges_filepath, special_tokens)

    try:
        # Get file size in bytes
        file_size_bytes = os.path.getsize(args.input_file)
        print(f"Input file raw size: {file_size_bytes:,} bytes")

        all_ids = []
        with open(args.input_file) as f:
            for _id in tqdm(tokenizer.encode_iterable(f), desc="Tokenizing", unit=" tokens"):
                all_ids.append(_id)
        
        # Total tokens
        total_tokens = len(all_ids)
        print(f"Total token count: {total_tokens:,}")

        # Compression ratio: bytes per token
        if total_tokens > 0:
            compression_ratio = file_size_bytes / total_tokens
            print(f"Compression ratio: {compression_ratio:.2f} bytes / token")
        else:
            print(f"No tokens generated, cannot compute compression ratio")

        # Save as numpy int16 array to specified path
        token_array = np.array(all_ids, dtype=np.int16)
        np.save(args.save_path, token_array)
        print(f"Tokenized result saved to: {args.save_path}")
        print(f"Saved array shape: {token_array.shape}, dtype: {token_array.dtype}")

        print("\nAll done!")
    except Exception as e:
        print(f"\nError during execution: {str(e)}")

if __name__ == "__main__":
    main()


