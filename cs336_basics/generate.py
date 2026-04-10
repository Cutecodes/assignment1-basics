
import torch
import argparse
from typing import List, Optional
import time
import os
from cs336_basics.model import TransformerLM, RoPE
from cs336_basics.utils import load_checkpoint
from cs336_basics.bpe_tokenizer import BPETokenizer
from cs336_basics.optimizer import AdamW

class InferenceEngine:
    def __init__(self, model, tokenizer, device="cuda"):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.model.eval()
        self.eos_token_id = tokenizer.encode("<|endoftext|>")[0]
        
    @torch.no_grad()
    def generate(
        self,
        prompt: str,
        max_tokens: int = 100,
        temperature: float = 1.0,
        top_p: float = 0.9,
        top_k: Optional[int] = None,
        repetition_penalty: float = 1.0,
        do_sample: bool = True,
        seed: Optional[int] = None,
    ) -> dict:
        """
        Args:
            prompt: Input prompt text
            max_new_tokens: Maximum number of tokens to generate
            temperature: Temperature parameter (controls randomness, 0=deterministic)
            top_p: Nucleus sampling probability threshold
            top_k: Top-k sampling parameter
            repetition_penalty: Repetition penalty coefficient
            do_sample: Whether to sample (False uses greedy decoding)
            seed: Random seed
            
        Returns:
            Dictionary containing generation results and statistics
        """
        if seed is not None:
            torch.manual_seed(seed)
        
        # Tokenize input
        input_ids = torch.tensor([self.tokenizer.encode(prompt)]).to(self.device)

        prompt_length = input_ids.shape[1]
        
        # start_time
        start_time = time.time()
        
        # generate
        with torch.no_grad():
            output_ids = self._generate_loop(
                input_ids=input_ids,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                repetition_penalty=repetition_penalty,
                do_sample=do_sample,
            )
        
        # generation time
        generation_time = time.time() - start_time
        generated_text = self.tokenizer.decode(output_ids[0].tolist())
        output_length = output_ids.shape[1]
        generated_tokens = output_length - prompt_length
        
        # generate speed
        tokens_per_second = generated_tokens / generation_time if generation_time > 0 else 0
        
        return {
            "prompt": prompt,
            "generated_text": generated_text,
            "prompt_length": prompt_length,
            "output_length": output_length,
            "generated_tokens": generated_tokens,
            "generation_time": generation_time,
            "tokens_per_second": tokens_per_second,
            "temperature": temperature,
            "top_p": top_p,
            "top_k": top_k,
        }
    
    def _generate_loop(
        self,
        input_ids: torch.Tensor,
        max_tokens: int,
        temperature: float,
        top_p: float,
        top_k: Optional[int],
        repetition_penalty: float,
        do_sample: bool,
    ) -> torch.Tensor:
        """generate loop"""
        
        while len(input_ids[0]) < max_tokens:

            logits = self.model(input_ids)
            next_token_logits = logits[0, -1, :] / temperature
            
            # apply repetition
            if repetition_penalty != 1.0:
                for token_id in set(input_ids[0].tolist()):
                    if next_token_logits[token_id] < 0:
                        next_token_logits[token_id] *= repetition_penalty
                    else:
                        next_token_logits[token_id] /= repetition_penalty
            
            if do_sample:
                # Top-p (nucleus) sampling
                if top_p < 1.0:
                    sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                    cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
                    
                    sorted_indices_to_remove = cumulative_probs > top_p
                    sorted_indices_to_remove[1:] = sorted_indices_to_remove[:-1].clone()
                    sorted_indices_to_remove[0] = False
                    
                    indices_to_remove = sorted_indices[sorted_indices_to_remove]
                    next_token_logits[indices_to_remove] = float('-inf')
                
                # Top-k sampling
                if top_k is not None and top_k > 0:
                    indices_to_remove = next_token_logits < torch.topk(next_token_logits, top_k)[0][..., -1, None]
                    next_token_logits[indices_to_remove] = float('-inf')
                
                # sample
                probs = torch.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
            else:
                # greedy
                next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
            
            input_ids = torch.cat([input_ids, next_token.unsqueeze(0)], dim=-1)
            
            # EOS token
            if next_token.item() == self.eos_token_id:
                break
        
        return input_ids


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Transformer Model Inference")
    
    # Model parameters
    parser.add_argument("--checkpoint", type=str, required=True, help="Model checkpoint path")

    # Tokenizer parameters
    parser.add_argument("--vocab_dir", type=str, required=True, help="Directory of trained BPE vocabulary (contains vocab.json + merges.txt)")

    # Inference parameters
    parser.add_argument("--prompt", type=str, default="Once upon a time,", help="Input prompt")
    parser.add_argument("--max_tokens", type=int, default=256, help="Maximum number of tokens")
    parser.add_argument("--temperature", type=float, default=1.0, help="Temperature parameter (0-2)")
    parser.add_argument("--top_p", type=float, default=0.9, help="Top-p sampling (0-1)")
    parser.add_argument("--top_k", type=int, default=None, help="Top-k sampling")
    parser.add_argument("--repetition_penalty", type=float, default=1.0, help="Repetition penalty")
    parser.add_argument("--do_sample", action="store_true", default=True, help="Whether to sample")
    parser.add_argument("--greedy", action="store_true", help="Use greedy decoding (overrides sampling)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    # Other parameters
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    
    return parser.parse_args()


def load_model_and_tokenizer(checkpoint, vocab_dir):
    vocab_filepath: str = os.path.join(vocab_dir, "vocab.json")
    merges_filepath: str = os.path.join(vocab_dir, "merges.txt")
    special_tokens = ["<|endoftext|>"]
    tokenizer = BPETokenizer.from_files(vocab_filepath, merges_filepath, special_tokens)
    
    rope = RoPE(10000, 32, 256)
    
    model = TransformerLM(
        10000,
        256, 
        4, 
        512, 
        16, 
        1344, 
        rope
    )
    optimizer = optimizer = AdamW(model.parameters())
    step = load_checkpoint(checkpoint, model, optimizer)
    print(step)
    return model, tokenizer

def main():
    args = parse_args()
    
    # Load model
    print(f"Loading model from {args.checkpoint}...")
    model, tokenizer = load_model_and_tokenizer(args.checkpoint, args.vocab_dir)
    
    model = model.to(args.device)
    
    # create inference engine
    engine = InferenceEngine(model, tokenizer, args.device)
    
    # Prepare generation parameters
    generate_kwargs = {
        "max_tokens": args.max_tokens,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "top_k": args.top_k,
        "repetition_penalty": args.repetition_penalty,
        "do_sample": not args.greedy,
        "seed": args.seed,
    }
    
    # show config
    print("\n" + "="*60)
    print("Inference Configuration")
    print("="*60)
    print(f"Device: {args.device}")
    print(f"Max tokens: {args.max_tokens}")
    print(f"Temperature: {args.temperature}")
    print(f"Top-p: {args.top_p}")
    print(f"Top-k: {args.top_k}")
    print(f"Repetition penalty: {args.repetition_penalty}")
    print(f"Sampling: {not args.greedy}")
    print("="*60)
    
    
    result = engine.generate(args.prompt, **generate_kwargs)
        
    print("\n" + "="*60)
    print("Generation Result")
    print("="*60)
    print(f"Prompt ({result['prompt_length']} tokens): {result['prompt']}")
    print(f"\nGenerated ({result['generated_tokens']} tokens):")
    print(result['generated_text'])
    print("\n" + "="*60)
    print(f"Statistics:")
    print(f"  Generation time: {result['generation_time']:.2f}s")
    print(f"  Tokens per second: {result['tokens_per_second']:.2f}")
    print(f"  Temperature: {result['temperature']}")
    print(f"  Top-p: {result['top_p']}")
    print("="*60)

if __name__ == "__main__":
    main()