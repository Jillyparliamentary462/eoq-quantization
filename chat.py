#!/usr/bin/env python3
"""Interactive chat with an EOQ-quantized model."""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TextStreamer
from core.utils import quantize_absmax, dequantize


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Chat with EOQ model")
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--bits", type=int, default=4)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.7)
    args = parser.parse_args()

    print(f"Loading {args.model} (Q{args.bits})...", flush=True)

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.float32, trust_remote_code=True
    )

    # Quantize weights
    count = 0
    for name, param in model.named_parameters():
        if param.ndim >= 2 and param.numel() >= 256:
            with torch.no_grad():
                qt = quantize_absmax(param.data, args.bits, 128)
                param.data.copy_(dequantize(qt))
                count += 1
    model.eval()

    print(f"Ready. {count} tensors quantized to Q{args.bits}.")
    print(f"Type your message. /quit to exit. /clear to reset.\n")

    streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
    history = []

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye!")
            break

        if not user_input:
            continue
        if user_input.lower() in ("/quit", "/exit", "quit", "exit"):
            print("Bye!")
            break
        if user_input.lower() in ("/clear", "/reset"):
            history = []
            print("-- History cleared --\n")
            continue

        history.append({"role": "user", "content": user_input})

        prompt = tokenizer.apply_chat_template(
            history, tokenize=False, add_generation_prompt=True
        )
        inputs = tokenizer(prompt, return_tensors="pt")

        print("AI: ", end="", flush=True)
        with torch.no_grad():
            output = model.generate(
                **inputs,
                max_new_tokens=args.max_tokens,
                do_sample=args.temperature > 0,
                temperature=args.temperature if args.temperature > 0 else None,
                top_p=0.9,
                streamer=streamer,
            )

        # Extract assistant response for history
        full_output = tokenizer.decode(output[0], skip_special_tokens=True)
        assistant_text = full_output[len(tokenizer.apply_chat_template(
            history, tokenize=False, add_generation_prompt=False
        )):].strip()
        history.append({"role": "assistant", "content": assistant_text})
        print()


if __name__ == "__main__":
    main()
