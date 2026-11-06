"""Fine-tune DistilGPT-2 on the train split shared with the Keras LSTM."""
import argparse
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]


def make_blocks(ids, block_size, pad_id):
    blocks = [ids[i:i + block_size] for i in range(0, len(ids) - block_size + 1, block_size)]
    return blocks


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--block-size", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=5e-5)
    parser.add_argument("--offline", action="store_true", help="Use only model files already in the Hugging Face cache.")
    args = parser.parse_args()
    torch.manual_seed(7)
    text = (ROOT / "data" / "tiny_corpus.txt").read_text(encoding="utf-8")
    split = int(len(text) * 0.85)
    train_text, valid_text = text[:split], text[split:]
    tokenizer = AutoTokenizer.from_pretrained("distilgpt2", local_files_only=args.offline)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained("distilgpt2", local_files_only=args.offline)
    train_ids = tokenizer(train_text, return_tensors="pt")["input_ids"][0]
    valid_ids = tokenizer(valid_text, return_tensors="pt")["input_ids"][0]
    train_blocks = make_blocks(train_ids, args.block_size, tokenizer.pad_token_id)
    valid_blocks = make_blocks(valid_ids, args.block_size, tokenizer.pad_token_id)
    if not train_blocks or not valid_blocks:
        raise ValueError("Corpus split is too short for the chosen block size.")
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    model.train()
    for epoch in range(args.epochs):
        total = 0.0
        count = 0
        for start in range(0, len(train_blocks), args.batch_size):
            batch = torch.stack(train_blocks[start:start + args.batch_size])
            loss = model(input_ids=batch, attention_mask=torch.ones_like(batch), labels=batch).loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad()
            total += float(loss.item())
            count += 1
        print(f"epoch={epoch + 1} train_loss={total / max(count, 1):.4f}")
    model.eval()
    with torch.no_grad():
        batch = torch.stack(valid_blocks)
        val_loss = float(model(input_ids=batch, attention_mask=torch.ones_like(batch), labels=batch).loss.item())
    output = ROOT / "models" / "distilgpt2-finetuned"
    output.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(output)
    tokenizer.save_pretrained(output)
    print(f"validation_perplexity={torch.exp(torch.tensor(val_loss)).item():.3f}")
    print(f"Saved fine-tuned checkpoint to {output}")


if __name__ == "__main__":
    main()

