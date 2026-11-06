"""Reproducible P16 evaluation: perplexity, NSP coherence, outputs, overlap."""
import csv
import json
import math
import re
import argparse
from pathlib import Path

import numpy as np
import tensorflow as tf
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoModelForNextSentencePrediction,
    AutoTokenizer,
)

ROOT = Path(__file__).resolve().parents[1]
PROMPT = "A traveler arrived at the observatory with a question."
SEED = 7
TEMPERATURE = 0.8
TOP_K = 40
GPT_NEW_TOKENS = 100
LSTM_NEW_CHARS = 240


def lstm_perplexity(model, mapping, text):
    ids = np.asarray([mapping.get(char, 0) for char in text], dtype=np.int32)
    width, nll, count = 80, 0.0, 0
    for start in range(0, len(ids) - width, width):
        x, y = ids[start:start + width], ids[start + 1:start + width + 1]
        logits = model(x[None, :], training=False)[0]
        losses = tf.keras.losses.sparse_categorical_crossentropy(y, logits, from_logits=True)
        nll += float(tf.reduce_sum(losses).numpy())
        count += width
    return math.exp(nll / count)


def generate_lstm(model, mapping):
    chars, to_id = mapping["chars"], mapping["to_id"]
    reverse = {v: k for k, v in to_id.items()}
    context = [to_id.get(char, 0) for char in PROMPT[-80:]] or [0]
    output = list(PROMPT)
    for _ in range(LSTM_NEW_CHARS):
        x = np.array([([0] * max(0, 80 - len(context)) + context[-80:])], dtype=np.int32)
        logits = model(x, training=False)[0, -1].numpy() / TEMPERATURE
        probs = tf.nn.softmax(logits).numpy()
        k = min(TOP_K, len(probs))
        top = np.argpartition(probs, -k)[-k:]
        probs = probs[top] / probs[top].sum()
        sampled = int(np.random.choice(top, p=probs))
        context.append(sampled)
        output.append(reverse[sampled])
    return "".join(output)


def nsp_score(model, tokenizer, first, second):
    inputs = tokenizer(first, second, return_tensors="pt", truncation=True, max_length=256)
    with torch.no_grad():
        logits = model(**inputs).logits[0]
    return float(torch.softmax(logits, dim=-1)[0].item())


def overlap(output, train_text):
    out = re.sub(r"\s+", " ", output.lower()).strip()
    source = re.sub(r"\s+", " ", train_text.lower())
    for width in (80, 60, 40, 30, 20):
        for i in range(max(0, len(out) - width + 1)):
            span = out[i:i + width]
            if span in source:
                return len(span), span
    return 0, ""


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--offline", action="store_true", help="Use only model files already in the Hugging Face cache.")
    args = parser.parse_args()
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    tf.random.set_seed(SEED)
    corpus = (ROOT / "data" / "tiny_corpus.txt").read_text(encoding="utf-8")
    split = int(len(corpus) * 0.85)
    train_text, held_out = corpus[:split], corpus[split:]

    char_data = json.loads((ROOT / "models" / "char_vocab.json").read_text(encoding="utf-8"))
    lstm = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(80,)),
        tf.keras.layers.Embedding(len(char_data["chars"]), 64),
        tf.keras.layers.LSTM(128, return_sequences=True),
        tf.keras.layers.Dense(len(char_data["chars"])),
    ])
    lstm.load_weights(ROOT / "models" / "char_lstm.weights.h5")
    lstm_ppl = lstm_perplexity(lstm, char_data["to_id"], held_out)
    lstm_output = generate_lstm(lstm, char_data)

    checkpoint_path = ROOT / "models" / "distilgpt2-finetuned"
    checkpoint = str(checkpoint_path) if checkpoint_path.exists() else "distilgpt2"
    checkpoint_label = "fine-tuned local DistilGPT-2" if checkpoint_path.exists() else "base distilgpt2 (fine-tuned checkpoint unavailable)"
    if not checkpoint_path.exists():
        print("WARNING: fine-tuned checkpoint missing; using base DistilGPT-2 for the transformer row.")
    tokenizer = AutoTokenizer.from_pretrained(checkpoint, local_files_only=args.offline)
    gpt = AutoModelForCausalLM.from_pretrained(checkpoint, local_files_only=args.offline)
    encoded = tokenizer(held_out, return_tensors="pt", truncation=True, max_length=1024)
    with torch.no_grad():
        gpt_ppl = float(torch.exp(gpt(**encoded, labels=encoded["input_ids"]).loss).item())
        prompt_ids = tokenizer(PROMPT, return_tensors="pt")
        generated_ids = gpt.generate(**prompt_ids, max_new_tokens=GPT_NEW_TOKENS,
                                     do_sample=True, temperature=TEMPERATURE, top_k=TOP_K,
                                     pad_token_id=tokenizer.eos_token_id)
    gpt_output = tokenizer.decode(generated_ids[0], skip_special_tokens=True)

    bert_tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased", local_files_only=args.offline)
    bert = AutoModelForNextSentencePrediction.from_pretrained("bert-base-uncased", local_files_only=args.offline)
    anchor = PROMPT
    def candidate(text):
        pieces = [x.strip() for x in re.split(r"(?<=[.!?])\s+", text[len(PROMPT):]) if x.strip()]
        return pieces[0] if pieces else text[len(PROMPT):].strip() or text
    lstm_nsp = nsp_score(bert, bert_tokenizer, anchor, candidate(lstm_output))
    gpt_nsp = nsp_score(bert, bert_tokenizer, anchor, candidate(gpt_output))
    lstm_overlap, lstm_span = overlap(lstm_output, train_text)
    gpt_overlap, gpt_span = overlap(gpt_output, train_text)

    results = {
        "prompt": PROMPT,
        "checkpoint": checkpoint_label,
        "train_chars": len(train_text),
        "held_out_chars": len(held_out),
        "settings": {"seed": SEED, "temperature": TEMPERATURE, "top_k": TOP_K,
                      "gpt_new_tokens": GPT_NEW_TOKENS, "lstm_new_chars": LSTM_NEW_CHARS},
        "metrics": {
            "lstm_character_perplexity": lstm_ppl,
            "distilgpt2_subword_perplexity": gpt_ppl,
            "lstm_bert_nsp_isnext": lstm_nsp,
            "distilgpt2_bert_nsp_isnext": gpt_nsp,
            "lstm_longest_exact_overlap_chars": lstm_overlap,
            "distilgpt2_longest_exact_overlap_chars": gpt_overlap,
        },
        "outputs": {"lstm": lstm_output, "distilgpt2": gpt_output},
        "overlap_examples": {"lstm": lstm_span, "distilgpt2": gpt_span},
        "limitations": "Perplexity units differ (characters vs subwords); BERT NSP is a proxy, not human coherence or fact verification.",
    }
    out = ROOT / "results"
    out.mkdir(exist_ok=True)
    (out / "evaluation.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    with (out / "metrics.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["Metric", "LSTM", "DistilGPT-2", "Notes"])
        writer.writerow(["Perplexity", f"{lstm_ppl:.4f}", f"{gpt_ppl:.4f}", "Character vs subword; not directly comparable"])
        writer.writerow(["BERT NSP IsNext", f"{lstm_nsp:.4f}", f"{gpt_nsp:.4f}", "Coherence proxy"])
        writer.writerow(["Longest exact overlap (chars)", lstm_overlap, gpt_overlap, "Manual review required"])
    print(json.dumps(results["metrics"], indent=2))
    print(f"Saved outputs and metrics to {out}")


if __name__ == "__main__":
    main()

