"""P16: LSTM vs Transformer Coherence Test — Streamlit comparison app."""
from __future__ import annotations

import math
import re
import random
import struct
import subprocess
import sys
import wave
from io import BytesIO
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parent
CORPUS_PATH = ROOT / "data" / "tiny_corpus.txt"
LSTM_PATH = ROOT / "models" / "char_lstm.weights.h5"
GPT_PATH = ROOT / "models" / "distilgpt2-finetuned"
MAX_NEW_TOKENS = 100
TEMPERATURE = 0.8
TOP_K = 40
RANDOM_SEED = 7

st.set_page_config(page_title="P16 | LSTM vs Transformer", page_icon="🧪", layout="wide")


@st.cache_data
def read_corpus() -> str:
    return CORPUS_PATH.read_text(encoding="utf-8")


def sentences(text: str) -> list[str]:
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", text.strip()) if s.strip()]


@st.cache_resource(show_spinner="Loading the character LSTM…")
def load_lstm():
    if not LSTM_PATH.exists():
        return None, None
    import json
    import tensorflow as tf

    vocab = json.loads((LSTM_PATH.parent / "char_vocab.json").read_text(encoding="utf-8"))
    model = build_lstm_model(len(vocab["chars"]))
    model.load_weights(LSTM_PATH)
    return model, vocab


def build_lstm_model(vocab_size: int):
    import tensorflow as tf

    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(80,)),
        tf.keras.layers.Embedding(vocab_size, 64),
        tf.keras.layers.LSTM(128, return_sequences=True),
        tf.keras.layers.Dense(vocab_size),
    ])
    return model


def train_lstm(corpus: str, epochs: int = 12):
    import json
    import numpy as np
    import tensorflow as tf

    tf.keras.utils.set_random_seed(RANDOM_SEED)
    chars = sorted(set(corpus))
    char_to_id = {char: i for i, char in enumerate(chars)}
    ids = np.array([char_to_id[c] for c in corpus], dtype=np.int32)
    seq_len = 80
    starts = range(0, len(ids) - seq_len, 3)
    x = np.stack([ids[i:i + seq_len] for i in starts])
    y = np.stack([ids[i + 1:i + seq_len + 1] for i in range(0, len(ids) - seq_len, 3)])
    ds = tf.data.Dataset.from_tensor_slices((x, y)).shuffle(min(len(x), 2048), seed=7).batch(64).prefetch(tf.data.AUTOTUNE)
    model = build_lstm_model(len(chars))
    model.compile(optimizer="adam", loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True))
    model.fit(ds, epochs=epochs, verbose=0)
    LSTM_PATH.parent.mkdir(parents=True, exist_ok=True)
    (LSTM_PATH.parent / "char_vocab.json").write_text(json.dumps({"chars": chars, "to_id": char_to_id}), encoding="utf-8")
    model.save_weights(LSTM_PATH)
    return model, {"chars": chars, "to_id": char_to_id}


def sample_lstm(model, vocab: dict, prompt: str, length: int = 240, temperature: float = TEMPERATURE) -> str:
    import numpy as np
    import tensorflow as tf

    to_id = vocab["to_id"]
    from_id = {v: k for k, v in to_id.items()}
    context = [to_id.get(c, 0) for c in prompt[-80:]] or [0]
    output = list(prompt)
    for _ in range(length):
        x = np.array([([0] * max(0, 80 - len(context)) + context[-80:])], dtype=np.int32)
        logits = model(x, training=False)[0, -1].numpy() / max(temperature, 0.05)
        probs = tf.nn.softmax(logits).numpy()
        k = min(TOP_K, len(probs))
        top = np.argpartition(probs, -k)[-k:]
        top_probs = probs[top] / probs[top].sum()
        chosen = int(np.random.choice(top, p=top_probs))
        context.append(chosen)
        output.append(from_id[chosen])
    return "".join(output)


@st.cache_resource(show_spinner="Loading DistilGPT-2…")
def load_gpt2():
    from transformers import AutoModelForCausalLM, AutoTokenizer

    checkpoint = str(GPT_PATH) if GPT_PATH.exists() else "distilgpt2"
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    model = AutoModelForCausalLM.from_pretrained(checkpoint)
    model.eval()
    return model, tokenizer, checkpoint


def ensure_finetuned_gpt2():
    if GPT_PATH.exists():
        return
    subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "finetune_gpt2.py"), "--epochs", "1"],
        cwd=ROOT,
        check=True,
    )
    load_gpt2.clear()


def generate_gpt2(model, tokenizer, prompt: str) -> str:
    import torch

    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
    with torch.no_grad():
        output = model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS, do_sample=True,
                                temperature=TEMPERATURE, top_k=TOP_K,
                                pad_token_id=tokenizer.eos_token_id)
    return tokenizer.decode(output[0], skip_special_tokens=True)


def gpt2_perplexity(model, tokenizer, text: str) -> float:
    import torch

    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=1024)
    with torch.no_grad():
        loss = model(**inputs, labels=inputs["input_ids"]).loss
    return float(torch.exp(loss).item())


def lstm_perplexity(model, vocab: dict, text: str) -> float:
    import numpy as np
    import tensorflow as tf

    ids = np.array([vocab["to_id"].get(c, 0) for c in text], dtype=np.int32)
    width = 80
    if len(ids) <= width:
        return float("nan")
    nll = 0.0
    count = 0
    for start in range(0, len(ids) - width, width):
        x = ids[start:start + width]
        y = ids[start + 1:start + width + 1]
        if len(y) != width:
            continue
        logits = model(x[None, :], training=False)[0]
        losses = tf.keras.losses.sparse_categorical_crossentropy(y, logits, from_logits=True)
        nll += float(tf.reduce_sum(losses).numpy())
        count += width
    return math.exp(nll / max(count, 1))


@st.cache_resource(show_spinner="Loading BERT next-sentence model…")
def load_nsp():
    from transformers import AutoModelForNextSentencePrediction, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    model = AutoModelForNextSentencePrediction.from_pretrained("bert-base-uncased")
    model.eval()
    return model, tokenizer


def coherence_score(model, tokenizer, first: str, second: str) -> float:
    import torch

    inputs = tokenizer(first, second, return_tensors="pt", truncation=True, max_length=256)
    with torch.no_grad():
        logits = model(**inputs).logits[0]
    return float(torch.softmax(logits, dim=-1)[0].item())


def memorization_report(output: str, corpus: str) -> tuple[float, str]:
    normalized_out = re.sub(r"\s+", " ", output.lower()).strip()
    normalized_corpus = re.sub(r"\s+", " ", corpus.lower())
    longest = ""
    for width in (80, 60, 40, 30, 20):
        for i in range(max(0, len(normalized_out) - width + 1)):
            span = normalized_out[i:i + width]
            if len(span) > len(longest) and span in normalized_corpus:
                longest = span
                break
    return len(longest) / max(len(normalized_out), 1), longest


def generate_music_cue(text: str, seconds_per_note: float = 0.24) -> bytes:
    """Generate a short C-major motif by sampling a note-transition Markov chain."""
    import numpy as np

    motifs = [[0, 2, 4, 2, 5, 4, 2, 0], [0, 1, 3, 5, 4, 2, 1, 0], [2, 4, 5, 4, 3, 2, 0, 2], [2, 4, 6, 5, 4, 2, 0, 2]]
    transitions = {i: {} for i in range(7)}
    for motif in motifs:
        for left, right in zip(motif, motif[1:] + motif[:1]):
            transitions[left][right] = transitions[left].get(right, 0) + 1
    words = re.findall(r"[A-Za-z]+", text)
    start = (sum(len(word) for word in words[:8]) % 7) if words else 0
    rng = random.Random(RANDOM_SEED + sum(ord(char) for char in text[:200]))
    notes = [start]
    for _ in range(23):
        options = transitions[notes[-1]]
        choices = list(options)
        weights = [options[note] for note in choices]
        notes.append(rng.choices(choices, weights=weights, k=1)[0])

    sample_rate = 22050
    note_hz = [261.63, 293.66, 329.63, 392.00, 440.00, 523.25, 587.33]
    pcm = bytearray()
    note_samples = int(sample_rate * seconds_per_note)
    for index, note in enumerate(notes):
        frequency = note_hz[note]
        t = np.arange(note_samples, dtype=np.float32) / sample_rate
        envelope = np.minimum(1.0, np.minimum(t * 18, (seconds_per_note - t) * 10))
        tone = 0.22 * np.sin(2 * np.pi * frequency * t) * np.maximum(envelope, 0)
        if index % 4 == 0:
            tone += 0.07 * np.sin(2 * np.pi * frequency * 2 * t) * np.maximum(envelope, 0)
        pcm.extend((tone * 32767).astype("<i2").tobytes())
    buffer = BytesIO()
    with wave.open(buffer, "wb") as audio:
        audio.setnchannels(1)
        audio.setsampwidth(2)
        audio.setframerate(sample_rate)
        audio.writeframes(pcm)
    return buffer.getvalue()


corpus = read_corpus()
st.title("P16 · LSTM vs Transformer Coherence Test")
st.caption("Compare a character-level Keras LSTM and fine-tuned DistilGPT-2 on one shared mini corpus.")
st.info("Fixed decoding: seed 7, temperature 0.8, top-k 40, and 100 new transformer tokens. Character and subword perplexities are reported separately because their tokenization differs.")

tab_compare, tab_u1, tab_u2, tab_u3, tab_u4, tab_u5 = st.tabs([
    "Live comparison", "U1 · Foundations", "U2 · Models", "U3 · Coherence", "U4 · Keras + TTS", "U5 · Responsible AI"
])

with tab_compare:
    st.subheader("Generate both continuations from the same prompt")
    prompt = st.text_area("Shared prompt", value="At dawn, the little observatory opened its doors. ", height=90)
    train_epochs = st.slider("LSTM epochs if training is needed", 2, 30, 12)
    if st.button("Run comparison", type="primary", use_container_width=True):
        if not prompt.strip():
            st.error("Enter a prompt to compare the models.")
        else:
            try:
                import numpy as np
                import torch
                np.random.seed(RANDOM_SEED)
                torch.manual_seed(RANDOM_SEED)
                lstm, vocab = load_lstm()
                if lstm is None:
                    with st.spinner("Training the Keras character LSTM…"):
                        split_at = int(len(corpus) * 0.85)
                        lstm, vocab = train_lstm(corpus[:split_at], train_epochs)
                    load_lstm.clear()
                lstm_text = sample_lstm(lstm, vocab, prompt)
                with st.spinner("Loading transformer and generating…"):
                    if not GPT_PATH.exists():
                        st.info("Fine-tuning DistilGPT-2 on first use. The first run can take a few minutes.")
                        ensure_finetuned_gpt2()
                    gpt, tok, checkpoint = load_gpt2()
                    gpt_text = generate_gpt2(gpt, tok, prompt)
                st.session_state["comparison"] = {"prompt": prompt, "lstm": lstm_text, "gpt": gpt_text, "checkpoint": checkpoint}
                st.success(f"Comparison complete. Transformer checkpoint: {checkpoint}")
            except Exception as exc:
                st.error(f"A model could not be loaded: {exc}")
                st.caption("Install requirements and allow first-run model downloads. Fine-tune DistilGPT-2 using scripts/finetune_gpt2.py for the assigned comparison.")
    result = st.session_state.get("comparison")
    if result:
        left, right = st.columns(2)
        left.markdown("#### Character LSTM")
        left.write(result["lstm"])
        right.markdown("#### DistilGPT-2")
        right.write(result["gpt"])
        try:
            lm, vocab = load_lstm()
            gm, tok, _ = load_gpt2()
            p1, p2 = st.columns(2)
            held_out = corpus[int(len(corpus) * 0.85):]
            p1.metric("LSTM character perplexity", f"{lstm_perplexity(lm, vocab, held_out):.2f}")
            p2.metric("GPT-2 subword perplexity", f"{gpt2_perplexity(gm, tok, held_out):.2f}")
            st.caption("Lower perplexity means greater likelihood for the evaluation passage. Values across tokenizers are not a like-for-like ranking.")
        except Exception as exc:
            st.warning(f"Perplexity scoring unavailable: {exc}")

with tab_u1:
    st.subheader("U1 · Foundations")
    st.markdown("**Task:** use fixed decoding settings and compute perplexity. The app uses T=0.8, top-k=40, and scores a held-out corpus passage.")
    st.markdown("**Generative vs discriminative:** LSTM/GPT-2 predict the next character or token to generate text. BERT NSP estimates whether a sentence pair is consecutive.")
    st.markdown("**Lifecycle:** corpus → cleaning/split → training or fine-tuning → prompt + decoding → generated text → perplexity/coherence/overlap checks → human review.")
    st.markdown("**Prompt techniques:** compare (1) zero-shot continuation from an opening sentence with (2) an instruction plus few-shot examples. Keep decoding settings fixed.")
    st.markdown("**Probability view:** temperature rescales logits before sampling; top-k keeps only the k most likely choices.")
    st.code("p(token | context) = softmax(logits / temperature)\nPerplexity = exp(mean negative log-likelihood)")
    if st.button("Compare zero-shot and few-shot prompts"):
        try:
            import torch
            if not GPT_PATH.exists():
                with st.spinner("Fine-tuning DistilGPT-2 before the prompt comparison…"):
                    ensure_finetuned_gpt2()
            model, tokenizer, checkpoint = load_gpt2()
            zero_shot = prompt.strip()
            few_shot = (
                "Continue a curious science story in clear, coherent prose. Keep details consistent.\n"
                "Example opening: The old lighthouse woke before sunrise.\n"
                "Example continuation: Its lamp swept slowly over the quiet harbor.\n"
                f"New opening: {prompt.strip()}\nContinuation:"
            )
            torch.manual_seed(RANDOM_SEED)
            zero_output = generate_gpt2(model, tokenizer, zero_shot)
            torch.manual_seed(RANDOM_SEED)
            few_output = generate_gpt2(model, tokenizer, few_shot)
            st.session_state["prompt_comparison"] = {"zero": zero_output, "few": few_output, "checkpoint": checkpoint}
        except Exception as exc:
            st.error(f"Prompt comparison could not load the transformer: {exc}")
    prompt_comparison = st.session_state.get("prompt_comparison")
    if prompt_comparison:
        zero_col, few_col = st.columns(2)
        zero_col.markdown("#### Zero-shot continuation")
        zero_col.write(prompt_comparison["zero"])
        few_col.markdown("#### Instruction + few-shot examples")
        few_col.write(prompt_comparison["few"])
        st.caption(f"Same DistilGPT-2 checkpoint and decoding settings: {prompt_comparison['checkpoint']}.")

with tab_u2:
    st.subheader("U2 · LSTM and Transformer")
    st.markdown("The LSTM carries a recurrent hidden state and reads in sequence. DistilGPT-2 uses causal self-attention so earlier tokens can contribute to each next-token prediction.")
    st.markdown("**Prompt-based learning vs fine-tuning:** a prompt changes inference context; fine-tuning updates weights. The project includes a fine-tuning script for DistilGPT-2.")
    st.markdown("**GAN/VAE element:** this project does not train a GAN/VAE. It explains causal attention and compares it with recurrent state, the relevant generative architectures for this task.")
    st.code("python scripts/finetune_gpt2.py --epochs 1\npython scripts/train_lstm.py --epochs 12", language="bash")
    st.caption("The app fine-tunes DistilGPT-2 on first use when no saved checkpoint exists; subsequent comparisons reuse that checkpoint.")

with tab_u3:
    st.subheader("U3 · BERT next-sentence coherence")
    st.markdown("BERT NSP scores whether the generated sentence follows the prompt's last sentence. This is a learned coherence proxy, not a fact-check or a complete human judgment.")
    result = st.session_state.get("comparison")
    if result and st.button("Score both continuations with BERT NSP"):
        try:
            model, tokenizer = load_nsp()
            anchor_sentences = sentences(result["prompt"])
            anchor = anchor_sentences[-1] if anchor_sentences else result["prompt"]
            scores = {}
            for key in ("lstm", "gpt"):
                generated = sentences(result[key])
                candidate = generated[-1] if generated else result[key]
                scores[key] = coherence_score(model, tokenizer, anchor, candidate)
            st.session_state["nsp_scores"] = scores
        except Exception as exc:
            st.error(f"BERT NSP could not load: {exc}")
    scores = st.session_state.get("nsp_scores")
    if scores:
        x, y = st.columns(2)
        x.metric("LSTM IsNext probability", f"{scores['lstm']:.3f}")
        y.metric("Transformer IsNext probability", f"{scores['gpt']:.3f}")
    else:
        st.caption("Run a comparison first, then load BERT NSP to score the outputs.")

with tab_u4:
    st.subheader("U4 · Keras training and text-to-speech")
    st.markdown("The character LSTM trains with TensorFlow/Keras. Generate speech with gTTS or a short C-major melody cue with a note-transition Markov chain.")
    result = st.session_state.get("comparison")
    if result:
        choice = st.radio("Output to read aloud", ["Transformer", "LSTM"], horizontal=True)
        selected = result["gpt"] if choice == "Transformer" else result["lstm"]
        if st.button("Create speech audio"):
            try:
                from gtts import gTTS
                from io import BytesIO
                audio_buffer = BytesIO()
                gTTS(text=selected[:900], lang="en", slow=False).write_to_fp(audio_buffer)
                st.audio(audio_buffer.getvalue(), format="audio/mp3")
                st.caption("gTTS sends selected text to Google's text-to-speech service; avoid private text.")
            except Exception as exc:
                st.error(f"TTS unavailable: {exc}. Check network access or install gTTS.")
        if st.button("Generate a short music cue"):
            st.audio(generate_music_cue(selected), format="audio/wav")
            st.caption("The melody samples note transitions from three short C-major motifs. It is a simple generative audio feature, not a trained music model.")
    else:
        st.caption("Generate both outputs first to enable TTS.")

with tab_u5:
    st.subheader("U5 · Memorization, safety, and limits")
    result = st.session_state.get("comparison")
    if result:
        for label, key in (("LSTM", "lstm"), ("DistilGPT-2", "gpt")):
            ratio, span = memorization_report(result[key], corpus)
            st.write(f"**{label}:** longest exact overlap: {len(span)} characters; {ratio:.1%} of generated text (overlap length / output length).")
            if span:
                st.code(span)
        st.caption("Exact-span scan is a screening measure: it misses paraphrases and can flag common phrases.")
    else:
        st.caption("Generate outputs to run the exact-overlap screen.")
    st.markdown("**Guardrail:** avoid sensitive prompts, supervise outputs, and do not treat generated text as fact. This teaching app does not automatically block harmful content.")
    st.markdown("**Privacy/IP:** the bundled corpus is original teaching text. Upstream model licenses apply. gTTS sends selected text to a third party.")
    st.markdown("**Failure case to document:** small training data or out-of-domain prompts may produce repetitive, ungrammatical text; NSP may reward topical adjacency without checking factual correctness.")

