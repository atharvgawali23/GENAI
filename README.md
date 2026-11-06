---
title: P16 LSTM vs Transformer Coherence Test
emoji: 🧪
colorFrom: blue
colorTo: indigo
sdk: streamlit
sdk_version: 1.40.2
app_file: app.py
pinned: false
---

# P16: LSTM vs Transformer Coherence Test

An individual Generative AI mini-project comparing a character-level Keras LSTM with fine-tuned DistilGPT-2 on one shared teaching corpus. The Streamlit app generates continuations, measures perplexity, scores next-sentence coherence with BERT NSP, reads an output with TTS, and screens for exact training-text overlap.

## Run locally

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
python scripts/train_lstm.py --epochs 12
python scripts/finetune_gpt2.py --epochs 1
streamlit run app.py
```

The first comparison downloads DistilGPT-2 and fine-tunes it for one epoch if `models/distilgpt2-finetuned` is missing. The app trains the LSTM on demand too. Keep trained weights out of the public repo. First use on a hosted CPU may take several minutes while weights download and training runs. Do not commit secrets.

## Evaluation method

- Corpus: `data/tiny_corpus.txt`; both models use the same 85/15 character-level train/held-out split.
- Shared decoding: seed 7, temperature 0.8, top-k 40, and 100 new transformer tokens. LSTM samples characters using the same temperature/top-k values for 240 characters.
- Perplexity: exponentiate each model's mean next-unit negative log-likelihood on the held-out passage. GPT-2 uses subword tokens; LSTM uses characters, so the absolute values are not directly comparable.
- Coherence: BERT NSP reports the IsNext probability for the prompt's final sentence and each generated output's final sentence. This is a proxy, not a human rating or fact-check.
- Memorization: scan generated output for exact spans of 20–80 characters against the training corpus; manually inspect any match.
- Document one failure case with its prompt, screenshot, and analysis.
- Run `python scripts/evaluate.py` after training to write `results/evaluation.json` and `results/metrics.csv`.

## Deployment

Create a **Streamlit** Hugging Face Space, upload this repository, and use `requirements.txt`. Add no secrets unless you add a private inference API. First load downloads model weights and may take several minutes; a free CPU instance may need a smaller model/cache strategy. Add the public Space and GitHub URLs to the report and demo.

## Project layers

1. **U1 Foundations:** generative vs discriminative roles, lifecycle, prompt comparison, sampling probability, and perplexity.
2. **U2 GANs & LLMs:** Keras LSTM and causal transformer; prompt-based vs fine-tuning discussion; attention explanation.
3. **U3 Models:** GPT-style generation with DistilGPT-2 and BERT-family NSP scoring.
4. **U4 Tools:** TensorFlow/Keras training, gTTS speech output, and a short Markov-chain melody generator.
5. **U5 Responsible AI:** overlap audit, limitations, privacy/IP notes, and model card.

## Submission checklist

- `RollNo_Name_P16.pdf` with measured results, failure case, screenshots, links, and ethics note.
- Public app URL and public GitHub URL.
- 2–3 minute screen-recorded demo.
- Upload all deliverables to `RollNo_Name` on the class shared drive.

See `MODEL_CARD.md` for intended use, data, evaluation, limitations, safety, and license information.

