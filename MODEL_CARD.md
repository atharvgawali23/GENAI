# Model Card: P16 LSTM vs Transformer Coherence Test

## Project

Educational comparison app for P16. It contrasts a character-level Keras LSTM trained from scratch with DistilGPT-2 fine-tuned on the same short corpus. A BERT next-sentence-prediction head supplies a coherence proxy.

## Intended use

Classroom demonstration of language-model training, decoding, perplexity, coherence proxies, and memorization checks. Not for production writing, factual answers, or decisions about people.

## Data

The included `data/tiny_corpus.txt` is original teaching prose written for this project. It is intentionally small and does not represent general language. DistilGPT-2 starts from upstream weights and is fine-tuned on the training split of this corpus.

## Evaluation

Report held-out perplexity for both models, BERT NSP probability on identical prompt/output pairs, exact-span overlap lengths, decoding settings, and one failure example. Perplexities use different tokenization units (characters versus subwords), so their magnitudes are not directly comparable. NSP is not a truth or complete coherence metric.

Measured run (seed 7, 85/15 character split, one GPT-2 fine-tuning epoch): LSTM character perplexity 9.816; DistilGPT-2 subword perplexity 44.321; BERT NSP IsNext scores 0.0004 (LSTM) and 0.9995 (DistilGPT-2); longest exact overlap 0 characters for both. See `results/metrics.csv` and `results/evaluation.json`. Re-run after any training or code changes before final submission.

## Limitations and risks

- Small corpus encourages overfitting and repetitive completions.
- Fluency and coherence can degrade on prompts outside the corpus domain.
- NSP may score topical word overlap rather than logical continuity.
- Exact-span overlap misses paraphrased memorization and can flag common phrases.
- gTTS transmits selected output text to an external service.

## Licenses and attribution

The included corpus is original project text. DistilGPT-2 and BERT weights retain their upstream model licenses and terms; verify the current model cards before redistribution. TensorFlow, Transformers, Streamlit, and gTTS retain their own licenses. This project does not redistribute model weights.

## Safety

Outputs are unverified. Avoid personal, private, or sensitive prompts. Human review is required. The demo screens for exact training text but does not automatically block all unsafe or memorized content.

## Maintainer

Student: [Your name] · Roll No. 16 · Course: Generative AI

