# Experiment protocol

All search methods use the same immutable training-derived subset, 100 candidate evaluations, and 50 server timing diagnostics per seed. Main seeds are 11, 22, and 33. Duplicate policies are cached by model, policy, dataset, runtime, and protocol hashes.

Selection uses validation CIDEr only. Final evaluation uses greedy decoding with at most 30 new tokens. Report CIDEr, BLEU-1/4, METEOR, ROUGE-L, SPICE, caption length, empty rate, prefix distortion, logit KL, BitOps, model size, and diagnostic server p50/p95.
