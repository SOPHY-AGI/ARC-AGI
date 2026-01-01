ARC-AGI architecture notes

Components:
- FunctionGemma: small function-calling model used as a fast gate/adapter.
- HRM specialists: small deterministic reasoners for structured puzzle reasoning.
- Seamless (Toroidal) wrappers: optional mixing modules applied to selected FFN blocks to give local toroidal inductive bias.
- Verifier: lightweight reranker / correctness classifier.

Design principles:
- Minimal, auditable changes; alpha-gating for the mixer to preserve baseline behavior; progressive fine-tuning and active selection.
