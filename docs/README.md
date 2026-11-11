# SentryScore (POC) Documentation

This folder has two documents, written for two different readers:

- [`business-domain-knowhow/`](business-domain-knowhow/README.md) — for anyone who wants to understand what problem this solves and what the score/threshold actually mean. No code reading required.
- [`technical/`](technical/README.md) — for engineers who need to maintain this code, retrain the model, or carry this POC toward a real service.

If you just want to get it running, the root [`README.md`](../README.md) has install steps and a curl example you can paste straight into a terminal.

One thing worth saying up front: **this is a proof of concept, not a system that's handling real transactions.** The model was trained on a public Kaggle competition dataset, not on any company's actual transaction history, so any number you see here (PR-AUC 0.5926, for instance) only tells you how this approach performs on that public dataset — it isn't a guarantee of how it would perform in production. Both documents come back to this boundary from their own angle rather than mentioning it once and moving on.
