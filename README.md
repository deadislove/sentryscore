# Credit Card Fraud Detection — ML + FastAPI

![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue)
![License: MIT](https://img.shields.io/badge/license-MIT-green)
![Model: LightGBM](https://img.shields.io/badge/model-LightGBM-success)

An end-to-end machine learning proof of concept for detecting credit-card
fraud, trained on the IEEE-CIS Fraud Detection competition dataset and
served as a low-latency inference API with FastAPI.

```
POST /predict  →  { "fraud_probability": 0.9999, "is_blocked": true, "action": "BLOCK" }
```

## Why this is worth a look

Most "fraud detection POC" repos stop at a notebook that prints a metric.
This one goes one step further and treats the handoff from notebook to API
as seriously as the modeling itself:

- **PR-AUC 0.5926** on a strict time-based hold-out split (not a random
  one — see [Engineering notes](#engineering-notes-worth-reading) #1 for
  why that distinction matters and isn't just pedantry).
- **Mostly-named request schema.** Most fields are real, documented
  business fields (`product_cd`, card network/type, email domains,
  address/distance codes) — not a wall of anonymous floats. A documented
  minority (`velocity_counts`, `time_deltas`, `match_flags`,
  `vesta_features`) remain vendor-anonymized because the dataset's
  publisher never disclosed their individual meaning; this project is
  explicit about that rather than inventing fake labels for them.
- **The serving code actually uses the encoder it loads.** (This sounds
  obvious. It is easy to get wrong — see Engineering notes #2.)

## Quick start

```bash
git clone <this-repo> && cd <this-repo>
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# The trained model/encoder are already committed, so you can run the API
# immediately without retraining. To reproduce training from scratch:
#   1. Download the IEEE-CIS Fraud Detection competition data from Kaggle
#      (train_transaction.csv + train_identity.csv) and point train.ipynb's
#      DATA_DIR at it.
#   2. Run every cell in train.ipynb (writes fraud_model.joblib, encoder.joblib).

uvicorn main:app --reload --port 8000
```

```bash
curl -X POST http://127.0.0.1:8000/predict \
  -H 'Content-Type: application/json' \
  -d '{"transaction_amt": 49.0, "product_cd": "W", "card": {"card1": 5000, "card2": 400, "card3": 150, "card4": "visa", "card5": 226, "card6": "debit"}, "addr1": 315, "addr2": 87, "dist1": 10, "dist2": null, "p_email_domain": "gmail.com", "r_email_domain": null, "velocity_counts": [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], "time_deltas": [0, null, null, null, null, null, null, null, null, null, null, null, null, null, null], "match_flags": ["T", "T", "T", "T", "T", "T", "T", "T", "T"], "vesta_features": [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]}'
# {"fraud_probability":0.0014,"is_blocked":false,"action":"PASS"}
```

`vesta_features` must be exactly 339 values, `velocity_counts` exactly 14,
`time_deltas` exactly 15, `match_flags` exactly 9. `identity` may be
omitted entirely (most transactions have no identity data captured).

## Tech stack

- **Data/ML:** pandas, NumPy, scikit-learn, LightGBM
- **Serving:** FastAPI, Uvicorn, joblib
- **Dataset:** Kaggle — IEEE-CIS Fraud Detection (competition data; 590,540
  labeled transactions, 3.5% fraud rate, left-joined with a separate
  identity table covering ~24% of transactions)

## Project structure

```
.
├── train.ipynb         # Load+merge, time-based split, encoding, training, evaluation
├── main.py             # FastAPI inference service
├── fraud_model.joblib  # Trained LightGBM classifier
├── encoder.joblib      # Fitted OrdinalEncoder (categorical columns)
└── requirements.txt
```

(The raw dataset itself isn't here — see Quick start. It's ~1.3GB and
gated behind Kaggle competition acceptance, so it's not ours to
redistribute.)

## Engineering notes worth reading

A few decisions here are deliberate, not incidental — worth calling out
because they're exactly the kind of thing that's easy to get wrong in a
quick ML demo:

1. **Time-based split, not random.** `train.ipynb` sorts by
   `TransactionDT` and trains on the earliest 80%, validating on the
   latest 20%, instead of a random split. Fraud patterns drift over
   time; a random split lets near-future transactions (correlated by
   card/device activity) leak into training, overstating how well the
   model would actually perform going forward. This is *why* this
   dataset's reported PR-AUC (0.59) looks lower than you might expect for
   a tuned gradient-boosted model — it's a harder, more honest number,
   not a weaker model.
2. **The serving code must apply the same transform training did — and
   actually call it.** It's easy to load an encoder object and forget to
   call `.transform()` on incoming requests, silently feeding the model
   raw categorical strings it never learned a mapping for. The model
   still returns a number — it's just wrong, with no exception or
   warning. `main.py` builds a properly-ordered, properly-encoded feature
   frame before every prediction specifically to avoid that trap — and
   normalizes missing values to NaN *before* encoding, since a
   single-row DataFrame built from Python `None` silently becomes
   `object` dtype instead of a real float NaN (a real bug this project
   hit and fixed, not a hypothetical).
3. **Categorical encoding, not scaling, is the transform that matters
   here.** Unlike a from-scratch logistic regression, LightGBM is
   tree-based and scale-invariant — no `StandardScaler` needed. What it
   does need is a consistent numeric encoding for categorical columns
   (`OrdinalEncoder`, fit on the training split only), since raw strings
   can't feed into the model at all.
4. **PR-AUC, not accuracy, is the metric that matters here.** With 3.5%
   fraud, a model that predicts "not fraud" for everything still scores
   96.5% accuracy and catches zero fraud. PR-AUC (precision-recall
   tradeoff across all thresholds) is what `train.ipynb` actually
   optimizes for and reports.

## Design highlights

- **Time-based evaluation** instead of a random split (see note #1).
- **Consistent categorical encoding:** one `OrdinalEncoder`, fit on the
  training split only, applied — never re-fit — to validation and to
  every live request.
- **Mostly-named API contract:** real business fields where the dataset
  provides them; clearly-labeled anonymized arrays where it doesn't.

## Limitations (stated plainly)

- `velocity_counts`/`time_deltas`/`match_flags`/`vesta_features` carry no
  disclosed per-feature business meaning — the dataset's publisher never
  released what each one represents.
- A single competition snapshot from one processor; may not represent
  any particular business's real fraud patterns.
- The `0.2891` block threshold is the F1-optimal cutoff on this model's
  own validation split — a real derivation, not an arbitrary guess, but
  still not a cost-calibrated business decision. That requires real-world
  precision/recall analysis against an actual cost-of-error model.
- No auth, rate limiting, request logging, or horizontal-scaling story —
  this is a focused ML-serving demo, not a hardened production deployment.

## License

MIT — see [LICENSE](LICENSE).
