# Technical Guide

Audience: engineers modifying, retraining, or maintaining this code. For what the score means in business terms, see [`../business-domain-knowhow/README.md`](../business-domain-knowhow/README.md).

## What this POC actually is

The whole service is one file, `main.py`, with one endpoint. On startup it loads the trained model and encoder from disk into memory, and every request after that reuses the same in-memory objects for inference. No database, no external services, no config file:

```
train.ipynb  ──produces──▶  fraud_model.joblib (LightGBM classifier)
                             encoder.joblib (OrdinalEncoder)
                                   │
                                   ▼
main.py (FastAPI app, joblib.load's both files at startup)
   └── POST /predict: assemble features → apply encoder → model.predict_proba → compare to threshold → respond
```

Worth listing what's deliberately *not* here, so it doesn't read as an oversight: no API key auth, no request logging, no `/healthz`, no rate limiting, no automated tests. This proves out "can the data-science artifact be wrapped in something you can actually call over HTTP" — it isn't a hardened, production-ready service.

## The training pipeline (`train.ipynb`)

The data is Kaggle's "IEEE-CIS Fraud Detection" competition dataset — `train_transaction.csv` and `train_identity.csv`, left-joined on `TransactionID`, 590,540 labeled transactions, roughly 3.5% fraud rate. Only about 24% of transactions have a matching identity record; the rest have every identity-related field as NaN. That's a meaningful signal in itself (no device data was captured for this transaction), not a gap to impute around.

A few training decisions are worth calling out specifically, because they're easy to get subtly wrong and getting them wrong doesn't throw an error — it just makes the numbers look better than they should:

**Time-based split, not random.** `train.ipynb` sorts by `TransactionDT` and trains on the earliest 80%, validating on the latest 20%. Fraud patterns drift over time, and the same card or device often shows up in bursts of related activity in a short window — a random split would let some of that "future" information leak into training through correlated rows, which inflates the validation score. That's also why the PR-AUC here (0.5926) looks lower than it might otherwise: it's a stricter, more honest evaluation, not a weaker model.

**Categorical columns get `OrdinalEncoder`; numeric columns get no scaling at all.** LightGBM is tree-based and scale-invariant, so there's no need for a `StandardScaler` the way a linear model would need one. String columns, though, can't go into the model as-is, so they're encoded to numbers with a single `OrdinalEncoder` — fit once on the training split, then only `.transform()`-ed on the validation split and on every live request, never refit. This distinction matters, and comes up again below.

**The hyperparameters that got picked**: `n_estimators=800`, `num_leaves=127`, `scale_pos_weight=2`, `min_child_samples=50`, `feature_fraction=0.7`, `bagging_fraction=0.8`, `random_state=42`. These came out of a few rounds of trying different configs — the notebook's comments keep the two runs that didn't make the cut (`scale_pos_weight=1` landed around 0.5605; `=3` with more trees landed around 0.581), so whoever picks this up next knows what's already been tried.

**The metric that matters here is PR-AUC, not accuracy.** With a 3.5% fraud rate, a model that predicts "not fraud" for every single transaction still scores 96.5% accuracy while catching exactly zero fraud — useless. PR-AUC (area under the precision-recall curve) is the number that actually reflects whether the model separates fraud from non-fraud, and it's what `train.ipynb` reports: 0.5926.

**The 0.2891 threshold** is the F1-optimal cutoff read off the validation split's precision-recall curve, hardcoded into `main.py` as `BLOCK_THRESHOLD`, with no environment variable to override it. How it's derived and what it means as a business decision is covered in the business doc — this section is only about how it's used in code.

## The API contract

One endpoint:

```
POST /predict
```

The request shape mirrors the training data's columns. Most of it is named business fields (transaction amount, product code, card details, address/distance codes, email domain), but four groups stay anonymized because Kaggle's data provider (Vesta) never disclosed what each individual column in them means:

| Request field | Maps to training columns | Length requirement |
|---|---|---|
| `velocity_counts` | `C1`–`C14` | exactly 14 |
| `time_deltas` | `D1`–`D15` | exactly 15 |
| `match_flags` | `M1`–`M9` | exactly 9 |
| `vesta_features` | `V1`–`V339` | exactly 339 |
| `identity.identity_features` | `id_01`–`id_38` | exactly 38 |

These lengths are enforced with Pydantic's `Field(..., min_length=..., max_length=...)` — send the wrong length and you get a 422 immediately, rather than the request limping forward with misaligned data. `identity` can be omitted entirely: most transactions genuinely have no identity data, and sending a fully-null object isn't semantically the same thing as not sending one at all, so this contract lets the latter case say so explicitly.

Response:

```json
{"fraud_probability": 0.0014, "is_blocked": false, "action": "PASS"}
```

`fraud_probability` is rounded to four decimal places; `action` is `"PASS"` or `"BLOCK"`, determined by whether `fraud_probability` exceeds `BLOCK_THRESHOLD`. There's no `request_id` and no `model_version` in the response — both would matter for anything closer to production, but they're out of scope for this POC.

## The few lines in `predict_fraud` that actually matter

The least flashy-looking lines in `main.py` exist for a specific reason:

```python
row = {k: (np.nan if v is None else v) for k, v in row.items()}
```

Here's why this line exists. Building a single-row `pd.DataFrame` from a Python dict where some value is Python's `None` (not numpy's `NaN`) makes pandas infer `object` dtype for that column rather than a proper float NaN. That breaks LightGBM's dtype check outright — and for categorical columns it's worse, because it's silent: the encoder saw the string `"nan"` during training (training-data missing values are numpy NaN, and `str(nan) == "nan"`), but if a live request sends `None`, `str(None)` becomes `"None"` — a category the encoder never saw, which it silently maps to `unknown_value=-1`. The model still returns a number. It's just the wrong number, with no exception or warning to flag it. That's why every `None` gets normalized to `np.nan` before the DataFrame is built.

These two lines right after it exist to guarantee every column gets the exact same treatment at inference time that it got during training:

```python
frame[numeric_columns] = frame[numeric_columns].apply(pd.to_numeric, errors="coerce")
frame[CATEGORICAL_COLUMNS] = encoder.transform(frame[CATEGORICAL_COLUMNS].astype(str))
```

Note it only ever calls `encoder.transform()` — never `.fit()` or `.fit_transform()`. The encoder was fit exactly once, in `train.ipynb`, on the training split. Both the validation split and every live request only ever apply that one fit's mapping. Refit it by accident at serving time and the encoder would assign different numbers to the same categorical values than it did during training — the model would receive a sequence of numbers that no longer matches what it actually learned, and nothing would throw an error, because the code is still logically valid. It's just wrong.

`CATEGORICAL_COLUMNS` is a hardcoded list in `main.py`, computed during training via `select_dtypes(include=["object", "string"])` on the merged dataframe — not guessed from Kaggle's documentation. That matters, because what actually determines whether a column is categorical is the dtype pandas assigns after the merge, not what the dataset's docs claim. The list is kept in sync with `train.ipynb` by hand, with no shared config file backing it — acceptable at this POC's scale, but not something a real service should keep doing.

## Known limitations

- **The threshold is a hardcoded constant.** `BLOCK_THRESHOLD = 0.2891` lives in Python source; changing it means editing code and redeploying, with no config or env var to flip.
- **No auth, no rate limiting.** Anyone who can reach the endpoint can call it, as often as they want.
- **Nothing gets logged.** No prediction is written to any log or store, so there's no way to look back and answer "what did this service decide at a given point in time."
- **No automated tests.** The only way this gets verified today is manually running `curl` — there's no coverage for edge cases like wrong-length arrays or an all-null request.
- **The model/encoder column mapping relies on naming convention, with nothing checking it.** If `train.ipynb` changes column order or adds a column, `FEATURE_COLUMNS` and `CATEGORICAL_COLUMNS` in `main.py` need a matching manual edit — skip it and nothing fails loudly, predictions just quietly go wrong.

## Running it locally

```bash
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
uvicorn main:app --reload --port 8000
```

`fraud_model.joblib` and `encoder.joblib` are already checked into this folder, so the API runs without retraining anything first. To reproduce training from scratch, download the IEEE-CIS Fraud Detection competition data from Kaggle (`train_transaction.csv` + `train_identity.csv`) and point `train.ipynb`'s `DATA_DIR` at it, then run the notebook top to bottom. The dataset itself is gated behind Kaggle's competition rules and can't be redistributed with this repo, which is why it isn't committed.
