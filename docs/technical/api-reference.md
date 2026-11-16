# API Reference: `POST /predict`

This is the field-by-field reference for the one endpoint this service exposes. For architecture and the reasoning behind the serving code, see [`README.md`](README.md). For a runnable version of this same contract, see the Postman artifacts in [`../../postman/`](../../postman/) — `postman/specs/sentryscore/index.yaml` is the OpenAPI source of truth, `postman/SentryScore-POC-API.postman_collection.json` is a single file you can import straight into Postman.

Every field below traces back to a specific column in Kaggle's "IEEE-CIS Fraud Detection" competition dataset (`train_transaction.csv` + `train_identity.csv`). Where that source column's meaning was never disclosed by the dataset's provider (Vesta), this doc says so plainly rather than guessing.

## Request body

| Field | Type | Required | Source column | Meaning |
|---|---|---|---|---|
| `transaction_amt` | number, &ge; 0 | yes | `TransactionAmt` | Transaction payment amount, in USD. |
| `product_cd` | string | yes | `ProductCD` | Product/category code for what was purchased. Observed values in the training data: `W`, `C`, `R`, `H`, `S` — what each code stands for wasn't individually disclosed by Vesta. |
| `card.card1` | number \| null | card object required | `card1` | Anonymized numeric card-issuer identifier. Exact encoding undisclosed by Vesta. |
| `card.card2` | number \| null | | `card2` | Anonymized numeric card-related identifier. Exact encoding undisclosed. |
| `card.card3` | number \| null | | `card3` | Anonymized numeric card-related identifier. Exact encoding undisclosed. |
| `card.card4` | string \| null | | `card4` | Card payment network — observed values: `"visa"`, `"mastercard"`, `"american express"`, `"discover"`. |
| `card.card5` | number \| null | | `card5` | Anonymized numeric card-related identifier. Exact encoding undisclosed. |
| `card.card6` | string \| null | | `card6` | Card funding type — observed values: `"debit"`, `"credit"`. |
| `addr1` | number \| null | no | `addr1` | Billing-address region code. Vesta never published an exact definition. |
| `addr2` | number \| null | no | `addr2` | Billing-address country code. Same disclosure caveat as `addr1`. |
| `dist1` | number \| null | no | `dist1` | A distance-type signal (e.g. between billing and shipping address, or billing address and IP location). Exact definition undisclosed. |
| `dist2` | number \| null | no | `dist2` | A second distance-type signal. Same caveat as `dist1`. |
| `p_email_domain` | string \| null | no | `P_emaildomain` | Purchaser's email domain, e.g. `"gmail.com"`. |
| `r_email_domain` | string \| null | no | `R_emaildomain` | Recipient's email domain, when known. |
| `velocity_counts` | array of (number \| null), exactly 14 | yes | `C1`–`C14` | Vesta-engineered count features. Per the competition's public documentation: "counting, such as how many addresses are found to be associated with the payment card." Individual column definitions were never disclosed. Values map positionally — index 0 is `C1`, index 13 is `C14`. |
| `time_deltas` | array of (number \| null), exactly 15 | yes | `D1`–`D15` | Vesta-engineered time-delta features. Per the competition's public documentation: "timedelta, such as days between previous transaction." Individual column definitions undisclosed. Maps positionally to `D1`...`D15`. |
| `match_flags` | array of (string \| null), exactly 9 | yes | `M1`–`M9` | Vesta-engineered match flags, typically `"T"`, `"F"`, or null in the raw dataset. Per the competition's public documentation: "match, such as names on card and address." Individual column definitions undisclosed. Maps positionally to `M1`...`M9`. |
| `vesta_features` | array of (number \| null), exactly 339 | yes | `V1`–`V339` | Vesta's proprietary engineered features. Per the competition's public documentation: "rich features, including ranking, counting, and other entity relations." This is the largest and least interpretable input block — individual column definitions were never disclosed. Maps positionally to `V1`...`V339`. |
| `identity` | object, or omitted entirely | no | `train_identity.csv`, left-joined on `TransactionID` | Device/network signals. Present for only ~24% of transactions in the training data. **Omit this field entirely** rather than sending it with null sub-fields when no identity data exists for a transaction — that's the common case, not an edge case. |
| `identity.device_type` | string \| null | | `DeviceType` | Observed values: `"mobile"`, `"desktop"`. |
| `identity.device_info` | string \| null | | `DeviceInfo` | Free-text device/browser string, e.g. `"Windows"`, `"iOS Device"`. |
| `identity.identity_features` | array of (number \| string \| null), exactly 38 | yes, if `identity` is sent | `id_01`–`id_38` | Network/device fingerprinting signals (IP/ISP/proxy information, browser and OS fingerprint, digital signature) collected by Vesta's fraud-protection partners. Individual column meanings were never disclosed, to protect the underlying detection method. Maps positionally to `id_01`...`id_38`. |

All five array fields are length-checked by Pydantic (`min_length`/`max_length` in `main.py`'s `TransactionInput`/`IdentityInfo`) — send the wrong count and the request fails with a 422 before it reaches the model.

## Response body (200)

```json
{"fraud_probability": 0.0014, "is_blocked": false, "action": "PASS"}
```

| Field | Type | Meaning |
|---|---|---|
| `fraud_probability` | number, 0–1 | `model.predict_proba(...)[0][1]`, rounded to 4 decimal places. A research-quality number from a public competition dataset — not a real-world-calibrated risk score. See [`../business-domain-knowhow/README.md`](../business-domain-knowhow/README.md) for what that distinction means. |
| `is_blocked` | boolean | `true` when `fraud_probability` is greater than `BLOCK_THRESHOLD` (`0.2891`, hardcoded in `main.py`). That threshold is the F1-optimal cutoff on this model's own validation split — a statistical result, not a business-calibrated decision. |
| `action` | `"PASS"` \| `"BLOCK"` | Human-readable restatement of `is_blocked`. |

## Validation error (422)

Standard FastAPI/Pydantic shape:

```json
{
  "detail": [
    {
      "loc": ["body", "vesta_features"],
      "msg": "List should have at least 339 items after validation, not 338",
      "type": "too_short"
    }
  ]
}
```

| Field | Meaning |
|---|---|
| `detail[].loc` | Path to the invalid field, e.g. `["body", "vesta_features"]`. |
| `detail[].msg` | Human-readable description of what failed validation. |
| `detail[].type` | Machine-readable error category, e.g. `"too_short"`, `"missing"`, `"greater_than_equal"`. |

There's no other error shape to document — a request that passes validation always reaches the model and always gets a 200. The model itself has no code path that raises.
