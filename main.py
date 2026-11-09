from fastapi import FastAPI
from pydantic import BaseModel, Field
import joblib
import numpy as np
import pandas as pd

app = FastAPI(title="Fraud Detection Inference Service")

# Load the model and categorical encoder upon startup
model = joblib.load("fraud_model.joblib")
encoder = joblib.load("encoder.joblib")

CARD_COLUMNS = ["card1", "card2", "card3", "card4", "card5", "card6"]
VELOCITY_COLUMNS = [f"C{i}" for i in range(1, 15)]
TIME_DELTA_COLUMNS = [f"D{i}" for i in range(1, 16)]
MATCH_FLAG_COLUMNS = [f"M{i}" for i in range(1, 10)]
VESTA_COLUMNS = [f"V{i}" for i in range(1, 340)]
IDENTITY_FEATURE_COLUMNS = [f"id_{i:02d}" for i in range(1, 39)]

FEATURE_COLUMNS = (
    ["TransactionAmt", "ProductCD"]
    + CARD_COLUMNS
    + ["addr1", "addr2", "dist1", "dist2", "P_emaildomain", "R_emaildomain"]
    + VELOCITY_COLUMNS
    + TIME_DELTA_COLUMNS
    + MATCH_FLAG_COLUMNS
    + VESTA_COLUMNS
    + IDENTITY_FEATURE_COLUMNS
    + ["DeviceType", "DeviceInfo"]
)
# Determined empirically by train.ipynb (string dtype after the
# transaction/identity merge) - hardcoded here since this POC has no
# model_registry.json to persist it in (see product/sentryscore for that).
CATEGORICAL_COLUMNS = [
    "ProductCD", "card4", "card6", "P_emaildomain", "R_emaildomain",
    "M1", "M2", "M3", "M4", "M5", "M6", "M7", "M8", "M9",
    "id_12", "id_15", "id_16", "id_23", "id_27", "id_28", "id_29", "id_30",
    "id_31", "id_33", "id_34", "id_35", "id_36", "id_37", "id_38",
    "DeviceType", "DeviceInfo",
]

BLOCK_THRESHOLD = 0.2891  # F1-optimal cutoff from train.ipynb's validation split


class CardInfo(BaseModel):
    card1: float | None = None
    card2: float | None = None
    card3: float | None = None
    card4: str | None = None
    card5: float | None = None
    card6: str | None = None


class IdentityInfo(BaseModel):
    """Present for only ~24% of real transactions - omit this whole
    object when no identity data was captured, rather than filling it
    with nulls."""

    device_type: str | None = None
    device_info: str | None = None
    identity_features: list[float | str | None] = Field(..., min_length=38, max_length=38)


class TransactionInput(BaseModel):
    transaction_amt: float = Field(..., ge=0)
    product_cd: str
    card: CardInfo
    addr1: float | None = None
    addr2: float | None = None
    dist1: float | None = None
    dist2: float | None = None
    p_email_domain: str | None = None
    r_email_domain: str | None = None
    velocity_counts: list[float | None] = Field(..., min_length=14, max_length=14)
    time_deltas: list[float | None] = Field(..., min_length=15, max_length=15)
    match_flags: list[str | None] = Field(..., min_length=9, max_length=9)
    vesta_features: list[float | None] = Field(..., min_length=339, max_length=339)
    identity: IdentityInfo | None = None


@app.post("/predict")
def predict_fraud(data: TransactionInput):
    row: dict[str, object] = {
        "TransactionAmt": data.transaction_amt,
        "ProductCD": data.product_cd,
        "addr1": data.addr1,
        "addr2": data.addr2,
        "dist1": data.dist1,
        "dist2": data.dist2,
        "P_emaildomain": data.p_email_domain,
        "R_emaildomain": data.r_email_domain,
    }
    for name in CARD_COLUMNS:
        row[name] = getattr(data.card, name)
    row.update(zip(VELOCITY_COLUMNS, data.velocity_counts))
    row.update(zip(TIME_DELTA_COLUMNS, data.time_deltas))
    row.update(zip(MATCH_FLAG_COLUMNS, data.match_flags))
    row.update(zip(VESTA_COLUMNS, data.vesta_features))

    identity = data.identity
    row["DeviceType"] = identity.device_type if identity else None
    row["DeviceInfo"] = identity.device_info if identity else None
    identity_features = identity.identity_features if identity else [None] * len(IDENTITY_FEATURE_COLUMNS)
    row.update(zip(IDENTITY_FEATURE_COLUMNS, identity_features))

    # A single-row DataFrame built from a dict infers `object` dtype for
    # any column whose only value is Python `None`, instead of float64
    # NaN - breaks LightGBM's dtype check and would make the encoder see
    # "None" instead of the "nan" string it was fit on. Normalize first.
    row = {k: (np.nan if v is None else v) for k, v in row.items()}
    frame = pd.DataFrame([row], columns=FEATURE_COLUMNS)

    numeric_columns = [c for c in FEATURE_COLUMNS if c not in CATEGORICAL_COLUMNS]
    frame[numeric_columns] = frame[numeric_columns].apply(pd.to_numeric, errors="coerce")
    frame[CATEGORICAL_COLUMNS] = encoder.transform(frame[CATEGORICAL_COLUMNS].astype(str))

    fraud_prob = model.predict_proba(frame)[0][1]
    is_blocked = bool(fraud_prob > BLOCK_THRESHOLD)

    return {
        "fraud_probability": round(float(fraud_prob), 4),
        "is_blocked": is_blocked,
        "action": "BLOCK" if is_blocked else "PASS"
    }
