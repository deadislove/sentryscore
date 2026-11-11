# Business & Domain Knowledge

Audience: anyone who wants to know what problem this solves and what the score and threshold mean — no code required. For how the system actually works, see [`../technical/README.md`](../technical/README.md).

## What question this service answers

Feed it one credit-card transaction, and it answers one question: how likely is this to be fraud, and should it be blocked? The output is a probability between 0 and 1, plus a binary decision — `PASS` or `BLOCK` — based on a threshold.

It only ever looks at a single transaction in isolation. It doesn't pull up that card's history, doesn't check whether this account is linked to other suspicious accounts, and doesn't do any after-the-fact investigation — those are different problems that need different tools and data. In a real deployment, a single-transaction score like this is typically one signal among several: it would sit alongside velocity rules (the same card swiped ten times in a minute, say), denylists, geolocation checks, and human review for the ambiguous cases. It isn't meant to be the sole decision-maker.

## Why this matters to the business

Every decision here — right or wrong — carries a cost on one side or the other:

| Outcome | What happened | Cost to the business |
|---|---|---|
| True Block | Fraud, correctly blocked | Loss avoided |
| False Block | A legitimate customer, blocked anyway | Worse customer experience, lost sale, support overhead from the dispute, reputational damage |
| True Pass | A legitimate customer, let through | Normal business |
| False Pass | Fraud, let through | Direct monetary loss, and possibly fees from the acquiring bank or card network |

The threshold is the dial that trades these off against each other: lower it and you catch more fraud but block more good customers; raise it and the opposite happens. Where exactly to set that dial isn't strictly an engineering question — it's a business decision. Engineering can produce the table showing precision and recall at different threshold values, but the call on "how many good customers are we willing to turn away to catch how much more fraud" belongs to whoever owns both the fraud-loss and the customer-experience side of the ledger, not to whoever wrote the code.

## How far this is from being ready to block real transactions

This needs to be said plainly, not buried: **this model was trained on Kaggle's public "IEEE-CIS Fraud Detection" competition dataset — not on any real company's transaction data.**

A few things follow from that:

- The PR-AUC of 0.5926 measures how well this approach separates fraud from legitimate transactions on that specific public dataset, under a deliberately strict evaluation (a time-based split, not a random one). It is not a guarantee of how it would perform on real traffic.
- Real-world fraud patterns differ by industry, region, and customer base, and may look nothing like what's represented in this one public dataset.
- Before this could actually be trusted to block real transactions, it needs to be validated against real traffic and real fraud outcomes (chargebacks, usually arriving weeks after the fact), and the threshold would need to be recalculated from that data — the current number shouldn't just carry over.

The right way to think about this project is as proof that the pipeline works end to end — data handling, model training, wrapping it all into something callable over an API — not as a validated, ready-to-use anti-fraud product.

## What the model actually looks at

Compared to a pile of unlabeled numbers, most of these fields have real, understandable business meaning:

- `transaction_amt`, `product_cd` — the transaction amount and a product category code.
- `card` — card network, card type (debit/credit), and issuer-related codes (`card1`–`card6`).
- `addr1`/`addr2`, `dist1`/`dist2` — billing address region codes and distance-related signals.
- `p_email_domain`/`r_email_domain` — purchaser and recipient email domains.
- `identity` (only present for about 24% of transactions) — device type and device info.

Four groups of fields stay anonymized on purpose: `velocity_counts` (maps to `C1`–`C14`), `time_deltas` (`D1`–`D15`), `match_flags` (`M1`–`M9`), and `vesta_features` (`V1`–`V339`). This isn't the project hiding anything — the dataset's provider (Vesta) never disclosed what each individual column in these groups represents. Rather than invent a plausible-sounding label, this service labels them honestly as anonymized groups. They stay in the model because, per the public writeups from this Kaggle competition, they carry real predictive signal — it's just that "which engineered feature mattered" can't be translated into "what it actually measures."

## The 0.2891 threshold: what it is, and what it isn't

The `0.2891` hardcoded into `main.py` is the cutoff that maximizes F1 score on this model's own validation data — a real, data-derived result, not an arbitrary guess. But it's still only "mathematically optimal on this validation split." It isn't the same thing as "calibrated to what this business is actually willing to pay in false blocks versus missed fraud." Before this gets used for real, whoever owns the fraud-loss and customer-experience numbers should look at precision/recall at a few candidate thresholds, weigh them against actual cost figures, and choose one deliberately — rather than defaulting to whatever the model's own optimum happens to be.

## What this service is not

Worth stating plainly, so expectations stay realistic:

- **Not anti-money laundering (AML).** AML looks at relationships and patterns across accounts over time — structuring, mule accounts, laundering networks. This service scores one transaction, in isolation, at one point in time.
- **Not a case-management or investigation tool.** It doesn't produce evidence, doesn't keep investigator notes, and doesn't plug into chargeback workflows.
- **Not a credit or underwriting decision.** Deciding "is this transaction fraud" and deciding "will this person default on a loan" are different problems with different data and different models.
- **Not a final decision on its own.** It's built to sit alongside other business rules and, for the ambiguous cases, human judgment — not to be the only vote that counts.

## Glossary

| Term | Meaning |
|---|---|
| Fraud probability | The model's estimated likelihood, between 0 and 1, that a transaction is fraudulent. |
| Threshold | The probability value above which a transaction gets blocked. |
| PR-AUC | Area under the precision-recall curve — a single number summarizing the tradeoff between precision and recall across every possible threshold. Closer to 1 is better; on a heavily imbalanced dataset like this one (~3.5% fraud), even a number that looks modest in absolute terms can represent a large improvement over guessing. |
| False block (false positive) | A legitimate transaction incorrectly blocked. |
| False pass (false negative) | A fraudulent transaction incorrectly let through. |
| Chargeback | A transaction a cardholder disputes and gets reversed — usually the real-world evidence that a transaction actually was fraud, typically surfacing weeks after the fact. |

## Where to look for common questions

| Question | Where to find it |
|---|---|
| Can these numbers be trusted enough to go live? | "How far this is from being ready to block real transactions" above |
| How does the system actually work, and how is it structured? | [`../technical/README.md`](../technical/README.md) |
| How do I run this locally, or retrain it? | Root [`README.md`](../../README.md) |
