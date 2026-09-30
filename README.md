# Feature Research Engine

## Current implementation

The repository includes the deterministic foundation described in
`INSTRUCTOR.md` and source-linked no-data research. Install locally with
`python3 -m pip install -e .`. Edit `run.json` for the problem and horizon, then
run one of these commands:

```bash
python3 -m ml_creator --config run.json --mode STRUCTURES --requested-model-structures 3 --output structures.json
python3 -m ml_creator --config run.json --mode FEATURES --requested-features 5 --output features.json
python3 -m ml_creator --config run.json --mode BOTH --requested-features 5 --output blueprint.json
```

Choose source categories with `--sources kaggle,papers,repos`, or select only
some of them, such as `--sources papers,repos`. The same list can be set under
`research.sources` in `run.json`. Search queries target the selected sites and
the returned URL domains are checked before they reach a seeker.

Structure admission uses a separate reviewer call through the configured
JustWoker API by default. The reviewer judges whether each proposal describes
a model architecture and whether its component flow is new relative to accepted
structures. It also rejects proposals with no support in the returned snippets.
Only approved proposals enter the list, and each decision is logged.
This account currently exposes only `claude-opus-4-8`, so the review is an
independent call and prompt using the same model ID. To use a different model
later, set `REVIEW_BASE_URL`, `REVIEW_MODEL`, `REVIEW_API_PROTOCOL` (`anthropic`
or `openai_chat`), and `REVIEW_API_KEY` in `.env`.

The run configuration controls the problem and optional horizon. Set
`--problem` to research another task and `--horizon-days` only when the
prediction window matters. Changing the problem on the command line clears
the configured horizon unless you supply a new one. These options work
with `FEATURES`, `STRUCTURES`, and `BOTH`:

```bash
python3 -m ml_creator --config run.json --mode FEATURES --requested-features 5 --problem customer_churn --output churn-features.json
python3 -m ml_creator --config run.json --mode STRUCTURES --problem customer_churn --horizon-days 90 --output churn-structures.json
```

For horizon-free research, omit `horizon_days`, set it to `null`, or use
`"horizon_days": "no-horizon"` in the config.
For a single run against a config that already has a horizon, use `--no-horizon`:

```bash
python3 -m ml_creator --config run.json --mode STRUCTURES --requested-model-structures 3 --no-horizon --output structures-no-horizon.json
```

Data-backed evaluation requires a positive horizon and remains a later CLI
milestone. Set
`JUSTWOKER_API_KEY`, `JUSTWOKER_BASE_URL`, `JUSTWOKER_MODEL`, and
`SEARXNG_URL` in `.env` as shown in `.env.example`. SearXNG search results
must contain snippets; the output labels source evidence `SNIPPET_ONLY` and
includes only URLs returned by search. These are research proposals, not
empirically selected features or models. The configured LLM endpoint is called
through `curl`, which must be available on the host. Run
`PYTHONPATH=src python3 -m unittest discover -s tests -v` for the tests.

Both no-data seekers use an iterative agent loop: plan search queries, retrieve
source snippets, extract proposals, reject missing-source and duplicate items,
critique coverage, and use the remaining gaps in the next round. `run.json`
currently requests 100 structures and allows up to 20 structure rounds. Each
round uses at most six searches (two per selected source category) and eight
model calls (planning, extraction, critique, and up to five admission reviews).
`max_feature_rounds`
controls the feature agent separately. The output records queries, source URLs,
accepted proposals, rejections, critic feedback, counts, and the stop reason.
`MAX_ROUNDS` or `LOW_NOVELTY` means the requested count was not met; it does not
mean the search converged. Long runs write an adjacent `.checkpoint.json` file
after every completed round and remove it after successful completion. This is
an audit snapshot; automatic resume is not implemented.
Structure research concentrates on model components and connections, including
hybrids such as XGBoost plus LSTM. Training and validation notes are stored
briefly for context; they do not determine novelty.

The package includes bitemporal fact filtering, a single-table FeatureSpec
compiler, deterministic variants, a fit provenance guard, a time-ordered OOF
test baseline, portfolio freeze, and a sealed holdout function. The router
supports all six intent/data-state combinations as a library and does not map
or evaluate in `NO_DATA` mode. The CLI rejects `DATA_AVAILABLE` until a real
dataset adapter exists. Structure applicability and experiments, full-page
source verification, statistical feature selection, and a production dataset
adapter remain unimplemented. Research alone establishes no measured value.

`feature_scout.py` is the older standalone feature-research script. It is not
used by the `ml-creator` CLI; install `requirements.txt` separately if you run it.

A leakage-safe system for **researching, mapping, validating, generating, and evaluating machine-learning features**.

> **LLMs propose. Deterministic systems verify. Experiments determine value.**

The key design rule is simple:

> The **Feature Seeker never sees the database**.

It researches potentially useful feature concepts from the prediction problem. A separate schema-aware mapper proposes how those concepts could be built from available data, and deterministic validators decide whether the mapping is safe and computable.

---

## MVP Scope

The first version intentionally stays small:

```text
Problem: credit_default
Entity: customer
Data: one transaction table

≤ 10 research concepts
≤ 30 generated variants
```

No feature or empirical claim is considered proven until the validation experiments execute successfully.

---

# Architecture

```text
                    Problem Contract
                           ↓
                    SearchContract
                           ↓
                 ┌─────────────────┐
                 │ FEATURE SEEKER  │
                 │                 │
                 │   NO DB ACCESS  │
                 └────────┬────────┘
                          ↓
                  ResearchFeature[]
                          ↓
                 Source Verification
                          ↓
                 Concept Diversity
                          ↓

              ═══════ DATA BOUNDARY ═══════

                          ↓
                       Mapper
                  schema-aware LLM
                          ↓
                   MappedFeature[]
                          ↓
              Deterministic Verifier
                ├── schema
                ├── joins
                ├── temporal rules
                ├── leakage
                └── computability
                          ↓
                     FeatureSpec
                          ↓
              Deterministic Generator
                          ↓
                    ≤30 variants
                          ↓
                    Coarse Eval
                          ↓
                     Fine Eval
                          ↓
                      Ablation
                          ↓
                     SELECTED
                          ↓
                 PORTFOLIO FREEZE
                          ↓
                  SEALED HOLDOUT
                          ↓
                HOLDOUT_EVALUATED
                          ↓
                   Research Memory
```

---

# 1. Search Contract

The search contract defines the problem and research limits.

```yaml
problem:
  task: credit_default
  horizon_days: 90
  entity: customer
  txn_table: transactions

research:
  max_concepts: 10
  max_variants: 30

reproducibility:
  research_artifact_hash: ...
  seeker_model_id: ...
  seeker_model_version: ...
  decoding_params:
    temperature: 0
```

The seeker output is frozen as a research artifact before downstream processing.

This makes downstream runs reproducible even if the LLM itself changes later.

---

# 2. Feature Seeker

The Feature Seeker researches:

- the target problem
- similar prediction problems
- feature mechanisms
- papers
- GitHub repositories
- Kaggle solutions

It may propose feature concepts such as:

```yaml
concept_id: repayment_irregularity

name: repayment_interval_volatility

concept_family: payment_behavior

mechanism:
  Increasing irregularity in repayment timing
  may indicate financial deterioration.

required_information:
  - customer identifier
  - repayment event
  - event timestamp
```

The seeker must use **generic information requirements**.

Allowed:

```text
repayment timestamp
transaction amount
customer identifier
```

Not allowed:

```text
payments.payment_date
transactions.amount
customer.customer_id
```

The seeker does not know the database schema.

---

# 3. ResearchFeature

Each discovered concept becomes a structured `ResearchFeature`.

```python
ResearchFeature(
    concept_id,
    name,
    concept_family,
    mechanism,
    required_information,
    source_ids,
    novelty_hash
)
```

Each feature must include at least one research source.

Supported source identifiers may include:

```text
arXiv ID
DOI
GitHub repository + commit
Kaggle reference
```

Unresolvable sources receive:

```text
UNVERIFIED_SOURCE
```

and cannot proceed.

---

# 4. Concept Diversity

Research should discover genuinely different concepts rather than trivial rewrites.

Example:

```text
payment irregularity
repayment timing instability
variance between repayment dates
```

should normally map to the same conceptual family.

The MVP limits research to:

```text
≤10 concepts
```

and applies deterministic concept deduplication before database mapping.

---

# 5. Feature Mapper

The Mapper receives:

```text
ResearchFeature
+
database schema
```

It proposes:

```text
mapping status
column bindings
join path
```

Supported states:

```text
APPLICABLE
DERIVABLE
PARTIALLY_APPLICABLE
NOT_AVAILABLE
```

Example:

```yaml
concept:
  transaction_frequency_change

mapping_status:
  DERIVABLE

column_bindings:
  entity: transactions.customer_id
  timestamp: transactions.transaction_ts

join_path:
  - transactions
```

The Mapper proposes only.

It cannot approve a feature.

---

# 6. Prediction Event

Every training example is tied to an explicit prediction time.

```python
PredictionEvent(
    entity_id,
    as_of_ts,
    label,
    label_ts,
    horizon
)
```

An event is eligible only when:

```text
as_of_ts + horizon <= data_max_ts
```

For the MVP, exactly one prediction event is used per entity.

This simplifies protection against customer identity leakage across folds.

---

# 7. Deterministic Verification

Only the deterministic verifier may emit a `FeatureSpec`.

Every proposed mapping must pass:

```text
Schema validation
        ↓
Type validation
        ↓
Join validation
        ↓
Temporal validation
        ↓
Leakage checks
        ↓
Computability checks
```

Any single veto blocks the feature.

---

## Temporal Rule

Every fact used by a feature must satisfy:

```text
valid_ts <= as_of_ts
```

before aggregation.

Conceptually:

```text
Feature(customer, T)

may only use

information available at or before T
```

---

## Stateful Transform Rule

All learned transforms must fit inside the training fold only.

This includes:

```text
target encoding
mean encoding
scaling
imputation statistics
mutual-information thresholds
variance thresholds
```

Validation data must never influence fitted transformation state.

---

# 8. FeatureSpec

A successful mapping becomes a frozen executable specification.

```yaml
concept_id: repayment_irregularity

join_path:
  table: transactions
  valid_ts: transaction_ts

aggregation:
  type: std
  expression: days_between_repayments

lookback_window:
  90d

variant_grammar_version:
  v1
```

The `FeatureSpec` defines **how the feature can be safely calculated**, not whether it is useful.

---

# 9. Deterministic Expansion

Validated concepts may be expanded into multiple variants.

Example concept:

```text
Recent vs historical transaction activity
```

Possible variants:

```text
transactions_7d / transactions_30d
transactions_14d / transactions_90d
transactions_7d - transactions_30d
transaction_slope_30d
transaction_acceleration_30d
```

Expansion is deterministic and versioned:

```text
variant_grammar_version
```

The MVP generates at most:

```text
30 variants
```

---

# 10. Evaluation

The evaluation pipeline is:

```text
Generated Variants
        ↓
Coarse Evaluation
        ↓
Fine Evaluation
        ↓
Ablation
        ↓
Feature Selection
        ↓
PORTFOLIO FREEZE
        ↓
SEALED HOLDOUT
```

The holdout is not used for feature selection.

It is used only after the portfolio has been frozen.

---

## Cross-Validation

The MVP uses:

```text
entity-grouped
+
time-blocked
cross-validation
```

to reduce both temporal leakage and identity leakage.

Out-of-fold predictions are pooled at the entity level.

---

## Statistical Evaluation

Candidate lift may be evaluated using:

```text
paired bootstrap confidence intervals
BH-FDR correction
minimum practical effect threshold
```

Statistical significance alone is not enough.

A feature should also provide meaningful and stable improvement.

---

# 11. Zero-Candidate Guard

If no feature survives verification:

```text
FeatureSpec count = 0
```

the run stops immediately with:

```text
NO_CANDIDATES
```

No model lift or baseline comparison is produced.

---

# 12. Run Reproducibility

Each deterministic run receives:

```text
run_id = hash(
    snapshot_hash,
    variant_grammar_version,
    seed,
    cv_config
)
```

The research artifact is also frozen:

```text
research_artifact_hash
```

This means the same research artifact and deterministic inputs can be replayed without calling the LLM again.

---

# 13. Statuses

Pipeline states include:

```text
SEEKING

MAPPED

VERIFIED

EXPANDED

COARSE

FINE

SELECTED

PORTFOLIO_FROZEN

HOLDOUT_EVALUATED

NO_CANDIDATES

VETOED

BUDGET_EXHAUSTED
```

`MAX_ROUNDS` or budget exhaustion does not imply convergence.

---

# 14. Budget Control

Every run has explicit limits.

```yaml
budget:
  max_variants: 30
  max_fine_evals: ...
  max_compute_seconds: ...
  max_source_verifications: ...
```

If a budget is exhausted:

```text
checkpoint
↓
halt
```

A partially completed stage must not silently promote candidates.

---

# 15. Research Memory

The system stores both successful and failed research outcomes.

```text
concept_id

source_ids

concept_family

mapping_status

leakage_verdict

CV lift

rejection reason

novelty information

compute cost
```

Example:

```text
✓ repayment_interval_volatility
  VERIFIED
  +0.006 validation AUC

✗ future_balance
  VETOED
  future information

✗ transaction_ratio_v2
  duplicate concept

✗ support_sentiment
  NOT_AVAILABLE
  required data absent
```

This prevents repeated research into already explored failures.

---

# 16. Auditability

Each run records:

```text
run_id

snapshot_hash

research_artifact_hash

variant_grammar_version

seed

CV configuration

terminal status
```

Each feature variant records:

```text
mapping status

leakage verdict

veto reasons

coarse metrics

fine metrics

OOF confidence interval

multiple-testing result

selection decision
```

---

# 17. MVP Validation Tests

The architecture is considered **PROPOSED** until the following tests pass.

### Source Verification

Inject fabricated references.

Expected:

```text
fabricated sources → UNVERIFIED_SOURCE
```

---

### Concept Diversity

Inject paraphrased duplicate concepts.

Expected:

```text
same mechanism → one concept
```

---

### Entity Leakage

Compare:

```text
time-only CV
```

against:

```text
time + entity-disjoint CV
```

using a seeded entity-memory feature.

The entity-safe configuration should remove artificial lift.

---

### Future Leakage

Inject:

```text
valid_ts = as_of_ts + horizon
```

Expected:

```text
VETOED before evaluation
```

---

### Null Feature

Run pure-noise features through the complete evaluation funnel.

Expected false-selection rate:

```text
≤ configured statistical threshold
```

---

### Zero Candidates

Make all mapped concepts unavailable.

Expected:

```text
NO_CANDIDATES
```

with no model-lift artifact.

---

### Replay

Replay identical deterministic inputs.

Expected:

```text
same selected feature set
```

without re-invoking the LLM.

---

# 18. Suggested Repository Structure

```text
feature-research-engine/
│
├── README.md
├── pyproject.toml
│
├── src/
│   │
│   ├── contracts/
│   │   ├── problem.py
│   │   ├── search_contract.py
│   │   ├── research_feature.py
│   │   ├── mapped_feature.py
│   │   ├── prediction_event.py
│   │   ├── feature_spec.py
│   │   ├── repro_spec.py
│   │   └── budget_contract.py
│   │
│   ├── seeker/
│   │   ├── seeker.py
│   │   ├── planner.py
│   │   ├── extractor.py
│   │   └── sources/
│   │
│   ├── mapping/
│   │   ├── mapper.py
│   │   └── join_paths.py
│   │
│   ├── validation/
│   │   ├── source.py
│   │   ├── diversity.py
│   │   ├── schema.py
│   │   ├── joins.py
│   │   ├── temporal.py
│   │   ├── leakage.py
│   │   └── computability.py
│   │
│   ├── generation/
│   │   └── grammar.py
│   │
│   ├── evaluation/
│   │   ├── coarse.py
│   │   ├── fine.py
│   │   ├── bootstrap.py
│   │   ├── ablation.py
│   │   └── holdout.py
│   │
│   ├── knowledge/
│   │   └── research_memory.py
│   │
│   └── orchestration/
│       ├── pipeline.py
│       ├── checkpoints.py
│       └── budget.py
│
├── tests/
│   ├── fixtures/
│   ├── test_temporal_leakage.py
│   ├── test_entity_leakage.py
│   ├── test_source_verification.py
│   ├── test_zero_candidates.py
│   └── test_replay.py
│
└── experiments/
```

---

# 19. Implementation Order

Build the deterministic system first.

```text
1. Contracts
        ↓
2. PredictionEvent + temporal rules
        ↓
3. Leakage + join verification
        ↓
4. Deterministic FeatureSpec compiler
        ↓
5. Deterministic feature expansion
        ↓
6. Evaluation pipeline
        ↓
7. Feature Seeker
        ↓
8. Source + diversity validation
        ↓
9. Research Memory
```

Do not start with a large multi-agent architecture.

Use LLMs where semantic reasoning is useful.

Use deterministic software wherever rules can be enforced.

---

# Non-Goals for MVP

The MVP does not attempt:

```text
multiple prediction problems

multiple entity types

multiple transaction tables

more than 10 research concepts

more than 30 generated variants

multiple prediction events per entity

LLM approval of feature correctness

LLM approval of empirical value

large-scale OpenFE / Featuretools search

cross-problem transfer learning

claims of validated performance
before the experiments run
```

---

# Core Principle

```text
Problem
   ↓
LLM researches WHAT may matter
   ↓
Freeze research artifact
   ↓
LLM proposes HOW it maps to the schema
   ↓
Deterministic software decides
whether it can safely be built
   ↓
Generate controlled variants
   ↓
Experiments determine usefulness
   ↓
Freeze selected portfolio
   ↓
Evaluate once on sealed holdout
   ↓
Remember results for future research
```

The project is not an LLM that generates feature code.

It is a **Feature Research Engine** built around the separation:

> **Discovery → Mapping → Verification → Generation → Experimentation → Learning**
