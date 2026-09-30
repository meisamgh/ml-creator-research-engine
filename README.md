# ML Creator Research Engine

ML Creator is a bounded, source-linked research engine for discovering **machine-learning feature ideas** and **model architectures** before a dataset is available.

It is designed for questions such as:

- What feature families should I investigate for this prediction problem?
- What model architectures or hybrid structures are worth testing?
- What data would I need to evaluate these ideas later?

The current CLI is intentionally a **NO_DATA research system**. It generates research proposals; it does **not** claim measured lift, a best model, production readiness, or empirical superiority.

## How it works

```text
Problem
  ↓
Search Planner
  ↓
Targeted web queries
  ↓
SearXNG
  ├── Kaggle
  ├── Papers
  └── Repositories
  ↓
Source-linked snippets
  ↓
Candidate Generator
  ↓
Validation / Admission Review
  ↓
Critic identifies gaps
  ↓
Next research round
  ↓
TARGET_REACHED | LOW_NOVELTY | MAX_ROUNDS
```

The loop is bounded. Each round uses previous discoveries and unresolved gaps to search for additional candidates instead of repeatedly asking for the same ideas.

## Research modes

ML Creator supports three modes:

| Mode | Purpose |
| --- | --- |
| `FEATURES` | Discover feature concepts, mechanisms, required information, and typical formulations |
| `STRUCTURES` | Discover complete model architectures, hybrids, stacks, and modeling strategies |
| `BOTH` | Run feature and structure research in the same job |

The core contracts also define `DATA_AVAILABLE`, but the current CLI rejects data-backed runs until a dataset adapter and mapping layer are implemented.

## Feature research

Feature discovery uses an iterative planner → search → generator → critic loop.

Each accepted `ResearchFeature` contains:

```text
concept_id
name
concept_family
mechanism
required_information
typical_formula
source_ids
similar_problems
```

Candidates must cite URLs returned by the current search round. Unsupported source URLs and duplicate concept IDs are rejected.

The critic then identifies missing feature families or predictive mechanisms and feeds those gaps into the next search round.

## Structure research

Structure discovery is stricter than feature discovery.

```text
Planner
  ↓
Search
  ↓
Structure Generator
  ↓
Basic deterministic checks
  ↓
Independent Admission Reviewer
  ├── Is this actually a model structure?
  ├── Is it semantically new?
  ├── Is source support DIRECT / ADAPTED / NONE?
  └── What is its canonical architecture?
  ↓
Accept / Reject
  ↓
Critic finds missing architecture families
```

This prevents the result set from filling with renamed versions of the same architecture. For example, `XGBoost + LSTM` and `LSTM/XGBoost hybrid` can be recognized as the same canonical structure.

A structure proposal includes:

```text
name
problem_type
architecture
components
rationale
assumptions
required_data_properties
training_protocol
validation_protocol
strengths
risks
sources
canonical_structure
reviewer_reason
```

## Evidence level

Current research evidence is explicitly labeled:

```text
SNIPPET_ONLY
```

The engine validates source domains and links candidates to returned search results, but it does not yet perform deep repository inspection, full-paper review, notebook execution, or empirical reproduction.

Treat outputs as **research hypotheses to test**, not verified results.

## Search sources

Research can be restricted to any combination of:

```text
KAGGLE
PAPERS
REPOS
```

Examples include Kaggle, arXiv/OpenReview/JMLR and other research hosts, plus GitHub/GitLab/Hugging Face and similar repository hosts.

Search is performed through SearXNG.

## Install

Requires Python 3.10+.

```bash
python3 -m pip install -e .
cp .env.example .env
```

Configure the generator and search provider in `.env`:

```env
JUSTWOKER_API_KEY=...
JUSTWOKER_BASE_URL=...
JUSTWOKER_MODEL=...
SEARXNG_URL=...
```

Structure admission can use a separate reviewer model:

```env
REVIEW_API_KEY=...
REVIEW_BASE_URL=...
REVIEW_MODEL=...
REVIEW_API_PROTOCOL=anthropic
```

If no separate reviewer is configured, the reviewer falls back to the configured JustWoker endpoint.

## Run

### Find model structures

```bash
ml-creator --config run.json \
  --mode STRUCTURES \
  --requested-model-structures 3 \
  --sources kaggle,papers,repos \
  --output structures.json
```

### Find features

```bash
ml-creator --config run.json \
  --mode FEATURES \
  --requested-features 5 \
  --output features.json
```

### Find both

```bash
ml-creator --config run.json \
  --mode BOTH \
  --requested-features 5 \
  --requested-model-structures 3 \
  --output blueprint.json
```

You can override the prediction problem at runtime:

```bash
ml-creator --config run.json \
  --problem "customer churn" \
  --entity customer \
  --horizon-days 30
```

For problems without a meaningful fixed prediction horizon:

```bash
ml-creator --config run.json --no-horizon
```

## Large searches

Structure research supports up to 100 requested structures and up to 20 rounds.

Each round generates at most five new candidates, reviews them, records accepted/rejected proposals, asks a critic what architecture families are still missing, and uses those gaps in the next planner step.

A run stops when:

```text
TARGET_REACHED
LOW_NOVELTY
MAX_ROUNDS
```

`LOW_NOVELTY` is triggered after repeated rounds fail to produce accepted new candidates.

## Checkpoints and outputs

When `--output` is provided, the CLI writes an intermediate checkpoint after each research round.

```text
<output>.checkpoint.json
```

The checkpoint is an audit snapshot, not a resumable execution state. The final JSON artifact is written atomically so a failed write does not corrupt the previous result.

Generated artifacts include:

- problem and run intent
- accepted feature or structure proposals
- research rounds
- stop reason
- selected source categories
- evidence level
- research artifact hash

## Deterministic evaluation foundation

The repository also contains the beginning of the future data-backed evaluation spine:

```text
PredictionEvent
bitemporal fact eligibility
FeatureSpec compilation
FitGuard
time-ordered OOF evaluation
portfolio freeze
sealed holdout
```

A fact is usable only when it was both valid and known by prediction time:

```text
valid_ts <= as_of_ts
knowledge_ts <= as_of_ts
```

`FitGuard` prevents fitting operations from touching validation or holdout rows.

These components are groundwork for future dataset-backed evaluation. They are **not yet wired into the current `ml-creator` CLI**.

## Current repository structure

```text
src/ml_creator/
├── __main__.py      # CLI, configuration, orchestration, checkpoints
├── core.py          # contracts and deterministic feature foundation
├── research.py      # search, planning, generation, critique, structure review
└── evaluation.py    # deterministic chronological evaluation helpers
```

`feature_scout.py` is a legacy standalone script and is not used by the `ml-creator` CLI.

`INSTRUCTOR.md` describes the larger target architecture and milestone plan. It should be read as a roadmap rather than the current module layout.

## Tests

```bash
PYTHONPATH=src python3 -m unittest discover -s tests -v
```

Tests cover mode routing, source validation, retries, optional horizons, iterative research, large bounded searches, structure admission, duplicate rejection, and deterministic core behavior.

## Design principle

> **Research discovers what may be worth testing. Data determines whether it is applicable. Experiments determine whether it actually works.**

No dataset is required for research. A dataset is required for empirical claims.
