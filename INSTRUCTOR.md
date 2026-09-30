# Feature & Structure Research Engine — Instructor

This file defines how to build and run the system step by step.

The system must support three research modes:

```text
FEATURES
STRUCTURES
BOTH
```

It must also support two data states:

```text
DATA_AVAILABLE
NO_DATA
```

The key rule is:

> **Research can happen without a dataset. Mapping and empirical validation require data.**

---

# 1. Decide the Mode First

Every run starts with a `RunIntent`.

```yaml
intent:
  mode: FEATURES | STRUCTURES | BOTH
  data_state: DATA_AVAILABLE | NO_DATA
```

Examples:

```yaml
intent:
  mode: FEATURES
  data_state: DATA_AVAILABLE
```

```yaml
intent:
  mode: STRUCTURES
  data_state: NO_DATA
```

```yaml
intent:
  mode: BOTH
  data_state: DATA_AVAILABLE
```

---

# 2. Supported Workflows

## A. FEATURES + DATA_AVAILABLE

Use when the user wants feature ideas and a dataset/schema exists.

```text
Problem
  ↓
Feature Seeker
  ↓
ResearchFeature[]
  ↓
Schema Mapper
  ↓
Deterministic Verification
  ↓
FeatureSpec
  ↓
Feature Expansion
  ↓
OOF Evaluation
  ↓
Feature Portfolio
  ↓
Sealed Holdout
```

---

## B. FEATURES + NO_DATA

Use when the user wants feature ideas but no dataset exists yet.

```text
Problem
  ↓
Feature Seeker
  ↓
ResearchFeature[]
  ↓
Feature Families
  ↓
Required Information
  ↓
Data Requirements Report
```

Stop here.

Do not fabricate:

```text
column mappings
SQL
FeatureSpec
model lift
```

Output should describe:

- feature concept
- mechanism
- required information
- typical formulation
- evidence/source
- similar problems
- expected data needed

This mode can also identify what data should be collected later.

---

## C. STRUCTURES + DATA_AVAILABLE

Use when the user wants model structures and a dataset/schema exists.

```text
Problem
  ↓
Structure Seeker
  ↓
ModelStructure[]
  ↓
Model Applicability Mapper
  ↓
ModelSpec[]
  ↓
Controlled Experiments
  ↓
Selected Structure
  ↓
Sealed Holdout
```

Example output:

```text
1. CatBoost baseline
2. XGBoost + CatBoost OOF stack
3. Survival model
```

The mapper checks whether each structure is compatible with:

- target type
- sample size
- categorical/numerical data
- event sequences
- timestamps
- censoring information
- class imbalance
- deployment constraints

---

## D. STRUCTURES + NO_DATA

This is a valid and important mode.

Use when the user asks:

> What are strong structures for churn?

or:

> Give me 3 strong structures for credit default.

The Structure Seeker researches the problem only.

```text
Problem
  ↓
Structure Seeker
  ↓
Research
  ├── papers
  ├── benchmarks
  ├── Kaggle
  ├── GitHub
  └── similar problems
  ↓
ModelStructure[]
  ↓
Structure Research Report
```

No dataset is required.

The output must clearly label each structure as:

```text
RESEARCH_PROPOSAL
```

not:

```text
BEST_MODEL
```

because no experiment has been run.

Each structure should include:

```yaml
name:
problem_type:
architecture:
components:
why_it_may_work:
assumptions:
required_data_properties:
training_protocol:
validation_protocol:
strengths:
risks:
sources:
```

---

## E. BOTH + DATA_AVAILABLE

Use when the user wants both feature research and model-structure research.

```text
                         Problem
                           │
            ┌──────────────┴──────────────┐
            ▼                             ▼
      Feature Seeker               Structure Seeker
            │                             │
    ResearchFeature[]              ModelStructure[]
            │                             │
            ▼                             ▼
      Feature Mapper                Model Mapper
            │                             │
            └──────────────┬──────────────┘
                           ▼
                  Feature × Model Search
                           ↓
                     OOF Evaluation
                           ↓
                 Portfolio + Structure
                           ↓
                    Portfolio Freeze
                           ↓
                    Sealed Holdout
```

Do not evaluate every possible feature × model combination.

Use coarse-to-fine search.

---

## F. BOTH + NO_DATA

Use when the user wants a full research plan before any dataset exists.

```text
Problem
  ↓
  ├── Feature Seeker
  │      ↓
  │   Feature Families
  │      ↓
  │   Required Information
  │
  └── Structure Seeker
         ↓
      Model Structures
         ↓
      Data Assumptions
```

Final output:

```text
Research Blueprint
├── Feature families to investigate
├── Required information/data
├── Candidate model structures
├── Assumptions for each structure
├── Recommended validation design
└── Data gaps to resolve later
```

No feature or model is empirically approved in this mode.

---

# 3. Core Contracts

Create:

```text
src/contracts/
├── run_intent.py
├── problem.py
├── search_contract.py
├── research_feature.py
├── model_structure.py
├── mapped_feature.py
├── feature_spec.py
├── model_spec.py
├── prediction_event.py
└── budget_contract.py
```

---

# 4. RunIntent

```python
class RunMode(str, Enum):
    FEATURES = "FEATURES"
    STRUCTURES = "STRUCTURES"
    BOTH = "BOTH"

class DataState(str, Enum):
    DATA_AVAILABLE = "DATA_AVAILABLE"
    NO_DATA = "NO_DATA"
```

```python
class RunIntent(BaseModel):
    mode: RunMode
    data_state: DataState
```

This determines which components are allowed to run.

---

# 5. SearchContract

Example:

```yaml
problem:
  task: churn
  prediction_horizon: 30_days
  entity: customer
  domain: subscription

research:
  requested_features: 100
  requested_feature_families: 20
  requested_model_structures: 3

search:
  direct_problem: true
  similar_problems: true
  mechanisms: true
  analogies: true

limits:
  max_rounds: 5
  max_queries: 50
```

Fields may be omitted when the corresponding research mode is disabled.

Example structure-only request:

```yaml
research:
  requested_model_structures: 3
```

---

# 6. Build the Deterministic Spine First

Before implementing any LLM Seeker, build:

```text
PredictionEvent
Bitemporal temporal validation
Join validation
FitGuard
FeatureSpec compiler
Feature grammar
OOF evaluator
Portfolio freeze
Sealed holdout
Replay
```

Milestone-1:

```text
LLM calls = 0
Network calls = 0
```

Use one hard-coded `ResearchFeature`.

---

# 7. Bitemporal Safety

Facts should include:

```text
valid_ts
knowledge_ts
```

A fact is usable only if:

```python
valid_ts <= as_of_ts
and
knowledge_ts <= as_of_ts
```

Test:

```text
valid=T-1, knowledge=T-1  → PASS
valid=T,   knowledge=T    → PASS
valid=T+1, knowledge=T    → VETO
valid=T-1, knowledge=T+1  → VETO
```

---

# 8. Runtime FitGuard

All stateful transforms must fit only on training rows.

Protect:

```text
imputation
scaling
target encoding
mean encoding
feature selection
mutual information
variance thresholds
```

Conceptually:

```python
with FitGuard(allowed_indices=train_idx):
    transformer.fit(X_train)
```

If `fit()` touches validation or holdout rows:

```text
VETOED
```

---

# 9. Feature Seeker

Implement after the deterministic feature pipeline passes tests.

The Feature Seeker sees:

```text
problem
domain
target meaning
prediction horizon
requested feature count
research memory
```

It must not see:

```text
database
schema
column names
raw data
target values
```

It searches:

```text
direct problem
similar problems
mechanisms
analogies
papers
GitHub
Kaggle
domain literature
```

Output:

```python
ResearchFeature(
    concept_id=...,
    name=...,
    concept_family=...,
    mechanism=...,
    required_information=...,
    typical_formula=...,
    source_ids=...,
)
```

---

# 10. Structure Seeker

The Structure Seeker is independent from the Feature Seeker.

It sees:

```text
problem
target meaning
prediction horizon
domain
requested structure count
```

Dataset access is not required for research.

It searches for complete modeling strategies, not only algorithm names.

Bad output:

```text
XGBoost
CatBoost
Random Forest
```

Better output:

```text
Structure 1:
CatBoost baseline

Structure 2:
XGBoost + CatBoost
→ out-of-fold predictions
→ L2 Logistic Regression meta-model

Structure 3:
Survival modeling
→ time-to-event prediction
→ censoring-aware validation
```

Each structure must include:

```yaml
name:
problem_type:
architecture:
components:
rationale:
assumptions:
required_data_properties:
training_protocol:
validation_protocol:
strengths:
risks:
sources:
```

---

# 11. Structure Seeker Without Data

When `data_state=NO_DATA`:

Do:

```text
research problem
research similar problems
research strong model structures
describe assumptions
describe required data
describe validation protocol
```

Do not:

```text
claim the structure is best
claim measured lift
claim applicability to a specific schema
claim production readiness
```

Output status:

```text
RESEARCH_PROPOSAL
```

Example:

```yaml
name: dual_boosting_stack

status: RESEARCH_PROPOSAL

architecture:
  base_models:
    - XGBoost
    - CatBoost
  meta_model:
    - L2 Logistic Regression

required_data_properties:
  - tabular observations
  - binary target

training_protocol:
  - generate OOF predictions from base models
  - train meta-model only on OOF predictions

risks:
  - highly correlated base learners
  - added complexity without lift
  - calibration issues
```

---

# 12. Feature Mapper

Run only when:

```text
mode includes FEATURES
AND
data_state = DATA_AVAILABLE
```

Input:

```text
ResearchFeature
+
schema metadata
```

Output:

```text
APPLICABLE
DERIVABLE
PARTIALLY_APPLICABLE
NOT_AVAILABLE
```

The Mapper proposes only.

The deterministic verifier approves constructability.

---

# 13. Model Mapper

Run only when:

```text
mode includes STRUCTURES
AND
data_state = DATA_AVAILABLE
```

Input:

```text
ModelStructure
+
dataset metadata
```

Check:

```text
target compatibility
sample size
categorical features
temporal information
event sequences
censoring
class imbalance
compute constraints
latency constraints
```

Output:

```text
APPLICABLE
PARTIALLY_APPLICABLE
NOT_APPLICABLE
```

Again, this is feasibility, not empirical superiority.

---

# 14. Feature Evaluation

When data exists:

```text
ResearchFeature
  ↓
MappedFeature
  ↓
FeatureSpec
  ↓
Variants
  ↓
OOF Evaluation
  ↓
Selection
  ↓
Portfolio Freeze
  ↓
Sealed Holdout
```

Initial acceptance rule may use:

```text
mean OOF lift >= min_effect
AND
positive lift in >= ceil(0.75 * n_blocks)
```

Treat this rule as provisional until the null-vs-signal experiment validates it.

---

# 15. Structure Evaluation

When data exists:

Evaluate structures under the same folds and metric definitions.

Example:

```text
Structure A
CatBoost

Structure B
XGBoost

Structure C
XGB + CatBoost OOF stack
```

Use identical:

```text
training data
folds
feature portfolio
metrics
budget
```

where possible.

Do not call a structure "best" based on literature alone.

Only executed experiments can support empirical superiority.

---

# 16. BOTH Mode Evaluation

Avoid a full Cartesian explosion.

Do not blindly run:

```text
1000 features × 10 structures
```

Use:

```text
Feature screening
      ↓
small feature portfolios
      ↓
Structure screening
      ↓
top structures
      ↓
Feature × Structure fine evaluation
```

Example:

```text
300 research features
      ↓
80 applicable
      ↓
20 fine features
      ↓
3 feature portfolios

5 researched structures
      ↓
3 applicable
      ↓
2 structure finalists

3 portfolios × 2 structures
      ↓
6 final experiments
```

---

# 17. No-Data Output Contracts

## Feature-only, no data

Return:

```text
Feature Research Report
```

with:

```text
feature family
feature concept
mechanism
required information
typical formula
sources
similar problems
```

---

## Structure-only, no data

Return:

```text
Structure Research Report
```

with:

```text
model structure
rationale
assumptions
required data properties
training protocol
validation protocol
risks
sources
```

---

## Both, no data

Return:

```text
ML Research Blueprint
```

containing:

```text
Feature families
+
Model structures
+
Required data
+
Data gaps
+
Recommended validation design
```

---

# 18. Statuses

Recommended research statuses:

```text
RESEARCHING
RESEARCH_PROPOSAL
MAPPED
VERIFIED
EXPANDED
COARSE
FINE
SELECTED
PORTFOLIO_FROZEN
HOLDOUT_EVALUATED
NO_DATA
NO_CANDIDATES
VETOED
BUDGET_EXHAUSTED
MAX_ROUNDS
```

Important:

```text
MAX_ROUNDS != CONVERGED
```

and:

```text
RESEARCH_PROPOSAL != EMPIRICALLY_VALIDATED
```

---

# 19. Repository Structure

```text
src/
├── contracts/
│   ├── run_intent.py
│   ├── problem.py
│   ├── search_contract.py
│   ├── research_feature.py
│   ├── model_structure.py
│   ├── mapped_feature.py
│   ├── feature_spec.py
│   ├── model_spec.py
│   ├── prediction_event.py
│   └── budget_contract.py
│
├── feature_seeker/
│   ├── seeker.py
│   ├── planner.py
│   ├── extractor.py
│   └── sources/
│
├── structure_seeker/
│   ├── seeker.py
│   ├── planner.py
│   ├── extractor.py
│   └── sources/
│
├── mapping/
│   ├── feature_mapper.py
│   ├── model_mapper.py
│   └── join_paths.py
│
├── validation/
│   ├── temporal.py
│   ├── fit_guard.py
│   ├── joins.py
│   ├── leakage.py
│   ├── schema.py
│   └── computability.py
│
├── generation/
│   └── grammar.py
│
├── evaluation/
│   ├── splits.py
│   ├── oof.py
│   ├── feature_eval.py
│   ├── structure_eval.py
│   ├── ablation.py
│   └── holdout.py
│
├── knowledge/
│   └── research_memory.py
│
└── orchestration/
    ├── router.py
    ├── pipeline.py
    ├── checkpoints.py
    └── budget.py
```

---

# 20. Router Logic

The orchestration layer should decide the pipeline automatically.

Pseudo-code:

```python
def run(intent, problem, data=None):

    if intent.mode in {"FEATURES", "BOTH"}:
        features = feature_seeker(problem)

        if intent.data_state == "DATA_AVAILABLE":
            mapped_features = feature_mapper(features, data.schema)
            verified_features = verifier(mapped_features)
            feature_results = evaluate_features(verified_features)
        else:
            feature_results = build_feature_research_report(features)

    if intent.mode in {"STRUCTURES", "BOTH"}:
        structures = structure_seeker(problem)

        if intent.data_state == "DATA_AVAILABLE":
            mapped_structures = model_mapper(structures, data.metadata)
            structure_results = evaluate_structures(mapped_structures)
        else:
            structure_results = build_structure_research_report(structures)

    return assemble_output(...)
```

---

# 21. Milestone Sequence

## Milestone 1 — Deterministic Feature Spine

```text
zero LLM
hard-coded ResearchFeature
bitemporal validation
FitGuard
OOF
portfolio freeze
sealed holdout
```

---

## Milestone 2 — Feature Seeker, No Data Required

```text
problem
↓
research
↓
ResearchFeature[]
↓
Feature Research Report
```

---

## Milestone 3 — Structure Seeker, No Data Required

```text
problem
↓
research
↓
ModelStructure[]
↓
Structure Research Report
```

This milestone is independently useful even if no dataset exists.

---

## Milestone 4 — Feature Mapping

```text
ResearchFeature
+
schema
↓
MappedFeature
↓
FeatureSpec
```

---

## Milestone 5 — Structure Mapping

```text
ModelStructure
+
dataset metadata
↓
ModelSpec
```

---

## Milestone 6 — BOTH Mode

```text
feature research
+
structure research
↓
controlled joint experiments
```

---

## Milestone 7 — Research Memory

Add memory only after feature and structure pipelines are trustworthy.

---

# 22. Decision Rules

Always ask the system internally:

```text
What did the user request?
```

### If user asks:

> Find features for churn.

Run:

```text
FEATURES
```

### If user asks:

> Give me 3 strong structures for churn.

Run:

```text
STRUCTURES
```

### If user asks:

> Find 300 features and 3 model structures.

Run:

```text
BOTH
```

### If no dataset is provided:

Run research normally.

Stop before mapping/evaluation.

### If a dataset/schema is later added:

Continue from the frozen research artifacts rather than re-running research automatically.

---

# 23. Core Principles

```text
Feature Seeker discovers WHAT information may matter.

Structure Seeker discovers HOW the problem may be modeled.

Mapper determines WHETHER the ideas fit available data.

Deterministic code determines WHETHER features are safe to construct.

Experiments determine WHETHER features or structures actually work.
```

And:

> **No dataset is required for research.<br>
> A dataset is required for applicability and empirical claims.**

---

# First Useful Product

The first broadly useful product should accept:

```yaml
mode: FEATURES | STRUCTURES | BOTH
data_state: DATA_AVAILABLE | NO_DATA
```

and behave correctly in all six combinations.

That is more valuable than building a system that assumes every user already has a dataset.
