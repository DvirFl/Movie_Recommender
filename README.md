# MovieLens RecSys

End-to-end recommender system on the MovieLens dataset: ingestion → feature
engineering → dual-architecture training with cross-distillation →
precomputed retrieval → FastAPI serving. Orchestrated by Airflow, tracked by
MLflow, tuned by Optuna, stored in PostgreSQL + MinIO.

## Architecture

```
┌─────────┐   ┌──────────┐   ┌───────────┐   ┌───────┐   ┌──────┐   ┌───────┐   ┌──────────────┐   ┌──────────┐   ┌────────────┐   ┌───────┐
│ startup │→ │ ingest   │→ │ validate  │→ │featur-│→ │split │→ │ tune  │→ │    train     │→ │  cross_  │→ │ evaluate → │→ │ serve │
│ (infra) │  │ (zip/dir)│  │(schema/   │  │ ize   │  │(temp.│  │(Optuna│  │ (per arch×loss│  │ distill  │  │ precompute │  │(API)  │
│         │  │          │  │ null/refs)│  │       │  │split)│  │ sweep)│  │  + SDFT)      │  │ (SDFT KL)│  │ (FAISS)    │  │       │
└─────────┘   └──────────┘   └───────────┘   └───────┘   └──────┘   └───────┘   └──────────────┘   └──────────┘   └────────────┘   └───────┘
```

Each stage is a standalone module in `stages/` (callable from `main.py` or an
Airflow task) that delegates real logic to `etl/`, `training/`, `precompute/`.

### 1. Ingest (`etl/ingest.py`)
Accepts a `.zip`, extracted directory, or single file. Auto-detects the
MovieLens variant (`ml-latest-small`, `ml-25m`, `ml-1m`, `ml-100k`) from
filenames and dispatches to a per-variant/per-file reader. Writes to
`raw.{ratings,movies,tags,links,genome_scores}` via batched
`INSERT ... ON CONFLICT` (upsert / skip / replace modes).

### 2. Validate (`etl/validate.py`)
Schema, null, range, and referential-integrity checks against `raw.*`.
Hard failures (empty tables, null keys) raise; soft issues (rating out of
range, orphan movie_ids) are warnings.

### 3. Featurize (`etl/featurize.py`)
- User features: time-decayed interaction history, genre affinity vector,
  rating count/mean.
- Item features: genre multi-hot, tag TF-IDF, release year (parsed from
  title).
Written to `features.user_features` / `features.item_features`.

### 4. Split (`etl/split.py`)
Global temporal split (sort by timestamp, first 80% train / next 10% val /
last 10% test) — no leakage. Indices in `features.split_indices`.

### 5. Tune (`training/hparam/tuner.py`, Optuna)
Per (architecture × loss) combination from the registry, sweeps a search
space merged from shared + arch-specific + loss-specific spaces
(`training/hparam/search_spaces.py`). RDB-backed (resumable), only the best
trial is logged to MLflow.

### 6. Train (`training/two_tower/trainer.py`, `training/infonce/trainer.py`)
Two independently registered architectures, same trainer contract:

| | Two-Tower | InfoNCE |
|---|---|---|
| User encoder | MLP over id + genre affinity + history | Transformer over interaction sequence |
| Item encoder | MLP over id + genre + release year | MLP over id + genre + release year |
| Loss | `TimedecayMSELoss` (weighted MSE) | `TimedecayInfoNCELoss` (in-batch contrastive) |

Both support **within-architecture SDFT** (self-distillation via an EMA
teacher, analytic-KL warmup) during their own training loop. Checkpoints go
to MLflow + MinIO.

> **Note:** module filenames under `training/two_tower/` and
> `training/infonce/` mirror each other (`towers.py`/`encoders.py`,
> `losses.py`, `trainer.py`) — same contract, independent implementations,
> no shared weights.

### 7. Cross-distill (`training/distillation/cross_distill.py`)
After both architectures are trained, each acts as EMA teacher for the
other (ordered pairs, reverse-KL on concatenated user+item embeddings).
Skipped automatically when fewer than 2 architectures are enabled.

### 8. Evaluate (`stages/stage_evaluate.py`)
Test-split loss per combination, logged to MLflow.

### 9. Precompute (`precompute/recommend.py`)
For every trained model: encode all items, build FAISS indices (cosine /
dot / L2, GPU if available) + optional learned scoring head, retrieve
Top-N per user × genre and cold-start aggregates. Written to
`serving.top_n_user_genre` / `serving.cold_start_genre`; indices uploaded
to MinIO.

### 10. Serve (`serving/api.py`, FastAPI)
| Route | Purpose |
|---|---|
| `POST /recommend` | Personalized or cold-start Top-N lookup |
| `POST /batch` | Batch recommend over a list of user_ids |
| `POST /ab_test` | Deterministic A/B split between two models |
| `POST /trigger` | Kick off the on-demand Airflow DAG |
| `GET /viz/*` | MLflow run metrics + pipeline row counts |
| `GET /health` | Liveness |

## Orchestration (Airflow)

Component registry (`training/registry.py`, backed by `config/registry.yaml`)
is the single source of truth for which arch×loss combinations exist —
adding an entry there automatically adds DAG tasks, no DAG edits needed.

| DAG | Trigger | Behavior |
|---|---|---|
| `movielens_data_trigger` | Watermark sensor on any `raw.*` table | Full pipeline, all combos |
| `movielens_daily` | Cron (midnight UTC) | Ingest skipped if no new data, rest always runs |
| `movielens_on_demand` | `POST /trigger` | Per-combo `ShortCircuitOperator` guards run only requested arch/loss filters |

All Airflow-free business logic lives in `airflow/dags/pipeline_logic.py`
(unit-testable without an Airflow runtime); `common.py`/`dag_*.py` are thin
Airflow-facing wrappers.

## Storage

- **PostgreSQL** — `raw`, `features`, `serving`, `pipeline` schemas
  (`db/models.py`, SQLAlchemy ORM, `Base.metadata.create_all` at startup).
- **MinIO** — buckets `model-checkpoints`, `faiss-indices`,
  `teacher-snapshots`, `cross-distill`.
- **MLflow** — experiment tracking + artifact logging; also runs as a
  self-managed subprocess if not already up (`stages/stage_startup.py`).
- **Optuna** — RDB storage (separate `optuna_studies` DB, see `init.sql`)
  for resumable sweeps.

## Running locally

```bash
cp .env.example .env                     # fill in DB/MinIO/MLflow settings
docker compose up -d                     # mlflow, postgres-mlflow, minio
python main.py --data-dir ./ml-25m.zip   # full pipeline
python main.py --stages serve            # API only
```

Key flags: `--stages`, `--from-stage/--to-stage`, `--skip-tune`,
`--losses/--architectures` (filter combos), `--ingest-mode
{upsert,skip,replace}`, `--dry-run`. See `python main.py --help`.

## Tests

```bash
pytest tests/ -v
```
Airflow DAG files themselves aren't imported in tests (they need a live
Airflow install); `pipeline_logic.py` — which has zero Airflow imports —
carries all the tested logic.

## Adding a new architecture or loss

1. Subclass `BaseRecommenderArchitecture` / `BaseRecommenderLoss`
   (`training/base/`).
2. Add an entry to `config/registry.yaml` with `enabled: true`.
3. Done — tune/train/cross_distill/evaluate/precompute and the Airflow DAGs
   pick it up automatically via `ComponentRegistry`.
