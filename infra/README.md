# infra

Docker orchestration for the full pipeline: Zookeeper, Kafka, ksqlDB,
MLflow, and the four application services (producer, pipeline, scoring,
dashboard).

- `docker-compose.yml` — full service topology, referenced by the root `docker-compose.yml` via `include:`.
- `docker-compose.dev.yml` — hot-reload overrides for local development.
- `.env.example` — infra-local port overrides (see the repo root `.env.example` for application config).
- `kafka/kafka_setup.sh` — creates all Kafka topics used across the pipeline.

## Start everything

```bash
cp .env.example .env      # from repo root
docker-compose up --build
```

## Development mode (hot reload)

```bash
docker-compose -f infra/docker-compose.yml -f infra/docker-compose.dev.yml up --build
```

## Environment variables

Repo-root `.env` (copied from `.env.example`), consumed by the application
containers via `env_file`:

| Variable | Default | Used by |
|---|---|---|
| `KAFKA_BOOTSTRAP_SERVERS` | `localhost:9092` | producer, pipeline |
| `KAFKA_TOPIC_TRANSACTIONS` | `pos-transactions` | producer, pipeline |
| `KAFKA_TOPIC_SCORES` | `pos-fraud-scores` | pipeline, scoring |
| `KAFKA_TOPIC_ALERTS` | `pos-fraud-alerts` | scoring, ksqlDB alert streams |
| `KSQLDB_URL` | `http://localhost:8088` | ksqlDB setup scripts |
| `SPARK_MASTER` | `local[*]` | pipeline |
| `SPARK_CHECKPOINT_DIR` | `/tmp/spark-checkpoints` | pipeline |
| `MLFLOW_TRACKING_URI` | `http://localhost:5000` | models, scoring |
| `MLFLOW_EXPERIMENT_NAME` | `pos-fraud-detection` | models |
| `SCORING_API_HOST` / `SCORING_API_PORT` | `0.0.0.0` / `8000` | scoring |
| `SCORING_API_URL` | `http://localhost:8000/score` | pipeline (posts each computed feature row here) |
| `MODEL_VERSION` | `v1` | scoring |
| `STREAMLIT_SERVER_PORT` | `8501` | dashboard |
| `DATA_DIR` and the `*_XLSX` paths | see `.env.example` | all Excel readers/writers |

`infra/.env` (optional, copied from `infra/.env.example`) only overrides the
*host-side* ports docker-compose publishes, independent of the container-internal
values above - useful when one of these ports is already taken on your machine:
`KAFKA_HOST_PORT`, `ZOOKEEPER_HOST_PORT`, `KSQLDB_HOST_PORT`, `MLFLOW_HOST_PORT`,
`SCORING_HOST_PORT`, `DASHBOARD_HOST_PORT`.

## Troubleshooting

**A service exits immediately after `docker-compose up`.**
Run `docker-compose logs <service>` (e.g. `docker-compose logs kafka`) to see
why. `kafka` and `ksqldb-server` both depend on `zookeeper`/`kafka` being
reachable and can crash-loop for a few cycles on first boot until their
dependency's health check settles - `docker-compose up` without `-d` makes
this visible as it happens.

**`producer`, `pipeline`, `scoring`, or `dashboard` fails with a Kafka
connection error.** These containers reach Kafka at the in-network
`kafka:9092` address (see `KAFKA_ADVERTISED_LISTENERS` in
`infra/docker-compose.yml`), not `localhost:9092` - `localhost:9092` only
works for a process running on the host itself, outside Docker.

**Port already in use on the host.** Copy `infra/.env.example` to
`infra/.env` and change the relevant `*_HOST_PORT` variable, then
`docker-compose up --build` again.

**`scoring` can't find a registered model / `/score` returns an error about
no model version.** The XGBoost and Isolation Forest models have to be
trained and registered to MLflow's `Production` stage before `scoring` can
load them - run `models/train_xgboost.py` and
`models/train_isolation_forest.py` once against the running `mlflow`
service first (see `models/README.md`).

**`data/*.xlsx` files are locked or a write hangs.** Every writer takes a
`filelock` (`<file>.xlsx.lock`) before touching a workbook. If a container
was killed mid-write, a stale `.lock` file can be left behind - safe to
delete once you've confirmed no writer container is actually still running.
