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
