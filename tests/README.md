# tests

- `unit/` — fast, no-network tests against each layer's pure logic (transaction generation, feature math, tier thresholds). Run in CI on every PR.
- `integration/` — end-to-end tests against a running `docker-compose` stack. Gated behind `RUN_INTEGRATION_TESTS=1` so they don't run by default.

## Run unit tests

```bash
pytest tests/unit/ -v
```

## Run integration tests

```bash
docker-compose up -d
RUN_INTEGRATION_TESTS=1 pytest tests/integration/ -v
```
