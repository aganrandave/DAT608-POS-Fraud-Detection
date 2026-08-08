# producer

Generates synthetic Nigerian POS transactions and streams them to Kafka.

- `nibss_distributions.py` — reference distributions (states, LGAs, card BINs, amount bands) used to keep generated volumes realistic.
- `fraud_scenarios.py` — injectors for the three labeled fraud types: `cloned_card`, `agent_collusion`, `fake_reversal`.
- `transaction_generator.py` — builds a single synthetic transaction record, optionally injecting fraud at `config.FRAUD_RATE`.
- `kafka_producer.py` — publishes generated transactions to the `pos-transactions` Kafka topic and mirrors them into `data/transactions_raw.xlsx`.
- `excel_writer.py` — appends transaction rows to `data/transactions_raw.xlsx` using the shared `openpyxl` + `filelock` pattern.
- `config.py` — environment-driven configuration (Kafka bootstrap servers, generation rate, fraud rate, file paths).

## Run locally

```bash
pip install -r producer/requirements.txt
python producer/kafka_producer.py
```

## Run via Docker

```bash
docker-compose up producer
```
