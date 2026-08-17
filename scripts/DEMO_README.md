# Demo: Ingest and Query

Run the demo script to ingest a local file into a `KnowledgeLibrary` and ask a query.

Usage (from repo root):

```bash
python3 scripts/demo_ingest_query.py --source scripts/sample.txt --force-dummy --query "Summarize the document."
```

- `--force-dummy` forces the script to use local dummy embedder/LLM so it runs offline.
- Without `--force-dummy`, the script will attempt to use `OllamaQwenProvider` if available.

Library data will be stored under `./data/libraries/demo-library` by default.
