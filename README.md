# Vexoo Labs AI Engineer Assignment

This repository is a human-oriented prototype that showcases:

- A document ingestion engine with a 2-page sliding window and a 4-layer Knowledge Pyramid.
- A GSM8K reasoning model training workflow with optional LoRA-style fine-tuning.
- A reasoning adapter that routes questions to math, legal, or general reasoning modes.

## Project Structure

- `app.py` — primary application entrypoint with CLI commands for ingestion, query, routing, and training.
- `ingestion.py` — document ingestion module with raw text, summaries, categories, and distilled keywords.
- `training.py` — GSM8K training pipeline demonstrating dataset loading, tokenization, training, and evaluation.
- `adapter.py` — routing component for classifying question intent.
- `data/sample_document.txt` — sample text used for ingestion tests.
- `summary.docx` — one-page design summary.
- `requirements.txt` — Python dependency list.

## Setup

1. Create and activate a Python virtual environment:

```bash
python -m venv .venv
.\.venv\Scripts\activate
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

## Application Usage

### Ingest a document

```bash
python app.py ingest --source-file data/sample_document.txt --show-chunks
```

This command reads the document, builds the Knowledge Pyramid, and prints a human-readable summary of the ingestion process.

### Query the ingested document

```bash
python app.py query --source-file data/sample_document.txt --query "What are the main themes in the document?"
```

If no query is provided, the app will prompt you for one interactively.

### Route a question to a reasoning mode

```bash
python app.py route --query "If I have 5 apples and I buy 3 more, what is the total?"
```

This shows the reasoning mode and strategy selected for the question.

### Train the GSM8K reasoning model

```bash
python app.py train --model-name gpt2 --train-samples 3000 --eval-samples 1000
```

If Hugging Face GSM8K is unavailable, the script falls back to a simulated dataset so the workflow remains runnable.

## Design Highlights

- The ingestion pipeline is intentionally transparent: each chunk is stored with text, summary, category, and keywords.
- The app is designed for human review and iteration, not only machine evaluation.
- Training is structured to support larger models later, while remaining practical with `gpt2` today.
