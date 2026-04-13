import argparse
from pathlib import Path
from typing import Optional

from adapter import ReasoningAdapter
from ingestion import DocumentIngestor, read_file
from training import TrainConfig, train


def human_header(title: str) -> None:
    print("\n" + "=" * 60)
    print(title)
    print("=" * 60 + "\n")


def run_ingest(source_file: Optional[str], show_chunks: bool) -> None:
    source_path = source_file or "data/sample_document.txt"
    path = Path(source_path)
    if not path.exists():
        raise FileNotFoundError(f"Source document not found: {path}")

    text = read_file(path)
    ingestor = DocumentIngestor()
    pyramid = ingestor.ingest(str(path), text)

    human_header("Document Ingestion Summary")
    print(f"Document: {path.name}")
    print(f"Loaded {len(text)} characters of source text.")
    print(f"Created {len(pyramid.chunks)} overlapping content windows.")
    print("Each window now carries a raw text segment, a short summary, a category label, and a distilled keywords layer.")

    if show_chunks:
        print("\nChunk details:")
        for index, chunk in enumerate(pyramid.chunks, start=1):
            print(f"  [{index}] Category: {chunk.category} | Keywords: {', '.join(chunk.distilled)[:80]}")
            print(f"      Summary: {chunk.summary[:120]}...\n")
    else:
        print("Use --show-chunks to review summaries, categories, and keywords for each window.")


def run_query(source_file: Optional[str], query: Optional[str]) -> None:
    source_path = source_file or "data/sample_document.txt"
    path = Path(source_path)
    if not path.exists():
        raise FileNotFoundError(f"Source document not found: {path}")

    text = read_file(path)
    ingestor = DocumentIngestor()
    ingestor.ingest(str(path), text)

    if not query:
        query = input("Enter your question about the document: ").strip()
        if not query:
            print("No query entered. Exiting.")
            return

    hit = ingestor.query(query)
    human_header("Query Result")
    print(f"Query: {query}")
    print(f"Best match level: {hit['level']} (score {hit['score']:.4f})")
    print(f"Source: {hit['source']}")
    print("\nBest answer content:\n")
    print(hit['answer'].strip())
    print("\n---\n")
    print(f"Chunk summary: {hit['summary']}")
    print(f"Distilled keywords: {hit['keywords']}\n")


def run_route(query: str) -> None:
    adapter = ReasoningAdapter()
    result = adapter.route(query)
    human_header("Reasoning Adapter")
    for key, value in result.items():
        print(f"{key}: {value}")
    print("\nThis adapter is designed to route questions to the correct reasoning strategy for math, legal, or general tasks.")


def run_train(args: argparse.Namespace) -> None:
    human_header("Training Workflow")
    print("This workflow is built as a prototype for GSM8K fine-tuning and reasoning model evaluation.")
    config = TrainConfig(
        model_name=args.model_name,
        train_samples=args.train_samples,
        eval_samples=args.eval_samples,
        output_dir=args.output_dir,
        use_lora=args.use_lora,
    )
    train(config)


def main() -> None:
    parser = argparse.ArgumentParser(description="Vexoo Labs reasoning system: ingestion, query, training, and routing.")
    subparsers = parser.add_subparsers(dest="command")

    parser_ingest = subparsers.add_parser("ingest", help="Ingest a document and build the knowledge pyramid.")
    parser_ingest.add_argument("--source-file", type=str, help="Path to the document text file.")
    parser_ingest.add_argument("--show-chunks", action="store_true", help="Display chunk metadata after ingestion.")

    parser_query = subparsers.add_parser("query", help="Query an ingested document.")
    parser_query.add_argument("--source-file", type=str, help="Path to the document text file.")
    parser_query.add_argument("--query", type=str, help="The question to ask about the document.")

    parser_route = subparsers.add_parser("route", help="Route a question to a reasoning strategy.")
    parser_route.add_argument("--query", type=str, required=True, help="The question to classify.")

    parser_train = subparsers.add_parser("train", help="Train the GSM8K reasoning model.")
    parser_train.add_argument("--model-name", type=str, default="gpt2", help="Pretrained base model name.")
    parser_train.add_argument("--train-samples", type=int, default=3000, help="Number of training samples.")
    parser_train.add_argument("--eval-samples", type=int, default=1000, help="Number of evaluation samples.")
    parser_train.add_argument("--output-dir", type=str, default="output", help="Directory to save model artifacts.")
    parser_train.add_argument("--use-lora", action="store_true", help="Use LoRA adapters if available.")

    args = parser.parse_args()
    if args.command == "ingest":
        run_ingest(args.source_file, args.show_chunks)
    elif args.command == "query":
        run_query(args.source_file, args.query)
    elif args.command == "route":
        run_route(args.query)
    elif args.command == "train":
        run_train(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
