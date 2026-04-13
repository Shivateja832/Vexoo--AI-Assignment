import argparse
import math
import os
import re
from collections import Counter
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from pathlib import Path
from typing import Dict, List, Optional, Tuple

PAGE_CHAR_LIMIT = 2400
OVERLAP_CHAR_LIMIT = PAGE_CHAR_LIMIT // 2
STOPWORDS = {
    "the", "and", "for", "that", "with", "this", "from", "are", "was", "were",
    "have", "has", "had", "will", "would", "could", "should", "then", "than",
    "because", "when", "where", "what", "which", "while", "into", "about", "over",
    "under", "after", "before", "between", "also", "other", "their", "there"
}
CATEGORY_KEYWORDS = {
    "strategy": ["strategy", "planning", "roadmap", "vision", "goal", "objective"],
    "engineering": ["architecture", "system", "platform", "integration", "pipeline", "development"],
    "operations": ["process", "workflow", "execution", "deployment", "delivery", "compliance"],
    "knowledge": ["learning", "insight", "concept", "summary", "review", "analysis"],
}


def normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().lower())


def split_sliding_windows(text: str, window_size: int = PAGE_CHAR_LIMIT, overlap: int = OVERLAP_CHAR_LIMIT) -> List[str]:
    """Split text into overlapping windows to preserve context across boundaries."""
    raw_text = re.sub(r"\s+", " ", text.strip())
    chunks = []
    start = 0
    while start < len(raw_text):
        end = min(start + window_size, len(raw_text))
        chunk = raw_text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end == len(raw_text):
            break
        start = max(0, end - overlap)
    return chunks


def placeholder_summarize(text: str, max_sentences: int = 2) -> str:
    """Create a short summary using simple sentence selection and keyword reduction."""
    sentences = re.split(r"(?<=[.!?])\s+", text)
    sentences = [s.strip() for s in sentences if s.strip()]
    if not sentences:
        return text[:200].strip()
    summary = " ".join(sentences[:max_sentences])
    if len(summary) < 120 and len(sentences) > max_sentences:
        summary += " " + sentences[max_sentences][:120].strip()
    return summary


def rule_based_label(text: str) -> str:
    lowered = text.lower()
    scores = {category: 0 for category in CATEGORY_KEYWORDS}
    for category, keywords in CATEGORY_KEYWORDS.items():
        for token in keywords:
            if token in lowered:
                scores[category] += 1
    best_category = max(scores, key=scores.get)
    if scores[best_category] == 0:
        return "general"
    return best_category


def extract_keywords(text: str, top_n: int = 8) -> List[str]:
    tokens = re.findall(r"\b[a-zA-Z]{3,}\b", text.lower())
    filtered = [t for t in tokens if t not in STOPWORDS]
    counts = Counter(filtered)
    return [term for term, _ in counts.most_common(top_n)]


def similarity_score(a: str, b: str) -> float:
    a_norm = normalize(a)
    b_norm = normalize(b)
    if not a_norm or not b_norm:
        return 0.0
    matcher = SequenceMatcher(None, a_norm, b_norm)
    ratio = matcher.ratio()
    overlap = len(set(a_norm.split()) & set(b_norm.split()))
    score = ratio + overlap * 0.08
    return round(score, 4)


@dataclass
class PyramidChunk:
    raw_text: str
    summary: str
    category: str
    distilled: List[str]

    def get_layer_text(self, level: str) -> str:
        if level == "raw":
            return self.raw_text
        if level == "summary":
            return self.summary
        if level == "category":
            return self.category
        if level == "distilled":
            return " ".join(self.distilled)
        return self.raw_text


@dataclass
class KnowledgePyramid:
    source: str
    chunks: List[PyramidChunk] = field(default_factory=list)

    def build(self, text: str) -> None:
        chunk_texts = split_sliding_windows(text)
        for chunk in chunk_texts:
            summary = placeholder_summarize(chunk)
            category = rule_based_label(chunk)
            distilled = extract_keywords(chunk)
            self.chunks.append(PyramidChunk(
                raw_text=chunk,
                summary=summary,
                category=category,
                distilled=distilled,
            ))

    def retrieve(self, query: str) -> Tuple[Optional[PyramidChunk], str, float]:
        best_chunk = None
        best_level = "raw"
        best_score = -1.0
        for chunk in self.chunks:
            for level in ["raw", "summary", "category", "distilled"]:
                text = chunk.get_layer_text(level)
                score = similarity_score(query, text)
                if score > best_score:
                    best_score, best_chunk, best_level = score, chunk, level
        return best_chunk, best_level, best_score


class DocumentIngestor:
    def __init__(self) -> None:
        self.documents: Dict[str, KnowledgePyramid] = {}

    def ingest(self, source: str, text: str) -> KnowledgePyramid:
        pyramid = KnowledgePyramid(source=source)
        pyramid.build(text)
        self.documents[source] = pyramid
        return pyramid

    def query(self, query_text: str) -> Dict[str, str]:
        best_hit = {
            "source": None,
            "level": None,
            "score": 0.0,
            "answer": None,
            "summary": None,
            "keywords": None,
        }
        for name, pyramid in self.documents.items():
            chunk, level, score = pyramid.retrieve(query_text)
            if chunk and score > best_hit["score"]:
                best_hit["source"] = name
                best_hit["level"] = level
                best_hit["score"] = score
                best_hit["answer"] = chunk.get_layer_text(level)
                best_hit["summary"] = chunk.summary
                best_hit["keywords"] = ", ".join(chunk.distilled)
        if best_hit["source"] is None:
            raise ValueError("No documents have been ingested yet.")
        return best_hit


def read_file(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def display_hit(hit: Dict[str, str]) -> None:
    print("\n=== Retrieval Result ===")
    print(f"Source: {hit['source']}")
    print(f"Match level: {hit['level']}")
    print(f"Similarity score: {hit['score']:.4f}")
    print(f"Answer snippet:\n{hit['answer'][:800]}\n")
    print(f"Chunk summary:\n{hit['summary']}\n")
    print(f"Distilled keywords: {hit['keywords']}\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Knowledge Pyramid ingestion and retrieval demo.")
    parser.add_argument("--source-file", type=str, help="Path to the document text file.")
    parser.add_argument("--query", type=str, help="Query to retrieve information from the pyramid.")
    parser.add_argument("--show-chunks", action="store_true", help="Print chunk summaries and categories.")
    args = parser.parse_args()

    if not args.source_file:
        default_path = Path("data/sample_document.txt")
        if default_path.exists():
            source_path = default_path
        else:
            raise ValueError("No source file provided and sample document is missing.")
    else:
        source_path = Path(args.source_file)
        if not source_path.exists():
            raise FileNotFoundError(f"Source file not found: {source_path}")

    text = read_file(source_path)
    ingestor = DocumentIngestor()
    pyramid = ingestor.ingest(str(source_path), text)

    print(f"Ingested document '{source_path.name}' into {len(pyramid.chunks)} overlapping windows.")
    if args.show_chunks:
        for index, chunk in enumerate(pyramid.chunks, start=1):
            print(f"\n--- Chunk {index} ---")
            print(f"Category: {chunk.category}")
            print(f"Summary: {chunk.summary}")
            print(f"Keywords: {', '.join(chunk.distilled)}")

    if args.query:
        hit = ingestor.query(args.query)
        display_hit(hit)
    else:
        print("No query provided. Use --query to ask a question against the ingested document.")


if __name__ == "__main__":
    main()
