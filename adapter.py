import re
from typing import Dict


class ReasoningAdapter:
    """A simple routing layer that chooses a reasoning path from input text."""

    CATEGORY_RULES = {
        "math": [r"\b(\d+|\bplus\b|\bminus\b|\btimes\b|\bdivide\b|\btotal\b|\badd\b|\bsubtract\b)", r"\bwhat is\b", r"\bcalculate\b"],
        "legal": [r"\bcontract\b", r"\blaw\b", r"\briefs\b", r"\bstatute\b", r"\blegal\b", r"\bclause\b"],
        "general": [r"\bexplain\b", r"\bdescribe\b", r"\bwhy\b", r"\bhow\b"],
    }

    def classify(self, query: str) -> str:
        query_lower = query.lower()
        for mode, patterns in self.CATEGORY_RULES.items():
            for pattern in patterns:
                if re.search(pattern, query_lower):
                    return mode
        return "general"

    def route(self, query: str) -> Dict[str, str]:
        """Return routing metadata that a reasoning system can use."""
        mode = self.classify(query)
        if mode == "math":
            strategy = "formula-driven reasoning with stepwise calculation"
        elif mode == "legal":
            strategy = "rule-based interpretation and precedent summarization"
        else:
            strategy = "conceptual retrieval and answer synthesis"
        return {
            "query": query,
            "reasoning_mode": mode,
            "strategy": strategy,
            "explanation": f"Selected {mode} reasoning mode based on query keywords."
        }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Reasoning-aware adapter demo.")
    parser.add_argument("--query", type=str, required=True, help="User query to classify.")
    args = parser.parse_args()

    adapter = ReasoningAdapter()
    result = adapter.route(args.query)
    print("Reasoning Adapter Output:\n")
    for key, value in result.items():
        print(f"{key}: {value}")
