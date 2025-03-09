import bz2
import json
import pandas as pd
from collections import Counter

CRAG_DATA_PATH = "data/crag_task_1_and_2_dev_v4.jsonl.bz2"
FILTERED_DATA_PATH = "data/filtered_long_tailed_questions.jsonl"

# Define long-tailed question criteria
RARE_THRESHOLD = 5  # Words appearing less than 5 times
LONG_TAIL_TYPES = ["multi-hop", "comparison", "false_premise"]  # Harder question types
SAMPLE_SIZE = 500  # Limit dataset size


def load_crag_data(filepath):
    """Load CRAG dataset from bz2 compressed JSONL format."""
    with bz2.open(filepath, "rt") as f:
        return [json.loads(line) for line in f]


def filter_long_tailed_questions(data):
    """Extract long-tailed questions using word frequency and question type."""
    df = pd.DataFrame(data)

    # Compute word frequencies
    all_words = " ".join(df["query"]).split()
    word_counts = Counter(all_words)

    def is_long_tailed(query, question_type):
        words = query.split()
        rare_words = [word for word in words if word_counts[word] < RARE_THRESHOLD]
        return len(rare_words) > 2 or question_type in LONG_TAIL_TYPES

    filtered_df = df[
        df.apply(lambda row: is_long_tailed(row["query"], row["question_type"]), axis=1)
    ]
    return filtered_df.sample(n=min(SAMPLE_SIZE, len(filtered_df)), random_state=42)


if __name__ == "__main__":
    crag_data = load_crag_data(CRAG_DATA_PATH)
    filtered_data = filter_long_tailed_questions(crag_data)

    # Save filtered dataset
    with open(FILTERED_DATA_PATH, "w") as f:
        for record in filtered_data.to_dict(orient="records"):
            f.write(json.dumps(record) + "\n")

    print(
        f"Filtered {len(filtered_data)} long-tailed questions saved to {FILTERED_DATA_PATH}"
    )
