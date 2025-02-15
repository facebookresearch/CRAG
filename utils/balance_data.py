import bz2
import json
import pandas as pd
from collections import Counter

CRAG_DATA_PATH = "data/crag_task_1_and_2_dev_v4.jsonl.bz2"
FILTERED_DATA_PATH = "data/balanced_100_questions.jsonl"
SAMPLE_PER_CATEGORY = 10  # Number of questions per category

def load_crag_data(filepath):
    """Load CRAG dataset from bz2 compressed JSONL format."""
    with bz2.open(filepath, "rt") as f:
        return [json.loads(line) for line in f]

def sample_balanced_questions(data):
    """Select an equal number of questions from each category."""
    df = pd.DataFrame(data)
    categories = df["question_type"].unique()
    
    sampled_dfs = []
    for category in categories:
        category_df = df[df["question_type"] == category]
        sampled_dfs.append(category_df.sample(n=min(SAMPLE_PER_CATEGORY, len(category_df)), random_state=42))
    
    return pd.concat(sampled_dfs)

if __name__ == "__main__":
    crag_data = load_crag_data(CRAG_DATA_PATH)
    balanced_data = sample_balanced_questions(crag_data)
    
    # Save balanced dataset
    with open(FILTERED_DATA_PATH, "w") as f:
        for record in balanced_data.to_dict(orient="records"):
            f.write(json.dumps(record) + "\n")

    print(f"Balanced dataset with {len(balanced_data)} questions saved to {FILTERED_DATA_PATH}")