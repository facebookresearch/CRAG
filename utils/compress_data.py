import bz2
import json


def compress_jsonl_to_bz2(jsonl_path, bz2_path):
    with open(jsonl_path, "rt") as jsonl_file, bz2.open(bz2_path, "wt") as bz2_file:
        for line in jsonl_file:
            bz2_file.write(line)


if __name__ == "__main__":
    jsonl_path = (
        "data/balanced_100_questions.jsonl"  # Replace with the path to your JSONL file
    )
    bz2_path = (
        "data/balanced_100_questions.jsonl.bz2"  # Replace with the desired output path
    )

    compress_jsonl_to_bz2(jsonl_path, bz2_path)
    print(f"Compressed {jsonl_path} to {bz2_path}")
