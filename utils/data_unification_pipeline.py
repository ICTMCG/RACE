import json
import argparse
import os
from tqdm import tqdm
from parse_dataset_hart import process as parse_hart


def run_pipeline(output_path):
    """
    Runs the data unification pipeline for the Hart dataset.

    This function calls the Hart dataset parser, which reads the raw JSON files
    from 'data/truth-mirror/benchmark/hart/', groups records by their base ID,
    and outputs a list of structured records conforming to Schema A.

    Args:
        output_path (str): Path to save the unified JSONL output file.
    """
    print("Running data unification pipeline for Hart dataset...")
    all_records = parse_hart()

    # Save the grouped records to the master file
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for group in tqdm(all_records, desc="Writing records"):
            f.write(json.dumps(group, ensure_ascii=False) + "\n")

    print(f"Master file created at: {output_path} ({len(all_records)} groups)")


def main():
    parser = argparse.ArgumentParser(
        description="Data Unification Pipeline for RACE (Hart dataset only)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/master_grouped.jsonl",
        help="Path to save the output JSONL file.",
    )
    args = parser.parse_args()
    run_pipeline(args.output)


if __name__ == "__main__":
    main()
