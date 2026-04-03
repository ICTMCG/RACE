import pandas as pd
import json
import os
import argparse
from sklearn.model_selection import train_test_split

# Define paths
INPUT_FILE = "data/flattened_articles.jsonl"
HART_OUTPUT_DIR = "data/hart_split"


def get_domain(filename):
    """Extracts domain from filename like 'news.dev.json' -> 'news'."""
    if not isinstance(filename, str):
        return "unknown"
    if "news" in filename:
        return "news"
    if "essay" in filename:
        return "essay"
    if "arxiv" in filename:
        return "arxiv"
    if "writing" in filename:
        return "writing"
    return "unknown"


def clean_original_file(value):
    """Cleans the 'original_file' column by extracting the first element from a list."""
    return value[0] if isinstance(value, list) and len(value) > 0 else value


def save_df_to_jsonl(df, path, cols_to_keep):
    """Saves a dataframe to a jsonl file, keeping only specified columns."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df[cols_to_keep].to_json(path, orient="records", lines=True)
    print(f"Saved {os.path.basename(path)} to {os.path.dirname(path)}")


def generate_and_save_stats(
    train_df, val_df, test_df, output_dir, dataset_name, group_by_cols
):
    """Generates a markdown statistics table and saves it."""
    print(f"Generating statistics for {dataset_name}...")

    # Calculate counts for each split
    train_counts = train_df.groupby(group_by_cols).size().reset_index(name="train_num")
    val_counts = val_df.groupby(group_by_cols).size().reset_index(name="val_num")
    test_counts = test_df.groupby(group_by_cols).size().reset_index(name="test_num")

    # Merge the counts into a single DataFrame
    stats_df = pd.merge(train_counts, val_counts, on=group_by_cols, how="outer")
    stats_df = pd.merge(stats_df, test_counts, on=group_by_cols, how="outer")

    # Clean up the merged DataFrame
    stats_df = stats_df.fillna(0)
    num_cols = ["train_num", "val_num", "test_num"]
    for col in num_cols:
        stats_df[col] = stats_df[col].astype(int)
    stats_df = stats_df.sort_values(group_by_cols).reset_index(drop=True)

    # Convert to custom markdown table
    header = "| " + " | ".join(stats_df.columns) + " |"
    separator = "| " + " | ".join(["---"] * len(stats_df.columns)) + " |"
    body = "\n".join(
        [
            "| " + " | ".join(map(str, row)) + " |"
            for row in stats_df.itertuples(index=False)
        ]
    )
    md_string = f"{header}\n{separator}\n{body}"

    # Prepend a title and save
    title = f"# Statistics for {dataset_name} Split\n\nThis table shows the distribution of data across the generated train, validation, and test sets.\n\n"
    md_content = title + md_string

    stats_path = os.path.join(output_dir, "stats.md")
    with open(stats_path, "w") as f:
        f.write(md_content)
    print(f"Saved stats.md to {output_dir}")


def split_data():
    """
    Loads, splits, and saves the datasets, generating statistics for each split.
    """
    print(f"Loading data from {INPUT_FILE}...")
    with open(INPUT_FILE, "r") as f:
        lines = [json.loads(line) for line in f]

    # Pop 'rst_structure' to prevent flattening
    rst_list = [l.pop("rst_structure", None) for l in lines]

    df = pd.json_normalize(lines, sep="_")

    # Add them back as columns of objects
    df["rst_structure"] = rst_list

    # Handle and merge potentially different keys for the original file name
    files_col = "source_metadata_original_files"
    file_col = "source_metadata_original_file"

    if files_col in df.columns and file_col in df.columns:
        # Combine both columns, preferring 'files_col' where it's not null
        df["original_file"] = df[files_col].combine_first(df[file_col])
        df.drop(columns=[files_col, file_col], inplace=True)
    elif files_col in df.columns:
        df.rename(columns={files_col: "original_file"}, inplace=True)
    elif file_col in df.columns:
        df.rename(columns={file_col: "original_file"}, inplace=True)
    else:
        raise KeyError(
            f"Could not find '{file_col}' or '{files_col}' in DataFrame columns"
        )

    df["original_file"] = df["original_file"].apply(clean_original_file)

    cols_to_keep = [
        "item_id",
        "group_id",
        "source_dataset",
        "article",
        "label",
        "model_name",
        "rst_structure",
    ]

    # --- Pre-processing ---
    hart_mask = df["source_dataset"].str.contains("truth-mirror-hart")
    df_hart = df[hart_mask].copy()
    df_hart["domain"] = df_hart["original_file"].apply(get_domain)

    # --- Splitting truth-mirror-hart (70/10/20) ---
    print("\nSplitting truth-mirror-hart dataset...")
    stratify_cols_hart = ["domain", "label", "model_name"]
    for col in stratify_cols_hart:
        df_hart[col] = df_hart[col].fillna("N/A")

    hart_train, hart_temp = train_test_split(
        df_hart, test_size=0.30, random_state=42, stratify=df_hart[stratify_cols_hart]
    )
    hart_val, hart_test = train_test_split(
        hart_temp,
        test_size=(2 / 3),
        random_state=42,
        stratify=hart_temp[stratify_cols_hart],
    )

    save_df_to_jsonl(
        hart_train, os.path.join(HART_OUTPUT_DIR, "train_graph.jsonl"), cols_to_keep
    )
    save_df_to_jsonl(
        hart_val, os.path.join(HART_OUTPUT_DIR, "val_graph.jsonl"), cols_to_keep
    )
    save_df_to_jsonl(
        hart_test, os.path.join(HART_OUTPUT_DIR, "test_graph.jsonl"), cols_to_keep
    )
    generate_and_save_stats(
        hart_train,
        hart_val,
        hart_test,
        HART_OUTPUT_DIR,
        "Truth-Mirror-Hart",
        stratify_cols_hart,
    )

    print(f"\nAll dataset splitting tasks completed.")


if __name__ == "__main__":

    split_data()
