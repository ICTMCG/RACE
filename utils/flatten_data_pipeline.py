import json
import os
import argparse

# Text type constants
TEXT_TYPES = [
    "human_written",
    "ai_humanized",
    "human_ai_polished",
    "ai_generated",
]
# Text types that have a nested 'texts' dictionary
NESTED_TEXT_TYPES = ["human_ai_polished", "ai_generated", "ai_humanized"]


def flatten_data(input_file, output_file):
    """
    Converts grouped JSONL data (Schema A + RST) into per-article flat JSONL (Schema B).

    Each input line is a "group" containing a human-written text and its various
    AI-generated/polished variants. This function "flattens" these groups so that
    each output line represents a single article with its associated RST structure,
    label, and model name.

    Args:
        input_file (str): Path to the input JSONL file (e.g., 'data/master_rst_parsed.jsonl').
            Expected schema per line (Schema A + RST):
            {
                "group_id": "...",
                "source_dataset": "...",
                "source_metadata": {...},
                "texts": {
                    "human_written": "...",
                    "ai_generated": {"prompt": "...", "texts": {"model_a": "...", ...}},
                    "human_ai_polished": {"prompt": "...", "texts": {"model_b": "...", ...}},
                    "ai_humanized": {"prompt": "...", "texts": {"human": "...", ...}}
                },
                "rst_structures": {
                    "human_written": {...},
                    "ai_generated": {"texts": {"model_a": {...}, ...}},
                    ...
                }
            }
        output_file (str): Path to save the output flattened JSONL file.
            Output schema per line (Schema B):
            {
                "item_id": "group_id_label_model",
                "group_id": "...",
                "source_dataset": "...",
                "article": "...",
                "label": "human_written|ai_generated|human_ai_polished|ai_humanized",
                "model_name": "model_a" or null,
                "rst_structure": {...}
            }
    """
    if not os.path.exists(input_file):
        print(f"Error: Input file not found at '{input_file}'")
        return

    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    total_articles = 0
    with open(input_file, "r", encoding="utf-8") as infile, open(
        output_file, "w", encoding="utf-8"
    ) as outfile:

        for line in infile:
            try:
                data = json.loads(line)
                group_id = data.get("group_id")
                if not group_id:
                    continue

                # Extract shared metadata
                shared_metadata = {
                    "group_id": group_id,
                    "source_dataset": data.get("source_dataset"),
                    "source_metadata": data.get("source_metadata"),
                }

                texts = data.get("texts", {})
                rst_structures = data.get("rst_structures", {})

                for article_type in TEXT_TYPES:
                    article_content = texts.get(article_type)
                    rst_data = rst_structures.get(article_type)

                    if not article_content:
                        continue

                    # Handle nested types (ai_generated, human_ai_polished, ai_humanized)
                    if article_type in NESTED_TEXT_TYPES:
                        if (
                            isinstance(article_content, dict)
                            and "texts" in article_content
                        ):
                            for model_name, article_text in article_content[
                                "texts"
                            ].items():
                                if not article_text:
                                    continue

                                # Find the corresponding RST tree
                                rst_tree = (
                                    rst_data.get("texts", {}).get(model_name)
                                    if rst_data
                                    else None
                                )

                                safe_model_name = model_name.replace("/", "_")
                                item_id = f"{group_id}_{article_type}_{safe_model_name}"

                                item = {
                                    "item_id": item_id,
                                    **shared_metadata,
                                    "article": article_text,
                                    "label": article_type,
                                    "model_name": model_name,
                                    "rst_structure": rst_tree,
                                }
                                outfile.write(
                                    json.dumps(item, ensure_ascii=False) + "\n"
                                )
                                total_articles += 1

                    # Handle simple types (human_written)
                    else:
                        item_id = f"{group_id}_{article_type}"
                        item = {
                            "item_id": item_id,
                            **shared_metadata,
                            "article": article_content,
                            "label": article_type,
                            "model_name": None,
                            "rst_structure": rst_data,
                        }
                        outfile.write(json.dumps(item, ensure_ascii=False) + "\n")
                        total_articles += 1

            except (json.JSONDecodeError, AttributeError) as e:
                print(
                    f"Warning: Could not process line due to an error: {e}. Line: {line.strip()[:100]}"
                )
                continue

    print(f"Data flattening complete. {total_articles} articles saved to '{output_file}'")


def main():
    """
    Main function to parse command-line arguments and run flattening.
    """
    parser = argparse.ArgumentParser(
        description="Flatten data from Schema A+RST to a per-article Schema B format."
    )
    parser.add_argument(
        "--input_file",
        type=str,
        default="data/master_rst_parsed.jsonl",
        help="Path to the input JSONL file (e.g., 'data/master_rst_parsed.jsonl').",
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default="data/flattened_articles.jsonl",
        help="Path to the output flattened JSONL file.",
    )

    args = parser.parse_args()
    flatten_data(args.input_file, args.output_file)


if __name__ == "__main__":
    main()
