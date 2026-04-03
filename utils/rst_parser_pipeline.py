import json
import argparse
import os
from dotenv import load_dotenv
from tqdm import tqdm
from isanlp_rst.parser import Parser


def tree_to_dict(node):
    """
    Recursively converts a DiscourseUnit object to a JSON-serializable dictionary.
    """
    if not node:
        return None

    # Get all attributes of the node object
    node_dict = {
        "id": getattr(node, "id", None),
        "relation": getattr(node, "relation", None),
        "nuclearity": getattr(node, "nuclearity", None),
        "start": getattr(node, "start", None),
        "end": getattr(node, "end", None),
        "text": getattr(node, "text", None),
    }

    # Recursively convert child nodes
    if hasattr(node, "left") and node.left:
        node_dict["left"] = tree_to_dict(node.left)
    else:
        node_dict["left"] = None

    if hasattr(node, "right") and node.right:
        node_dict["right"] = tree_to_dict(node.right)
    else:
        node_dict["right"] = None

    return node_dict


def parse_text(parser, text):
    """
    Safely parses a single text string using the RST parser.
    Returns the RST tree structure as a dictionary or None if parsing fails.
    """
    if not text or not isinstance(text, str):
        return None
    try:
        result = parser(text)
        if result and result.get("rst"):
            # Convert the DiscourseUnit tree to a dictionary
            return tree_to_dict(result["rst"][0])
        return None
    except Exception as e:
        print(f"Warning: RST parsing failed for a text snippet. Error: {e}")
        return None


def main():
    """
    Main function to run the RST parsing pipeline.
    This script reads the master_grouped.jsonl file, processes each text type
    through the isanlp_rst parser, and saves the output to a new JSONL file.
    """
    parser = argparse.ArgumentParser(description="RST Parsing Pipeline")
    parser.add_argument(
        "--input",
        type=str,
        help="Path to the input jsonl file.",
    )
    parser.add_argument(
        "--output",
        type=str,
        help="Path to save the output JSONL file with RST structures.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional: Process only the first N records for testing.",
    )
    args = parser.parse_args()

    # --- 1. Initialization ---
    print("Initializing RST parser...")
    load_dotenv()  # Load .env file from the root directory
    model_path = os.getenv("RST_PARSER_PATH")

    if not model_path:
        print(
            "Error: RST_PARSER_PATH not found in .env file. Please check your .env configuration."
        )
        return

    # Initialize the parser with the 'rstdt' version
    rst_parser = Parser(
        model_dir=model_path, hf_model_version="rstdt", cuda_device=0
    )  # Use 0 for GPU, -1 for CPU

    print(f"RST parser initialized with model: {model_path}, version: rstdt")

    # --- 2. Processing Loop ---
    print(f"Starting processing of {args.input}")

    # Ensure output directory exists
    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    with open(args.input, "r", encoding="utf-8") as infile, open(
        args.output, "w", encoding="utf-8"
    ) as outfile:

        # Use tqdm for a progress bar
        line_iterator = (
            line
            for i, line in enumerate(infile)
            if args.limit is None or i < args.limit
        )
        total_lines = (
            args.limit
            if args.limit is not None
            else sum(1 for line in open(args.input, "r"))
        )

        for line in tqdm(line_iterator, total=total_lines, desc="Processing records"):
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                print(f"Warning: Skipping malformed JSON line.")
                continue

            # Create the new record structure, starting with the original data
            parsed_record = record.copy()
            parsed_record["rst_structures"] = {
                "human_written": None,
                "ai_generated": {"texts": {}},
                "human_ai_polished": {"texts": {}},
                "ai_humanized": {"texts": {}},
            }

            texts_to_parse = record.get("texts", {})

            # Parse human_written
            if texts_to_parse.get("human_written"):
                parsed_record["rst_structures"]["human_written"] = parse_text(
                    rst_parser, texts_to_parse["human_written"]
                )

            # Parse nested text types
            for text_type in ["ai_generated", "human_ai_polished", "ai_humanized"]:
                if texts_to_parse.get(text_type) and texts_to_parse[text_type].get(
                    "texts"
                ):
                    for model, text in texts_to_parse[text_type]["texts"].items():
                        parsed_record["rst_structures"][text_type]["texts"][model] = (
                            parse_text(rst_parser, text)
                        )

            # clean up empty nested dicts
            for k, v in parsed_record["rst_structures"].items():
                if isinstance(v, dict) and "texts" in v and not v["texts"]:
                    parsed_record["rst_structures"][k] = None

            # --- 3. Saving ---
            outfile.write(json.dumps(parsed_record, ensure_ascii=False) + "\n")

    print(f"Processing complete. Output saved to {args.output}")


if __name__ == "__main__":
    main()
