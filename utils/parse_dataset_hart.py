import json
import glob
import os
import re


def normalize_model_name(name):
    """
    Normalizes known variations of model names to a canonical version.

    This mapping ensures consistent naming across different data sources,
    e.g., 'ChatGPT' and 'gpt-3.5-turbo-0125' both map to 'gpt-3.5-turbo'.

    Args:
        name (str): The raw model name string from the dataset.

    Returns:
        str: The normalized, canonical model name.
    """
    if not isinstance(name, str):
        return name

    # Lowercase and strip whitespace for easier matching
    norm_name = name.lower().strip()

    # Mapping: key is the variation, value is the canonical name
    model_map = {
        "chatgpt-turbo": "gpt-3.5-turbo",
        "gpt-3.5-turbo-0125": "gpt-3.5-turbo",
        "chatgpt": "gpt-3.5-turbo",
        "gpt4": "gpt-4",
        "gpt-4": "gpt-4",
        "2llama": "meta-llama/Llama-2-70b-chat-hf",
    }

    if norm_name in model_map:
        return model_map[norm_name]

    # If no specific rule matches, return the original name
    return name


def create_empty_group(group_id, source_dataset, file_path, original_id):
    """
    Creates an empty group structure according to the master schema (Schema A).

    A "group" represents a single source document and all its associated variants
    (human-written original + AI-generated/polished/humanized versions).

    Args:
        group_id (str): Unique identifier for the group.
        source_dataset (str): Name of the source dataset.
        file_path (str): Original file name from the dataset.
        original_id (str): Original record ID in the source dataset.

    Returns:
        dict: An empty group structure ready to be populated.
    """
    return {
        "group_id": group_id,
        "source_dataset": source_dataset,
        "source_metadata": {
            "original_file": file_path,
            "original_id": original_id,
            "raw_data_hash": None,
        },
        "texts": {
            "human_written": None,
            "ai_generated": {"prompt": None, "texts": {}},
            "human_ai_polished": {"prompt": None, "texts": {}},
            "ai_humanized": {"prompt": None, "texts": {}},
        },
        "domain": None,
    }


def get_base_id(record_id: str):
    """
    Extracts the base ID from a prefixed record ID.

    Hart dataset uses prefixes to distinguish text types:
    - No prefix: human-written original
    - 'gen/': AI-generated text
    - 'rep/': AI-rephrased (human content, AI language)
    - 'hum/gen/': Humanized AI text

    Args:
        record_id (str): The full record ID with potential prefix.

    Returns:
        str: The base ID without any prefix.
    """
    prefixes = ["gen/", "rep/", "hum/gen/"]
    for prefix in prefixes:
        if record_id.startswith(prefix):
            return record_id[len(prefix):]
    return record_id


def process(base_path="data/truth-mirror/benchmark/hart"):
    """
    Processes the truth-mirror/hart dataset.

    Reads all English JSON files from the Hart benchmark directory, groups
    records by their base ID, and classifies each text variant into the
    appropriate category (human_written, ai_generated, human_ai_polished,
    ai_humanized).

    Args:
        base_path (str): Path to the Hart dataset directory containing JSON files.
            Default: 'data/truth-mirror/benchmark/hart'

    Returns:
        list[dict]: A list of group dictionaries conforming to Schema A.
    """
    processed_data = {}

    # Filter out non-English files (e.g., 'news-ar.dev.json')
    non_english_pattern = re.compile(r".*-(ar|es|fr|zh)\.(dev|test)\.json$")

    all_json_files = glob.glob(os.path.join(base_path, "*.json"))
    english_json_files = [
        f for f in all_json_files if not non_english_pattern.match(os.path.basename(f))
    ]

    print(
        f"Found {len(english_json_files)} English JSON files to process for Hart dataset."
    )

    # Metadata fields to copy from the base human-written record
    fields_to_keep = [
        "domain",
        "date",
        "title",
        "prompt",
        "note",
        "task_level1",
        "task_level2",
        "task_level3",
    ]

    for file_path in english_json_files:
        filename = os.path.basename(file_path)
        source_dataset_name = f"truth-mirror-hart_{filename.split('.')[1]}"

        with open(file_path, "r", encoding="utf-8") as f:
            records = json.load(f)

        for record in records:
            record_id = record.get("id")
            if not record_id:
                continue

            base_id = get_base_id(record_id)

            # Find or create the group for this base_id
            if base_id not in processed_data:
                processed_data[base_id] = create_empty_group(
                    base_id, source_dataset_name, filename, base_id
                )

            group = processed_data[base_id]

            # --- Map record to the correct text type based on ID prefix ---
            generation_text = record.get("generation")
            content_source = record.get("content_source", "")
            language_source = record.get("language_source", "")

            if not record_id.startswith(("gen/", "rep/", "hum/gen/")):
                # Type 0: human_written (no prefix = original human text)
                group["texts"]["human_written"] = generation_text
                for key in fields_to_keep:
                    if key in record and record[key] is not None:
                        group[key] = record[key]

            elif record_id.startswith("gen/"):
                # Type 3: ai_generated (content generated by AI from scratch)
                if content_source.startswith("machine:"):
                    model_name = normalize_model_name(content_source.split(":", 1)[1])
                    group["texts"]["ai_generated"]["texts"][
                        model_name
                    ] = generation_text
                    group["texts"]["ai_generated"]["prompt"] = record.get("prompt")

            elif record_id.startswith("rep/"):
                # Type 1: human_ai_polished (human content, rephrased by AI)
                if language_source.startswith("rephrase:"):
                    model_name = normalize_model_name(language_source.split(":", 1)[1])
                    group["texts"]["human_ai_polished"]["texts"][
                        model_name
                    ] = generation_text

            elif record_id.startswith("hum/gen/"):
                # Type 2: ai_humanized or ai_generated (depends on humanization method)
                if language_source in ["humanize:human", "humanize:tool"]:
                    # Humanized by a human or a tool -> ai_humanized
                    group["texts"]["ai_humanized"]["texts"][
                        language_source.split(":", 1)[1]
                    ] = generation_text
                elif language_source.startswith("humanize:"):
                    # Humanized by another AI model -> still ai_generated (AI-AI chain)
                    if content_source.startswith("machine:"):
                        revising_model = normalize_model_name(
                            language_source.split(":", 1)[1]
                        )
                        original_model = normalize_model_name(
                            content_source.split(":", 1)[1]
                        )
                        # Use compound key to preserve provenance
                        group["texts"]["ai_generated"]["texts"][
                            f"{original_model}_THEN_{revising_model}"
                        ] = generation_text

    # --- Cleanup of empty text fields ---
    final_data = list(processed_data.values())
    for group_data in final_data:
        for text_type in list(group_data["texts"].keys()):
            content = group_data["texts"][text_type]
            if content is None:
                continue
            is_empty_dict = isinstance(content, dict) and not content.get("texts")
            if is_empty_dict:
                group_data["texts"][text_type] = None

    print(f"Processed {len(final_data)} groups from Hart dataset.")

    return final_data


if __name__ == "__main__":
    results = process()
    if results:
        print("--- Sample Hart Record ---")
        sample_to_print = results[0]
        for r in results:
            if (
                r["texts"]["human_written"]
                and r["texts"]["ai_generated"]
                and r["texts"]["human_ai_polished"]
            ):
                sample_to_print = r
                break
        print(json.dumps(sample_to_print, indent=2, ensure_ascii=False))

    print(f"\nProcessed {len(results)} groups from Hart dataset.")
