import json
from src.assessment import Assessment, LLMPowerSeekingAssessment
from src.reviser import Reviser, LLMPowerSeekingReviser
import os
from typing import Any, List

def clean_data(file_path: str, limit: int, assess: Assessment, reviser: Reviser) -> None:
    """
    Clean the data in the provided file and save the cleaned data to a new file.

    Args:
        file_path (str): The path to the file containing the data to clean.
        limit (int): The maximum number of lines to process.
        assess (AssessmentMethod): The method used to assess the data.

    Returns:
        None
    """
    if not os.path.exists("data/processed"):
        os.makedirs("data/processed")

    batch_size = 1000  # Process data in batches to avoid storing everything in memory
    cleaned_data_batch: List[Any] = []
    removed_data_batch: List[Any] = []

    with open(file_path, 'r', encoding='utf-8') as file:
        for line_index, line in enumerate(file):
            if line_index >= limit:
                break 
            data = json.loads(line.strip())
            probability_estimate = assess.assess(data['text'])
            data["estimate"] = probability_estimate

            if probability_estimate > FILTER_THRESHOLD:
                removed_data_batch.append(data)
            elif probability_estimate > REVISE_THRESHOLD:
                data["original_text"] = data["text"]
                data["text"] = reviser.revise(data["text"])
                cleaned_data_batch.append(data)
            else:
                cleaned_data_batch.append(data)

            # Write batches to files as they get full
            if len(cleaned_data_batch) >= batch_size:
                with open("data/processed/cleaned_data.jsonl", "a", encoding="utf-8") as cleaned_file:
                    for item in cleaned_data_batch:
                        cleaned_file.write(json.dumps(item) + "\n")
                cleaned_data_batch = []

            if len(removed_data_batch) >= batch_size:
                with open("data/processed/removed_data.jsonl", "a", encoding="utf-8") as removed_file:
                    for item in removed_data_batch:
                        removed_file.write(json.dumps(item) + "\n")
                removed_data_batch = []

    # Write any remaining data
    if cleaned_data_batch:
        with open("data/processed/cleaned_data.jsonl", "a", encoding="utf-8") as cleaned_file:
            for item in cleaned_data_batch:
                cleaned_file.write(json.dumps(item) + "\n")

    if removed_data_batch:
        with open("data/processed/removed_data.jsonl", "a", encoding="utf-8") as removed_file:
            for item in removed_data_batch:
                removed_file.write(json.dumps(item) + "\n")

FILTER_THRESHOLD = 80
REVISE_THRESHOLD = 50

if __name__ == "__main__":
    input_file = "data/raw/webtext.train.jsonl"
    # Limit the number of lines to process for testing
    # clean_data(input_file, 10, LLMDeceptionAssessment(), LLMDeceptionReviser())
    clean_data(input_file, 50, LLMPowerSeekingAssessment(), LLMPowerSeekingReviser())
    print(f"Data cleaned and saved to processed files.")
