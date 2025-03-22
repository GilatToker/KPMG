import json
import logging
import os
from typing import Dict, Any
from ocr_extraction import extract_text_from_pdf
from parse_ocr_to_json import generate_json_from_text


def flatten_json(nested: Dict[str, Any], parent_key: str = '', sep: str = '.') -> Dict[str, str]:
    """
    Flattens a nested JSON structure into a flat dictionary for easier comparison.
    For example: {"address": {"city": "TLV"}} becomes {"address.city": "TLV"}
    """
    items = {}
    for k, v in nested.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.update(flatten_json(v, new_key, sep=sep))
        else:
            items[new_key] = str(v).strip()
    return items


def evaluate_extraction_result(predicted_json: Dict, ground_truth_json: Dict) -> Dict:
    """
       Compares the extracted (predicted) JSON with the manually labeled ground truth JSON,
       and returns an evaluation report.

       The report includes:
         - A summary (total fields, correct, incorrect, missing (false negatives),
           extra filled (false positives), and overall accuracy percentage)
         - A detailed per-field analysis.
       """
    pred_flat = flatten_json(predicted_json)
    true_flat = flatten_json(ground_truth_json)

    total_fields = len(true_flat)
    correct = 0
    incorrect = 0
    missing = 0
    false_positives = 0

    field_reports = []

    for field, true_value in true_flat.items():
        pred_value = pred_flat.get(field, "").strip()

        if true_value == "" and pred_value != "":
            false_positives += 1
            field_reports.append((field, "False Positive", pred_value, true_value))

        elif true_value != "" and pred_value == "":
            missing += 1
            field_reports.append((field, "Missing", pred_value, true_value))

        elif true_value == pred_value:
            correct += 1
            field_reports.append((field, "Correct", pred_value, true_value))

        else:
            incorrect += 1
            field_reports.append((field, "Incorrect", pred_value, true_value))

    accuracy = round(correct / total_fields * 100, 2) if total_fields > 0 else 0

    return {
        "summary": {
            "Total Fields": total_fields,
            "Correct": correct,
            "Incorrect": incorrect,
            "Missing (False Negative)": missing,
            "Extra Filled (False Positive)": false_positives,
            "Accuracy (%)": accuracy
        },
        "details": field_reports
    }


def run_evaluation(form_file: str, ground_truth_file: str, report_file: str):
    """
       Runs the extraction and comparison process for a given form.
         1. Performs OCR on the form.
         2. Calls GPT to extract the JSON structure.
         3. Loads the ground truth JSON.
         4. Generates an evaluation report and saves it to a file.
       """
    logging.info(f"Processing form: {form_file}")
    # Step 1: Run OCR on the form
    extracted_text, word_confidences = extract_text_from_pdf(form_file)
    if extracted_text.startswith("ERROR"):
        logging.error(f"OCR Error: {extracted_text}")
        return

    # Step 2: Run GPT extraction to get the predicted JSON
    predicted_json = generate_json_from_text(extracted_text, word_confidences)

    # Step 3: Load the ground truth JSON
    with open(ground_truth_file, encoding="utf-8") as f:
        ground_truth = json.load(f)

    # Step 4: Evaluate the extraction result
    report = evaluate_extraction_result(predicted_json, ground_truth)

    # Print evaluation report to console
    print("\nEvaluation Summary:")
    print(json.dumps(report["summary"], indent=2, ensure_ascii=False))
    print("\nField-level Analysis:")
    for field, status, pred_val, true_val in report["details"]:
        print(f"{status} - {field}:\n    expected: {true_val}\n    got:      {pred_val}\n")

    os.makedirs(os.path.dirname(report_file), exist_ok=True)
    with open(report_file, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    logging.info(f"Report saved to {report_file}")


if __name__ == "__main__":
    # Define evaluation cases for all three forms
    evaluations = [
        {
            "form_file": "phase1_data/283_ex1.pdf",
            "ground_truth_file": "phase1_data/283_ex1_label.json",
            "report_file": "evaluation_reports/283_ex1_evaluation.json"
        },
        {
            "form_file": "phase1_data/283_ex2.pdf",
            "ground_truth_file": "phase1_data/283_ex2_label.json",
            "report_file": "evaluation_reports/283_ex2_evaluation.json"
        },
        {
            "form_file": "phase1_data/283_ex3.pdf",
            "ground_truth_file": "phase1_data/283_ex3_label.json",
            "report_file": "evaluation_reports/283_ex3_evaluation.json"
        },
        {
            "form_file": "phase1_data/283_ex1-image.jpg",
            "ground_truth_file": "phase1_data/283_ex1_label.json",
            "report_file": "evaluation_reports/283_ex1-image_evaluation.json"
        }
    ]

    for case in evaluations:
        run_evaluation(case["form_file"], case["ground_truth_file"], case["report_file"])
