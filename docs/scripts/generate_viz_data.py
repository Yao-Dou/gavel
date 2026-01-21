#!/usr/bin/env python3
"""
Generate visualization data for the Gavel webpage.
Reads model summaries, reference summaries, checklists, and evaluations,
then outputs a single JSON file for the frontend to consume.
"""

import json
import os
from pathlib import Path

# Base paths
BASE_DIR = Path(__file__).parent.parent.parent  # /Users/douy/gavel
DATA_DIR = BASE_DIR / "data"
OUTPUT_DIR = BASE_DIR / "docs" / "data"

# Data file patterns
BENCHMARK_FILES = [
    "50_cases_for_benchmarking",
    "50_cases_for_benchmarking_2"
]

MODEL_SUMMARY_SUFFIX = "_complex_order_by_date_True_thinking_True"
MODEL_CHECKLIST_SUFFIX = "_complex_order_by_date_True_thinking_True_thinking_True"
EVALUATION_SUFFIX = "_complex_order_by_date_True_thinking_True_thinking_True_thinking_False"


def load_json_safe(filepath):
    """Load JSON file, return None if file doesn't exist or is empty."""
    try:
        if not os.path.exists(filepath):
            return None
        if os.path.getsize(filepath) == 0:
            return None
        with open(filepath, 'r', encoding='utf-8') as f:
            return json.load(f)
    except (json.JSONDecodeError, IOError):
        return None


def get_model_folders(base_path):
    """Get list of model folders (excluding non-model items)."""
    if not os.path.exists(base_path):
        return []

    models = []
    for item in os.listdir(base_path):
        item_path = os.path.join(base_path, item)
        if os.path.isdir(item_path) and not item.startswith('.'):
            # Skip utility scripts
            if item.endswith('.py'):
                continue
            models.append(item)
    return sorted(models)


def load_reference_summaries():
    """Load reference summaries from benchmark files."""
    references = {}

    for benchmark in BENCHMARK_FILES:
        filepath = DATA_DIR / "summaries" / f"{benchmark}.json"
        data = load_json_safe(filepath)
        if data:
            for case_id, case_data in data.items():
                if isinstance(case_data, dict) and "summary/long" in case_data:
                    references[case_id] = {
                        "summary": case_data["summary/long"],
                        "benchmark": benchmark
                    }

    return references


def load_model_summaries(model_name):
    """Load model summaries for a given model."""
    summaries = {}
    model_dir = DATA_DIR / "summaries" / model_name

    for benchmark in BENCHMARK_FILES:
        filepath = model_dir / f"{benchmark}{MODEL_SUMMARY_SUFFIX}.json"
        data = load_json_safe(filepath)

        if data and "results" in data:
            for case_id, summary in data["results"].items():
                if summary:  # Skip empty summaries
                    summaries[case_id] = {
                        "summary": summary if isinstance(summary, str) else summary.get("answer", ""),
                        "benchmark": benchmark
                    }

    return summaries


def load_reference_checklists():
    """Load reference checklists (currently empty files)."""
    checklists = {}

    for benchmark in BENCHMARK_FILES:
        filepath = DATA_DIR / "summary_checklists" / f"{benchmark}_thinking_True.json"
        data = load_json_safe(filepath)

        if data and "results" in data:
            for case_id, checklist in data["results"].items():
                checklists[case_id] = checklist

    return checklists


def load_model_checklists(model_name):
    """Load model checklists for a given model."""
    checklists = {}
    model_dir = DATA_DIR / "summary_checklists" / f"{model_name}_summarization"

    for benchmark in BENCHMARK_FILES:
        filepath = model_dir / f"{benchmark}{MODEL_CHECKLIST_SUFFIX}.json"
        data = load_json_safe(filepath)

        if data and "results" in data:
            for case_id, checklist in data["results"].items():
                # Process checklist items
                processed_checklist = {}
                for item_name, item_data in checklist.items():
                    if isinstance(item_data, dict):
                        processed_checklist[item_name] = {
                            "extracted": item_data.get("extracted", []),
                            "reasoning": item_data.get("reasoning", "")
                        }
                    else:
                        processed_checklist[item_name] = {"extracted": [], "reasoning": ""}
                checklists[case_id] = processed_checklist

    return checklists


def load_evaluations(model_name):
    """Load evaluation results for a given model."""
    evaluations = {}
    model_dir = DATA_DIR / "evaluation" / f"{model_name}_summarization"

    for benchmark in BENCHMARK_FILES:
        filepath = model_dir / f"{benchmark}{EVALUATION_SUFFIX}.json"
        data = load_json_safe(filepath)

        if data and "results" in data:
            for case_id, evaluation in data["results"].items():
                evaluations[case_id] = evaluation

    return evaluations


def extract_checklist_items(model_checklists):
    """Extract the list of checklist item names from model checklists."""
    items = set()
    for case_id, checklist in model_checklists.items():
        for item_name in checklist.keys():
            items.add(item_name)
    return sorted(list(items))


def compute_relation(model_values, ref_values, evaluation):
    """
    Compute the relation between model and reference values.
    Returns: {"text": "Equal", "class": "relation-equal", "matched_model": [], "matched_ref": []}
    """
    # Extract value lists
    model_extracted = [v.get("value", "") for v in model_values] if model_values else []
    ref_extracted = [v.get("value", "") for v in ref_values] if ref_values else []

    # Both empty
    if len(model_extracted) == 0 and len(ref_extracted) == 0:
        return {"text": "N/A", "class": "relation-na", "matched_model": [], "matched_ref": []}

    # One empty
    if len(model_extracted) == 0:
        return {"text": "Missing", "class": "relation-missing", "matched_model": [], "matched_ref": []}
    if len(ref_extracted) == 0:
        return {"text": "Ref N/A", "class": "relation-na", "matched_model": [], "matched_ref": []}

    # Both have identical values (quick check)
    if model_extracted == ref_extracted:
        matched_indices = list(range(1, len(model_extracted) + 1))
        return {
            "text": "Equal",
            "class": "relation-equal",
            "matched_model": matched_indices,
            "matched_ref": matched_indices
        }

    # Use evaluation data if available
    if evaluation:
        if isinstance(evaluation, str):
            # String-wise evaluation
            if "equals" in evaluation.lower():
                return {"text": "Equal", "class": "relation-equal", "matched_model": [1], "matched_ref": [1]}
            elif "reference contains model" in evaluation.lower():
                return {"text": "Ref ⊃ Model", "class": "relation-contains", "matched_model": [1], "matched_ref": []}
            elif "model contains reference" in evaluation.lower():
                return {"text": "Model ⊃ Ref", "class": "relation-contains", "matched_model": [], "matched_ref": [1]}
            elif "different" in evaluation.lower():
                return {"text": "Different", "class": "relation-different", "matched_model": [], "matched_ref": []}

        elif isinstance(evaluation, dict):
            # List-wise evaluation
            common = evaluation.get("common", [])
            only_in_model = evaluation.get("only_in_model", [])
            only_in_reference = evaluation.get("only_in_reference", [])

            matched_model = set()
            matched_ref = set()

            for match in common:
                if isinstance(match, dict):
                    mi = match.get("model_index")
                    ri = match.get("reference_index")
                    if isinstance(mi, int):
                        matched_model.add(mi)
                    elif isinstance(mi, list):
                        matched_model.update(mi)
                    if isinstance(ri, int):
                        matched_ref.add(ri)
                    elif isinstance(ri, list):
                        matched_ref.update(ri)

            if len(common) > 0 and len(only_in_model) == 0 and len(only_in_reference) == 0:
                return {
                    "text": "Equal",
                    "class": "relation-equal",
                    "matched_model": sorted(matched_model),
                    "matched_ref": sorted(matched_ref)
                }
            elif len(common) > 0:
                return {
                    "text": "Partial",
                    "class": "relation-contains",
                    "matched_model": sorted(matched_model),
                    "matched_ref": sorted(matched_ref)
                }
            else:
                return {
                    "text": "Different",
                    "class": "relation-different",
                    "matched_model": [],
                    "matched_ref": []
                }

    # Fallback: simple comparison
    return {"text": "Unknown", "class": "relation-na", "matched_model": [], "matched_ref": []}


def main():
    print("Generating visualization data...")

    # Load reference data
    reference_summaries = load_reference_summaries()
    reference_checklists = load_reference_checklists()
    print(f"Loaded {len(reference_summaries)} reference summaries")
    print(f"Loaded {len(reference_checklists)} reference checklists")

    # Get available models from summaries folder
    summary_models = get_model_folders(DATA_DIR / "summaries")
    print(f"Found {len(summary_models)} models in summaries folder")

    # Build visualization data
    viz_data = {
        "models": [],
        "cases": {},
        "checklist_items": []
    }

    all_checklist_items = set()

    for model_name in summary_models:
        print(f"Processing model: {model_name}")

        # Load model data
        model_summaries = load_model_summaries(model_name)
        model_checklists = load_model_checklists(model_name)
        evaluations = load_evaluations(model_name)

        if not model_summaries:
            print(f"  Skipping {model_name}: no summaries found")
            continue

        print(f"  Found {len(model_summaries)} summaries, {len(model_checklists)} checklists, {len(evaluations)} evaluations")

        # Add model to list
        viz_data["models"].append(model_name)
        viz_data["cases"][model_name] = {}

        # Collect checklist items
        if model_checklists:
            items = extract_checklist_items(model_checklists)
            all_checklist_items.update(items)

        # Process each case
        for case_id, model_data in model_summaries.items():
            if case_id not in reference_summaries:
                continue

            ref_data = reference_summaries[case_id]
            model_checklist = model_checklists.get(case_id, {})
            ref_checklist = reference_checklists.get(case_id, {})
            case_evaluation = evaluations.get(case_id, {})

            # Build checklist comparison
            checklist_comparison = {}
            for item_name in model_checklist.keys():
                model_item = model_checklist.get(item_name, {})
                ref_item = ref_checklist.get(item_name, {}) if ref_checklist else {}
                item_eval = case_evaluation.get(item_name) if case_evaluation else None

                model_values = model_item.get("extracted", [])
                ref_values = ref_item.get("extracted", []) if isinstance(ref_item, dict) else []

                relation = compute_relation(model_values, ref_values, item_eval)

                checklist_comparison[item_name] = {
                    "model_values": [v.get("value", "") for v in model_values],
                    "reference_values": [v.get("value", "") for v in ref_values],
                    "relation": relation
                }

            viz_data["cases"][model_name][case_id] = {
                "model_summary": model_data["summary"],
                "reference_summary": ref_data["summary"],
                "checklist": checklist_comparison
            }

    # Set checklist items
    viz_data["checklist_items"] = sorted(list(all_checklist_items))

    # Write output
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_file = OUTPUT_DIR / "viz_summaries.json"

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(viz_data, f, indent=2, ensure_ascii=False)

    print(f"\nGenerated: {output_file}")
    print(f"Models: {len(viz_data['models'])}")
    print(f"Checklist items: {len(viz_data['checklist_items'])}")

    # Print summary
    total_cases = sum(len(cases) for cases in viz_data["cases"].values())
    print(f"Total case-model pairs: {total_cases}")


if __name__ == "__main__":
    main()
