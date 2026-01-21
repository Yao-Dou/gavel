#!/usr/bin/env python3
"""
Generate visualization data for Document Extracted Checklist visualization.
Compares model-extracted checklists from documents vs human-extracted checklists from summaries.
"""

import json
import os
from pathlib import Path

# Base paths
BASE_DIR = Path(__file__).parent.parent.parent  # /Users/douy/gavel
DATA_DIR = BASE_DIR / "data"
OUTPUT_DIR = BASE_DIR / "docs" / "data"

# 26 checklist items
CHECKLIST_ITEMS = [
    'Filing Date',
    'Cause of Action',
    'Statutory or Constitutional Basis for the Case',
    'Remedy Sought',
    'Type of Counsel',
    'First and Last name of Judge',
    'All Reported Opinions Cited with Shortened Bluebook Citation',
    'Class Action or Individual Plaintiffs',
    'Related Cases Listed by Their Case Code Number',
    'How Long Decrees will Last',
    'Date of Settlement',
    'How Long Settlement will Last',
    'Whether the Settlement is Court-enforced or Not',
    'Name of the Monitor',
    'Appeal',
    'Who are the Parties',
    'Consolidated Cases Noted',
    'Dates of All Decrees',
    'Factual Basis of Case',
    'Note Important Filings',
    'Significant Terms of Decrees',
    'Significant Terms of Settlement',
    'Monitor Reports',
    'Trials',
    'Court Rulings',
    'Disputes Over Settlement Enforcement'
]


def normalize_item_name(name):
    """Convert underscore names to space-separated names."""
    result = name.replace('_', ' ')
    result = result.replace(' Court enforced ', ' Court-enforced ')
    return result


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


def load_summaries():
    """Load human summaries from 20_human_eval_cases files."""
    summaries = {}

    for filename in ["20_human_eval_cases.json", "20_human_eval_cases_2.json"]:
        filepath = DATA_DIR / "summaries" / filename
        data = load_json_safe(filepath)
        if data:
            for case_id, case_data in data.items():
                if isinstance(case_data, dict) and "summary/long" in case_data:
                    summaries[case_id] = case_data["summary/long"]

    return summaries


def load_human_reference():
    """Load human reference checklist from human.json."""
    filepath = DATA_DIR / "document_checklists" / "human.json"
    data = load_json_safe(filepath)

    if not data or "results" not in data:
        return {}

    return data["results"]


def load_end_to_end_data():
    """Load end-to-end extraction data from batch files."""
    base_path = DATA_DIR / "document_checklists" / "end_to_end"

    batch_files = []
    for prefix in ["20_human_eval_cases", "20_human_eval_cases_2"]:
        for batch_num in range(1, 6):
            batch_files.append(f"{prefix}_order_by_date_True_batch_{batch_num}.json")

    all_results = {}
    for batch_file in batch_files:
        filepath = base_path / batch_file
        data = load_json_safe(filepath)

        if data and "results" in data:
            for case_id, items in data["results"].items():
                if case_id not in all_results:
                    all_results[case_id] = {}

                for item_name, item_data in items.items():
                    normalized_name = normalize_item_name(item_name)
                    if isinstance(item_data, dict) and "extracted" in item_data:
                        all_results[case_id][normalized_name] = item_data

    return all_results


def load_chunk_by_chunk_data(model):
    """Load chunk-by-chunk extraction data for a model."""
    base_path = DATA_DIR / "document_checklists" / "chunk_by_chunk" / model

    all_results = {}
    for filename in ["20_human_eval_cases_thinking_True.json", "20_human_eval_cases_2_thinking_True.json"]:
        filepath = base_path / filename
        data = load_json_safe(filepath)

        if data and "results" in data:
            for case_id, items in data["results"].items():
                if case_id not in all_results:
                    all_results[case_id] = {}

                for item_name, item_data in items.items():
                    normalized_name = normalize_item_name(item_name)
                    all_results[case_id][normalized_name] = item_data

    return all_results


def load_gavel_agent_data(model, config):
    """Load GAVEL-AGENT extraction data for a model and config."""
    base_path = DATA_DIR / "document_checklists" / "gavel_agent" / model

    if not base_path.exists():
        return {}

    all_results = {}

    # Config to directory mapping
    config_dirs = {
        "all": "all/all_26_items",
        "grouped": "grouped",
        "individual": "individual"
    }

    for case_dir in base_path.iterdir():
        if not case_dir.is_dir():
            continue

        case_id = case_dir.name

        if config == "all":
            checklist_path = case_dir / "all" / "all_26_items" / "checklist.json"
            data = load_json_safe(checklist_path)
            if data:
                normalized_data = {}
                for item_name, item_value in data.items():
                    normalized_name = normalize_item_name(item_name)
                    normalized_data[normalized_name] = item_value
                all_results[case_id] = normalized_data

        elif config == "grouped":
            groups = [
                "01_basic_case_info", "02_legal_foundation", "03_judge_info",
                "04_related_cases", "05_filings_proceedings", "06_decrees",
                "07_settlements", "08_monitoring", "09_context"
            ]
            case_data = {}
            for group in groups:
                checklist_path = case_dir / "grouped" / group / "checklist.json"
                data = load_json_safe(checklist_path)
                if data:
                    for item_name, item_value in data.items():
                        normalized_name = normalize_item_name(item_name)
                        case_data[normalized_name] = item_value
            if case_data:
                all_results[case_id] = case_data

        elif config == "individual":
            items = [
                "01_filing_date", "02_parties", "03_class_action", "04_type_of_counsel",
                "05_cause_of_action", "06_statutory_basis", "07_remedy_sought", "08_judge_name",
                "09_consolidated_cases", "10_related_cases", "11_important_filings", "12_court_rulings",
                "13_reported_opinions", "14_trials", "15_appeals", "16_decree_terms",
                "17_decree_dates", "18_decree_duration", "19_settlement_terms", "20_settlement_date",
                "21_settlement_duration", "22_court_enforced", "23_enforcement_disputes", "24_monitor_name",
                "25_monitor_reports", "26_factual_basis"
            ]
            case_data = {}
            for item in items:
                checklist_path = case_dir / "individual" / item / "checklist.json"
                data = load_json_safe(checklist_path)
                if data:
                    for item_name, item_value in data.items():
                        normalized_name = normalize_item_name(item_name)
                        case_data[normalized_name] = item_value
            if case_data:
                all_results[case_id] = case_data

    return all_results


def load_evaluation(method, model, config=None):
    """Load evaluation results for a method/model/config combination."""
    base_path = DATA_DIR / "evaluation_documents_checklist"

    if method == "end_to_end":
        filepath = base_path / "end_to_end" / f"{model}_thinking_False.json"
    elif method == "chunk_by_chunk":
        filepath = base_path / "chunk_by_chunk" / f"{model}_thinking_False.json"
    elif method == "gavel_agent":
        filepath = base_path / "agent" / f"{model}_{config}_thinking_False.json"
    else:
        return {}

    data = load_json_safe(filepath)
    if data and "results" in data:
        return data["results"]
    return {}


def compute_relation(model_values, ref_values, evaluation):
    """Compute relation between model and reference values."""
    model_extracted = []
    if model_values:
        for v in model_values:
            if isinstance(v, dict):
                val = v.get("value", "")
                if val and val != "Not Applicable":
                    model_extracted.append(val)
            elif isinstance(v, str):
                model_extracted.append(v)

    ref_extracted = []
    if ref_values:
        for v in ref_values:
            if isinstance(v, dict):
                val = v.get("value", "")
                if val:
                    ref_extracted.append(val)
            elif isinstance(v, str):
                ref_extracted.append(v)

    # Both empty
    if len(model_extracted) == 0 and len(ref_extracted) == 0:
        return {"text": "N/A", "class": "relation-na", "matched_model": [], "matched_ref": []}

    # One empty
    if len(model_extracted) == 0:
        return {"text": "Missing", "class": "relation-missing", "matched_model": [], "matched_ref": []}
    if len(ref_extracted) == 0:
        return {"text": "Ref N/A", "class": "relation-na", "matched_model": [], "matched_ref": []}

    # Both have identical values
    if model_extracted == ref_extracted:
        matched_indices = list(range(1, len(model_extracted) + 1))
        return {
            "text": "Equal",
            "class": "relation-equal",
            "matched_model": matched_indices,
            "matched_ref": matched_indices
        }

    # Use evaluation if available
    if evaluation:
        if isinstance(evaluation, str):
            eval_lower = evaluation.lower()
            if "equals" in eval_lower or "equal" in eval_lower:
                return {"text": "Equal", "class": "relation-equal", "matched_model": [1], "matched_ref": [1]}
            elif "model contains" in eval_lower or "a contains b" in eval_lower:
                return {"text": "Model > Ref", "class": "relation-contains", "matched_model": [], "matched_ref": [1]}
            elif "reference contains" in eval_lower or "b contains a" in eval_lower:
                return {"text": "Ref > Model", "class": "relation-contains", "matched_model": [1], "matched_ref": []}
            elif "different" in eval_lower:
                return {"text": "Different", "class": "relation-different", "matched_model": [], "matched_ref": []}

        elif isinstance(evaluation, dict):
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

    # Fallback
    return {"text": "Unknown", "class": "relation-na", "matched_model": [], "matched_ref": []}


def main():
    print("Generating document checklist visualization data...")

    # Load human summaries
    summaries = load_summaries()
    print(f"Loaded {len(summaries)} human summaries")

    # Load human reference checklist
    human_reference = load_human_reference()
    print(f"Loaded {len(human_reference)} human reference checklists")

    # Define method configurations
    method_configs = [
        {"method": "end_to_end", "model": "gpt-4.1-2025-04-14", "config": None},
        {"method": "chunk_by_chunk", "model": "gpt-oss-20b-BF16", "config": None},
        {"method": "chunk_by_chunk", "model": "Qwen3-32B", "config": None},
        {"method": "chunk_by_chunk", "model": "Qwen3-30B-A3B-Thinking-2507", "config": None},
        {"method": "gavel_agent", "model": "gpt-oss-20b-BF16", "config": "all"},
        {"method": "gavel_agent", "model": "gpt-oss-20b-BF16", "config": "grouped"},
        {"method": "gavel_agent", "model": "gpt-oss-20b-BF16", "config": "individual"},
        {"method": "gavel_agent", "model": "Qwen3-30B-A3B-Thinking-2507", "config": "all"},
        {"method": "gavel_agent", "model": "Qwen3-30B-A3B-Thinking-2507", "config": "grouped"},
        {"method": "gavel_agent", "model": "Qwen3-30B-A3B-Thinking-2507", "config": "individual"},
    ]

    # Build visualization data
    viz_data = {
        "methods": [],
        "cases": {},
        "checklist_items": CHECKLIST_ITEMS
    }

    for mc in method_configs:
        method = mc["method"]
        model = mc["model"]
        config = mc["config"]

        # Generate method ID
        method_id = f"{method}__{model}"
        if config:
            method_id += f"__{config}"

        print(f"\nProcessing: {method} / {model}" + (f" / {config}" if config else ""))

        # Load method data
        if method == "end_to_end":
            method_data = load_end_to_end_data()
        elif method == "chunk_by_chunk":
            method_data = load_chunk_by_chunk_data(model)
        elif method == "gavel_agent":
            method_data = load_gavel_agent_data(model, config)
        else:
            method_data = {}

        if not method_data:
            print(f"  Skipping: no data found")
            continue

        print(f"  Found {len(method_data)} cases")

        # Load evaluation
        evaluations = load_evaluation(method, model, config)
        print(f"  Found {len(evaluations)} evaluations")

        # Add method to list
        viz_data["methods"].append({
            "method": method,
            "model": model,
            "config": config,
            "id": method_id
        })

        viz_data["cases"][method_id] = {}

        # Process each case
        for case_id, model_checklist in method_data.items():
            # Skip cases without human reference
            if case_id not in human_reference:
                continue

            # Get summary
            summary = summaries.get(case_id, "No summary available")

            # Build checklist comparison
            checklist_comparison = {}
            ref_checklist = human_reference.get(case_id, {})
            case_evaluation = evaluations.get(case_id, {})

            for item_name in CHECKLIST_ITEMS:
                model_item = model_checklist.get(item_name, {})
                ref_item = ref_checklist.get(item_name, [])
                item_eval = case_evaluation.get(item_name)

                # Get extracted values
                model_values = model_item.get("extracted", []) if isinstance(model_item, dict) else []
                ref_values = ref_item if isinstance(ref_item, list) else []

                # Compute relation
                relation = compute_relation(model_values, ref_values, item_eval)

                # Extract value strings
                model_val_strs = []
                for v in model_values:
                    if isinstance(v, dict):
                        val = v.get("value", "")
                        if val and val != "Not Applicable":
                            model_val_strs.append(val)
                    elif isinstance(v, str):
                        model_val_strs.append(v)

                ref_val_strs = []
                for v in ref_values:
                    if isinstance(v, dict):
                        val = v.get("value", "")
                        if val:
                            ref_val_strs.append(val)
                    elif isinstance(v, str):
                        ref_val_strs.append(v)

                checklist_comparison[item_name] = {
                    "model_values": model_val_strs,
                    "reference_values": ref_val_strs,
                    "relation": relation
                }

            viz_data["cases"][method_id][case_id] = {
                "summary": summary,
                "checklist": checklist_comparison
            }

    # Write output
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_file = OUTPUT_DIR / "viz_doc_checklist.json"

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(viz_data, f, indent=2, ensure_ascii=False)

    print(f"\n\nGenerated: {output_file}")
    print(f"Methods: {len(viz_data['methods'])}")
    print(f"Checklist items: {len(viz_data['checklist_items'])}")

    # Print summary
    total_cases = sum(len(cases) for cases in viz_data["cases"].values())
    print(f"Total case-method pairs: {total_cases}")


if __name__ == "__main__":
    main()
