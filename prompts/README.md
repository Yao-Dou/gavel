# GAVEL Prompts

This folder contains all prompts used in the GAVEL framework for legal case summarization and evaluation.

---

## Folder Structure

```
prompts/
├── extract_checklist_item/          # Extract checklist from summaries
│   ├── general_template.txt         # Prompt template
│   └── item_specific_info.json      # 26 checklist item definitions
├── extract_checklist_item_from_docs/ # Extract checklist from case documents
│   ├── end_to_end_template.txt      # Template for end-to-end extraction
│   ├── chunk_by_chunk_template.txt  # Template for chunk-by-chunk extraction
│   └── item_specific_info.json      # 26 checklist item definitions
├── evaluate_checklist/              # Checklist evaluation prompts
│   ├── evaluate_checklist_item.txt  # String-wise comparison (single values)
│   └── evaluate_checklist_item_list.txt  # List-wise comparison (multiple values)
├── generate_summary.txt             # Generate case summary from documents
├── evaluate_writing_style.txt       # Compare writing style of two summaries
└── extract_facts_from_residual_spans.txt  # Extract residual facts
```

---

## Prompt Descriptions

### 1. `extract_checklist_item/`

**Purpose:** Extract 26 checklist items from case summaries (human or model-generated).

#### How It Works
The `general_template.txt` contains a prompt template with a `{item_description}` placeholder. For each checklist item, the corresponding definition from `item_specific_info.json` is inserted into this placeholder.

#### Template Variables
| Variable | Description |
|----------|-------------|
| `{item_description}` | Definition of the specific checklist item (from JSON) |
| `{case_summary}` | The case summary text to extract from |

#### Output Format
```json
{
  "reasoning": "<analysis of how information was identified>",
  "extracted": [
    {
      "evidence": ["<verbatim snippet from summary>"],
      "value": "<extracted information>"
    }
  ]
}
```

---

### 2. `extract_checklist_item_from_docs/`

**Purpose:** Extract 26 checklist items directly from case documents (not summaries).

This folder contains **two different templates** for two extraction methods:

#### `end_to_end_template.txt`
- **Method:** Feed all case documents to a long-context LLM at once
- **Use case:** Models with large context windows (e.g., GPT-4.1)
- **Key difference:** Processes all documents together, extracts with full context

#### `chunk_by_chunk_template.txt`
- **Method:** Process documents in 16K-token chunks iteratively
- **Use case:** Models with limited context windows
- **Key difference:** Maintains a `{current_state}` that accumulates across chunks

#### Template Variables

**End-to-End:**
| Variable | Description |
|----------|-------------|
| `{item_description}` | Definition of the specific checklist item |
| `{case_documents}` | All concatenated case documents |

**Chunk-by-Chunk:**
| Variable | Description |
|----------|-------------|
| `{item_description}` | Definition of the specific checklist item |
| `{current_state}` | Accumulated extraction state from previous chunks |
| `{document_name}` | Name of the current document |
| `{chunk_id}` | Current chunk number |
| `{total_chunks}` | Total number of chunks |
| `{document_chunk}` | The current chunk of text |

#### Output Format (both methods)
```json
{
  "reasoning": "<analysis>",
  "extracted": [
    {
      "evidence": [
        {
          "text": "<verbatim snippet>",
          "source_document": "<document name>",
          "location": "<page or chunk identifier>"
        }
      ],
      "value": "<extracted information>"
    }
  ]
}
```

---

### 3. `evaluate_checklist/`

**Purpose:** Compare extracted checklist values between model and reference.

#### `evaluate_checklist_item.txt` (String-wise Comparison)
For single-value checklist items (e.g., Filing Date, Judge Name).

**Relationship Options:**
- `"A contains B"` - A includes all info in B plus more
- `"B contains A"` - B includes all info in A plus more
- `"A equals B"` - Semantically equivalent
- `"A and B are different"` - Different or conflicting

#### `evaluate_checklist_item_list.txt` (List-wise Comparison)
For multi-value checklist items (e.g., Court Rulings, Parties).

**Output Format:**
```json
{
  "common": [{"A_index": 1, "B_index": 2}],
  "only_in_A": [3, 4],
  "only_in_B": [1]
}
```

---

### 4. `generate_summary.txt`

**Purpose:** Generate a legal case summary from case documents.

The prompt includes a 26-item checklist as guidance for what information to include in the summary. The model produces a narrative summary covering:
- Basic case information (filing date, parties, counsel)
- Legal foundation (cause of action, statutory basis, remedy sought)
- Judge information
- Related/consolidated cases
- Filings and proceedings
- Decrees and settlements
- Monitoring information
- Factual basis

---

### 5. `evaluate_writing_style.txt`

**Purpose:** Compare the writing style of two summaries across 5 dimensions.

**Evaluation Dimensions (1-5 scale):**
| Dimension | Description |
|-----------|-------------|
| Readability & Jargon | Technical vs. plain language balance |
| Narrative Order | Sequence of information presentation |
| Sentence Structure & Voice | Active/passive voice, sentence variety |
| Formatting & Layout | Headings, lists, paragraphing |
| Citation Style | Presence and formatting of references |

**Output Format:**
```json
{
  "readability_jargon": 4,
  "narrative_order": 3,
  "sentence_structure": 4,
  "formatting_layout": 2,
  "citation_style": 5
}
```

---

### 6. `extract_facts_from_residual_spans.txt`

**Purpose:** Extract atomic facts from text spans not covered by the 26-item checklist.

Used in residual fact evaluation to capture important information beyond the standard checklist items.

**Output Format:**
```json
{
  "reasoning": "<analysis>",
  "extracted": [
    {
      "fact": "<atomic fact>",
      "evidence_spans": [1, 3]
    }
  ]
}
```

---

## The 26 Checklist Items

Both `item_specific_info.json` files define the same 26 checklist items:

| # | Item Key | Description |
|---|----------|-------------|
| 1 | `Filing_Date` | When the lawsuit was first initiated |
| 2 | `Who_are_the_Parties` | Plaintiffs and defendants with descriptions |
| 3 | `Cause_of_Action` | Legal vehicle used to bring claims (e.g., 42 U.S.C. § 1983) |
| 4 | `Type_of_Counsel` | Private, public interest, government, pro se |
| 5 | `Statutory_or_Constitutional_Basis_for_the_Case` | Rights/laws allegedly violated |
| 6 | `Remedy_Sought` | What each party asks the court to grant |
| 7 | `First_and_Last_name_of_Judge` | Presiding judge(s) |
| 8 | `Class_Action_or_Individual_Plaintiffs` | Type of plaintiff representation |
| 9 | `Consolidated_Cases_Noted` | Cases combined for joint proceedings |
| 10 | `Related_Cases_Listed_by_Their_Case_Code_Number` | Connected cases |
| 11 | `Note_Important_Filings` | Key motions and filings |
| 12 | `Court_Rulings` | Decisions on motions and issues |
| 13 | `All_Reported_Opinions_Cited_with_Shortened_Bluebook_Citation` | Cited opinions |
| 14 | `Trials` | Trial proceedings and outcomes |
| 15 | `Appeal` | Appellate proceedings |
| 16 | `Significant_Terms_of_Decrees` | Court-ordered obligations |
| 17 | `Dates_of_All_Decrees` | When decrees were issued |
| 18 | `How_Long_Decrees_will_Last` | Duration of decree obligations |
| 19 | `Significant_Terms_of_Settlement` | Settlement obligations (not court-ordered) |
| 20 | `Date_of_Settlement` | Settlement-related dates |
| 21 | `How_Long_Settlement_will_Last` | Duration of settlement obligations |
| 22 | `Whether_the_Settlement_is_Court-enforced_or_Not` | Court enforcement status |
| 23 | `Disputes_Over_Settlement_Enforcement` | Compliance disputes |
| 24 | `Name_of_the_Monitor` | Court-appointed monitor name |
| 25 | `Monitor_Reports` | Monitoring findings and compliance status |
| 26 | `Factual_Basis_of_Case` | Underlying facts and evidence |

---

## Usage Example

To extract the "Filing Date" from a summary:

1. Load `general_template.txt`
2. Get the `Filing_Date` definition from `item_specific_info.json`
3. Replace `{item_description}` with the definition
4. Replace `{case_summary}` with the summary text
5. Send to the LLM (GPT-oss 20B in our experiments)
