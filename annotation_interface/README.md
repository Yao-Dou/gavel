# Annotation Interfaces

This folder contains annotation interfaces used for human evaluation in the GAVEL framework. These interfaces support three key annotation tasks: checklist extraction, checklist comparison, and writing style comparison.

## 1. Checklist Extraction

Extract 26 checklist items from legal case summaries, processing the summary paragraph by paragraph.

**Interface**: [thresh_mod](https://github.com/Yao-Dou/thresh_mod) (modified from [Thresh](https://github.com/davidheineman/thresh))

**Template**: [legal_extract_checklist_paragraph/1.yml](https://github.com/Yao-Dou/thresh_mod/blob/main/public/templates/legal_extract_checklist_paragraph/1.yml)

**Key Modification**: Added `data_format` parameter to support paragraph-level annotation:
```yaml
# Data format configuration: "paragraph" or "item"
# - "item": Each instance is one checklist item (original format)
# - "paragraph": Each instance is one paragraph containing multiple checklist items
data_format: paragraph
```

This allows annotators to extract all applicable checklist items from each paragraph in a single annotation instance, improving efficiency over the original item-by-item approach.

## 2. Checklist Comparison

Compare extracted checklist values between two sources (e.g., model vs. human summaries).

**Location**: [`checklist_comparison/`](./checklist_comparison/)

**Features**:
- **String Comparison**: Determine semantic relationships (equal, A contains B, B contains A, different)
- **List Matching**: Drag-and-drop interface for matching multi-value checklist items
- **Deployment**: Flask app deployable to Heroku
- **Storage**: Annotations saved to HuggingFace datasets

See [`checklist_comparison/README.md`](./checklist_comparison/README.md) for setup and deployment instructions.

## 3. Writing Style Comparison

Rate writing style similarity between two summaries across 5 dimensions.

**Location**: [`writing_comparison/`](./writing_comparison/)

**5 Evaluation Dimensions**:
| Dimension | Description |
|-----------|-------------|
| Readability & Jargon | Level of technical language and accessibility |
| Narrative Order | Chronological vs. thematic organization |
| Sentence Structure | Active/passive voice, sentence complexity |
| Formatting & Layout | Use of headers, lists, paragraphs |
| Citation Style | How legal citations are formatted and referenced |

**Features**:
- Side-by-side summary display
- 1-5 Likert scale ratings per dimension
- Deployment: Flask app deployable to Heroku
- Storage: Annotations saved to HuggingFace datasets

See [`writing_comparison/README.md`](./writing_comparison/README.md) for setup and deployment instructions.

## Folder Structure

```
annotation_interface/
├── README.md                    # This file
├── checklist_comparison/        # Checklist value comparison interface
│   ├── app.py
│   ├── data/
│   │   └── example.json
│   ├── templates/
│   ├── Procfile
│   ├── requirements.txt
│   └── README.md
└── writing_comparison/          # Writing style comparison interface
    ├── app.py
    ├── data/
    │   └── example.json
    ├── templates/
    ├── Procfile
    ├── requirements.txt
    └── README.md
```
