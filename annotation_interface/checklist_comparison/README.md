# Legal Case Summary Checklist Comparison Annotation Interface

A web application for annotating legal case summary checklist item comparisons.

## Heroku Deployment

### Prerequisites
- Heroku CLI installed
- HuggingFace account with API token

### Deployment Steps

1. Create a new Heroku app:
   ```bash
   heroku create your-app-name
   ```

2. Set environment variables:
   ```bash
   heroku config:set SECRET_KEY="your-secret-key"
   heroku config:set HF_TOKEN="your-huggingface-token"
   ```

3. Deploy the application:
   ```bash
   git add .
   git commit -m "Deploy checklist comparison interface"
   git push heroku main
   ```

4. Open the app:
   ```bash
   heroku open
   ```

---

## Features

- **String Comparison**: Select semantic relationships between two text items
- **List Comparison**: Match semantically equivalent items between two lists using drag-and-drop
- **Feedback System**: Optional feedback and problem flagging for each instance
- **HuggingFace Integration**: Automatically saves annotations to a HuggingFace dataset

## Local Development

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Set environment variables:
   ```bash
   export SECRET_KEY="your-secret-key"
   export HF_TOKEN="your-huggingface-token"
   ```

3. Run the application:
   ```bash
   python app.py
   ```

## Configuration

### Adding Users
Edit `USER_DATA_MAPPING` in `app.py` to add annotator usernames:
```python
USER_DATA_MAPPING = {
    'user1': 'data/example.json',
    'user2': 'data/example.json',
    # Add more users as needed
}
```

### Data Format
The application expects data in the following JSON format:
```json
{
    "keys": [
        {"case_id": "123", "checklist_item": "Item Name"},
        ...
    ],
    "value_pairs": [
        {"item_A": "...", "item_B": "...", "is_list": false},
        ...
    ],
    "orders": [["source_1", "source_2"], ...]
}
```

## Annotation Guidelines

### String Comparison
- **A contains B**: A includes all information in B, plus additional information
- **B contains A**: B includes all information in A, plus additional information
- **A equals B**: A and B convey the same information (semantically equivalent)
- **A and B are different**: A and B contain different or conflicting information

### List Comparison
- Drag items from List B to matching items in List A
- Click matched pairs to unmatch them
- Items may be paraphrased but convey the same meaning
- Some items may not have matches

## Data Storage

Annotations are automatically saved to your HuggingFace dataset repository with:
- Session ID
- Timestamp
- Annotation results
- Optional feedback
- Problem flags

## File Structure
```
checklist_comparison/
├── app.py
├── data/
│   └── example.json
├── templates/
│   ├── annotate.html
│   ├── complete.html
│   └── login.html
├── Procfile
├── requirements.txt
└── README.md
```
