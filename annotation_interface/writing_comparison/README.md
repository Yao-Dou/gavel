# Legal Summary Writing Style Comparison Interface

A Flask-based web application for annotating and comparing legal case summaries across various writing style dimensions.

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
   git commit -m "Deploy writing comparison interface"
   git push heroku main
   ```

4. Open the app:
   ```bash
   heroku open
   ```

---

## Features

- Compare two summaries side-by-side
- Evaluate 5 key writing style dimensions:
  - Readability & Jargon
  - Narrative Order
  - Sentence Structure
  - Formatting & Layout
  - Citation Style
- Track annotation progress
- Save annotations to HuggingFace dataset
- User authentication system
- Time tracking for each annotation
- Load existing annotations when returning to a case

## Local Development

1. Install dependencies:
   ```bash
   pip install flask huggingface_hub pytz
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
Edit `VALID_USERS` in `app.py` to add annotator usernames:
```python
VALID_USERS = ['user1', 'user2']  # Add your usernames here
```

### Data Configuration
Edit `DATA_CONFIG` in `app.py` to configure your HuggingFace repository:
```python
DATA_CONFIG = {
    'default': {
        'hf_repo_id': 'your-username/your-repo-name',
        'data_path': 'data/example.json'
    }
}
```

## Usage

1. Navigate to the login page: `/login`
2. Enter your username
3. Begin annotating summary pairs
4. Your progress is automatically saved
5. Previous annotations are loaded when you return to a case

## Data Format

The application expects data in the following JSON format:
```json
{
    "keys": ["case_id_1", "case_id_2", ...],
    "summary_pairs": [
        {
            "summary_A": "...",
            "summary_B": "..."
        },
        ...
    ],
    "orders": [["source_1", "source_2"], ...]
}
```

## File Structure
```
writing_comparison/
├── app.py
├── data/
│   └── example.json
├── templates/
│   ├── adjudication.html
│   ├── annotate.html
│   ├── complete.html
│   └── login.html
├── Procfile
├── requirements.txt
└── README.md
```

## Data Storage

Annotations are automatically saved to your HuggingFace dataset repository with:
- Username
- Session ID
- Timestamp
- Annotation results (preferences, ratings, reasons)
- Optional feedback
- Problem flags
- Time spent on each annotation
