import os
import json
import random
from flask import Flask, render_template, request, jsonify, session, redirect, url_for
from datetime import datetime
import pytz
import uuid
from huggingface_hub import HfApi, HfFolder
import tempfile

app = Flask(__name__)
app.secret_key = os.environ.get('SECRET_KEY', 'dev-key-change-in-production')

# Initialize HuggingFace API
hf_api = HfApi()
HF_TOKEN = os.environ.get('HF_TOKEN')

# Data configuration mapping
DATA_CONFIG = {
    'default': {
        'hf_repo_id': 'your-username/your-repo-name',  # Change to your HuggingFace dataset repo
        'data_path': 'data/example.json'
    }
}

# Default data source
DEFAULT_DATA_SOURCE = 'default'

# List of valid users - add your annotator usernames here
VALID_USERS = ['user1', 'user2']

def get_data_source():
    """Get the current data source from session or default"""
    return session.get('data_source', DEFAULT_DATA_SOURCE)

def get_hf_repo_id():
    """Get the HuggingFace repo ID for current data source"""
    data_source = get_data_source()
    return DATA_CONFIG.get(data_source, DATA_CONFIG[DEFAULT_DATA_SOURCE])['hf_repo_id']

def load_annotation_data(username, data_source=None):
    """Load annotation data for specific user and data source"""
    if data_source is None:
        data_source = get_data_source()

    # Validate data source
    if data_source not in DATA_CONFIG:
        data_source = DEFAULT_DATA_SOURCE

    data_file = DATA_CONFIG[data_source]['data_path']

    if not os.path.exists(data_file):
        return None

    with open(data_file, 'r') as f:
        return json.load(f)

def load_user_progress(username):
    """Load user progress from HuggingFace"""
    try:
        progress_file = f"progress/{username}_progress.json"
        hf_repo_id = get_hf_repo_id()

        # Try to download progress file from HuggingFace
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as tmp_file:
            tmp_path = tmp_file.name

        try:
            hf_api.hf_hub_download(
                repo_id=hf_repo_id,
                filename=progress_file,
                repo_type="dataset",
                token=HF_TOKEN,
                local_dir_use_symlinks=False,
                local_dir=os.path.dirname(tmp_path)
            )
            
            with open(os.path.join(os.path.dirname(tmp_path), progress_file), 'r') as f:
                progress = json.load(f)
            
            # Clean up
            os.unlink(tmp_path)
            return progress.get('current_index', 0)
            
        except Exception:
            # File doesn't exist or error downloading - start from 0
            os.unlink(tmp_path)
            return 0
            
    except Exception as e:
        print(f"Error loading progress for {username}: {e}")
        return 0

def save_user_progress(username, current_index):
    """Save user progress to HuggingFace"""
    try:
        hf_repo_id = get_hf_repo_id()

        # Get NYC Eastern timezone
        eastern = pytz.timezone('America/New_York')
        nyc_time = datetime.now(eastern)

        progress_data = {
            'username': username,
            'current_index': current_index,
            'last_updated': nyc_time.isoformat()
        }

        progress_file = f"progress/{username}_progress.json"
        
        # Save to temporary file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as tmp_file:
            json.dump(progress_data, tmp_file, indent=2)
            tmp_path = tmp_file.name
        
        # Upload to HuggingFace
        if HF_TOKEN:
            hf_api.upload_file(
                path_or_fileobj=tmp_path,
                path_in_repo=progress_file,
                repo_id=hf_repo_id,
                repo_type="dataset",
                token=HF_TOKEN
            )
        
        # Clean up
        os.unlink(tmp_path)
        return True
        
    except Exception as e:
        print(f"Error saving progress for {username}: {e}")
        return False

def load_existing_annotation(username, data_source, case_id):
    """Load existing annotation for a specific case if it exists"""
    annotation_file = f"annotations/{data_source}/{username}.json"

    if os.path.exists(annotation_file):
        try:
            with open(annotation_file, 'r') as f:
                annotations = json.load(f)
                if case_id in annotations:
                    case_data = annotations[case_id]
                    # Return both annotation data and metadata
                    return {
                        'annotation': case_data.get('annotation', {}),
                        'has_problem': case_data.get('has_problem', False),
                        'feedback': case_data.get('feedback', '')
                    }
        except Exception as e:
            print(f"Error loading existing annotation: {e}")
    return None

def load_all_annotators_ratings(case_id):
    """Load ratings from annotators for adjudication"""
    # Add annotator usernames here for adjudication mode
    annotators = ['user1', 'user2']
    all_ratings = {}

    # Define the aspect mapping
    aspects = {
        'readability_jargon': 'readability_jargon',
        'narrative_order': 'narrative_order',
        'sentence_structure': 'sentence_structure',
        'formatting_layout': 'formatting_layout',
        'citation_style': 'citation_style'
    }

    data_source = get_data_source()
    for aspect_key, aspect_name in aspects.items():
        ratings = []
        for annotator in annotators:
            annotation_file = f"annotations/{data_source}/{annotator}.json"
            if os.path.exists(annotation_file):
                try:
                    with open(annotation_file, 'r') as f:
                        data = json.load(f)
                        if case_id in data:
                            annotation = data[case_id].get('annotation', {})
                            rating = annotation.get(aspect_name)
                            if rating is not None:
                                ratings.append(rating)
                except Exception as e:
                    print(f"Error loading ratings from {annotator}: {e}")

        # Randomize order to prevent bias
        if ratings:
            random.shuffle(ratings)
        all_ratings[aspect_key] = ratings

    return all_ratings

def load_adjudication_progress(username):
    """Load adjudication progress from HuggingFace"""
    try:
        progress_file = f"adjudicated_progress/{username}_progress.json"
        hf_repo_id = get_hf_repo_id()  # Will be pilot repo for adjudication

        # Try to download progress file from HuggingFace
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as tmp_file:
            tmp_path = tmp_file.name

        try:
            hf_api.hf_hub_download(
                repo_id=hf_repo_id,
                filename=progress_file,
                repo_type="dataset",
                token=HF_TOKEN,
                local_dir_use_symlinks=False,
                local_dir=os.path.dirname(tmp_path)
            )

            with open(os.path.join(os.path.dirname(tmp_path), progress_file), 'r') as f:
                progress = json.load(f)

            # Clean up
            os.unlink(tmp_path)
            return progress.get('current_index', 0)

        except Exception:
            # File doesn't exist or error downloading - start from 0
            os.unlink(tmp_path)
            return 0

    except Exception as e:
        print(f"Error loading adjudication progress for {username}: {e}")
        return 0

def save_adjudication_progress(username, current_index):
    """Save adjudication progress to HuggingFace"""
    try:
        hf_repo_id = get_hf_repo_id()

        # Get NYC Eastern timezone
        eastern = pytz.timezone('America/New_York')
        nyc_time = datetime.now(eastern)

        progress_data = {
            'username': username,
            'current_index': current_index,
            'mode': 'adjudication',
            'last_updated': nyc_time.isoformat()
        }

        progress_file = f"adjudicated_progress/{username}_progress.json"

        # Save to temporary file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as tmp_file:
            json.dump(progress_data, tmp_file, indent=2)
            tmp_path = tmp_file.name

        # Upload to HuggingFace
        if HF_TOKEN:
            hf_api.upload_file(
                path_or_fileobj=tmp_path,
                path_in_repo=progress_file,
                repo_id=hf_repo_id,
                repo_type="dataset",
                token=HF_TOKEN
            )

        # Clean up
        os.unlink(tmp_path)
        return True

    except Exception as e:
        print(f"Error saving adjudication progress for {username}: {e}")
        return False

def save_adjudicated_annotation(annotation_result):
    """Save adjudicated annotation to HuggingFace dataset"""
    try:
        hf_repo_id = get_hf_repo_id()

        # Get NYC Eastern timezone for filename
        eastern = pytz.timezone('America/New_York')
        nyc_time = datetime.now(eastern)
        timestamp = nyc_time.strftime('%Y%m%d_%H%M%S')

        username = annotation_result['username']
        session_id = annotation_result['session_id']
        filename = f"adjudicated_annotations/{username}_{session_id}_{timestamp}.json"

        # Save to temporary file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as tmp_file:
            json.dump(annotation_result, tmp_file, indent=2)
            tmp_path = tmp_file.name

        # Upload to HuggingFace
        if HF_TOKEN:
            hf_api.upload_file(
                path_or_fileobj=tmp_path,
                path_in_repo=filename,
                repo_id=hf_repo_id,
                repo_type="dataset",
                token=HF_TOKEN
            )

        # Clean up
        os.unlink(tmp_path)
        return True
    except Exception as e:
        print(f"Error saving adjudicated annotation to HuggingFace: {e}")
        return False

def load_existing_adjudication(username, case_id):
    """Load existing adjudication for a specific case if it exists"""
    annotation_file = f"adjudicated_annotations/{username}.json"

    if os.path.exists(annotation_file):
        try:
            with open(annotation_file, 'r') as f:
                annotations = json.load(f)
                if case_id in annotations:
                    case_data = annotations[case_id]
                    return {
                        'annotation': case_data.get('annotation', {}),
                        'feedback': case_data.get('feedback', '')
                    }
        except Exception as e:
            print(f"Error loading existing adjudication: {e}")
    return None

def check_reannotation_mode(username, data_source):
    """Check if user should be in re-annotation mode"""
    annotation_file = f"annotations/{data_source}/{username}.json"
    return os.path.exists(annotation_file)

def load_reannotation_progress(username):
    """Load re-annotation progress from HuggingFace"""
    try:
        progress_file = f"reannotations_progress/{username}_progress.json"
        hf_repo_id = get_hf_repo_id()

        # Try to download progress file from HuggingFace
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as tmp_file:
            tmp_path = tmp_file.name

        try:
            hf_api.hf_hub_download(
                repo_id=hf_repo_id,
                filename=progress_file,
                repo_type="dataset",
                token=HF_TOKEN,
                local_dir_use_symlinks=False,
                local_dir=os.path.dirname(tmp_path)
            )

            with open(os.path.join(os.path.dirname(tmp_path), progress_file), 'r') as f:
                progress = json.load(f)

            # Clean up
            os.unlink(tmp_path)
            return progress.get('current_index', 0)

        except Exception:
            # File doesn't exist or error downloading - start from 0
            os.unlink(tmp_path)
            return 0

    except Exception as e:
        print(f"Error loading re-annotation progress for {username}: {e}")
        return 0

def save_reannotation_progress(username, current_index):
    """Save re-annotation progress to HuggingFace"""
    try:
        hf_repo_id = get_hf_repo_id()

        # Get NYC Eastern timezone
        eastern = pytz.timezone('America/New_York')
        nyc_time = datetime.now(eastern)

        progress_data = {
            'username': username,
            'current_index': current_index,
            'mode': 'reannotation',
            'last_updated': nyc_time.isoformat()
        }

        progress_file = f"reannotations_progress/{username}_progress.json"

        # Save to temporary file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as tmp_file:
            json.dump(progress_data, tmp_file, indent=2)
            tmp_path = tmp_file.name

        # Upload to HuggingFace
        if HF_TOKEN:
            hf_api.upload_file(
                path_or_fileobj=tmp_path,
                path_in_repo=progress_file,
                repo_id=hf_repo_id,
                repo_type="dataset",
                token=HF_TOKEN
            )

        # Clean up
        os.unlink(tmp_path)
        return True

    except Exception as e:
        print(f"Error saving re-annotation progress for {username}: {e}")
        return False

def save_to_reannotations(annotation_result):
    """Save re-annotation result to HuggingFace dataset"""
    try:
        hf_repo_id = get_hf_repo_id()

        # Get NYC Eastern timezone for filename
        eastern = pytz.timezone('America/New_York')
        nyc_time = datetime.now(eastern)
        timestamp = nyc_time.strftime('%Y%m%d_%H%M%S')

        username = annotation_result['username']
        session_id = annotation_result['session_id']
        filename = f"reannotations/{username}_{session_id}_{timestamp}.json"

        # Save to temporary file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as tmp_file:
            json.dump(annotation_result, tmp_file, indent=2)
            tmp_path = tmp_file.name

        # Upload to HuggingFace
        if HF_TOKEN:
            hf_api.upload_file(
                path_or_fileobj=tmp_path,
                path_in_repo=filename,
                repo_id=hf_repo_id,
                repo_type="dataset",
                token=HF_TOKEN
            )

        # Clean up
        os.unlink(tmp_path)
        return True
    except Exception as e:
        print(f"Error saving re-annotation to HuggingFace: {e}")
        return False

def get_or_create_session_id():
    if 'session_id' not in session:
        session['session_id'] = str(uuid.uuid4())
    return session['session_id']

@app.route('/login', methods=['GET', 'POST'])
def login():
    # Get data source and adjudicated mode from URL parameters
    data_source = request.args.get('data', DEFAULT_DATA_SOURCE)
    adjudicated = request.args.get('adjudicated', 'false').lower() == 'true'

    # Validate data source
    if data_source not in DATA_CONFIG:
        data_source = DEFAULT_DATA_SOURCE

    # Adjudication mode requires multiple annotators to have completed annotations
    # Update the data_source check if using a different dataset for adjudication

    if request.method == 'POST':
        username = request.form.get('username', '').strip()

        if username in VALID_USERS:
            session['username'] = username
            session['data_source'] = data_source
            session['adjudicated'] = adjudicated

            # Auto-detect re-annotation mode
            is_reannotation = check_reannotation_mode(username, data_source)
            session['reannotation'] = is_reannotation

            # Load appropriate progress based on mode
            if adjudicated:
                current_index = load_adjudication_progress(username)
            elif is_reannotation:
                current_index = load_reannotation_progress(username)
            else:
                current_index = load_user_progress(username)
            session['current_index'] = current_index

            # Preserve parameters in redirect
            redirect_url = f'/?data={data_source}'
            if adjudicated:
                redirect_url += '&adjudicated=true'
            return redirect(redirect_url)
        else:
            return render_template('login.html',
                                 error="Username not found. Please contact the administrator.",
                                 data_source=data_source,
                                 adjudicated=adjudicated)

    return render_template('login.html', data_source=data_source, adjudicated=adjudicated)

def save_to_huggingface(annotation_result):
    """Save annotation result to HuggingFace dataset"""
    try:
        hf_repo_id = get_hf_repo_id()

        # Get NYC Eastern timezone for filename
        eastern = pytz.timezone('America/New_York')
        nyc_time = datetime.now(eastern)
        timestamp = nyc_time.strftime('%Y%m%d_%H%M%S')

        username = annotation_result['username']
        session_id = annotation_result['session_id']
        filename = f"annotations/{username}_{session_id}_{timestamp}.json"
        
        # Save to temporary file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as tmp_file:
            json.dump(annotation_result, tmp_file, indent=2)
            tmp_path = tmp_file.name
        
        # Upload to HuggingFace
        if HF_TOKEN:
            hf_api.upload_file(
                path_or_fileobj=tmp_path,
                path_in_repo=filename,
                repo_id=hf_repo_id,
                repo_type="dataset",
                token=HF_TOKEN
            )
        
        # Clean up
        os.unlink(tmp_path)
        return True
    except Exception as e:
        print(f"Error saving to HuggingFace: {e}")
        return False

@app.route('/')
def index():
    # Get parameters from URL if provided
    data_source_param = request.args.get('data')
    adjudicated_param = request.args.get('adjudicated', 'false').lower() == 'true'

    # Update session if parameters provided
    if data_source_param and data_source_param in DATA_CONFIG:
        session['data_source'] = data_source_param
    if adjudicated_param is not None:
        session['adjudicated'] = adjudicated_param

    # Get current mode
    adjudicated = session.get('adjudicated', False)
    data_source = get_data_source()

    # Adjudication mode requires existing annotations from multiple annotators
    # The mode will work with the current data source if annotations exist

    # Check if user is logged in
    if 'username' not in session:
        # Show preview with blurred content
        dummy_data = {
            'case_id': 'preview-case',
            'summary_A': 'This is a sample summary for demonstration purposes. It contains legal information about a case including facts, procedural history, and the court\'s decision.',
            'summary_B': 'This is another sample summary to show the comparison interface. It might have different structure and writing style compared to the first summary.',
            'current_index': 1,
            'total_pairs': 100
        }
        template = 'adjudication.html' if adjudicated else 'annotate.html'
        return render_template(template,
                             case_id=dummy_data['case_id'],
                             summary_a=dummy_data['summary_A'],
                             summary_b=dummy_data['summary_B'],
                             current_index=dummy_data['current_index'],
                             total_pairs=dummy_data['total_pairs'],
                             username=None,
                             data_source=data_source,
                             adjudicated=adjudicated,
                             preview_mode=True)

    username = session['username']

    # Auto-detect re-annotation mode
    is_reannotation = check_reannotation_mode(username, data_source) and not adjudicated
    session['reannotation'] = is_reannotation

    # Load user's annotation data
    annotation_data = load_annotation_data(username, data_source)
    if annotation_data is None:
        session.clear()
        return redirect(url_for('login'))

    session_id = get_or_create_session_id()

    # Load appropriate progress based on mode
    if 'current_index' not in session:
        if adjudicated:
            session['current_index'] = load_adjudication_progress(username)
        elif is_reannotation:
            session['current_index'] = load_reannotation_progress(username)
        else:
            session['current_index'] = load_user_progress(username)

    current_index = session['current_index']
    total_pairs = len(annotation_data['keys'])

    # Handle completion differently for re-annotation mode
    if current_index >= total_pairs:
        if is_reannotation:
            # Reset to beginning for re-annotation
            session['current_index'] = 0
            current_index = 0
            save_reannotation_progress(username, 0)
        else:
            return render_template('complete.html',
                                 username=username,
                                 data_source=data_source,
                                 adjudicated=adjudicated,
                                 reannotation=is_reannotation)

    # Get current case
    case_id = annotation_data['keys'][current_index]
    summary_pair = annotation_data['summary_pairs'][current_index]

    if adjudicated:
        # Load all annotators' ratings for adjudication
        all_ratings = load_all_annotators_ratings(case_id)

        # Check if we have enough ratings for adjudication
        has_enough_ratings = all([len(ratings) >= 2 for ratings in all_ratings.values()])

        if not has_enough_ratings:
            # Skip this case in adjudication mode
            session['current_index'] = current_index + 1
            if adjudicated:
                save_adjudication_progress(username, current_index + 1)
            else:
                save_user_progress(username, current_index + 1)
            return redirect('/')

        # Load existing adjudication if it exists
        existing_adjudication = load_existing_adjudication(username, case_id)

        return render_template('adjudication.html',
                             case_id=case_id,
                             summary_a=summary_pair['summary_A'],
                             summary_b=summary_pair['summary_B'],
                             current_index=current_index + 1,
                             total_pairs=total_pairs,
                             username=username,
                             data_source=data_source,
                             all_ratings=all_ratings,
                             existing_annotation=existing_adjudication,
                             preview_mode=False)
    else:
        # Regular annotation mode or re-annotation mode
        existing_annotation = load_existing_annotation(username, data_source, case_id)

        return render_template('annotate.html',
                             case_id=case_id,
                             summary_a=summary_pair['summary_A'],
                             summary_b=summary_pair['summary_B'],
                             current_index=current_index + 1,
                             total_pairs=total_pairs,
                             username=username,
                             data_source=data_source,
                             existing_annotation=existing_annotation,
                             reannotation=is_reannotation,
                             preview_mode=False)

@app.route('/submit', methods=['POST'])
def submit_annotation():
    # Check if user is logged in
    if 'username' not in session:
        return jsonify({'success': False, 'error': 'Not logged in'})

    username = session['username']
    data_source = get_data_source()
    adjudicated = session.get('adjudicated', False)
    is_reannotation = session.get('reannotation', False)

    annotation_data = load_annotation_data(username, data_source)
    if annotation_data is None:
        return jsonify({'success': False, 'error': 'Invalid user data'})

    data = request.json
    session_id = get_or_create_session_id()
    current_index = session.get('current_index', 0)

    # Get NYC Eastern timezone
    eastern = pytz.timezone('America/New_York')
    nyc_time = datetime.now(eastern)

    if adjudicated:
        # Prepare adjudicated annotation result (only ratings)
        annotation_result = {
            'username': username,
            'session_id': session_id,
            'timestamp': nyc_time.isoformat(),
            'index': current_index,
            'key': annotation_data['keys'][current_index],
            'annotation': {
                'readability_jargon': data.get('annotation', {}).get('readability_jargon'),
                'narrative_order': data.get('annotation', {}).get('narrative_order'),
                'sentence_structure': data.get('annotation', {}).get('sentence_structure'),
                'formatting_layout': data.get('annotation', {}).get('formatting_layout'),
                'citation_style': data.get('annotation', {}).get('citation_style')
            },
            'feedback': data.get('feedback', ''),
            'time_spent_seconds': data.get('time_spent_seconds', 0),
            'mode': 'adjudication'
        }

        # Save to HuggingFace
        success = save_adjudicated_annotation(annotation_result)

        # Move to next item and save progress
        new_index = current_index + 1
        session['current_index'] = new_index
        save_adjudication_progress(username, new_index)
    elif is_reannotation:
        # Re-annotation submission
        annotation_result = {
            'username': username,
            'session_id': session_id,
            'timestamp': nyc_time.isoformat(),
            'index': current_index,
            'key': annotation_data['keys'][current_index],
            'annotation': data.get('annotation'),
            'has_problem': data.get('has_problem', False),
            'feedback': data.get('feedback', ''),
            'time_spent_seconds': data.get('time_spent_seconds', 0),
            'mode': 'reannotation'
        }

        # Save to reannotations folder
        success = save_to_reannotations(annotation_result)

        # Move to next item and save re-annotation progress
        new_index = current_index + 1
        session['current_index'] = new_index
        save_reannotation_progress(username, new_index)
    else:
        # Prepare regular annotation result
        annotation_result = {
            'username': username,
            'session_id': session_id,
            'timestamp': nyc_time.isoformat(),
            'index': current_index,
            'key': annotation_data['keys'][current_index],
            'order': annotation_data.get('orders', [None] * len(annotation_data['keys']))[current_index],
            'annotation': data.get('annotation'),
            'has_problem': data.get('has_problem', False),
            'feedback': data.get('feedback', ''),
            'time_spent_seconds': data.get('time_spent_seconds', 0)
        }

        # Save to HuggingFace
        success = save_to_huggingface(annotation_result)

        # Move to next item and save progress
        new_index = current_index + 1
        session['current_index'] = new_index
        save_user_progress(username, new_index)

    return jsonify({'success': success})

@app.route('/skip', methods=['POST'])
def skip_annotation():
    # Check if user is logged in
    if 'username' not in session:
        return jsonify({'success': False, 'error': 'Not logged in'})

    username = session['username']
    data_source = get_data_source()
    adjudicated = session.get('adjudicated', False)

    annotation_data = load_annotation_data(username, data_source)
    if annotation_data is None:
        return jsonify({'success': False, 'error': 'Invalid user data'})

    data = request.json if request.json else {}
    session_id = get_or_create_session_id()
    current_index = session.get('current_index', 0)

    # Get NYC Eastern timezone
    eastern = pytz.timezone('America/New_York')
    nyc_time = datetime.now(eastern)

    # Save skip information with time tracking
    skip_result = {
        'username': username,
        'session_id': session_id,
        'timestamp': nyc_time.isoformat(),
        'index': current_index,
        'key': annotation_data['keys'][current_index] if current_index < len(annotation_data['keys']) else None,
        'annotation': {'skipped': True},
        'time_spent_seconds': data.get('time_spent_seconds', 0)
    }

    # Check if in re-annotation mode
    is_reannotation = session.get('reannotation', False)

    if adjudicated:
        skip_result['mode'] = 'adjudication'
        # Save skip record to HuggingFace
        save_adjudicated_annotation(skip_result)

        # Move to next item and save progress
        new_index = current_index + 1
        session['current_index'] = new_index
        save_adjudication_progress(username, new_index)
    elif is_reannotation:
        skip_result['order'] = annotation_data.get('orders', [None] * len(annotation_data['keys']))[current_index]
        skip_result['mode'] = 're-annotation'
        # Save skip record to re-annotations
        save_to_reannotations(skip_result)

        # Move to next item and save progress
        new_index = current_index + 1
        session['current_index'] = new_index
        save_reannotation_progress(username, new_index)
    else:
        skip_result['order'] = annotation_data.get('orders', [None] * len(annotation_data['keys']))[current_index]
        # Save skip record to HuggingFace
        save_to_huggingface(skip_result)

        # Move to next item and save progress
        new_index = current_index + 1
        session['current_index'] = new_index
        save_user_progress(username, new_index)

    return jsonify({'success': True})

@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('login'))

if __name__ == '__main__':
    app.run(debug=True)