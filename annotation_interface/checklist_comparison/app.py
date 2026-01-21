import os
import json
from flask import Flask, render_template, request, jsonify, session, redirect, url_for
from datetime import datetime
import uuid
from huggingface_hub import HfApi, HfFolder
import tempfile

app = Flask(__name__)
app.secret_key = os.environ.get('SECRET_KEY', 'dev-key-change-in-production')

# Initialize HuggingFace API
hf_api = HfApi()
HF_TOKEN = os.environ.get('HF_TOKEN')
HF_REPO_ID = 'your-username/your-repo-name'  # Change to your HuggingFace dataset repo

# User to data file mapping
USER_DATA_MAPPING = {
    'user1': 'data/example.json',
    'user2': 'data/example.json',
    # Add more users as needed
}

def load_annotation_data(username):
    """Load annotation data for specific user"""
    data_file = USER_DATA_MAPPING.get(username)
    if not data_file or not os.path.exists(data_file):
        return None
    
    with open(data_file, 'r') as f:
        return json.load(f)

def load_user_progress(username):
    """Load user progress from HuggingFace"""
    try:
        progress_file = f"progress/{username}_progress.json"
        
        # Try to download progress file from HuggingFace
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as tmp_file:
            tmp_path = tmp_file.name
        
        try:
            hf_api.hf_hub_download(
                repo_id=HF_REPO_ID,
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
        progress_data = {
            'username': username,
            'current_index': current_index,
            'last_updated': datetime.now().isoformat()
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
                repo_id=HF_REPO_ID,
                repo_type="dataset",
                token=HF_TOKEN
            )
        
        # Clean up
        os.unlink(tmp_path)
        return True
        
    except Exception as e:
        print(f"Error saving progress for {username}: {e}")
        return False

def get_or_create_session_id():
    if 'session_id' not in session:
        session['session_id'] = str(uuid.uuid4())
    return session['session_id']

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form.get('username', '').strip()
        
        if username in USER_DATA_MAPPING:
            session['username'] = username
            
            # Load user's progress from HuggingFace
            current_index = load_user_progress(username)
            session['current_index'] = current_index
            
            return redirect('/')
        else:
            return render_template('login.html', error="Username not found. Please contact the administrator.")
    
    return render_template('login.html')

def save_to_huggingface(annotation_result):
    """Save annotation result to HuggingFace dataset"""
    try:
        # Create a filename with timestamp and username
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
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
                repo_id=HF_REPO_ID,
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
    # Check if user is logged in
    if 'username' not in session:
        # Show preview with blurred content
        # Use dummy data for preview
        dummy_data = {
            'case_id': 'preview-case',
            'checklist_item': 'Sample Checklist Category',
            'item_A': 'This is a sample text for demonstration purposes.',
            'item_B': 'This is another sample text to show the comparison interface.',
            'is_list': False,
            'current_index': 1,
            'total_pairs': 100
        }
        return render_template('annotate.html', 
                             case_id=dummy_data['case_id'],
                             checklist_item=dummy_data['checklist_item'],
                             item_a=dummy_data['item_A'],
                             item_b=dummy_data['item_B'],
                             is_list=dummy_data['is_list'],
                             current_index=dummy_data['current_index'],
                             total_pairs=dummy_data['total_pairs'],
                             username=None,
                             preview_mode=True)
    
    username = session['username']
    
    # Load user's annotation data
    annotation_data = load_annotation_data(username)
    if annotation_data is None:
        session.clear()
        return redirect(url_for('login'))
    
    session_id = get_or_create_session_id()
    if 'current_index' not in session:
        # Load progress from HuggingFace
        session['current_index'] = load_user_progress(username)
    
    current_index = session['current_index']
    total_pairs = len(annotation_data['keys'])
    
    if current_index >= total_pairs:
        return render_template('complete.html', username=username)
    
    # Get current annotation pair
    key = annotation_data['keys'][current_index]
    value_pair = annotation_data['value_pairs'][current_index]
    
    return render_template('annotate.html', 
                         case_id=key['case_id'],
                         checklist_item=key['checklist_item'],
                         item_a=value_pair['item_A'],
                         item_b=value_pair['item_B'],
                         is_list=value_pair['is_list'],
                         current_index=current_index + 1,
                         total_pairs=total_pairs,
                         username=username,
                         preview_mode=False)

@app.route('/submit', methods=['POST'])
def submit_annotation():
    # Check if user is logged in
    if 'username' not in session:
        return jsonify({'success': False, 'error': 'Not logged in'})
    
    username = session['username']
    annotation_data = load_annotation_data(username)
    if annotation_data is None:
        return jsonify({'success': False, 'error': 'Invalid user data'})
    
    data = request.json
    session_id = get_or_create_session_id()
    current_index = session.get('current_index', 0)
    
    # Prepare annotation result
    annotation_result = {
        'username': username,
        'session_id': session_id,
        'timestamp': datetime.now().isoformat(),
        'index': current_index,
        'key': annotation_data['keys'][current_index],
        'order': annotation_data['orders'][current_index],
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
    annotation_data = load_annotation_data(username)
    if annotation_data is None:
        return jsonify({'success': False, 'error': 'Invalid user data'})
    
    data = request.json if request.json else {}
    session_id = get_or_create_session_id()
    current_index = session.get('current_index', 0)
    
    # Save skip information with time tracking
    skip_result = {
        'username': username,
        'session_id': session_id,
        'timestamp': datetime.now().isoformat(),
        'index': current_index,
        'key': annotation_data['keys'][current_index] if current_index < len(annotation_data['keys']) else None,
        'order': annotation_data['orders'][current_index] if current_index < len(annotation_data['orders']) else None,
        'annotation': {'skipped': True},
        'time_spent_seconds': data.get('time_spent_seconds', 0)
    }
    
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