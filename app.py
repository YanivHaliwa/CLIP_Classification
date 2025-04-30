#!/usr/bin/env python3
import os
import torch
import clip
import uuid
import nltk
import gc  # For garbage collection
from PIL import Image
from flask import Flask, render_template, request, redirect, url_for, flash, session
from werkzeug.utils import secure_filename
import time  # For sleep between GPU operations
# Use NLTK's WordNet to generate object categories
from nltk.corpus import wordnet as wn

# Set PyTorch memory allocation configuration to avoid fragmentation
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

# Download NLTK data if not already present
try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet')
    nltk.download('omw-1.4')


def get_top_level_categories():
    # These are the "unique beginners" — top-level noun synsets
    top_synsets = wn.synset('entity.n.01').hyponyms()
    
    # Return the lemma names as human-readable categories
    return sorted(set(lemma.name().replace('_', ' ') for syn in top_synsets for lemma in syn.lemmas()))

def get_default_labels(max_labels=6000):
    """Generate default labels using NLTK WordNet"""
    # Categories we want to focus on
    interesting_categories = get_top_level_categories()
    
    # Add additional categories that might not be in the top level
    additional_categories = [
        'animal', 'plant', 'food', 'furniture', 'artifact', 
        'vehicle', 'building', 'person', 'location', 'object',
        'device', 'tool', 'clothing', 'structure', 'natural_object'
    ]
    
    # Combine both sets of categories
    interesting_categories.extend([cat for cat in additional_categories 
                                if cat not in interesting_categories])
    # Collect words from these categories
    labels = set()
    
    # Get common nouns from WordNet
    for category in interesting_categories:
        for synset in wn.synsets(category, pos=wn.NOUN):
            # Get hyponyms (subcategories)
            for hyponym in synset.hyponyms():
                # Get words from each synset
                for lemma in hyponym.lemmas():
                    word = lemma.name().replace('_', ' ')
                    # Only add words that are 3-20 letters and don't contain digits
                    if 3 <= len(word) <= 20 and word.isalpha():
                        labels.add(word)
                        
                # Get words from each hyponym's hyponyms (go deeper)
                for hypo_hypo in hyponym.hyponyms():
                    for lemma in hypo_hypo.lemmas():
                        word = lemma.name().replace('_', ' ')
                        if 3 <= len(word) <= 20 and word.isalpha():
                            labels.add(word)
    
    #Ensure we have most common objects
    must_include = [
        "person", "man", "woman", "child", "dog", "cat", "car", "tree", 
        "house", "building", "computer", "phone", "table", "chair", 
        "food", "book", "clothing", "water", "sky", "mountain", "water", 
        "ocean", "lake", "bed", "door", "window", "light", "fire", "smoke",
        "earth", "sun", "moon", "star", "cloud", "tree", "flower", "leaf", "tv","unbrella"
    ]
    
    for word in must_include:
        labels.add(word)
    
    # Convert to list and limit size
    labels_list = list(labels)
    if len(labels_list) > max_labels:
        labels_list = labels_list[:max_labels]
    
    return sorted(labels_list)

# Generate default labels
DEFAULT_LABELS = get_default_labels(6000)

# Configure application
app = Flask(__name__)
app.config['SECRET_KEY'] = os.urandom(24)  # Generate a secure random key
app.config['SESSION_TYPE'] = 'filesystem'  # Store sessions on filesystem for persistence
app.config['UPLOAD_FOLDER'] = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'static/uploads')
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg', 'gif'}

# Ensure upload directory exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Initialize CLIP model
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = None, None
 

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def load_model(force_cpu=False):
    """
    Load the CLIP model on the specified device.
    
    Args:
        force_cpu (bool): Force the model to run on CPU even if GPU is available
        
    Returns:
        tuple: (model, preprocess) CLIP model and preprocessing function
    """
    # Use CPU if forced or if GPU isn't available
    actual_device = "cpu" if force_cpu else device
    
    # Before loading model, clean up any existing GPU memory
    if torch.cuda.is_available() and actual_device == "cuda":
        # Force garbage collection
        gc.collect()
        # Empty CUDA cache
        torch.cuda.empty_cache()
        
        # Report available memory
        total_memory = torch.cuda.get_device_properties(0).total_memory
        reserved = torch.cuda.memory_reserved(0)
        allocated = torch.cuda.memory_allocated(0)
        free_memory = total_memory - reserved
        # print(f"GPU memory: {free_memory/1024**3:.2f} GB available out of {total_memory/1024**3:.2f} GB total")
    
    # Load the model
    model, preprocess = clip.load("ViT-B/32", device=actual_device)
    print(f"Model loaded on: {actual_device}")
    return model, preprocess

@app.route('/')
def index():
    # Clear any previous session data
    session.pop('uploaded_images', None)
    session.pop('labels', None)
    return render_template('index.html')

@app.route('/process', methods=['POST'])
def process():
    """Processes the image and labels outside of the redirect cycle"""
    print("*** Process route called ***")
    
    # Check if form contains files
    if 'files[]' not in request.files:
        print("No files[] in request.files")
        flash('No files selected')
        return render_template('index.html')
    
    files = request.files.getlist('files[]')
    print(f"Number of files in request: {len(files)}")
    
    # Check if user selected at least one file
    if not files or files[0].filename == '':
        print(f"First file empty: {files[0].filename == ''}")
        flash('No files selected')
        return render_template('index.html')
    
    # Get labels from form
    labels = request.form.get('labels', '').strip()
    using_default_labels = False
    
    # Process custom labels if provided, otherwise use defaults
    if labels:
        # Process labels and remove empty ones
        labels_list = [label.strip() for label in labels.split(',') if label.strip()]
        
        # If no valid labels after cleaning, use defaults
        if not labels_list:
            labels_list = DEFAULT_LABELS
            using_default_labels = True
            flash('No valid labels provided. Using default labels.')
    else:
        # No labels provided, use defaults
        labels_list = DEFAULT_LABELS
        using_default_labels = True
        flash('No labels provided. Using default labels.')
    
    # Save uploaded files
    uploaded_files = []
    valid_files_found = False
    print("Starting to process files...")
    for file in files:
        print(f"Processing file: {file.filename}")
        if file and allowed_file(file.filename):
            valid_files_found = True
            # Create unique filename
            filename = secure_filename(file.filename)
            unique_filename = f"{uuid.uuid4().hex}_{filename}"
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
            
            print(f"Saving file to: {file_path}")
            # Save file
            file.save(file_path)
            
            # Add to uploaded files list
            uploaded_files.append({
                'original_name': filename,
                'path': os.path.join('static/uploads', unique_filename),
                'full_path': file_path
            })
        else:
            print(f"Skipping file {file.filename}, allowed: {allowed_file(file.filename) if file else False}")
    
    # Check if we actually saved any valid files
    if not valid_files_found:
        flash('No valid image files selected. Please select PNG, JPG, JPEG or GIF files.')
        return render_template('index.html')
        
    # Now process the images directly without relying on session
    print(f"Processing {len(uploaded_files)} images with {len(labels_list)} labels")
    
    # Process each image individually to avoid GPU memory issues
    results = {}
    classifications = {}
    
    # Create text inputs for CLIP - just create once as they're the same for all images
    text_inputs = [f"a photo of {label}" for label in labels_list]
    
    # Only force CPU if we really need to (extreme case with too many labels)
    force_cpu_threshold = 20000  # Much higher threshold since we're processing one image at a time
    force_global_cpu = len(labels_list) > force_cpu_threshold
    
    if force_global_cpu:
        print(f"Using CPU mode due to extremely large number of labels ({len(labels_list)} > {force_cpu_threshold})")
    
    print(f"Processing {len(uploaded_files)} images individually on {'CPU' if force_global_cpu else 'GPU'}")
    
    # Process text tokens once if we're always using CPU
    if force_global_cpu:
        model, preprocess = load_model(force_cpu=True)
        text_tokens = clip.tokenize(text_inputs).to("cpu")
        with torch.no_grad():
            text_features = model.encode_text(text_tokens)
            text_features /= text_features.norm(dim=-1, keepdim=True)
    
    # Process each image individually
    for i, img_data in enumerate(uploaded_files):
        try:
            print(f"Processing image {i+1}/{len(uploaded_files)}: {img_data['original_name']}")
            image_path = img_data['full_path']
            
            # Use GPU for this image if available and not globally forced to CPU
            use_gpu_for_image = not force_global_cpu and device == "cuda"
            
            # Load/reload model for each image if using GPU
            if use_gpu_for_image or (i == 0 and force_global_cpu):
                # If we're using GPU, load fresh model for each image
                # If using CPU globally, load only once for the first image
                if use_gpu_for_image:
                    print(f"  Using GPU for this image")
                    model, preprocess = load_model(force_cpu=False)
                    # Create new text features for this image
                    text_tokens = clip.tokenize(text_inputs).to("cuda")
                    with torch.no_grad():
                        text_features = model.encode_text(text_tokens)
                        text_features /= text_features.norm(dim=-1, keepdim=True)
            
            # Determine device for this specific image
            actual_device = "cuda" if use_gpu_for_image else "cpu"
            
            # Load and preprocess image
            image = Image.open(image_path)
            processed_image = preprocess(image).unsqueeze(0).to(actual_device)
            
            # Get prediction
            with torch.no_grad():
                # Calculate features
                image_features = model.encode_image(processed_image)
                
                # Normalize features
                image_features /= image_features.norm(dim=-1, keepdim=True)
                
                # Calculate similarity
                similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)
                values, indices = similarity[0].topk(min(len(labels_list), 5))
            
            # Get results
            image_results = []
            for value, index in zip(values, indices):
                label = labels_list[index]
                confidence = value.item() * 100
                image_results.append({
                    'label': label,
                    'confidence': confidence
                })
            
            # Store results
            results[img_data['path']] = image_results
            
            # Get top class for grouping
            if image_results:  # Make sure we have results
                top_label = image_results[0]['label'] if image_results[0]['label'] else "Unlabeled"
                if top_label not in classifications:
                    classifications[top_label] = []
                classifications[top_label].append({
                    'path': img_data['path'],
                    'confidence': image_results[0]['confidence']
                })
            
            # Clean up GPU memory if we used it for this image
            if use_gpu_for_image:
                # Explicitly delete variables that hold tensors
                del processed_image, image_features, similarity, values, indices
                
                # Clear CUDA cache to free up memory
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    print(f"  GPU memory cleared for next image")
            
        except Exception as e:
            print(f"Error processing image {image_path}: {e}")
    
    # Render results page directly without redirecting
    return render_template('results.html', 
                           results=results, 
                           classifications=classifications,
                           labels=labels_list,
                           using_default_labels=using_default_labels)

@app.route('/upload', methods=['POST'])
def upload():
    """Simple redirect to the process route"""
    return redirect(url_for('process'))

@app.route('/classify')
def classify():
    print("*** Classify route called ***")
    print(f"Session keys: {list(session.keys())}")
    
    # Check if we have images and labels in session
    if 'uploaded_images' not in session:
        print("'uploaded_images' not in session")
        flash('No images found in session')
        return redirect(url_for('index'))
        
    if 'labels' not in session:
        print("'labels' not in session")
        flash('No labels found in session')
        return redirect(url_for('index'))
    
    # Get uploaded images and labels from session
    uploaded_images = session.get('uploaded_images', [])
    labels = session.get('labels', [])
    
    # Double-check that we have images
    if not uploaded_images:
        flash('No images found. Please upload some images.')
        return redirect(url_for('index'))
    
    # Process each image individually to avoid GPU memory issues
    results = {}
    classifications = {}
    
    # Prepare text tokens for all labels
    # Make sure each label is not empty and properly formatted
    cleaned_labels = [label.strip() for label in labels if isinstance(label, str) and label.strip()]
    
    # If no valid labels, use default labels instead of redirecting
    if not cleaned_labels:
        cleaned_labels = DEFAULT_LABELS
        session['labels'] = cleaned_labels
        session['using_default_labels'] = True
        flash('Using default AI-generated labels for classification.')
    
    # Update labels in session
    labels = cleaned_labels
    session['labels'] = labels
    
    # Create text inputs for CLIP
    text_inputs = [f"a photo of {label}" for label in labels]
    
    # Only force CPU if we really need to (extreme case with too many labels)
    force_cpu_threshold = 20000  # Much higher threshold since we're processing one image at a time
    force_global_cpu = len(labels) > force_cpu_threshold
    
    if force_global_cpu:
        print(f"Using CPU mode due to extremely large number of labels ({len(labels)} > {force_cpu_threshold})")
    
    print(f"Processing {len(uploaded_images)} images individually on {'CPU' if force_global_cpu else 'GPU'}")
    
    # Process text tokens once if we're always using CPU
    if force_global_cpu:
        model, preprocess = load_model(force_cpu=True)
        text_tokens = clip.tokenize(text_inputs).to("cpu")
        with torch.no_grad():
            text_features = model.encode_text(text_tokens)
            text_features /= text_features.norm(dim=-1, keepdim=True)
    
    # Process each image individually
    for i, img_data in enumerate(uploaded_images):
        try:
            print(f"Processing image {i+1}/{len(uploaded_images)}: {img_data['original_name'] if 'original_name' in img_data else 'Image'}")
            image_path = img_data['full_path']
            
            # Use GPU for this image if available and not globally forced to CPU
            use_gpu_for_image = not force_global_cpu and device == "cuda"
            
            # Load/reload model for each image if using GPU
            if use_gpu_for_image or (i == 0 and force_global_cpu):
                # If we're using GPU, load fresh model for each image
                # If using CPU globally, load only once for the first image
                if use_gpu_for_image:
                    print(f"  Using GPU for this image")
                    model, preprocess = load_model(force_cpu=False)
                    # Create new text features for this image
                    text_tokens = clip.tokenize(text_inputs).to("cuda")
                    with torch.no_grad():
                        text_features = model.encode_text(text_tokens)
                        text_features /= text_features.norm(dim=-1, keepdim=True)
            
            # Determine device for this specific image
            actual_device = "cuda" if use_gpu_for_image else "cpu"
            
            # Load and preprocess image
            image = Image.open(image_path)
            processed_image = preprocess(image).unsqueeze(0).to(actual_device)
            
            # Get prediction
            with torch.no_grad():
                # Calculate features
                image_features = model.encode_image(processed_image)
                
                # Normalize features
                image_features /= image_features.norm(dim=-1, keepdim=True)
                
                # Calculate similarity
                similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)
                values, indices = similarity[0].topk(min(len(labels), 5))
            
            # Get results
            image_results = []
            for value, index in zip(values, indices):
                label = labels[index]
                confidence = value.item() * 100
                image_results.append({
                    'label': label,
                    'confidence': confidence
                })
            
            # Store results
            results[img_data['path']] = image_results
            
            # Get top class for grouping
            if image_results:  # Make sure we have results
                top_label = image_results[0]['label'] if image_results[0]['label'] else "Unlabeled"
                if top_label not in classifications:
                    classifications[top_label] = []
                classifications[top_label].append({
                    'path': img_data['path'],
                    'confidence': image_results[0]['confidence']
                })
            
            # Clean up GPU memory if we used it for this image
            if use_gpu_for_image:
                # Explicitly delete variables that hold tensors
                del processed_image, image_features, similarity, values, indices
                
                # Clear CUDA cache to free up memory
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    print(f"  GPU memory cleared for next image")
            
        except Exception as e:
            print(f"Error processing image {image_path}: {e}")
    
    # Render results page
    return render_template('results.html', 
                          results=results, 
                          classifications=classifications,
                          labels=labels,
                          using_default_labels=session.get('using_default_labels', False))

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)