from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import pandas as pd
import numpy as np
import os
import traceback
import time
import requests


app = Flask(__name__)
CORS(app)

# Hugging Face Configuration - Using NEW Router API
HF_API_TOKEN = os.getenv('HF_API_TOKEN')
HF_API_URL = "https://api-inference.huggingface.co/models/google/flan-t5-base"  # Fast, reliable, free model

if not HF_API_TOKEN:
    print("⚠️ WARNING: HF_API_TOKEN environment variable not set!")
    print("   LLM reasoning will be disabled. Get your free token at:")
    print("   https://huggingface.co/settings/tokens")
    print("   Then set it with: export HF_API_TOKEN='your-token-here'")
    llm_enabled = False
else:
    llm_enabled = True
    print("✅ Hugging Face LLM configured successfully!")

# Rate limiting: track last request time
last_llm_request_time = 0
MIN_REQUEST_INTERVAL = 0.5  # Minimum 0.5 second between requests


MODEL_PATH = 'best_eladfp_model.pkl'
if os.path.exists(MODEL_PATH):
    model = joblib.load(MODEL_PATH)
    print("ELADFP Ensemble Model loaded successfully!")
else:
    print(f"❌ ERROR: {MODEL_PATH} not found! Please run the training script first.")
    exit(1)

INPUT_FEATURES = [
    'profile pic', 'nums/length username', 'fullname words', 'nums/length fullname',
    'name==username', 'description length', 'external URL', 'private',
    '#posts', '#followers', '#following'
]

MODEL_FEATURES = [
    'profile pic', 'nums/length username', 'fullname words', 'nums/length fullname',
    'name==username', 'description length', 'external URL', 'private',
    'log_posts', 'log_followers', 'log_following', 'followers_to_following', 'posts_per_follower'
]


def preprocess_features(input_df):
    df = input_df.copy()

    # Log transforms
    df['log_followers'] = np.log1p(df['#followers'])
    df['log_following'] = np.log1p(df['#following'])
    df['log_posts'] = np.log1p(df['#posts'])

    # Ratios
    df['followers_to_following'] = df['#followers'] / (df['#following'] + 1)
    df['posts_per_follower'] = df['#posts'] / (df['#followers'] + 1)

    # Clip extreme values
    df['followers_to_following'] = np.clip(df['followers_to_following'], 0, 10)
    df['posts_per_follower'] = np.clip(df['posts_per_follower'], 0, 5)

    # Drop unused raw count columns
    df.drop(columns=['#followers', '#following', '#posts'], inplace=True)

    # Reorder columns to match training
    df = df[MODEL_FEATURES]

    print("\n🧹 Preprocessed input:")
    print(df.to_string(index=False))
    return df


def generate_llm_reasoning(features, prediction, confidence):
    """
    Generate AI reasoning using Hugging Face LLM with rate limiting and retry logic.
    If LLM fails, returns fallback reasoning - prediction still works!
    """
    global last_llm_request_time
    
    # Extract values first for fallback
    posts = features.get('#posts', 0)
    followers = features.get('#followers', 0)
    following = features.get('#following', 0)
    nums_ratio = features.get('nums/length username', 0)
    bio_len = features.get('description length', 0)
    
    # If LLM not configured, use fallback immediately
    if not llm_enabled:
        print("⚠️ LLM not configured - using fallback reasoning")
        return _get_fallback_reasoning(prediction, posts, followers, following, nums_ratio, bio_len)
    
    try:
        print("\n🤖 Generating LLM reasoning...")
        
        # Rate limiting: ensure minimum interval between requests
        current_time = time.time()
        time_since_last_request = current_time - last_llm_request_time
        if time_since_last_request < MIN_REQUEST_INTERVAL:
            sleep_time = MIN_REQUEST_INTERVAL - time_since_last_request
            print(f"⏳ Rate limiting: waiting {sleep_time:.2f}s before next request...")
            time.sleep(sleep_time)
        
        last_llm_request_time = time.time()

        # Build concise prompt for the LLM (FLAN-T5 format)
        prompt = f"""Explain why this Instagram profile is {"fake" if prediction == 1 else "real"}: {int(posts)} posts, {int(followers)} followers, {int(following)} following, username is {int(nums_ratio * 100)}% numbers, bio has {int(bio_len)} characters, {"has" if features.get('profile pic', 0) == 1 else "no"} picture."""

        print(f"📝 Sending prompt to Hugging Face...")
        
        headers = {
            "Authorization": f"Bearer {HF_API_TOKEN}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "inputs": prompt,
            "parameters": {
                "max_new_tokens": 80,
                "temperature": 0.7,
                "do_sample": True
            },
            "options": {
                "use_cache": False,
                "wait_for_model": True
            }
        }
        
        # Retry logic with exponential backoff
        max_retries = 3
        for attempt in range(max_retries):
            try:
                response = requests.post(
                    HF_API_URL,
                    headers=headers,
                    json=payload,
                    timeout=25
                )
                
                if response.status_code == 200:
                    result = response.json()
                    
                    # Handle different response formats
                    if isinstance(result, list) and len(result) > 0:
                        generated_text = result[0].get('generated_text', '').strip()
                    elif isinstance(result, dict):
                        generated_text = result.get('generated_text', '').strip()
                    else:
                        generated_text = str(result).strip()
                    
                    if generated_text and len(generated_text) > 10:
                        # Clean up the response
                        generated_text = ' '.join(generated_text.split())[:250]
                        print(f"✅ LLM response: {generated_text}")
                        return generated_text
                    else:
                        print("⚠️ LLM returned empty/short response, using fallback")
                        return _get_fallback_reasoning(prediction, posts, followers, following, nums_ratio, bio_len)
                
                elif response.status_code == 503:
                    # Model is loading
                    print(f"⚠️ Model loading, retry {attempt + 1}/{max_retries}...")
                    if attempt < max_retries - 1:
                        time.sleep(8)  # Wait for model to load
                        continue
                    else:
                        print("⚠️ Model still loading after retries, using fallback")
                        return _get_fallback_reasoning(prediction, posts, followers, following, nums_ratio, bio_len)
                
                else:
                    error_msg = f"Status {response.status_code}: {response.text[:150]}"
                    print(f"⚠️ API error: {error_msg}")
                    if attempt < max_retries - 1:
                        time.sleep(2 ** attempt)
                        continue
                    raise Exception(error_msg)
                    
            except requests.exceptions.RequestException as req_error:
                if attempt < max_retries - 1:
                    wait_time = (2 ** attempt)  # Exponential backoff: 1s, 2s, 4s
                    print(f"⚠️ Request failed, retry {attempt + 1}/{max_retries}, waiting {wait_time}s...")
                    time.sleep(wait_time)
                else:
                    raise req_error

    except Exception as e:
        error_msg = str(e)
        print(f"⚠️ LLM API failed: {error_msg[:150]}")
        
        # Check error type
        if "429" in error_msg or "rate limit" in error_msg.lower():
            print("⚠️ Rate limit exceeded - using fallback reasoning")
        elif "401" in error_msg or "403" in error_msg or "authentication" in error_msg.lower():
            print("⚠️ Authentication failed - check your HF_API_TOKEN")
            print("   Get your token at: https://huggingface.co/settings/tokens")
        else:
            print(f"⚠️ LLM error: {error_msg}")
        
        # Return fallback reasoning instead of crashing
        return _get_fallback_reasoning(prediction, posts, followers, following, nums_ratio, bio_len)


def _get_fallback_reasoning(prediction, posts, followers, following, nums_ratio, bio_len):
    """Generate fallback reasoning based on profile features when Gemini unavailable"""
    if prediction == 1:
        # Fake profile reasoning
        reasons = []
        if nums_ratio > 0.5:
            reasons.append("high number ratio in username")
        if followers == 0:
            reasons.append("no followers")
        if posts == 0:
            reasons.append("no posts")
        if bio_len == 0:
            reasons.append("no bio")
        
        if reasons:
            return f"Profile classified as fake due to: {', '.join(reasons)}."
        else:
            return "Profile exhibits characteristics typical of fake accounts based on engagement patterns and account metadata."
    else:
        # Real profile reasoning
        reasons = []
        if posts > 5:
            reasons.append("active posting history")
        if followers > 0 and bio_len > 0:
            reasons.append("complete profile information")
        if nums_ratio < 0.3:
            reasons.append("natural username pattern")
        
        if reasons:
            return f"Profile appears legitimate with {', '.join(reasons)}."
        else:
            return "Profile exhibits characteristics typical of authentic accounts based on engagement patterns and account metadata."

# ==============================================================
# Prediction Endpoint
# ==============================================================
@app.route('/predict', methods=['POST'])
def predict_fake():
    try:
        data = request.get_json()
        print("\n" + "="*70)
        print("Received data for prediction:")
        print(data)
        print("="*70)

        # Check missing fields
        missing = [f for f in INPUT_FEATURES if f not in data]
        if missing:
            return jsonify({'error': f'Missing required features: {missing}'}), 400

        # Convert to DataFrame
        input_data = {f: float(data.get(f, 0)) for f in INPUT_FEATURES}
        input_df = pd.DataFrame([input_data])

        # Apply preprocessing
        processed_df = preprocess_features(input_df)

        # Predict
        prediction = model.predict(processed_df)[0]
        proba = model.predict_proba(processed_df)[0]
        real_prob, fake_prob = float(proba[0]), float(proba[1])

        print(f"✅ Prediction → {'FAKE' if prediction == 1 else 'REAL'} | Fake Prob = {fake_prob:.3f}")

        confidence = {'real_profile_prob': real_prob, 'fake_profile_prob': fake_prob}
        reasoning = generate_llm_reasoning(input_data, prediction, confidence)

        response = {
            'prediction': {'is_fake': int(prediction)},
            'confidence': confidence,
            'reasoning': reasoning,
            'message': 'FAKE PROFILE DETECTED!' if prediction == 1 else '✅ Real Profile',
            'features_used': input_data
        }

        print("\n📤 Response Sent:")
        print(response)
        print("="*70)
        return jsonify(response)

    except Exception as e:
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({
        'status': 'ELADFP Flask API Running',
        'model_loaded': True,
        'expected_features': len(INPUT_FEATURES),
        'note': 'Ensure consistent preprocessing between training and inference'
    })

@app.route('/', methods=['GET'])
def home():
    return jsonify({
        'project': 'ELADFP: Ensemble Learning Fake Profile Detection',
        'team': ['Fattah Ali 21L-5187', 'Ubaid Ur Rehman 21L-5189', 'Umer Shafiq 21L-5401'],
        'supervisor': 'Rana Waqas Ali',
        'endpoints': {
            'POST /predict': 'Predict fake or real profile',
            'GET /health': 'Check API status',
            'GET /': 'API documentation'
        },
        'expected_features': INPUT_FEATURES
    })


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
