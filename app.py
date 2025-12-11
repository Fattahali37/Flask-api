from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import pandas as pd
import numpy as np
import os
import traceback
import time
from huggingface_hub import InferenceClient


app = Flask(__name__)
CORS(app)

# Hugging Face Configuration
HF_API_TOKEN = os.getenv('HF_API_TOKEN')

if not HF_API_TOKEN:
    print("⚠️ WARNING: HF_API_TOKEN environment variable not set!")
    print("   LLM reasoning will be disabled. Get your free token at:")
    print("   https://huggingface.co/settings/tokens")
    print("   Then set it with: export HF_API_TOKEN='your-token-here'")
    hf_client = None
    llm_enabled = False
else:
    try:
        # Initialize Hugging Face Inference Client with the new API
        hf_client = InferenceClient(token=HF_API_TOKEN)
        llm_enabled = True
        print("✅ Hugging Face LLM configured successfully!")
    except Exception as e:
        print(f"⚠️ Failed to initialize Hugging Face client: {e}")
        hf_client = None
        llm_enabled = False

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
    if not llm_enabled or hf_client is None:
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

        # Build concise prompt for the LLM
        prompt = f"""Analyze this Instagram profile and explain in 1-2 sentences why it's {"FAKE" if prediction == 1 else "REAL"}.

Result: {"FAKE" if prediction == 1 else "REAL"} ({confidence['fake_profile_prob'] * 100:.0f}% confidence)
Stats: {int(posts)} posts, {int(followers)} followers, {int(following)} following
Username: {int(nums_ratio * 100)}% numbers, Bio: {int(bio_len)} chars
Has picture: {"Yes" if features.get('profile pic', 0) == 1 else "No"}, Has website: {"Yes" if features.get('external URL', 0) == 1 else "No"}
Status: {"Private" if features.get('private', 0) == 1 else "Public"}

Explanation:"""

        print(f"📝 Sending prompt to Hugging Face...")
        
        # Retry logic with exponential backoff
        max_retries = 2
        for attempt in range(max_retries):
            try:
                # Use text_generation with a fast, free model
                response = hf_client.text_generation(
                    prompt,
                    model="microsoft/Phi-3-mini-4k-instruct",
                    max_new_tokens=100,
                    temperature=0.7,
                    top_p=0.9,
                    return_full_text=False
                )
                
                if response:
                    # Clean up the response
                    generated_text = response.strip()
                    # Limit to 300 characters
                    generated_text = ' '.join(generated_text.split())[:300]
                    print(f"✅ LLM response: {generated_text}")
                    return generated_text
                else:
                    print("⚠️ LLM returned empty response, using fallback")
                    return _get_fallback_reasoning(prediction, posts, followers, following, nums_ratio, bio_len)
                    
            except Exception as retry_error:
                error_msg = str(retry_error)
                
                # Check if model is loading
                if "loading" in error_msg.lower() or "503" in error_msg:
                    print(f"⚠️ Model loading, retry {attempt + 1}/{max_retries}...")
                    if attempt < max_retries - 1:
                        time.sleep(10)  # Wait for model to load
                        continue
                
                if attempt < max_retries - 1:
                    wait_time = (2 ** attempt)  # Exponential backoff: 1s, 2s
                    print(f"⚠️ Retry {attempt + 1}/{max_retries} failed, waiting {wait_time}s...")
                    time.sleep(wait_time)
                else:
                    raise retry_error

    except Exception as e:
        error_msg = str(e)
        print(f"⚠️ LLM API failed: {error_msg[:150]}")
        
        # Check error type
        if "429" in error_msg or "rate limit" in error_msg.lower():
            print("⚠️ Rate limit exceeded - using fallback reasoning")
        elif "401" in error_msg or "authentication" in error_msg.lower() or "unauthorized" in error_msg.lower():
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
