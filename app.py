from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import pandas as pd
import numpy as np
import os
import traceback
import google.generativeai as genai


app = Flask(__name__)
CORS(app)

# Get Gemini API key from environment variable only (no hard-coded fallback!)
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')

if not GEMINI_API_KEY:
    print("⚠️ WARNING: GEMINI_API_KEY environment variable not set!")
    print("   AI reasoning will be disabled. Set it with:")
    print("   export GEMINI_API_KEY='your-api-key-here'")
    gemini_model = None
else:
    genai.configure(api_key=GEMINI_API_KEY)
    gemini_model = genai.GenerativeModel(
        'gemini-2.0-flash-exp',
        generation_config={'temperature': 0.7, 'max_output_tokens': 150, 'top_p': 0.8, 'top_k': 40}
    )
    print("✅ Gemini API configured successfully!")


MODEL_PATH = 'best_eladfp_model.pkl'
if os.path.exists(MODEL_PATH):
    model = joblib.load(MODEL_PATH)
    print("✅ ELADFP Ensemble Model loaded successfully!")
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


def generate_gemini_reasoning(features, prediction, confidence):
    """
    Generate AI reasoning for the prediction. 
    If Gemini fails or not configured, returns fallback reasoning - prediction still works!
    """
    # Extract values for fallback
    posts = features.get('#posts', 0)
    followers = features.get('#followers', 0)
    following = features.get('#following', 0)
    nums_ratio = features.get('nums/length username', 0)
    bio_len = features.get('description length', 0)
    
    # If Gemini not configured, use fallback immediately
    if gemini_model is None:
        print("⚠️ Gemini not configured - using fallback reasoning")
        return _get_fallback_reasoning(prediction, posts, followers, following, nums_ratio, bio_len)
    
    try:
        print("\n🤖 Generating Gemini reasoning...")

        prompt = f"""Analyze this Instagram profile briefly and explain the result.

Result: {"FAKE" if prediction == 1 else "REAL"} ({confidence['fake_profile_prob'] * 100:.0f}% confidence)
Profile Stats: {int(posts)} posts, {int(followers)} followers, {int(following)} following
Username: {int(nums_ratio * 100)}% numbers, Bio: {int(bio_len)} chars
Picture: {"Yes" if features.get('profile pic', 0) == 1 else "No"}, Website: {"Yes" if features.get('external URL', 0) == 1 else "No"}
Account Status: {"Private" if features.get('private', 0) == 1 else "Public"}

Provide a brief 1-2 sentence explanation for why this is a {"fake" if prediction == 1 else "real"} profile."""

        print(f"📝 Sending prompt to Gemini...")
        response = gemini_model.generate_content(prompt, request_options={'timeout': 15})
        
        if response and response.text:
            result = response.text.strip()
            print(f"✅ Gemini response received: {result}")
            return result
        else:
            print("⚠️ Gemini returned empty response, using fallback")
            return _get_fallback_reasoning(prediction, posts, followers, following, nums_ratio, bio_len)

    except Exception as e:
        error_msg = str(e)
        print(f"⚠️ Gemini API failed: {error_msg[:100]}")
        
        # Check if it's a rate limit error
        if "429" in error_msg or "quota" in error_msg.lower() or "ResourceExhausted" in str(type(e)):
            print("⚠️ Gemini API rate limit exceeded - using fallback reasoning")
            print("   💡 Get a new API key at: https://aistudio.google.com/app/apikey")
        else:
            print(f"⚠️ Gemini error: {error_msg}")
        
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
        print("📩 Received data for prediction:")
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
        reasoning = generate_gemini_reasoning(input_data, prediction, confidence)

        response = {
            'prediction': {'is_fake': int(prediction)},
            'confidence': confidence,
            'reasoning': reasoning,
            'message': '⚠️ FAKE PROFILE DETECTED!' if prediction == 1 else '✅ Real Profile',
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
