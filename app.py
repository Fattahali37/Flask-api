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

GEMINI_API_KEY = os.getenv('GEMINI_API_KEY', 'AIzaSyBhf2dcc8cxQb2uNsVAgRc5b2LwRfWYg9k')
genai.configure(api_key=GEMINI_API_KEY)
gemini_model = genai.GenerativeModel(
    'gemini-2.0-flash-exp',
    generation_config={'temperature': 0.7, 'max_output_tokens': 150, 'top_p': 0.8, 'top_k': 40}
)


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
    try:
        feature_analysis = []
        feature_analysis.append("Has profile picture" if features['profile pic'] == 1 else "No profile picture")

        if features['nums/length username'] > 0.5:
            feature_analysis.append(f"Username contains {int(features['nums/length username'] * 100)}% numbers (bot-like)")
        else:
            feature_analysis.append(f"Username contains few numbers ({int(features['nums/length username'] * 100)}%)")

        feature_analysis.append("Name equals username (suspicious)" if features['name==username'] == 1 else "Name differs from username")

        if features['description length'] == 0:
            feature_analysis.append("No bio/description")
        elif features['description length'] < 20:
            feature_analysis.append(f"Short bio ({features['description length']} chars)")
        else:
            feature_analysis.append(f"Detailed bio ({features['description length']} chars)")

        feature_analysis.append("Has external website" if features['external URL'] == 1 else "No external website")

        if features['#followers'] > 0:
            ratio = features['#following'] / (features['#followers'] + 1)
            if ratio > 5:
                feature_analysis.append(f"High following/follower ratio ({ratio:.1f}:1)")
            elif ratio < 0.2:
                feature_analysis.append(f"Low following/follower ratio ({ratio:.1f}:1)")
            else:
                feature_analysis.append(f"Balanced follower ratio ({ratio:.1f}:1)")

        if features['#posts'] == 0:
            feature_analysis.append("❌ No posts")
        elif features['#posts'] < 5:
            feature_analysis.append(f"Few posts ({features['#posts']})")
        else:
            feature_analysis.append(f"Active account ({features['#posts']} posts)")

        prompt = f"""
Analyze this Instagram profile briefly.

Result: {"FAKE" if prediction == 1 else "REAL"} ({confidence['fake_profile_prob'] * 100:.0f}% fake)
Profile: {features['#posts']} posts, {features['#followers']} followers, {features['#following']} following
Username: {int(features['nums/length username'] * 100)}% numbers, Bio: {features['description length']} chars
Picture: {"Yes" if features['profile pic'] == 1 else "No"}, Link: {"Yes" if features['external URL'] == 1 else "No"}

Explain WHY it's {"fake" if prediction == 1 else "real"} based on these metrics.
"""
        response = gemini_model.generate_content(prompt, request_options={'timeout': 10})
        return response.text.strip()

    except Exception:
        return (
            "Fake profile indicators include incomplete bio, unusual ratio patterns, and repetitive or number-heavy usernames."
            if prediction == 1 else
            "Real profile indicators include detailed bio, normal ratios, and natural username behavior."
        )

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
