from flask import Flask, render_template, request, jsonify
import pickle
import numpy as np
import os

app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, 'model')

# Load model artifacts
with open(os.path.join(MODEL_DIR, 'mlp_model.pkl'), 'rb') as f:
    model = pickle.load(f)
with open(os.path.join(MODEL_DIR, 'scaler.pkl'), 'rb') as f:
    scaler = pickle.load(f)
with open(os.path.join(MODEL_DIR, 'encoders.pkl'), 'rb') as f:
    enc = pickle.load(f)

AGE_CATEGORIES = enc['age_categories']
GENERAL_HEALTH_OPTS = enc['general_health_opts']
CHECKUP_OPTS = enc['checkup_opts']

@app.route('/')
def index():
    return render_template('index.html',
        age_categories=AGE_CATEGORIES,
        general_health_opts=GENERAL_HEALTH_OPTS,
        checkup_opts=CHECKUP_OPTS
    )

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()

        # Encode categorical features
        gen_health_enc = enc['health'].transform([data['general_health']])[0]
        checkup_enc = enc['checkup'].transform([data['checkup']])[0]
        age_enc = enc['age'].transform([data['age_category']])[0]

        features = np.array([[
            gen_health_enc,
            checkup_enc,
            int(data['exercise']),
            int(data['skin_cancer']),
            int(data['other_cancer']),
            int(data['depression']),
            int(data['diabetes']),
            int(data['arthritis']),
            int(data['sex']),
            age_enc,
            float(data['bmi']),
            int(data['smoking']),
            float(data['alcohol']),
            float(data['fruit']),
            float(data['green_veg']),
            float(data['fried_potato']),
        ]])

        features_scaled = scaler.transform(features)
        prob = model.predict_proba(features_scaled)[0][1]
        prediction = int(prob >= 0.5)

        # Risk tier
        if prob < 0.25:
            risk_level = 'Low'
            risk_color = '#27ae60'
            advice = [
                'Maintain your current healthy lifestyle.',
                'Schedule annual checkups to stay proactive.',
                'Keep up with regular physical activity.',
                'Continue a balanced, vegetable-rich diet.'
            ]
        elif prob < 0.50:
            risk_level = 'Moderate'
            risk_color = '#f39c12'
            advice = [
                'Consult your doctor for a cardiovascular assessment.',
                'Aim for at least 150 min of moderate exercise per week.',
                'Reduce fried and processed food intake.',
                'Monitor your blood pressure and cholesterol levels.'
            ]
        elif prob < 0.75:
            risk_level = 'High'
            risk_color = '#e67e22'
            advice = [
                'Seek medical consultation promptly.',
                'Closely monitor blood pressure, glucose, and BMI.',
                'Quit smoking if applicable — highest single impact action.',
                'Adopt a heart-healthy DASH or Mediterranean diet.',
                'Limit alcohol consumption to under 14 units/week.'
            ]
        else:
            risk_level = 'Very High'
            risk_color = '#c0392b'
            advice = [
                'Schedule an urgent appointment with a cardiologist.',
                'Do not delay treatment — immediate lifestyle changes required.',
                'Strict dietary control: eliminate fried food, reduce salt and sugar.',
                'Supervised cardiac exercise rehabilitation recommended.',
                'Medication review with your physician may be necessary.'
            ]

        # Feature contributions for display
        contributions = compute_contributions(data, prob)

        return jsonify({
            'success': True,
            'probability': round(float(prob) * 100, 1),
            'prediction': prediction,
            'risk_level': risk_level,
            'risk_color': risk_color,
            'advice': advice,
            'contributions': contributions
        })

    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 400


def compute_contributions(data, prob):
    """Simplified risk factor contribution labels."""
    factors = []

    age_idx = AGE_CATEGORIES.index(data['age_category'])
    age_risk = age_idx / (len(AGE_CATEGORIES) - 1)
    if age_risk > 0.6:
        factors.append({'label': 'Age (55+)', 'impact': 'high', 'direction': 'risk'})
    elif age_risk > 0.3:
        factors.append({'label': 'Age (35-54)', 'impact': 'medium', 'direction': 'risk'})

    bmi = float(data['bmi'])
    if bmi >= 30:
        factors.append({'label': f'BMI {bmi:.1f} (Obese)', 'impact': 'high', 'direction': 'risk'})
    elif bmi >= 25:
        factors.append({'label': f'BMI {bmi:.1f} (Overweight)', 'impact': 'medium', 'direction': 'risk'})
    else:
        factors.append({'label': f'BMI {bmi:.1f} (Normal)', 'impact': 'low', 'direction': 'protective'})

    if int(data['smoking']):
        factors.append({'label': 'Smoking History', 'impact': 'high', 'direction': 'risk'})

    if not int(data['exercise']):
        factors.append({'label': 'No Regular Exercise', 'impact': 'medium', 'direction': 'risk'})
    else:
        factors.append({'label': 'Regular Exercise', 'impact': 'medium', 'direction': 'protective'})

    if float(data['fried_potato']) > 2:
        factors.append({'label': 'High Fried Potato Intake', 'impact': 'medium', 'direction': 'risk'})

    if float(data['fruit']) > 1 and float(data['green_veg']) > 1:
        factors.append({'label': 'Good Fruit & Veg Intake', 'impact': 'medium', 'direction': 'protective'})

    if float(data['alcohol']) > 14:
        factors.append({'label': 'High Alcohol Consumption', 'impact': 'medium', 'direction': 'risk'})

    if int(data['diabetes']):
        factors.append({'label': 'Diabetes', 'impact': 'medium', 'direction': 'risk'})

    return factors


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(debug=True, host='0.0.0.0', port=port)