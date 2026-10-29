import unittest
import json
import os
import sys
import pandas as pd

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from app import app


class CardioSenseAppTests(unittest.TestCase):
    def setUp(self):
        app.config['TESTING'] = True
        self.client = app.test_client()

    def test_home_page(self):
        """Verify the main landing page loads with HTTP 200 and renders HTML."""
        response = self.client.get('/')
        self.assertEqual(response.status_code, 200)
        self.assertIn(b'CardioSense', response.data)
        self.assertIn(b'CVD Risk Predictor', response.data)
        self.assertIn(b'Medical Disclaimer', response.data)

    def test_health_check(self):
        """Verify the health check endpoint returns status healthy."""
        response = self.client.get('/health')
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        self.assertEqual(data.get('status'), 'healthy')
        self.assertTrue(data.get('model_loaded'))

    def test_prediction_low_risk(self):
        """Verify a healthy lifestyle yields low/moderate risk prediction."""
        payload = {
            'general_health': 'Excellent',
            'checkup': 'Within the past year',
            'exercise': '1',
            'sex': '0',
            'age_category': '25-29',
            'bmi': '21.5',
            'smoking': '0',
            'alcohol': '1.0',
            'fruit': '3.0',
            'green_veg': '3.0',
            'fried_potato': '0.5',
            'diabetes': '0',
            'depression': '0',
            'arthritis': '0',
            'skin_cancer': '0',
            'other_cancer': '0'
        }
        response = self.client.post('/predict',
                                    data=json.dumps(payload),
                                    content_type='application/json')
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        self.assertTrue(data.get('success'))
        self.assertEqual(data.get('prediction'), 0)
        self.assertIn(data.get('risk_level'), ['Low', 'Moderate'])
        self.assertIsInstance(data.get('advice'), list)
        self.assertGreater(len(data.get('advice')), 0)

    def test_prediction_high_risk(self):
        """Verify a high-risk patient profile yields high/very high risk."""
        payload = {
            'general_health': 'Poor',
            'checkup': '5 or more years ago',
            'exercise': '0',
            'sex': '1',
            'age_category': '75-79',
            'bmi': '35.0',
            'smoking': '1',
            'alcohol': '15.0',
            'fruit': '0.5',
            'green_veg': '0.5',
            'fried_potato': '5.0',
            'diabetes': '1',
            'depression': '1',
            'arthritis': '1',
            'skin_cancer': '0',
            'other_cancer': '0'
        }
        response = self.client.post('/predict',
                                    data=json.dumps(payload),
                                    content_type='application/json')
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        self.assertTrue(data.get('success'))
        self.assertEqual(data.get('prediction'), 1)
        self.assertIn(data.get('risk_level'), ['High', 'Very High'])
        self.assertGreater(data.get('probability'), 50.0)

    def test_prediction_missing_field(self):
        """Verify malformed payload returns HTTP 400 error."""
        payload = {'general_health': 'Good'}  # missing required fields
        response = self.client.post('/predict',
                                    data=json.dumps(payload),
                                    content_type='application/json')
        self.assertEqual(response.status_code, 400)
        data = json.loads(response.data)
        self.assertFalse(data.get('success'))

    def test_dashboard_dataset_integrity(self):
        """Verify the health dataset exists and contains expected schema."""
        base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
        csv_path = os.path.join(base_dir, 'data', 'health_dataset.csv')
        self.assertTrue(os.path.exists(csv_path), "health_dataset.csv must exist in data/")
        df = pd.read_csv(csv_path, nrows=50)
        expected_cols = [
            'General_Health', 'Checkup', 'Exercise', 'Heart_Disease',
            'Sex', 'Age_Category', 'BMI', 'Smoking_History'
        ]
        for col in expected_cols:
            self.assertIn(col, df.columns)


if __name__ == '__main__':
    unittest.main()
