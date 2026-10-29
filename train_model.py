import numpy as np
import pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, roc_auc_score
from sklearn.utils import resample
import pickle, os

np.random.seed(42)
N = 8000

age_categories = ['18-24','25-29','30-34','35-39','40-44','45-49','50-54','55-59','60-64','65-69','70-74','75-79','80+']
general_health_opts = ['Poor', 'Fair', 'Good', 'Very Good', 'Excellent']
checkup_opts = ['Never', '5 or more years ago', 'Within the past 5 years', 'Within the past 2 years', 'Within the past year']

age_idx = np.random.choice(len(age_categories), N, p=[0.05,0.06,0.07,0.08,0.08,0.09,0.09,0.1,0.1,0.09,0.08,0.07,0.04])
age_risk = age_idx / (len(age_categories) - 1)

bmi = np.clip(np.random.normal(27 + age_risk*5, 5), 15, 60)
smoking = np.random.binomial(1, 0.3 + 0.2*age_risk)
alcohol = np.clip(np.random.exponential(4), 0, 30)
exercise = np.random.binomial(1, 0.6 - 0.2*age_risk)
fruit = np.clip(np.random.exponential(1.5), 0, 5)
green_veg = np.clip(np.random.exponential(1.5), 0, 5)
fried_potato = np.clip(np.random.exponential(1.2), 0, 5)
sex = np.random.binomial(1, 0.5)
diabetes = np.random.binomial(1, 0.1 + 0.15*age_risk + 0.05*(bmi > 30))
depression = np.random.binomial(1, 0.15)
arthritis = np.random.binomial(1, 0.05 + 0.25*age_risk)
skin_cancer = np.random.binomial(1, 0.05)
other_cancer = np.random.binomial(1, 0.07 + 0.05*age_risk)
general_health_idx = np.random.choice(len(general_health_opts), N)
checkup_idx = np.random.choice(len(checkup_opts), N)

cvd_prob = (0.03 + 0.30*age_risk + 0.10*(bmi>30) + 0.12*smoking
            + 0.06*(alcohol>14) - 0.06*exercise - 0.05*(fruit>1)
            - 0.05*(green_veg>1) + 0.07*(fried_potato>2) + 0.05*diabetes + 0.03*(sex==1))
cvd_prob = np.clip(cvd_prob + np.random.normal(0, 0.04, N), 0.01, 0.99)
heart_disease = (np.random.uniform(0,1,N) < cvd_prob).astype(int)

df = pd.DataFrame({
    'GeneralHealth': [general_health_opts[i] for i in general_health_idx],
    'Checkup': [checkup_opts[i] for i in checkup_idx],
    'Exercise': exercise, 'HeartDisease': heart_disease,
    'SkinCancer': skin_cancer, 'OtherCancer': other_cancer,
    'Depression': depression, 'Diabetes': diabetes, 'Arthritis': arthritis,
    'Sex': sex, 'AgeCategory': [age_categories[i] for i in age_idx],
    'BMI': np.round(bmi,1), 'SmokingHistory': smoking,
    'AlcoholConsumption': np.round(alcohol,1), 'FruitConsumption': np.round(fruit,1),
    'GreenVegetableConsumption': np.round(green_veg,1), 'FriedPotatoConsumption': np.round(fried_potato,1),
})

df.to_csv('model/data.csv', index=False)
print(f"Dataset: {df.shape}, CVD prevalence: {heart_disease.mean():.2%}")

le_health = LabelEncoder(); le_checkup = LabelEncoder(); le_age = LabelEncoder()
df['GeneralHealth_enc'] = le_health.fit_transform(df['GeneralHealth'])
df['Checkup_enc'] = le_checkup.fit_transform(df['Checkup'])
df['AgeCategory_enc'] = le_age.fit_transform(df['AgeCategory'])

feature_cols = ['GeneralHealth_enc','Checkup_enc','Exercise','SkinCancer','OtherCancer',
                'Depression','Diabetes','Arthritis','Sex','AgeCategory_enc',
                'BMI','SmokingHistory','AlcoholConsumption','FruitConsumption',
                'GreenVegetableConsumption','FriedPotatoConsumption']

X = df[feature_cols].values; y = df['HeartDisease'].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

scaler = StandardScaler()
X_tr = scaler.fit_transform(X_train); X_ts = scaler.transform(X_test)

# Balance classes via oversampling
df_tr = pd.DataFrame(X_tr); df_tr['y'] = y_train
maj = df_tr[df_tr.y==0]; minr = df_tr[df_tr.y==1]
min_up = resample(minr, replace=True, n_samples=len(maj), random_state=42)
bal = pd.concat([maj, min_up])
X_bal = bal.drop('y', axis=1).values; y_bal = bal['y'].values

mlp = MLPClassifier(hidden_layer_sizes=(128,64,32), activation='relu', solver='adam',
                    max_iter=500, random_state=42, early_stopping=True,
                    validation_fraction=0.1, learning_rate_init=0.001)
mlp.fit(X_bal, y_bal)

y_pred = mlp.predict(X_ts); y_prob = mlp.predict_proba(X_ts)[:,1]
print(f"\nAccuracy: {accuracy_score(y_test,y_pred):.4f}")
print(f"ROC-AUC:  {roc_auc_score(y_test,y_prob):.4f}")
print(classification_report(y_test, y_pred))

with open('model/mlp_model.pkl','wb') as f: pickle.dump(mlp, f)
with open('model/scaler.pkl','wb') as f: pickle.dump(scaler, f)
with open('model/encoders.pkl','wb') as f:
    pickle.dump({'health': le_health, 'checkup': le_checkup, 'age': le_age,
                 'age_categories': age_categories,
                 'general_health_opts': general_health_opts,
                 'checkup_opts': checkup_opts}, f)
print("\nModel saved!")