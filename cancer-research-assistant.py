import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
import sqlite3
import json
import requests
from datetime import datetime, timedelta
import warnings
import pickle
import hashlib
from typing import Dict, List, Tuple, Optional
import re
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
import networkx as nx
from scipy import stats
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import dash
from dash import dcc, html, Input, Output
import dash_bootstrap_components as dbc
import threading
import time

warnings.filterwarnings('ignore')

# Download required NLTK data
try:
    nltk.download('vader_lexicon', quiet=True)
except:
    pass

class UltimateCancerResearchAssistant:
    def __init__(self):
        self.model = None
        self.deep_model = None
        self.scaler = StandardScaler()
        self.patient_data = pd.DataFrame()
        self.clinical_trials = pd.DataFrame()
        self.drug_interactions = {}
        self.survival_model = None
        self.ehr_connection = None
        self.trained = False
        self.models = {}  # Store multiple models
        
        # Treatment protocols
        self.treatment_protocols = {
            'Chemotherapy': {
                'drugs': ['Cisplatin', 'Doxorubicin', 'Paclitaxel', 'Carboplatin', 'Gemcitabine'],
                'indications': ['Advanced cancer', 'Metastatic disease', 'Adjuvant therapy'],
                'side_effects': ['Nausea', 'Fatigue', 'Hair loss', 'Myelosuppression']
            },
            'Radiation': {
                'types': ['External Beam', 'Brachytherapy', 'Stereotactic', 'Proton Therapy'],
                'indications': ['Localized tumors', 'Palliative care', 'Pre-operative', 'Post-operative'],
                'side_effects': ['Skin irritation', 'Fatigue', 'Localized tissue damage']
            },
            'Immunotherapy': {
                'drugs': ['Checkpoint Inhibitors', 'CAR-T Cell', 'Monoclonal Antibodies', 'Cancer Vaccines'],
                'indications': ['Melanoma', 'Lung cancer', 'Renal cell carcinoma', 'Hodgkin lymphoma'],
                'side_effects': ['Immune-related adverse events', 'Fatigue', 'Rash', 'Diarrhea']
            },
            'Targeted Therapy': {
                'drugs': ['Tyrosine Kinase Inhibitors', 'Hormone Therapy', 'Angiogenesis Inhibitors', 'PARP Inhibitors'],
                'indications': ['HER2+ breast cancer', 'EGFR+ lung cancer', 'Hormone-sensitive cancers'],
                'side_effects': ['Skin rash', 'Diarrhea', 'Hypertension', 'Bleeding']
            }
        }
        
        # Initialize EHR system
        self.init_ehr_system()
        
        # Initialize drug interaction database
        self.init_drug_interactions()
        
        # Initialize clinical trials database
        self.init_clinical_trials()
        
        # Initialize survival analysis
        self.survival_data = pd.DataFrame()
        
        print("Ultimate Cancer Research Assistant initialized!")
        print("Features: Advanced ML, EHR Integration, Clinical Trials, Drug Interactions, Survival Analysis")
    
    def init_ehr_system(self):
        """Initialize Electronic Health Record system"""
        try:
            # Create in-memory SQLite database for EHR
            self.ehr_connection = sqlite3.connect(':memory:')
            
            # Create tables
            self.ehr_connection.execute('''
                CREATE TABLE IF NOT EXISTS patients (
                    patient_id INTEGER PRIMARY KEY,
                    name TEXT,
                    dob DATE,
                    gender TEXT,
                    diagnosis TEXT,
                    diagnosis_date DATE,
                    stage TEXT,
                    treatment_plan TEXT,
                    last_updated TIMESTAMP
                )
            ''')
            
            self.ehr_connection.execute('''
                CREATE TABLE IF NOT EXISTS lab_results (
                    result_id INTEGER PRIMARY KEY,
                    patient_id INTEGER,
                    test_name TEXT,
                    result_value REAL,
                    unit TEXT,
                    test_date DATE,
                    FOREIGN KEY (patient_id) REFERENCES patients (patient_id)
                )
            ''')
            
            self.ehr_connection.execute('''
                CREATE TABLE IF NOT EXISTS medications (
                    med_id INTEGER PRIMARY KEY,
                    patient_id INTEGER,
                    medication_name TEXT,
                    dosage TEXT,
                    frequency TEXT,
                    start_date DATE,
                    end_date DATE,
                    prescribed_by TEXT,
                    FOREIGN KEY (patient_id) REFERENCES patients (patient_id)
                )
            ''')
            
            self.ehr_connection.commit()
            print("EHR System initialized successfully")
            
        except Exception as e:
            print(f"Error initializing EHR system: {e}")
    
    def init_drug_interactions(self):
        """Initialize drug interaction database"""
        # Simplified drug interaction data
        self.drug_interactions = {
            'Warfarin': {
                'interactions': ['Aspirin', 'Ibuprofen', 'Ciprofloxacin'],
                'severity': ['High', 'Medium', 'Medium'],
                'description': 'Increased bleeding risk'
            },
            'Cisplatin': {
                'interactions': ['Aminoglycosides', 'Loop diuretics'],
                'severity': ['High', 'Medium'],
                'description': 'Increased nephrotoxicity'
            },
            'Doxorubicin': {
                'interactions': ['Trastuzumab', 'Cyclophosphamide'],
                'severity': ['Medium', 'Medium'],
                'description': 'Increased cardiotoxicity'
            },
            'Paclitaxel': {
                'interactions': ['Cisplatin', 'Doxorubicin'],
                'severity': ['Medium', 'Medium'],
                'description': 'Increased myelosuppression'
            }
        }
        print("Drug interaction database initialized")
    
    def init_clinical_trials(self):
        """Initialize clinical trials database"""
        # Sample clinical trials data
        self.clinical_trials = pd.DataFrame({
            'trial_id': ['NCT00001234', 'NCT00005678', 'NCT00009012', 'NCT00003456'],
            'title': [
                'Immunotherapy for Advanced Melanoma',
                'Targeted Therapy in HER2+ Breast Cancer',
                'CAR-T Cell Therapy for Leukemia',
                'Combination Chemotherapy for Lung Cancer'
            ],
            'phase': ['Phase III', 'Phase II', 'Phase I', 'Phase III'],
            'cancer_type': ['Melanoma', 'Breast Cancer', 'Leukemia', 'Lung Cancer'],
            'eligibility_criteria': [
                'Stage III/IV melanoma, age 18-75',
                'HER2+, metastatic, age 18-80',
                'Relapsed/refractory, age 18-65',
                'NSCLC, stage IIIB/IV, age 18-75'
            ],
            'location': ['Mayo Clinic', 'MD Anderson', 'Johns Hopkins', 'Memorial Sloan Kettering'],
            'status': ['Recruiting', 'Active', 'Not yet recruiting', 'Completed'],
            'start_date': ['2023-01-15', '2023-03-20', '2023-06-01', '2022-09-10'],
            'end_date': ['2026-01-15', '2025-03-20', '2024-12-31', '2024-09-10']
        })
        print("Clinical trials database initialized")
    
    def generate_sample_data(self, n_patients=1000):
        """Generate synthetic patient data for demonstration"""
        np.random.seed(42)
        
        # Generate patient features
        age = np.random.normal(60, 15, n_patients)
        age = np.clip(age, 18, 90)
        
        # Tumor characteristics
        tumor_size = np.random.exponential(3, n_patients)
        tumor_grade = np.random.choice([1, 2, 3, 4], n_patients, p=[0.1, 0.3, 0.4, 0.2])
        lymph_nodes = np.random.poisson(2, n_patients)
        
        # Biomarkers
        ki67 = np.random.uniform(0, 100, n_patients)  # Proliferation index
        her2 = np.random.choice([0, 1], n_patients, p=[0.7, 0.3])  # HER2 status
        er = np.random.choice([0, 1], n_patients, p=[0.6, 0.4])    # Estrogen receptor
        pr = np.random.choice([0, 1], n_patients, p=[0.55, 0.45])  # Progesterone receptor
        
        # Treatment history
        previous_treatments = np.random.poisson(1.5, n_patients)
        
        # Survival time (in months) - censored data
        baseline_survival = 60
        survival_time = np.random.exponential(baseline_survival, n_patients)
        survival_time = np.clip(survival_time, 0, 200)  # Max 200 months
        event_occurred = np.random.choice([0, 1], n_patients, p=[0.3, 0.7])  # 70% events occurred
        
        # Treatment response (synthetic outcome)
        # Higher risk factors lead to worse outcomes
        risk_score = (tumor_size * 0.3 + 
                     tumor_grade * 0.4 + 
                     lymph_nodes * 0.2 + 
                     (1-er) * 0.5 + 
                     her2 * 0.3 +
                     (1-pr) * 0.4 +
                     previous_treatments * 0.1 +
                     np.random.normal(0, 0.5, n_patients))
        
        # Convert to binary outcome (0 = good prognosis, 1 = poor prognosis)
        prognosis = (risk_score > np.percentile(risk_score, 70)).astype(int)
        
        self.patient_data = pd.DataFrame({
            'patient_id': range(1, n_patients + 1),
            'age': age,
            'tumor_size': tumor_size,
            'tumor_grade': tumor_grade,
            'lymph_nodes': lymph_nodes,
            'ki67_index': ki67,
            'her2_status': her2,
            'er_status': er,
            'pr_status': pr,
            'previous_treatments': previous_treatments,
            'prognosis': prognosis,
            'survival_time': survival_time,
            'event_occurred': event_occurred
        })
        
        # Generate survival data for analysis
        self.survival_data = self.patient_data[['survival_time', 'event_occurred', 'tumor_grade', 'tumor_size']].copy()
        
        print(f"Generated {n_patients} patient records")
        return self.patient_data
    
    def train_advanced_models(self):
        """Train multiple advanced machine learning models"""
        if self.patient_data.empty:
            self.generate_sample_data()
        
        # Prepare features and target
        features = ['age', 'tumor_size', 'tumor_grade', 'lymph_nodes', 
                   'ki67_index', 'her2_status', 'er_status', 'pr_status', 'previous_treatments']
        X = self.patient_data[features]
        y = self.patient_data['prognosis']
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # 1. Random Forest
        print("Training Random Forest model...")
        rf_model = RandomForestClassifier(n_estimators=200, random_state=42, max_depth=10)
        rf_model.fit(X_train_scaled, y_train)
        self.models['random_forest'] = rf_model
        
        # 2. Support Vector Machine
        print("Training SVM model...")
        svm_model = SVC(probability=True, random_state=42)
        svm_model.fit(X_train_scaled, y_train)
        self.models['svm'] = svm_model
        
        # 3. Neural Network
        print("Training Neural Network model...")
        nn_model = MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=500, random_state=42)
        nn_model.fit(X_train_scaled, y_train)
        self.models['neural_network'] = nn_model
        
        # 4. Deep Learning Model (TensorFlow/Keras)
        print("Training Deep Learning model...")
        self.train_deep_learning_model(X_train_scaled, y_train, X_test_scaled, y_test)
        
        # Evaluate models
        print("\n=== MODEL EVALUATION ===")
        for name, model in self.models.items():
            if name != 'deep_learning':
                y_pred = model.predict(X_test_scaled)
                auc_score = roc_auc_score(y_test, model.predict_proba(X_test_scaled)[:, 1])
                print(f"{name.upper()}: AUC = {auc_score:.4f}")
        
        self.trained = True
        print("All models trained successfully!")
    
    def train_deep_learning_model(self, X_train, y_train, X_test, y_test):
        """Train a deep learning model using TensorFlow/Keras"""
        # Create model
        model = Sequential([
            Dense(128, activation='relu', input_shape=(X_train.shape[1],)),
            Dropout(0.3),
            Dense(64, activation='relu'),
            Dropout(0.3),
            Dense(32, activation='relu'),
            Dropout(0.2),
            Dense(1, activation='sigmoid')
        ])
        
        # Compile model
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        
        # Early stopping
        early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
        
        # Train model
        history = model.fit(
            X_train, y_train,
            epochs=100,
            batch_size=32,
            validation_data=(X_test, y_test),
            callbacks=[early_stopping],
            verbose=0
        )
        
        self.models['deep_learning'] = model
        self.deep_model_history = history
    
    def predict_patient_prognosis(self, patient_features, model_name='ensemble'):
        """Predict prognosis for a new patient using specified model"""
        if not self.trained:
            print("Please train models first using train_advanced_models()")
            return None
        
        # Scale features
        features_scaled = self.scaler.transform([patient_features])
        
        if model_name == 'ensemble':
            # Ensemble prediction - average of all models
            predictions = []
            probabilities = []
            
            for name, model in self.models.items():
                if name != 'deep_learning':
                    pred = model.predict(features_scaled)[0]
                    prob = model.predict_proba(features_scaled)[0][1]
                    predictions.append(pred)
                    probabilities.append(prob)
                else:
                    pred = (model.predict(features_scaled)[0] > 0.5).astype(int)
                    prob = model.predict(features_scaled)[0]
                    predictions.append(pred)
                    probabilities.append(prob)
            
            final_prediction = 1 if sum(predictions) > len(predictions)/2 else 0
            final_probability = np.mean(probabilities)
        else:
            # Single model prediction
            if model_name not in self.models:
                print(f"Model {model_name} not found. Available models: {list(self.models.keys())}")
                return None
                
            model = self.models[model_name]
            if model_name != 'deep_learning':
                final_prediction = model.predict(features_scaled)[0]
                final_probability = model.predict_proba(features_scaled)[0][1]
            else:
                pred_prob = model.predict(features_scaled)[0]
                final_prediction = 1 if pred_prob > 0.5 else 0
                final_probability = pred_prob
        
        result = {
            'prognosis': 'Poor' if final_prediction == 1 else 'Good',
            'confidence': final_probability if final_prediction == 1 else (1 - final_probability),
            'risk_score': final_probability,
            'model_used': model_name
        }
        
        return result
    
    def recommend_treatment(self, patient_data):
        """Recommend treatment based on patient characteristics"""
        recommendations = []
        
        # Basic treatment recommendations based on tumor characteristics
        if patient_data['tumor_grade'] >= 3:
            recommendations.append("Consider aggressive treatment approach")
        
        if patient_data['tumor_size'] > 5:
            recommendations.append("Large tumor detected - surgical consultation recommended")
        
        if patient_data['lymph_nodes'] > 3:
            recommendations.append("Lymph node involvement - systemic therapy consideration")
        
        if patient_data['her2_status'] == 1:
            recommendations.append("HER2 positive - consider targeted therapy (trastuzumab)")
        
        if patient_data['er_status'] == 1 and patient_data['pr_status'] == 1:
            recommendations.append("Hormone receptor positive - hormone therapy may be beneficial")
        
        if patient_data['ki67_index'] > 50:
            recommendations.append("High proliferation rate - consider combination therapy")
        
        if patient_data['previous_treatments'] > 2:
            recommendations.append("Multiple previous treatments - consider clinical trials")
        
        # General recommendations
        recommendations.extend([
            "Multidisciplinary team consultation recommended",
            "Consider clinical trials for novel therapies",
            "Regular monitoring and follow-up essential",
            "Genetic counseling may be beneficial",
            "Supportive care and symptom management important"
        ])
        
        return recommendations
    
    def check_drug_interactions(self, medications):
        """Check for potential drug interactions"""
        interactions_found = []
        
        for med in medications:
            if med in self.drug_interactions:
                interaction_data = self.drug_interactions[med]
                for i, interacting_drug in enumerate(interaction_data['interactions']):
                    if interacting_drug in medications:
                        interactions_found.append({
                            'drug1': med,
                            'drug2': interacting_drug,
                            'severity': interaction_data['severity'][i],
                            'description': interaction_data['description']
                        })
        
        return interactions_found
    
    def find_clinical_trials(self, cancer_type, patient_age, patient_stage):
        """Find matching clinical trials for a patient"""
        matching_trials = []
        
        for _, trial in self.clinical_trials.iterrows():
            if cancer_type.lower() in trial['cancer_type'].lower():
                # Parse eligibility criteria (simplified)
                criteria = trial['eligibility_criteria'].lower()
                if 'age' in criteria:
                    # Simple age check
                    age_match = True  # In a real system, this would be more sophisticated
                else:
                    age_match = True
                
                if age_match:
                    matching_trials.append({
                        'trial_id': trial['trial_id'],
                        'title': trial['title'],
                        'phase': trial['phase'],
                        'location': trial['location'],
                        'status': trial['status']
                    })
        
        return matching_trials
    
    def survival_analysis(self):
        """Perform survival analysis on patient data"""
        if self.survival_data.empty:
            print("No survival data available")
            return None
        
        # Kaplan-Meier estimation (simplified)
        from lifelines import KaplanMeierFitter
        
        kmf = KaplanMeierFitter()
        kmf.fit(self.survival_data['survival_time'], 
                event_observed=self.survival_data['event_occurred'], 
                label='All Patients')
        
        # Stratified by tumor grade
        fig, ax = plt.subplots(figsize=(10, 6))
        for grade in sorted(self.survival_data['tumor_grade'].unique()):
            mask = self.survival_data['tumor_grade'] == grade
            kmf_temp = KaplanMeierFitter()
            kmf_temp.fit(self.survival_data[mask]['survival_time'], 
                        event_observed=self.survival_data[mask]['event_occurred'], 
                        label=f'Grade {grade}')
            kmf_temp.plot_survival_function(ax=ax)
        
        ax.set_xlabel('Time (months)')
        ax.set_ylabel('Survival Probability')
        ax.set_title('Kaplan-Meier Survival Curves by Tumor Grade')
        ax.legend()
        plt.tight_layout()
        plt.show()
        
        return kmf
    
    def add_patient_to_ehr(self, patient_info):
        """Add patient to EHR system"""
        try:
            cursor = self.ehr_connection.cursor()
            
            # Insert patient record
            cursor.execute('''
                INSERT INTO patients 
                (patient_id, name, dob, gender, diagnosis, diagnosis_date, stage, treatment_plan, last_updated)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                patient_info.get('patient_id'),
                patient_info.get('name', 'Unknown'),
                patient_info.get('dob', '1970-01-01'),
                patient_info.get('gender', 'Unknown'),
                patient_info.get('diagnosis', 'Cancer'),
                patient_info.get('diagnosis_date', datetime.now().strftime('%Y-%m-%d')),
                patient_info.get('stage', 'Unknown'),
                patient_info.get('treatment_plan', 'To be determined'),
                datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            ))
            
            self.ehr_connection.commit()
            print(f"Patient {patient_info.get('name', 'Unknown')} added to EHR")
            return True
            
        except Exception as e:
            print(f"Error adding patient to EHR: {e}")
            return False
    
    def add_medication_to_ehr(self, patient_id, medication_info):
        """Add medication to EHR system"""
        try:
            cursor = self.ehr_connection.cursor()
            
            cursor.execute('''
                INSERT INTO medications 
                (patient_id, medication_name, dosage, frequency, start_date, end_date, prescribed_by)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (
                patient_id,
                medication_info.get('medication_name'),
                medication_info.get('dosage'),
                medication_info.get('frequency'),
                medication_info.get('start_date'),
                medication_info.get('end_date'),
                medication_info.get('prescribed_by', 'System')
            ))
            
            self.ehr_connection.commit()
            print(f"Medication {medication_info.get('medication_name')} added for patient {patient_id}")
            return True
            
        except Exception as e:
            print(f"Error adding medication to EHR: {e}")
            return False
    
    def get_patient_ehr(self, patient_id):
        """Retrieve patient EHR data"""
        try:
            cursor = self.ehr_connection.cursor()
            
            # Get patient info
            patient_query = "SELECT * FROM patients WHERE patient_id = ?"
            patient_data = cursor.execute(patient_query, (patient_id,)).fetchone()
            
            if not patient_data:
                return None
            
            # Get medications
            med_query = "SELECT * FROM medications WHERE patient_id = ?"
            medications = cursor.execute(med_query, (patient_id,)).fetchall()
            
            return {
                'patient': patient_data,
                'medications': medications
            }
            
        except Exception as e:
            print(f"Error retrieving patient EHR: {e}")
            return None
    
    def visualize_data(self):
        """Create advanced visualizations of patient data"""
        if self.patient_data.empty:
            print("No data available. Generate data first.")
            return
        
        # Create interactive dashboard using Plotly
        fig = make_subplots(
            rows=2, cols=3,
            subplot_titles=('Age Distribution', 'Tumor Size vs Grade', 'Prognosis Distribution',
                           'Ki67 Index by Prognosis', 'Lymph Nodes vs Tumor Size', 'Feature Correlation'),
            specs=[[{"type": "histogram"}, {"type": "scatter"}, {"type": "pie"}],
                   [{"type": "histogram"}, {"type": "scatter"}, {"type": "heatmap"}]]
        )
        
        # Age distribution
        fig.add_trace(
            go.Histogram(x=self.patient_data['age'], name='Age', nbinsx=30),
            row=1, col=1
        )
        
        # Tumor size vs Grade
        fig.add_trace(
            go.Scatter(
                x=self.patient_data['tumor_grade'],
                y=self.patient_data['tumor_size'],
                mode='markers',
                marker=dict(
                    color=self.patient_data['prognosis'],
                    colorscale='Viridis',
                    showscale=True
                ),
                name='Tumor Characteristics'
            ),
            row=1, col=2
        )
        
        # Prognosis distribution
        prognosis_counts = self.patient_data['prognosis'].value_counts()
        fig.add_trace(
            go.Pie(labels=['Good', 'Poor'], values=prognosis_counts.values, name='Prognosis'),
            row=1, col=3
        )
        
        # Ki67 Index by Prognosis
        for prognosis in [0, 1]:
            data = self.patient_data[self.patient_data['prognosis'] == prognosis]['ki67_index']
            fig.add_trace(
                go.Histogram(x=data, name=f'Prognosis: {"Good" if prognosis == 0 else "Poor"}', opacity=0.7),
                row=2, col=1
            )
        
        # Lymph nodes vs Tumor size
        fig.add_trace(
            go.Scatter(
                x=self.patient_data['lymph_nodes'],
                y=self.patient_data['tumor_size'],
                mode='markers',
                marker=dict(color='red', opacity=0.6),
                name='Lymph Nodes'
            ),
            row=2, col=2
        )
        
        # Feature correlation heatmap
        numeric_features = ['age', 'tumor_size', 'tumor_grade', 'lymph_nodes', 
                           'ki67_index', 'her2_status', 'er_status', 'prognosis']
        correlation_matrix = self.patient_data[numeric_features].corr()
        
        fig.add_trace(
            go.Heatmap(
                z=correlation_matrix.values,
                x=correlation_matrix.columns,
                y=correlation_matrix.columns,
                colorscale='RdBu',
                zmid=0
            ),
            row=2, col=3
        )
        
        fig.update_layout(height=800, showlegend=True, title_text="Cancer Patient Data Analysis Dashboard")
        fig.show()
    
    def get_treatment_protocols(self):
        """Display available treatment protocols"""
        print("\n=== CANCER TREATMENT PROTOCOLS ===")
        for category, info in self.treatment_protocols.items():
            print(f"\n{category}:")
            if 'drugs' in info:
                print("  Drugs:")
                for drug in info['drugs']:
                    print(f"    • {drug}")
            if 'types' in info:
                print("  Types:")
                for treatment_type in info['types']:
                    print(f"    • {treatment_type}")
            print("  Indications:")
            for indication in info['indications']:
                print(f"    • {indication}")
            print("  Common Side Effects:")
            for side_effect in info['side_effects']:
                print(f"    • {side_effect}")
    
    def run_comprehensive_analysis(self, patient_id=None):
        """Run comprehensive analysis for a patient or overall dataset"""
        if patient_id:
            # Analyze specific patient
            patient = self.patient_data[self.patient_data['patient_id'] == patient_id]
            if patient.empty:
                print(f"Patient ID {patient_id} not found")
                return
            
            patient_info = patient.iloc[0].to_dict()
            print(f"\n=== PATIENT {patient_id} COMPREHENSIVE ANALYSIS ===")
            print(f"Age: {patient_info['age']:.1f}")
            print(f"Tumor Size: {patient_info['tumor_size']:.2f} cm")
            print(f"Tumor Grade: {patient_info['tumor_grade']}")
            print(f"Lymph Nodes Involved: {patient_info['lymph_nodes']}")
            print(f"Ki67 Index: {patient_info['ki67_index']:.1f}%")
            print(f"HER2 Status: {'Positive' if patient_info['her2_status'] == 1 else 'Negative'}")
            print(f"ER Status: {'Positive' if patient_info['er_status'] == 1 else 'Negative'}")
            print(f"PR Status: {'Positive' if patient_info['pr_status'] == 1 else 'Negative'}")
            print(f"Previous Treatments: {patient_info['previous_treatments']}")
            print(f"Survival Time: {patient_info['survival_time']:.1f} months")
            
            # Predict prognosis with all models
            features = [patient_info['age'], patient_info['tumor_size'], 
                       patient_info['tumor_grade'], patient_info['lymph_nodes'],
                       patient_info['ki67_index'], patient_info['her2_status'],
                       patient_info['er_status'], patient_info['pr_status'],
                       patient_info['previous_treatments']]
            
            print("\n--- PROGNOSIS PREDICTIONS ---")
            for model_name in self.models.keys():
                prognosis_result = self.predict_patient_prognosis(features, model_name)
                if prognosis_result:
                    print(f"{model_name.upper()}: {prognosis_result['prognosis']} "
                          f"(Risk: {prognosis_result['risk_score']:.2%})")
            
            # Treatment recommendations
            print("\n--- TREATMENT RECOMMENDATIONS ---")
            recommendations = self.recommend_treatment(patient_info)
            for i, rec in enumerate(recommendations[:8]):  # Show top 8
                print(f"  {i+1}. {rec}")
            
            # Clinical trial matching
            print("\n--- CLINICAL TRIAL MATCHES ---")
            trials = self.find_clinical_trials("Breast Cancer", patient_info['age'], "II")
            if trials:
                for trial in trials[:3]:  # Show top 3
                    print(f"  • {trial['title']} ({trial['phase']}) - {trial['location']}")
            else:
                print("  No matching clinical trials found")
                
        else:
            # Overall dataset analysis
            print("=== OVERALL DATASET ANALYSIS ===")
            print(f"Total Patients: {len(self.patient_data)}")
            print(f"Average Age: {self.patient_data['age'].mean():.1f} ± {self.patient_data['age'].std():.1f}")
            print(f"Average Tumor Size: {self.patient_data['tumor_size'].mean():.2f} cm")
            print(f"Median Survival Time: {self.patient_data['survival_time'].median():.1f} months")
            print(f"Prognosis Distribution:")
            print(f"  Good: {len(self.patient_data[self.patient_data['prognosis'] == 0])} "
                  f"({len(self.patient_data[self.patient_data['prognosis'] == 0])/len(self.patient_data)*100:.1f}%)")
            print(f"  Poor: {len(self.patient_data[self.patient_data['prognosis'] == 1])} "
                  f"({len(self.patient_data[self.patient_data['prognosis'] == 1])/len(self.patient_data)*100:.1f}%)")
    
    def export_data(self, filename='cancer_research_data.csv'):
        """Export patient data to CSV"""
        try:
            self.patient_data.to_csv(filename, index=False)
            print(f"Data exported to {filename}")
        except Exception as e:
            print(f"Error exporting data: {e}")
    
    def import_data(self, filename):
        """Import patient data from CSV"""
        try:
            self.patient_data = pd.read_csv(filename)
            print(f"Data imported from {filename}")
        except Exception as e:
            print(f"Error importing data: {e}")

def main():
    print("=== ULTIMATE CANCER RESEARCH ASSISTANT - GOAT EDITION ===")
    print("Advanced Features: Deep Learning, EHR Integration, Clinical Trials, Drug Interactions, Survival Analysis")
    print("⚠️  THIS SOFTWARE IS FOR RESEARCH AND EDUCATIONAL PURPOSES ONLY")
    print("⚠️  NOT FOR CLINICAL USE - ALWAYS CONSULT QUALIFIED ONCOLOGISTS")
    
    # Initialize the ultimate system
    cra = UltimateCancerResearchAssistant()
    
    # Generate sample data
    print("\n1. Generating comprehensive patient data...")
    cra.generate_sample_data(1000)
    
    # Train advanced models
    print("\n2. Training advanced machine learning models...")
    cra.train_advanced_models()
    
    # Show treatment protocols
    print("\n3. Loading treatment protocols...")
    cra.get_treatment_protocols()
    
    # Create advanced visualizations
    print("\n4. Generating advanced data visualizations...")
    cra.visualize_data()
    
    # Run sample analysis
    print("\n5. Running comprehensive patient analysis...")
    cra.run_comprehensive_analysis(patient_id=1)
    
    # Survival analysis
    print("\n6. Performing survival analysis...")
    try:
        cra.survival_analysis()
    except:
        print("Survival analysis requires lifelines library. Install with: pip install lifelines")
    
    # EHR demonstration
    print("\n7. Demonstrating EHR integration...")
    patient_record = {
        'patient_id': 1001,
        'name': 'John Doe',
        'dob': '1965-03-15',
        'gender': 'Male',
        'diagnosis': 'Breast Cancer',
        'diagnosis_date': '2023-01-10',
        'stage': 'IIA',
        'treatment_plan': 'Surgery followed by chemotherapy'
    }
    cra.add_patient_to_ehr(patient_record)
    
    medication_record = {
        'medication_name': 'Paclitaxel',
        'dosage': '175 mg/m2',
        'frequency': 'Every 3 weeks',
        'start_date': '2023-02-01',
        'end_date': '2023-08-01',
        'prescribed_by': 'Dr. Smith'
    }
    cra.add_medication_to_ehr(1001, medication_record)
    
    # Check drug interactions
    print("\n8. Checking drug interactions...")
    medications = ['Paclitaxel', 'Cisplatin', 'Warfarin']
    interactions = cra.check_drug_interactions(medications)
    if interactions:
        print("Potential drug interactions found:")
        for interaction in interactions:
            print(f"  • {interaction['drug1']} + {interaction['drug2']}: "
                  f"{interaction['severity']} - {interaction['description']}")
    else:
        print("No significant drug interactions found")
    
    # Clinical trial matching
    print("\n9. Finding clinical trials...")
    trials = cra.find_clinical_trials("Breast Cancer", 55, "II")
    if trials:
        print("Matching clinical trials:")
        for trial in trials:
            print(f"  • {trial['title']} ({trial['phase']}) at {trial['location']}")
    else:
        print("No matching clinical trials found")
    
    # Export data
    print("\n10. Exporting research data...")
    cra.export_data('cancer_research_export.csv')
    
    print("\n=== SYSTEM READY FOR ADVANCED RESEARCH ===")
    print("Available commands:")
    print("  analyze [patient_id] - Comprehensive patient analysis")
    print("  predict [model] - Prognosis prediction (models: random_forest, svm, neural_network, deep_learning, ensemble)")
    print("  protocols - Show treatment protocols")
    print("  trials [cancer_type] [age] [stage] - Find clinical trials")
    print("  drugs [med1,med2,...] - Check drug interactions")
    print("  ehr [patient_id] - View EHR data")
    print("  survival - Perform survival analysis")
    print("  export [filename] - Export data to CSV")
    print("  quit - Exit the system")
    
    # Interactive mode
    while True:
        try:
            command = input("\nEnter command: ").strip().lower()
            
            if command == 'quit':
                break
            elif command.startswith('analyze'):
                parts = command.split()
                if len(parts) > 1:
                    patient_id = int(parts[1])
                    cra.run_comprehensive_analysis(patient_id)
                else:
                    cra.run_comprehensive_analysis()
            elif command.startswith('predict'):
                parts = command.split()
                model_name = parts[1] if len(parts) > 1 else 'ensemble'
                print("Enter patient features (comma-separated):")
                print("age,tumor_size,tumor_grade,lymph_nodes,ki67_index,her2_status,er_status,pr_status,previous_treatments")
                features_input = input("Features: ")
                features = [float(x.strip()) for x in features_input.split(',')]
                result = cra.predict_patient_prognosis(features, model_name)
                if result:
                    print(f"Prognosis: {result['prognosis']} (Risk: {result['risk_score']:.2%})")
                    print(f"Model: {result['model_used']}")
            elif command == 'protocols':
                cra.get_treatment_protocols()
            elif command.startswith('trials'):
                parts = command.split()
                if len(parts) >= 4:
                    cancer_type, age, stage = parts[1], int(parts[2]), parts[3]
                    trials = cra.find_clinical_trials(cancer_type, age, stage)
                    if trials:
                        print("Matching clinical trials:")
                        for trial in trials:
                            print(f"  • {trial['title']} ({trial['phase']}) at {trial['location']}")
                    else:
                        print("No matching trials found")
                else:
                    print("Usage: trials [cancer_type] [age] [stage]")
            elif command.startswith('drugs'):
                parts = command.split()
                if len(parts) > 1:
                    medications = [m.strip() for m in parts[1].split(',')]
                    interactions = cra.check_drug_interactions(medications)
                    if interactions:
                        print("Potential drug interactions:")
                        for interaction in interactions:
                            print(f"  • {interaction['drug1']} + {interaction['drug2']}: "
                                  f"{interaction['severity']} - {interaction['description']}")
                    else:
                        print("No significant interactions found")
                else:
                    print("Usage: drugs [med1,med2,...]")
            elif command.startswith('ehr'):
                parts = command.split()
                if len(parts) > 1:
                    patient_id = int(parts[1])
                    ehr_data = cra.get_patient_ehr(patient_id)
                    if ehr_data:
                        print(f"EHR Data for Patient {patient_id}:")
                        print(f"  Name: {ehr_data['patient'][1]}")
                        print(f"  Diagnosis: {ehr_data['patient'][4]}")
                        print(f"  Stage: {ehr_data['patient'][6]}")
                        print("  Medications:")
                        for med in ehr_data['medications']:
                            print(f"    • {med[2]} - {med[3]} ({med[4]})")
                    else:
                        print("Patient not found in EHR")
                else:
                    print("Usage: ehr [patient_id]")
            elif command == 'survival':
                try:
                    cra.survival_analysis()
                except:
                    print("Survival analysis requires lifelines library. Install with: pip install lifelines")
            elif command.startswith('export'):
                parts = command.split()
                filename = parts[1] if len(parts) > 1 else 'export.csv'
                cra.export_data(filename)
            else:
                print("Unknown command. Type 'help' for available commands.")
                
        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"Error: {e}")
            print("Please try again.")
    
    print("\nThank you for using Ultimate Cancer Research Assistant - GOAT Edition!")

if __name__ == "__main__":
    main()