// Treatment protocols data
const treatmentProtocols = {
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
};

// Drug interactions data
const drugInteractions = {
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
};

// Clinical trials data
const clinicalTrials = [
    {
        'trial_id': 'NCT00001234',
        'title': 'Immunotherapy for Advanced Melanoma',
        'phase': 'Phase III',
        'cancer_type': 'Melanoma',
        'eligibility_criteria': 'Stage III/IV melanoma, age 18-75',
        'location': 'Mayo Clinic',
        'status': 'Recruiting'
    },
    {
        'trial_id': 'NCT00005678',
        'title': 'Targeted Therapy in HER2+ Breast Cancer',
        'phase': 'Phase II',
        'cancer_type': 'Breast Cancer',
        'eligibility_criteria': 'HER2+, metastatic, age 18-80',
        'location': 'MD Anderson',
        'status': 'Active'
    },
    {
        'trial_id': 'NCT00009012',
        'title': 'CAR-T Cell Therapy for Leukemia',
        'phase': 'Phase I',
        'cancer_type': 'Leukemia',
        'eligibility_criteria': 'Relapsed/refractory, age 18-65',
        'location': 'Johns Hopkins',
        'status': 'Not yet recruiting'
    },
    {
        'trial_id': 'NCT00003456',
        'title': 'Combination Chemotherapy for Lung Cancer',
        'phase': 'Phase III',
        'cancer_type': 'Lung Cancer',
        'eligibility_criteria': 'NSCLC, stage IIIB/IV, age 18-75',
        'location': 'Memorial Sloan Kettering',
        'status': 'Completed'
    }
];