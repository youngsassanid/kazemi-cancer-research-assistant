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

// Predict patient prognosis (simplified model)
function predictPatientPrognosis(patientData) {
    // Simplified risk calculation
    const riskScore = (
        patientData.tumor_size * 0.3 +
        patientData.tumor_grade * 0.4 +
        patientData.lymph_nodes * 0.2 +
        (1 - patientData.er_status) * 0.5 +
        patientData.her2_status * 0.3 +
        (1 - patientData.pr_status) * 0.4 +
        patientData.previous_treatments * 0.1 +
        Math.random() * 0.5
    ) / 5;
    
    const prognosis = riskScore > 0.5 ? 'Poor' : 'Good';
    
    // Simulate different model predictions
    const models = {
        'Random Forest': Math.min(0.95, Math.max(0.05, riskScore + (Math.random() * 0.2 - 0.1))),
        'SVM': Math.min(0.95, Math.max(0.05, riskScore + (Math.random() * 0.15 - 0.075))),
        'Neural Network': Math.min(0.95, Math.max(0.05, riskScore + (Math.random() * 0.25 - 0.125))),
        'Deep Learning': Math.min(0.95, Math.max(0.05, riskScore + (Math.random() * 0.18 - 0.09)))
    };
    
    return {
        prognosis: prognosis,
        riskScore: riskScore,
        models: models
    };
}

// Recommend treatment based on patient data
function recommendTreatment(patientData) {
    const recommendations = [];
    
    if (patientData.tumor_grade >= 3) {
        recommendations.push("Consider aggressive treatment approach");
    }
    if (patientData.tumor_size > 5) {
        recommendations.push("Large tumor detected - surgical consultation recommended");
    }
    if (patientData.lymph_nodes > 3) {
        recommendations.push("Lymph node involvement - systemic therapy consideration");
    }
    if (patientData.her2_status === 1) {
        recommendations.push("HER2 positive - consider targeted therapy (trastuzumab)");
    }
    if (patientData.er_status === 1 && patientData.pr_status === 1) {
        recommendations.push("Hormone receptor positive - hormone therapy may be beneficial");
    }
    if (patientData.ki67_index > 50) {
        recommendations.push("High proliferation rate - consider combination therapy");
    }
    if (patientData.previous_treatments > 2) {
        recommendations.push("Multiple previous treatments - consider clinical trials");
    }
    
    recommendations.push("Multidisciplinary team consultation recommended");
    recommendations.push("Consider clinical trials for novel therapies");
    recommendations.push("Regular monitoring and follow-up essential");
    recommendations.push("Genetic counseling may be beneficial");
    recommendations.push("Supportive care and symptom management important");
    
    return recommendations;
}

// Check drug interactions
function checkDrugInteractions(medications) {
    const interactions = [];
    
    for (const med of medications) {
        if (drugInteractions[med]) {
            const interactionData = drugInteractions[med];
            for (let i = 0; i < interactionData.interactions.length; i++) {
                const interactingDrug = interactionData.interactions[i];
                if (medications.includes(interactingDrug)) {
                    interactions.push({
                        drug1: med,
                        drug2: interactingDrug,
                        severity: interactionData.severity[i],
                        description: interactionData.description
                    });
                }
            }
        }
    }
    
    return interactions;
}

// Find matching clinical trials
function findClinicalTrials(cancerType, patientAge, patientStage) {
    const matchingTrials = [];
    
    for (const trial of clinicalTrials) {
        if (trial.cancer_type.toLowerCase().includes(cancerType.toLowerCase())) {
            matchingTrials.push(trial);
        }
    }
    
    return matchingTrials;
}