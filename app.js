// Sample patient data for visualization
function generateSampleData() {
    const data = [];
    for (let i = 0; i < 1000; i++) {
        data.push({
            age: Math.floor(Math.random() * 60) + 20,
            tumor_size: Math.random() * 10,
            tumor_grade: Math.floor(Math.random() * 4) + 1,
            lymph_nodes: Math.floor(Math.random() * 10),
            ki67_index: Math.random() * 100,
            her2_status: Math.random() > 0.7 ? 1 : 0,
            er_status: Math.random() > 0.6 ? 1 : 0,
            pr_status: Math.random() > 0.55 ? 1 : 0,
            previous_treatments: Math.floor(Math.random() * 5),
            prognosis: Math.random() > 0.7 ? 1 : 0
        });
    }
    return data;
}

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

// Generate sample visualization
function generateVisualization() {
    const data = generateSampleData();
    
    // Age distribution
    const ages = data.map(d => d.age);
    
    // Tumor size vs Grade
    const tumorGrades = data.map(d => d.tumor_grade);
    const tumorSizes = data.map(d => d.tumor_size);
    const prognosisColors = data.map(d => d.prognosis === 1 ? 'red' : 'blue');
    
    // Prognosis distribution
    const prognosisCounts = { Good: 0, Poor: 0 };
    data.forEach(d => {
        if (d.prognosis === 0) prognosisCounts.Good++;
        else prognosisCounts.Poor++;
    });
    
    // Ki67 Index by Prognosis
    const ki67Good = data.filter(d => d.prognosis === 0).map(d => d.ki67_index);
    const ki67Poor = data.filter(d => d.prognosis === 1).map(d => d.ki67_index);
    
    // Create subplots
    const traces = [
        {
            x: ages,
            type: 'histogram',
            name: 'Age Distribution',
            xbins: { size: 5 },
            marker: { color: '#2563eb' }
        },
        {
            x: tumorGrades,
            y: tumorSizes,
            mode: 'markers',
            type: 'scatter',
            name: 'Tumor Characteristics',
            marker: {
                color: prognosisColors,
                colorscale: [['0.0', 'blue'], ['1.0', 'red']],
                showscale: false
            }
        },
        {
            labels: Object.keys(prognosisCounts),
            values: Object.values(prognosisCounts),
            type: 'pie',
            name: 'Prognosis Distribution'
        },
        {
            x: ki67Good,
            type: 'histogram',
            name: 'Ki67 - Good Prognosis',
            opacity: 0.7,
            marker: { color: 'blue' }
        },
        {
            x: ki67Poor,
            type: 'histogram',
            name: 'Ki67 - Poor Prognosis',
            opacity: 0.7,
            marker: { color: 'red' }
        }
    ];
    
    const layout = {
        title: 'Cancer Patient Data Analysis Dashboard',
        grid: { rows: 2, columns: 3, pattern: 'independent' },
        height: 600
    };
    
    Plotly.newPlot('chart-container', traces, layout);
}

// Event listeners
document.getElementById('prediction-form').addEventListener('submit', function(e) {
    e.preventDefault();
    
    const patientData = {
        age: parseFloat(document.getElementById('age').value),
        tumor_size: parseFloat(document.getElementById('tumor-size').value),
        tumor_grade: parseInt(document.getElementById('tumor-grade').value),
        lymph_nodes: parseInt(document.getElementById('lymph-nodes').value),
        ki67_index: parseFloat(document.getElementById('ki67').value),
        her2_status: parseInt(document.getElementById('her2').value),
        er_status: parseInt(document.getElementById('er').value),
        pr_status: parseInt(document.getElementById('pr').value),
        previous_treatments: parseInt(document.getElementById('previous-treatments').value)
    };
    
    const result = predictPatientPrognosis(patientData);
    
    document.getElementById('overall-prognosis').textContent = result.prognosis;
    document.getElementById('risk-score').textContent = (result.riskScore * 100).toFixed(2) + '%';
    
    let modelPredictionsHTML = '';
    for (const [model, score] of Object.entries(result.models)) {
        modelPredictionsHTML += `<div>${model}: ${(score * 100).toFixed(2)}%</div>`;
    }
    document.getElementById('model-predictions').innerHTML = modelPredictionsHTML;
    
    document.getElementById('prediction-result').style.display = 'block';
});

document.getElementById('recommend-btn').addEventListener('click', function() {
    const patientData = {
        tumor_grade: parseInt(document.getElementById('tumor-grade').value),
        tumor_size: parseFloat(document.getElementById('tumor-size').value),
        lymph_nodes: parseInt(document.getElementById('lymph-nodes').value),
        her2_status: parseInt(document.getElementById('her2').value),
        er_status: parseInt(document.getElementById('er').value),
        pr_status: parseInt(document.getElementById('pr').value),
        ki67_index: parseFloat(document.getElementById('ki67').value),
        previous_treatments: parseInt(document.getElementById('previous-treatments').value)
    };
    
    const recommendations = recommendTreatment(patientData);
    let recommendationsHTML = '<ul>';
    recommendations.forEach(rec => {
        recommendationsHTML += `<li>${rec}</li>`;
    });
    recommendationsHTML += '</ul>';
    
    document.getElementById('recommendations-list').innerHTML = recommendationsHTML;
    document.getElementById('recommendations-result').style.display = 'block';
});

document.getElementById('check-interactions-btn').addEventListener('click', function() {
    const medicationsInput = document.getElementById('medications').value;
    const medications = medicationsInput.split(',').map(m => m.trim());
    
    const interactions = checkDrugInteractions(medications);
    
    if (interactions.length > 0) {
        let interactionsHTML = '<ul>';
        interactions.forEach(interaction => {
            interactionsHTML += `<li><strong>${interaction.drug1} + ${interaction.drug2}</strong>: ${interaction.severity} - ${interaction.description}</li>`;
        });
        interactionsHTML += '</ul>';
        document.getElementById('interactions-list').innerHTML = interactionsHTML;
    } else {
        document.getElementById('interactions-list').innerHTML = '<p>No significant drug interactions found</p>';
    }
    
    document.getElementById('interactions-result').style.display = 'block';
});

document.getElementById('find-trials-btn').addEventListener('click', function() {
    const cancerType = document.getElementById('cancer-type').value;
    const patientAge = parseInt(document.getElementById('patient-age').value);
    const patientStage = document.getElementById('patient-stage').value;
    
    const trials = findClinicalTrials(cancerType, patientAge, patientStage);
    
    if (trials.length > 0) {
        let trialsHTML = '<ul>';
        trials.forEach(trial => {
            trialsHTML += `<li><strong>${trial.title}</strong> (${trial.phase}) at ${trial.location}</li>`;
        });
        trialsHTML += '</ul>';
        document.getElementById('trials-list').innerHTML = trialsHTML;
    } else {
        document.getElementById('trials-list').innerHTML = '<p>No matching clinical trials found</p>';
    }
    
    document.getElementById('trials-result').style.display = 'block';
});

document.getElementById('generate-chart-btn').addEventListener('click', function() {
    generateVisualization();
});

// Smooth scrolling for navigation links
document.querySelectorAll('a[href^="#"]').forEach(anchor => {
    anchor.addEventListener('click', function (e) {
        e.preventDefault();
        document.querySelector(this.getAttribute('href')).scrollIntoView({
            behavior: 'smooth'
        });
    });
});

// Simple animation on scroll
const observerOptions = {
    root: null,
    rootMargin: '0px',
    threshold: 0.1
};

const observer = new IntersectionObserver((entries) => {
    entries.forEach(entry => {
        if (entry.isIntersecting) {
            entry.target.classList.add('animated');
        }
    });
}, observerOptions);

document.querySelectorAll('.feature-card').forEach(card => {
    observer.observe(card);
});