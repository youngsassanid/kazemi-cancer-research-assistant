// Tab functionality
document.addEventListener('DOMContentLoaded', function() {
    const tabButtons = document.querySelectorAll('.tab-button');
    const tabContents = document.querySelectorAll('.tab-content');
    
    tabButtons.forEach(button => {
        button.addEventListener('click', () => {
            const tabId = button.getAttribute('data-tab');
            
            // Remove active class from all buttons and contents
            tabButtons.forEach(btn => btn.classList.remove('active'));
            tabContents.forEach(content => content.classList.remove('active'));
            
            // Add active class to clicked button and corresponding content
            button.classList.add('active');
            document.getElementById(`${tabId}-tab`).classList.add('active');
        });
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
    
    // Initialize the application
    initializeApp();
});

function initializeApp() {
    // Set up event listeners for the research assistant
    setupPredictionForm();
    setupTreatmentRecommendations();
    setupDrugInteractions();
    setupClinicalTrials();
    setupVisualization();
}

function setupPredictionForm() {
    const form = document.getElementById('prediction-form');
    if (form) {
        form.addEventListener('submit', function(e) {
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
    }
}

function setupTreatmentRecommendations() {
    const button = document.getElementById('recommend-btn');
    if (button) {
        button.addEventListener('click', function() {
            const patientData = {
                tumor_grade: parseInt(document.getElementById('treatment-grade').value),
                tumor_size: parseFloat(document.getElementById('treatment-size').value),
                lymph_nodes: parseInt(document.getElementById('treatment-lymph').value),
                her2_status: parseInt(document.getElementById('treatment-her2').value),
                er_status: parseInt(document.getElementById('treatment-er').value),
                pr_status: parseInt(document.getElementById('treatment-pr').value),
                ki67_index: parseFloat(document.getElementById('treatment-ki67').value),
                previous_treatments: parseInt(document.getElementById('treatment-prev').value)
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
    }
}

function setupDrugInteractions() {
    const button = document.getElementById('check-interactions-btn');
    if (button) {
        button.addEventListener('click', function() {
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
    }
}

function setupClinicalTrials() {
    const button = document.getElementById('find-trials-btn');
    if (button) {
        button.addEventListener('click', function() {
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
    }
}

function setupVisualization() {
    const button = document.getElementById('generate-chart-btn');
    if (button) {
        button.addEventListener('click', function() {
            generateVisualization();
        });
    }
}