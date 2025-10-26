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