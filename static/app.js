document.addEventListener('DOMContentLoaded', () => {
    const form = document.getElementById('predict-form');
    const textArea = document.getElementById('email-text');
    const submitBtn = document.getElementById('submit-btn');
    const submitText = submitBtn.querySelector('span');
    const spinner = document.getElementById('loading-spinner');
    
    const resultSection = document.getElementById('result-section');
    const predictionLabel = document.getElementById('prediction-label');
    const confidenceLabel = document.getElementById('prediction-confidence');
    const highlightedTextContainer = document.getElementById('highlighted-text');

    form.addEventListener('submit', async (e) => {
        e.preventDefault();
        
        const text = textArea.value.trim();
        if (!text) return;

        // UI Loading State
        submitBtn.disabled = true;
        submitText.textContent = "Analyzing...";
        spinner.classList.remove('hidden');
        resultSection.classList.add('hidden');

        try {
            const response = await fetch('/predict', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({ text })
            });

            if (!response.ok) {
                throw new Error("API Request Failed");
            }

            const data = await response.json();
            
            // Populate Results
            predictionLabel.textContent = data.prediction;
            predictionLabel.className = data.prediction === 'Phishing' ? 'text-red' : 'text-green';
            
            const confidencePercent = (data.confidence * 100).toFixed(1);
            confidenceLabel.textContent = `${confidencePercent}%`;

            // Render Explainability Highlights
            renderAttributions(data.attributions);

            // Show Result Section
            resultSection.classList.remove('hidden');
            
            // Scroll to results smoothly
            resultSection.scrollIntoView({ behavior: 'smooth', block: 'nearest' });

        } catch (error) {
            alert('An error occurred while analyzing the text. Please ensure the backend is running.');
            console.error(error);
        } finally {
            // Restore UI State
            submitBtn.disabled = false;
            submitText.textContent = "Analyze Content";
            spinner.classList.add('hidden');
        }
    });

    function renderAttributions(attributions) {
        highlightedTextContainer.innerHTML = '';
        
        // Find max absolute score for normalization in the frontend
        let maxScore = 0;
        attributions.forEach(attr => {
            if (Math.abs(attr.score) > maxScore) {
                maxScore = Math.abs(attr.score);
            }
        });

        // Safety against zero division
        if (maxScore === 0) maxScore = 1;

        attributions.forEach(attr => {
            let token = attr.token;
            // Clean up BERT tokenization artifacts
            if (token.startsWith('##')) {
                token = token.substring(2);
            } else if (highlightedTextContainer.children.length > 0) {
                // Add a space before new words, but not before subwords
                highlightedTextContainer.appendChild(document.createTextNode(' '));
            }

            const span = document.createElement('span');
            span.className = 'token';
            span.textContent = token;

            // Calculate background color opacity based on score magnitude
            // Score > 0 -> Phishing (Red)
            // Score < 0 -> Legitimate (Green)
            const normalizedScore = attr.score / maxScore; // range [-1, 1]
            const intensity = Math.min(Math.abs(normalizedScore) * 1.5, 0.9); // Boost intensity for visibility, max 0.9 opacity

            if (attr.score > 0) {
                span.style.backgroundColor = `rgba(239, 68, 68, ${intensity})`; // Red
            } else {
                span.style.backgroundColor = `rgba(16, 185, 129, ${intensity})`; // Green
            }

            // Title tooltip for the exact score
            span.title = `Score: ${attr.score.toFixed(4)}`;

            highlightedTextContainer.appendChild(span);
        });
    }
});
