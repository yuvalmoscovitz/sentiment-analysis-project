document.addEventListener("DOMContentLoaded", function() {
    const form = document.getElementById("stock-form");
    const resultDiv = document.getElementById("result");
    const stockTickerInput = document.getElementById("stock-ticker");

    // Enforce uppercase input
    stockTickerInput.addEventListener("input", function() {
        this.value = this.value.toUpperCase();
    });

    function renderCharts(basicStats, advancedStats) {
        // Basic Statistics Chart
        const basicCtx = document.getElementById('basic-chart').getContext('2d');
        new Chart(basicCtx, {
            type: 'bar',
            data: {
                labels: ['Total News Articles', 'Neutral Sentiments', 'Positive Sentiments', 'Negative Sentiments'],
                datasets: [{
                    label: 'Count',
                    data: [
                        basicStats['Total News Articles'],
                        basicStats['Neutral Sentiments'],
                        basicStats['Positive Sentiments'],
                        basicStats['Negative Sentiments']
                    ],
                    backgroundColor: [
                        'rgba(54, 162, 235, 0.5)',  // Total
                        'rgba(255, 206, 86, 0.5)',   // Neutral
                        'rgba(75, 192, 192, 0.5)',  // Positive
                        'rgba(255, 99, 132, 0.5)'   // Negative
                    ],
                    borderColor: [
                        'rgba(54, 162, 235, 1)',    // Total
                        'rgba(255, 206, 86, 1)',     // Neutral
                        'rgba(75, 192, 192, 1)',    // Positive
                        'rgba(255, 99, 132, 1)'     // Negative
                    ],
                    borderWidth: 1
                }]
            }
        });

        // Advanced Statistics Chart (Example: Sentiment Percentage)
        const advancedCtx = document.getElementById('advanced-chart').getContext('2d');
        new Chart(advancedCtx, {
            type: 'pie',
            data: {
                labels: ['Positive Percentage', 'Neutral Percentage', 'Negative Percentage'],
                datasets: [{
                    data: [
                        advancedStats['Positive Percentage'],
                        advancedStats['Neutral Percentage'],
                        advancedStats['Negative Percentage']
                    ],
                    backgroundColor: [
                        'rgba(75, 192, 192, 0.5)',
                        'rgba(255, 206, 86, 0.5)',
                        'rgba(255, 99, 132, 0.5)'
                    ]
                }]
            }
        });
    }

    form.addEventListener("submit", function(event) {
        event.preventDefault();
        const stockTicker = stockTickerInput.value;

        fetch("/analyze_sentiment", {
            method: "POST",
            headers: {
                "Content-Type": "application/json",
            },
            body: JSON.stringify({ stock_ticker: stockTicker }),
        })
        .then(response => response.json())
        .then(data => {
            if (data.error) {
                resultDiv.innerHTML = `<div class="error-message">Error: Invalid or unknown stock ticker.</div>`;
                return;
            }

            let basicStatsHtml = "<h2 class='basic-stats-header'>Basic Statistics</h2><ul class='basic-stats-list'>";
            const basicStatsOrder = ['Total News Articles', 'Neutral Sentiments', 'Positive Sentiments', 'Negative Sentiments'];
            for (const key of basicStatsOrder) {
                basicStatsHtml += `<li>${key}: ${data.statistics['Basic Statistics'][key]}</li>`;
            }
            basicStatsHtml += "</ul>";

            let advancedStatsHtml = "<h2 class='advanced-stats-header'>Advanced Statistics</h2><ul class='advanced-stats-list'>";
            for (const [key, value] of Object.entries(data.statistics['Advanced Statistics'])) {
                advancedStatsHtml += `<li>${key}: ${value.toFixed(2)}</li>`;
            }
            advancedStatsHtml += "</ul>";

            resultDiv.innerHTML = `
                <h2>Stock Ticker: ${data.stock_ticker}</h2>
                ${basicStatsHtml}
                ${advancedStatsHtml}
                <div id="basic-chart-container" class="chart-container">
                    <canvas id="basic-chart"></canvas>
                </div>
                <div id="advanced-chart-container" class="chart-container">
                    <canvas id="advanced-chart"></canvas>
                </div>
            `;

            // Render the charts
            renderCharts(data.statistics['Basic Statistics'], data.statistics['Advanced Statistics']);
        })
        .catch(error => {
            resultDiv.innerHTML = `<div class="error-message">Error: ${error}</div>`;
        });
    });
});
