// Frontend JavaScript for AI Misinformation Detector

class MisinformationDetector {
    constructor() {
        this.initializeEventListeners();
        this.loadStats();
    }

    initializeEventListeners() {
        // Single claim form
        document.getElementById('claimForm').addEventListener('submit', (e) => {
            e.preventDefault();
            this.analyzeClaim();
        });

        // Batch upload form
        document.getElementById('batchForm').addEventListener('submit', (e) => {
            e.preventDefault();
            this.analyzeBatch();
        });
    }

    async analyzeClaim() {
        const claimText = document.getElementById('claimText').value.trim();
        const sourceUrl = document.getElementById('sourceUrl').value.trim();

        if (!claimText) {
            this.showError('Please enter a claim to analyze');
            return;
        }

        this.showLoading();

        try {
            const response = await fetch('/analyze', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    text: claimText,
                    source_url: sourceUrl || null
                })
            });

            if (!response.ok) {
                throw new Error(`HTTP ${response.status}: ${response.statusText}`);
            }

            const result = await response.json();
            this.displayResults(result);

        } catch (error) {
            console.error('Analysis failed:', error);
            this.showError(`Analysis failed: ${error.message}`);
        } finally {
            this.hideLoading();
        }
    }

    async analyzeBatch() {
        const fileInput = document.getElementById('csvFile');
        const file = fileInput.files[0];

        if (!file) {
            this.showError('Please select a CSV file');
            return;
        }

        this.showLoading();

        try {
            const formData = new FormData();
            formData.append('file', file);

            const response = await fetch('/analyze/upload', {
                method: 'POST',
                body: formData
            });

            if (!response.ok) {
                throw new Error(`HTTP ${response.status}: ${response.statusText}`);
            }

            const result = await response.json();
            this.displayBatchResults(result);

        } catch (error) {
            console.error('Batch analysis failed:', error);
            this.showError(`Batch analysis failed: ${error.message}`);
        } finally {
            this.hideLoading();
        }
    }

    displayResults(result) {
        const resultsContent = document.getElementById('resultsContent');
        
        // Determine score color
        let scoreClass = 'score-low';
        if (result.reliability >= 70) scoreClass = 'score-high';
        else if (result.reliability >= 40) scoreClass = 'score-medium';

        // Determine classification badge class
        const classificationClass = `classification-${result.classification.toLowerCase()}`;

        let html = `
            <div class="row">
                <div class="col-md-8">
                    <h5>Claim:</h5>
                    <p class="lead">"${result.claim}"</p>
                    
                    <div class="mt-3">
                        <span class="classification-badge ${classificationClass}">
                            ${result.classification}
                        </span>
                    </div>
                </div>
                <div class="col-md-4 text-center">
                    <div class="reliability-score ${scoreClass}">
                        ${Math.round(result.reliability)}%
                    </div>
                    <small class="text-muted">Reliability Score</small>
                    <div class="mt-2">
                        <small class="processing-time">
                            <i class="fas fa-clock me-1"></i>
                            ${result.processing_time_ms}ms
                        </small>
                    </div>
                </div>
            </div>

            <div class="mt-4">
                <h6><i class="fas fa-brain me-2"></i>Reasoning:</h6>
                <p class="text-muted">${result.reasoning}</p>
            </div>
        `;

        if (result.evidence && result.evidence.length > 0) {
            html += `
                <div class="mt-4">
                    <h6><i class="fas fa-quote-left me-2"></i>Supporting Evidence:</h6>
                    ${result.evidence.map(snippet => `
                        <div class="evidence-snippet">
                            "${snippet}"
                        </div>
                    `).join('')}
                </div>
            `;
        }

        if (result.evidence_sources && result.evidence_sources.length > 0) {
            html += `
                <div class="mt-4">
                    <h6><i class="fas fa-link me-2"></i>Sources:</h6>
                    ${result.evidence_sources.map(source => `
                        <div class="source-item">
                            <div class="d-flex justify-content-between align-items-start">
                                <div>
                                    <div class="source-name">${source.source_name}</div>
                                    <div class="text-truncate">${source.title}</div>
                                    <small class="text-muted">
                                        <a href="${source.source_url}" target="_blank" rel="noopener">
                                            ${source.source_url}
                                        </a>
                                    </small>
                                </div>
                                <span class="relevance-score">
                                    ${Math.round(source.relevance_score * 100)}% relevance
                                </span>
                            </div>
                        </div>
                    `).join('')}
                </div>
            `;
        }

        resultsContent.innerHTML = html;
        document.getElementById('resultsSection').style.display = 'block';
        
        // Scroll to results
        document.getElementById('resultsSection').scrollIntoView({ behavior: 'smooth' });
    }

    displayBatchResults(result) {
        const resultsContent = document.getElementById('resultsContent');
        
        const summary = result.summary;
        const results = result.results;

        let html = `
            <div class="alert alert-info alert-custom">
                <h5><i class="fas fa-info-circle me-2"></i>Batch Analysis Summary</h5>
                <div class="row mt-3">
                    <div class="col-md-3">
                        <strong>Total Analyzed:</strong> ${summary.total_analyzed}
                    </div>
                    <div class="col-md-3">
                        <strong>Avg. Reliability:</strong> ${summary.average_reliability}%
                    </div>
                    <div class="col-md-6">
                        <strong>Classifications:</strong>
                        True: ${summary.classification_counts.True || 0}, 
                        False: ${summary.classification_counts.False || 0}, 
                        Misleading: ${summary.classification_counts.Misleading || 0}, 
                        Unverified: ${summary.classification_counts.Unverified || 0}
                    </div>
                </div>
            </div>

            <div class="mt-4">
                <h6>Individual Results:</h6>
                <div class="table-responsive">
                    <table class="table table-striped">
                        <thead>
                            <tr>
                                <th>Claim</th>
                                <th>Classification</th>
                                <th>Reliability</th>
                                <th>Evidence Count</th>
                            </tr>
                        </thead>
                        <tbody>
                            ${results.map(r => {
                                const classificationClass = `classification-${r.classification.toLowerCase()}`;
                                const scoreClass = r.reliability >= 70 ? 'score-high' : 
                                                 r.reliability >= 40 ? 'score-medium' : 'score-low';
                                return `
                                    <tr>
                                        <td class="text-truncate" style="max-width: 300px;" title="${r.claim}">
                                            ${r.claim.substring(0, 100)}${r.claim.length > 100 ? '...' : ''}
                                        </td>
                                        <td>
                                            <span class="badge ${classificationClass.replace('classification-', 'bg-')}">
                                                ${r.classification}
                                            </span>
                                        </td>
                                        <td class="${scoreClass}">${Math.round(r.reliability)}%</td>
                                        <td>${r.evidence.length}</td>
                                    </tr>
                                `;
                            }).join('')}
                        </tbody>
                    </table>
                </div>
            </div>
        `;

        resultsContent.innerHTML = html;
        document.getElementById('resultsSection').style.display = 'block';
        
        // Scroll to results
        document.getElementById('resultsSection').scrollIntoView({ behavior: 'smooth' });
    }

    async loadStats() {
        try {
            const response = await fetch('/stats');
            if (!response.ok) {
                throw new Error(`HTTP ${response.status}: ${response.statusText}`);
            }

            const stats = await response.json();
            this.displayStats(stats);

        } catch (error) {
            console.error('Failed to load stats:', error);
            document.getElementById('statsContent').innerHTML = `
                <div class="alert alert-warning alert-custom">
                    <i class="fas fa-exclamation-triangle me-2"></i>
                    Failed to load statistics: ${error.message}
                </div>
            `;
        }
    }

    displayStats(stats) {
        const statsContent = document.getElementById('statsContent');
        
        const html = `
            <div class="stats-grid">
                <div class="stat-card">
                    <div class="stat-number">${stats.total_claims}</div>
                    <div class="stat-label">Total Claims Analyzed</div>
                </div>
                <div class="stat-card">
                    <div class="stat-number">${stats.average_reliability}%</div>
                    <div class="stat-label">Average Reliability</div>
                </div>
                <div class="stat-card">
                    <div class="stat-number">${stats.classification_breakdown.True || 0}</div>
                    <div class="stat-label">True Claims</div>
                </div>
                <div class="stat-card">
                    <div class="stat-number">${stats.classification_breakdown.False || 0}</div>
                    <div class="stat-label">False Claims</div>
                </div>
                <div class="stat-card">
                    <div class="stat-number">${stats.classification_breakdown.Misleading || 0}</div>
                    <div class="stat-label">Misleading Claims</div>
                </div>
                <div class="stat-card">
                    <div class="stat-number">${stats.classification_breakdown.Unverified || 0}</div>
                    <div class="stat-label">Unverified Claims</div>
                </div>
            </div>
        `;

        statsContent.innerHTML = html;
    }

    showLoading() {
        document.getElementById('loadingSection').style.display = 'block';
        document.getElementById('resultsSection').style.display = 'none';
        document.getElementById('analyzeBtn').disabled = true;
        document.getElementById('analyzeBtn').innerHTML = `
            <span class="spinner-border spinner-border-sm me-2" role="status"></span>
            Analyzing...
        `;
    }

    hideLoading() {
        document.getElementById('loadingSection').style.display = 'none';
        document.getElementById('analyzeBtn').disabled = false;
        document.getElementById('analyzeBtn').innerHTML = `
            <i class="fas fa-cog me-2"></i>
            Analyze Claim
        `;
    }

    showError(message) {
        const resultsContent = document.getElementById('resultsContent');
        resultsContent.innerHTML = `
            <div class="alert alert-danger alert-custom">
                <i class="fas fa-exclamation-circle me-2"></i>
                <strong>Error:</strong> ${message}
            </div>
        `;
        document.getElementById('resultsSection').style.display = 'block';
    }
}

// Global function for stats button
function loadStats() {
    if (window.detector) {
        window.detector.loadStats();
    }
}

// Initialize the application
document.addEventListener('DOMContentLoaded', function() {
    window.detector = new MisinformationDetector();
});
