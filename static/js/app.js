// Deepfake Detection App JavaScript

document.addEventListener('DOMContentLoaded', function() {
    const uploadArea = document.getElementById('uploadArea');
    const fileInput = document.getElementById('fileInput');
    const uploadForm = document.getElementById('uploadForm');
    const analyzeBtn = document.getElementById('analyzeBtn');
    const resultsSection = document.getElementById('resultsSection');
    const loadingSection = document.getElementById('loadingSection');
    const resultsBody = document.getElementById('resultsBody');

    let selectedFile = null;

    // Upload area click handler
    uploadArea.addEventListener('click', () => {
        fileInput.click();
    });

    // File input change handler
    fileInput.addEventListener('change', handleFileSelect);

    // Drag and drop handlers
    uploadArea.addEventListener('dragover', handleDragOver);
    uploadArea.addEventListener('dragleave', handleDragLeave);
    uploadArea.addEventListener('drop', handleDrop);

    // Form submit handler
    uploadForm.addEventListener('submit', handleFormSubmit);

    function handleFileSelect(event) {
        const file = event.target.files[0];
        if (file) {
            selectedFile = file;
            updateUploadArea(file);
            analyzeBtn.disabled = false;
        }
    }

    function handleDragOver(event) {
        event.preventDefault();
        uploadArea.classList.add('dragover');
    }

    function handleDragLeave(event) {
        event.preventDefault();
        uploadArea.classList.remove('dragover');
    }

    function handleDrop(event) {
        event.preventDefault();
        uploadArea.classList.remove('dragover');
        
        const files = event.dataTransfer.files;
        if (files.length > 0) {
            const file = files[0];
            if (isValidFile(file)) {
                selectedFile = file;
                fileInput.files = files;
                updateUploadArea(file);
                analyzeBtn.disabled = false;
            } else {
                showAlert('Please select a valid file format (MP4, AVI, MOV, MKV, WebM, JPG, PNG)', 'warning');
            }
        }
    }

    function isValidFile(file) {
        const allowedTypes = [
            'video/mp4', 'video/avi', 'video/quicktime', 'video/x-matroska', 'video/webm',
            'image/jpeg', 'image/jpg', 'image/png'
        ];
        return allowedTypes.includes(file.type) || 
               /\.(mp4|avi|mov|mkv|webm|jpg|jpeg|png)$/i.test(file.name);
    }

    function updateUploadArea(file) {
        const fileType = file.type.startsWith('video/') ? 'video' : 'image';
        const fileSize = formatFileSize(file.size);
        
        uploadArea.innerHTML = `
            <i class="fas fa-${fileType === 'video' ? 'video' : 'image'} fa-3x text-primary mb-3"></i>
            <h5>${file.name}</h5>
            <p class="text-muted">${fileType.toUpperCase()} • ${fileSize}</p>
            <small class="text-muted">Click to change file</small>
        `;
    }

    function formatFileSize(bytes) {
        if (bytes === 0) return '0 Bytes';
        const k = 1024;
        const sizes = ['Bytes', 'KB', 'MB', 'GB'];
        const i = Math.floor(Math.log(bytes) / Math.log(k));
        return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
    }

    function handleFormSubmit(event) {
        event.preventDefault();
        
        if (!selectedFile) {
            showAlert('Please select a file first', 'warning');
            return;
        }

        // Show loading state
        showLoading();
        
        // Create FormData
        const formData = new FormData();
        formData.append('file', selectedFile);

        // Send request
        fetch('/upload', {
            method: 'POST',
            body: formData
        })
        .then(response => response.json())
        .then(data => {
            hideLoading();
            if (data.success) {
                displayResults(data.result);
            } else {
                showAlert(data.error || 'Analysis failed', 'danger');
            }
        })
        .catch(error => {
            hideLoading();
            console.error('Error:', error);
            showAlert('Network error occurred', 'danger');
        });
    }

    function showLoading() {
        loadingSection.style.display = 'block';
        resultsSection.style.display = 'none';
        analyzeBtn.disabled = true;
    }

    function hideLoading() {
        loadingSection.style.display = 'none';
        analyzeBtn.disabled = false;
    }

    function displayResults(result) {
        resultsSection.style.display = 'block';
        resultsSection.scrollIntoView({ behavior: 'smooth' });
        
        const isDeepfake = result.is_deepfake;
        const confidence = result.confidence || result.average_confidence || 0;
        const realProb = result.real_probability || (1 - confidence);
        const fakeProb = result.fake_probability || confidence;
        
        let html = `
            <div class="result-card ${isDeepfake ? 'fake-detected' : 'real-detected'} fade-in-up">
                <div class="row">
                    <div class="col-md-8">
                        <h5 class="mb-3">
                            <i class="fas fa-${isDeepfake ? 'exclamation-triangle text-danger' : 'check-circle text-success'} me-2"></i>
                            ${isDeepfake ? 'Deepfake Detected' : 'Authentic Content'}
                        </h5>
                        
                        <div class="file-info">
                            <h6><i class="fas fa-file me-2"></i>File Information</h6>
                            <p><strong>Type:</strong> ${result.type.toUpperCase()}</p>
                            <p><strong>Confidence:</strong> ${(confidence * 100).toFixed(2)}%</p>
                        </div>
                        
                        <div class="mb-3">
                            <h6>Analysis Results</h6>
                            <div class="row">
                                <div class="col-6">
                                    <div class="text-center">
                                        <div class="h4 text-success">${(realProb * 100).toFixed(1)}%</div>
                                        <small class="text-muted">Real Probability</small>
                                    </div>
                                </div>
                                <div class="col-6">
                                    <div class="text-center">
                                        <div class="h4 text-danger">${(fakeProb * 100).toFixed(1)}%</div>
                                        <small class="text-muted">Fake Probability</small>
                                    </div>
                                </div>
                            </div>
                        </div>
                        
                        <div class="confidence-bar mb-3">
                            <div class="confidence-indicator" style="width: ${confidence * 100}%"></div>
                        </div>
                    </div>
                    <div class="col-md-4">
                        <div class="text-center">
                            <div class="display-6 ${isDeepfake ? 'text-danger' : 'text-success'}">
                                ${isDeepfake ? '⚠️' : '✅'}
                            </div>
                            <h6 class="mt-2">${isDeepfake ? 'High Risk' : 'Low Risk'}</h6>
                        </div>
                    </div>
                </div>
        `;

        // Add video-specific information
        if (result.type === 'video' && result.video_info) {
            const videoInfo = result.video_info;
            html += `
                <div class="mt-3">
                    <h6><i class="fas fa-video me-2"></i>Video Analysis</h6>
                    <div class="row">
                        <div class="col-md-3">
                            <small class="text-muted">Duration</small><br>
                            <strong>${videoInfo.duration_seconds.toFixed(1)}s</strong>
                        </div>
                        <div class="col-md-3">
                            <small class="text-muted">Frames Analyzed</small><br>
                            <strong>${videoInfo.processed_frames}/${videoInfo.total_frames}</strong>
                        </div>
                        <div class="col-md-3">
                            <small class="text-muted">FPS</small><br>
                            <strong>${videoInfo.fps.toFixed(1)}</strong>
                        </div>
                        <div class="col-md-3">
                            <small class="text-muted">Confidence Std</small><br>
                            <strong>${(result.confidence_std * 100).toFixed(1)}%</strong>
                        </div>
                    </div>
                </div>
            `;
        }

        // Add frame-by-frame analysis for videos
        if (result.frame_analysis && result.frame_analysis.length > 0) {
            html += `
                <div class="mt-3">
                    <h6><i class="fas fa-list me-2"></i>Frame-by-Frame Analysis</h6>
                    <div class="table-responsive">
                        <table class="table table-sm">
                            <thead>
                                <tr>
                                    <th>Frame</th>
                                    <th>Timestamp</th>
                                    <th>Confidence</th>
                                    <th>Result</th>
                                </tr>
                            </thead>
                            <tbody>
            `;
            
            result.frame_analysis.slice(0, 10).forEach(frame => {
                html += `
                    <tr>
                        <td>${frame.frame_number}</td>
                        <td>${frame.timestamp.toFixed(2)}s</td>
                        <td>${(frame.confidence * 100).toFixed(1)}%</td>
                        <td>
                            <span class="badge ${frame.is_deepfake ? 'bg-danger' : 'bg-success'}">
                                ${frame.is_deepfake ? 'Fake' : 'Real'}
                            </span>
                        </td>
                    </tr>
                `;
            });
            
            if (result.frame_analysis.length > 10) {
                html += `
                    <tr>
                        <td colspan="4" class="text-center text-muted">
                            ... and ${result.frame_analysis.length - 10} more frames
                        </td>
                    </tr>
                `;
            }
            
            html += `
                            </tbody>
                        </table>
                    </div>
                </div>
            `;
        }

        html += `
                <div class="mt-3">
                    <div class="alert alert-info">
                        <i class="fas fa-info-circle me-2"></i>
                        <strong>Note:</strong> This analysis is based on current AI models and should be used as a reference. 
                        Always verify results through multiple methods for critical decisions.
                    </div>
                </div>
            </div>
        `;

        resultsBody.innerHTML = html;
    }

    function showAlert(message, type) {
        const alertDiv = document.createElement('div');
        alertDiv.className = `alert alert-${type} alert-dismissible fade show`;
        alertDiv.innerHTML = `
            ${message}
            <button type="button" class="btn-close" data-bs-dismiss="alert"></button>
        `;
        
        // Insert at the top of the container
        const container = document.querySelector('.container');
        container.insertBefore(alertDiv, container.firstChild);
        
        // Auto-dismiss after 5 seconds
        setTimeout(() => {
            if (alertDiv.parentNode) {
                alertDiv.remove();
            }
        }, 5000);
    }
});

