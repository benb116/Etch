/**
 * Etch Generator Preview App
 * Handles image upload, parameter tuning, live preview, and saving.
 */

class EtchPreview {
    constructor() {
        this.sessionId = null;
        this.currentStyle = 'sketch';
        this.params = {};
        this.paramSpecs = {};
        this.overlayVisible = true;
        this.animationDuration = 10; // seconds

        this.setupDOMElements();
        this.setupEventListeners();
    }

    setupDOMElements() {
        this.uploadArea = document.getElementById('uploadArea');
        this.imageInput = document.getElementById('imageInput');
        this.sourceImage = document.getElementById('sourceImage');
        this.previewLine = document.getElementById('previewLine');
        this.retraceOverlay = document.getElementById('retraceOverlay');
        this.connectorOverlay = document.getElementById('connectorOverlay');
        this.previewSvg = document.getElementById('previewSvg');
        this.paramsList = document.getElementById('paramsList');
        this.styleSelect = document.getElementById('styleSelect');
        this.statusEl = document.getElementById('status');
        this.renderTimeEl = document.getElementById('renderTime');
        this.imageInfoEl = document.getElementById('imageInfo');

        // Stat elements
        this.statPoints = document.getElementById('statPoints');
        this.statInk = document.getElementById('statInk');
        this.statLength = document.getElementById('statLength');
        this.statTime = document.getElementById('statTime');

        // Buttons
        this.replayBtn = document.getElementById('replayBtn');
        this.toggleOverlayBtn = document.getElementById('toggleOverlayBtn');
        this.finalBtn = document.getElementById('finalBtn');
    }

    setupEventListeners() {
        // Image upload
        this.uploadArea.addEventListener('click', () => this.imageInput.click());
        this.uploadArea.addEventListener('dragover', (e) => {
            e.preventDefault();
            this.uploadArea.classList.add('dragover');
        });
        this.uploadArea.addEventListener('dragleave', () => {
            this.uploadArea.classList.remove('dragover');
        });
        this.uploadArea.addEventListener('drop', (e) => {
            e.preventDefault();
            this.uploadArea.classList.remove('dragover');
            if (e.dataTransfer.files.length) {
                this.imageInput.files = e.dataTransfer.files;
                this.handleImageUpload();
            }
        });

        this.imageInput.addEventListener('change', () => this.handleImageUpload());

        // Style selection
        this.styleSelect.addEventListener('change', () => {
            this.currentStyle = this.styleSelect.value;
            this.loadParamSchema();
            this.debouncedRender();
        });

        // Buttons
        this.replayBtn.addEventListener('click', () => this.replayAnimation());
        this.toggleOverlayBtn.addEventListener('click', () => this.toggleOverlay());
        this.finalBtn.addEventListener('click', () => this.saveFinal());
    }

    setStatus(msg, duration = 3000) {
        this.statusEl.textContent = msg;
        if (duration > 0) {
            setTimeout(() => {
                this.statusEl.textContent = 'Ready';
            }, duration);
        }
    }

    async handleImageUpload() {
        const file = this.imageInput.files[0];
        if (!file) return;

        this.setStatus('Uploading image...');

        const formData = new FormData();
        formData.append('file', file);

        try {
            const response = await fetch('/api/image', {
                method: 'POST',
                body: formData,
            });

            const data = await response.json();
            if (!response.ok) {
                this.setStatus(`Error: ${data.error}`);
                return;
            }

            this.sessionId = data.session_id;
            this.imageInfoEl.textContent = `${file.name} (${data.width}×${data.height}px)`;

            // Load source image preview
            this.sourceImage.src = `/api/preview_image/${this.sessionId}`;

            // Load parameters for current style
            await this.loadParamSchema();

            this.setStatus('Image loaded. Rendering...', 2000);
            this.debouncedRender();

        } catch (error) {
            this.setStatus(`Upload failed: ${error.message}`);
        }
    }

    async loadParamSchema() {
        try {
            const response = await fetch(`/api/params?style=${this.currentStyle}`);
            const specs = await response.json();

            if (!Array.isArray(specs)) {
                console.error('Invalid param schema:', specs);
                return;
            }

            this.paramSpecs = {};
            specs.forEach(spec => {
                this.paramSpecs[spec.key] = spec;
            });

            this.renderParamControls();
        } catch (error) {
            console.error('Failed to load parameters:', error);
        }
    }

    renderParamControls() {
        this.paramsList.innerHTML = '';

        Object.values(this.paramSpecs).forEach(spec => {
            const group = document.createElement('div');
            group.className = 'param-group';

            const label = document.createElement('div');
            label.className = 'param-label';

            const name = document.createElement('span');
            name.textContent = spec.label;

            const value = document.createElement('span');
            value.className = 'param-value';
            value.textContent = this.params[spec.key] !== undefined ? this.params[spec.key] : spec.default;

            label.appendChild(name);
            label.appendChild(value);

            let control;

            if (spec.type === 'bool') {
                control = document.createElement('input');
                control.type = 'checkbox';
                control.checked = this.params[spec.key] !== undefined ? this.params[spec.key] : spec.default;
                control.addEventListener('change', (e) => {
                    this.params[spec.key] = e.target.checked;
                    value.textContent = e.target.checked ? 'on' : 'off';
                    this.debouncedRender();
                });
            } else if (spec.type === 'choice') {
                control = document.createElement('select');
                spec.choices.forEach(choice => {
                    const option = document.createElement('option');
                    option.value = choice;
                    option.textContent = choice;
                    option.selected = (this.params[spec.key] || spec.default) === choice;
                    control.appendChild(option);
                });
                control.addEventListener('change', (e) => {
                    this.params[spec.key] = e.target.value;
                    value.textContent = e.target.value;
                    this.debouncedRender();
                });
            } else {
                control = document.createElement('input');
                control.type = 'range';
                control.min = spec.lo;
                control.max = spec.hi;
                control.step = spec.step || 0.01;
                control.value = this.params[spec.key] !== undefined ? this.params[spec.key] : spec.default;
                control.addEventListener('input', (e) => {
                    let v = parseFloat(e.target.value);
                    if (spec.type === 'int') {
                        v = Math.round(v);
                    }
                    this.params[spec.key] = v;
                    value.textContent = v.toFixed(spec.type === 'int' ? 0 : 2);
                    this.debouncedRender();
                });
            }

            group.appendChild(label);
            group.appendChild(control);

            if (spec.tooltip) {
                const tooltip = document.createElement('div');
                tooltip.className = 'param-tooltip';
                tooltip.textContent = spec.tooltip;
                group.appendChild(tooltip);
            }

            this.paramsList.appendChild(group);
        });
    }

    debouncedRender() {
        if (this._renderTimeout) {
            clearTimeout(this._renderTimeout);
        }
        this._renderTimeout = setTimeout(() => this.render('preview'), 500);
    }

    async render(quality = 'preview') {
        if (!this.sessionId) return;

        this.setStatus(`Rendering (${quality})...`, 0);
        const t0 = performance.now();

        try {
            const response = await fetch('/api/generate', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    session_id: this.sessionId,
                    style: this.currentStyle,
                    params: this.params,
                    quality: quality,
                }),
            });

            const data = await response.json();
            if (!response.ok) {
                this.setStatus(`Render error: ${data.error}`);
                return;
            }

            const elapsed = ((performance.now() - t0) / 1000).toFixed(2);
            this.renderTimeEl.textContent = `${elapsed}s`;

            // Update SVG
            this.updatePreview(data);

            // Update stats
            this.updateStats(data.metrics);

            this.setStatus(`Rendered in ${elapsed}s`, 2000);

        } catch (error) {
            this.setStatus(`Render failed: ${error.message}`);
        }
    }

    pointsToPath(points, indexes = null) {
        if (!points || points.length === 0) return '';

        if (indexes) {
            const selected = indexes.map(i => points[i]);
            return selected.map(p => `${p[0]},${p[1]}`).join(' ');
        }

        return points.map(p => `${p[0]},${p[1]}`).join(' ');
    }

    updatePreview(data) {
        const points = data.points;
        const segClass = data.segment_classes;

        if (!points || points.length === 0) {
            this.previewLine.setAttribute('points', '');
            this.retraceOverlay.setAttribute('points', '');
            this.connectorOverlay.setAttribute('points', '');
            return;
        }

        // Full line
        this.previewLine.setAttribute('points', this.pointsToPath(points));

        // Collect retrace and connector segments
        const INK = 0, RETRACE = 1, CONNECTOR = 2;

        const retraceSegments = [];
        const connectorSegments = [];

        for (let i = 0; i < segClass.length; i++) {
            if (segClass[i] === RETRACE) {
                retraceSegments.push(points[i], points[i + 1]);
            } else if (segClass[i] === CONNECTOR) {
                connectorSegments.push(points[i], points[i + 1]);
            }
        }

        this.retraceOverlay.setAttribute('points', this.pointsToPath(retraceSegments));
        this.connectorOverlay.setAttribute('points', this.pointsToPath(connectorSegments));

        // Estimate animation duration (rough)
        const totalLen = data.metrics.total_len;
        this.animationDuration = Math.max(5, Math.min(30, totalLen / 50));
        this.previewSvg.style.setProperty('--animation-duration', `${this.animationDuration}s`);
    }

    updateStats(metrics) {
        this.statPoints.textContent = metrics.points.toLocaleString();
        this.statInk.textContent = `${(metrics.ink_fraction * 100).toFixed(1)}%`;
        this.statLength.textContent = metrics.total_len.toFixed(0);
        this.statTime.textContent = `${metrics.draw_seconds}s`;
    }

    toggleOverlay() {
        this.overlayVisible = !this.overlayVisible;
        this.retraceOverlay.style.opacity = this.overlayVisible ? '0.6' : '0';
        this.connectorOverlay.style.opacity = this.overlayVisible ? '0.6' : '0';
        this.toggleOverlayBtn.textContent = this.overlayVisible ? '👁 Toggle Overlay' : '👁‍🗨 Overlay Hidden';
    }

    replayAnimation() {
        this.previewSvg.classList.remove('animating');
        // Trigger reflow to restart animation
        void this.previewSvg.offsetWidth;
        this.previewSvg.classList.add('animating');
    }

    async saveFinal() {
        if (!this.sessionId) {
            this.setStatus('No image loaded');
            return;
        }

        const name = prompt('Enter drawing name:', 'drawing');
        if (!name) return;

        this.setStatus('Rendering final quality and saving...');

        try {
            // First, render at final quality
            await this.render('final');

            // Then save
            const response = await fetch('/api/save', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    session_id: this.sessionId,
                    style: this.currentStyle,
                    params: this.params,
                    name: name,
                }),
            });

            const data = await response.json();
            if (!response.ok) {
                this.setStatus(`Save error: ${data.error}`);
                return;
            }

            this.setStatus(`Saved to ${data.path}`, 5000);

        } catch (error) {
            this.setStatus(`Save failed: ${error.message}`);
        }
    }
}

// Initialize when DOM is ready
document.addEventListener('DOMContentLoaded', () => {
    new EtchPreview();
});
