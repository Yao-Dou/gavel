// Gavel Webpage JavaScript
// Right-side minimalist navigation and Visualization

// Visualization data
let vizData = null;
let vizDocData = null;

document.addEventListener('DOMContentLoaded', function() {
    // Load visualization data
    loadVizData();
    loadVizDocData();
    const sections = document.querySelectorAll('section[id]');
    const navLinks = document.querySelectorAll('.side-nav-link');
    const indicator = document.querySelector('.side-nav-indicator');
    const track = document.querySelector('.side-nav-track');

    // Calculate indicator position based on active section
    function updateIndicatorPosition(index) {
        if (!indicator || !track) return;

        const totalSections = navLinks.length;
        const trackHeight = track.offsetHeight;
        const indicatorHeight = indicator.offsetHeight;
        const availableSpace = trackHeight - indicatorHeight;
        const position = (index / (totalSections - 1)) * availableSpace;

        indicator.style.top = position + 'px';
    }

    // Update active section on scroll
    function updateActiveSection() {
        const scrollPosition = window.scrollY + window.innerHeight / 3;

        let currentIndex = 0;
        let found = false;

        sections.forEach((section, index) => {
            const sectionTop = section.offsetTop;
            const sectionHeight = section.offsetHeight;

            if (scrollPosition >= sectionTop && scrollPosition < sectionTop + sectionHeight) {
                currentIndex = index;
                found = true;
            }
        });

        // If we're past all sections, highlight the last one
        if (!found && window.scrollY > 0) {
            const lastSection = sections[sections.length - 1];
            if (lastSection && window.scrollY >= lastSection.offsetTop) {
                currentIndex = sections.length - 1;
            }
        }

        // Update nav links
        navLinks.forEach((link, index) => {
            link.classList.remove('active');
            if (index === currentIndex) {
                link.classList.add('active');
            }
        });

        // Update track indicator
        updateIndicatorPosition(currentIndex);
    }

    // Throttled scroll handler
    let ticking = false;
    window.addEventListener('scroll', function() {
        if (!ticking) {
            window.requestAnimationFrame(function() {
                updateActiveSection();
                ticking = false;
            });
            ticking = true;
        }
    });

    // Initial update
    updateActiveSection();
});

// =============================================
// Visualization Functions
// =============================================

async function loadVizData() {
    try {
        const response = await fetch('data/viz_summaries.json');
        if (!response.ok) return;
        vizData = await response.json();
        initializeVizControls();
    } catch (error) {
        // Data not yet available
    }
}

function formatModelName(model) {
    // Mapping for standardized display names
    const nameMap = {
        'Qwen3-14B': 'Qwen3 14B',
        'Qwen3-30B-A3B-Thinking-2507': 'Qwen3 30B-A3B',
        'Qwen3-32B': 'Qwen3 32B',
        'claude-opus-4-1-20250805-thinking': 'Claude Opus 4',
        'claude-sonnet-4-20250514-thinking': 'Claude Sonnet 4',
        'gemini-2.5-flash': 'Gemini 2.5 Flash',
        'gemini-2.5-pro': 'Gemini 2.5 Pro',
        'gpt-oss-20b-BF16': 'GPT-oss 20B'
    };
    return nameMap[model] || model;
}

function initializeVizControls() {
    const modelSelect = document.getElementById('model-select');
    const caseSelect = document.getElementById('case-select');

    if (!modelSelect || !caseSelect || !vizData) return;

    // Populate model dropdown with formatted names
    vizData.models.forEach(model => {
        const option = document.createElement('option');
        option.value = model;
        option.textContent = formatModelName(model);
        modelSelect.appendChild(option);
    });

    modelSelect.addEventListener('change', updateCaseDropdown);
    caseSelect.addEventListener('change', renderVisualization);

    // Initial population
    updateCaseDropdown();
}

function updateCaseDropdown() {
    const modelSelect = document.getElementById('model-select');
    const caseSelect = document.getElementById('case-select');
    const model = modelSelect.value;

    caseSelect.innerHTML = '';

    if (vizData && vizData.cases[model]) {
        const caseIds = Object.keys(vizData.cases[model]).sort();
        caseIds.forEach(caseId => {
            const option = document.createElement('option');
            option.value = caseId;
            option.textContent = `Case ${caseId}`;
            caseSelect.appendChild(option);
        });
    }

    renderVisualization();
}

function renderVisualization() {
    const modelSelect = document.getElementById('model-select');
    const caseSelect = document.getElementById('case-select');
    const modelSummaryContent = document.getElementById('model-summary-content');
    const referenceSummaryContent = document.getElementById('reference-summary-content');
    const checklistContainer = document.getElementById('checklist-container');
    const caseLink = document.getElementById('case-link');

    if (!modelSelect || !caseSelect || !vizData) return;

    const model = modelSelect.value;
    const caseId = caseSelect.value;

    // Update case link
    if (caseLink && caseId) {
        caseLink.href = `https://clearinghouse.net/case/${caseId}`;
    }

    if (!model || !caseId || !vizData.cases[model] || !vizData.cases[model][caseId]) {
        modelSummaryContent.textContent = 'Select a model and case to view the summary.';
        referenceSummaryContent.textContent = 'Select a model and case to view the reference.';
        checklistContainer.innerHTML = '<p class="checklist-placeholder">Select a model and case to view checklist evaluation.</p>';
        return;
    }

    const caseData = vizData.cases[model][caseId];

    // Render summaries
    modelSummaryContent.textContent = caseData.model_summary || 'No summary available.';
    referenceSummaryContent.textContent = caseData.reference_summary || 'No reference available.';

    // Render checklist
    renderChecklist(caseData.checklist, checklistContainer);
}

function renderChecklist(checklist, container) {
    container.innerHTML = '';

    if (!checklist || Object.keys(checklist).length === 0) {
        container.innerHTML = '<p class="checklist-placeholder">No checklist data available for this case.</p>';
        return;
    }

    // Sort checklist items alphabetically
    const sortedItems = Object.keys(checklist).sort();

    sortedItems.forEach(itemName => {
        const item = checklist[itemName];
        const itemDiv = document.createElement('div');
        itemDiv.className = 'checklist-item';

        // Item name header
        const nameDiv = document.createElement('div');
        nameDiv.className = 'checklist-item-name';
        nameDiv.textContent = itemName;
        itemDiv.appendChild(nameDiv);

        // Model values
        const modelSide = document.createElement('div');
        modelSide.className = 'checklist-side';
        modelSide.innerHTML = `
            <div class="checklist-side-label">Model</div>
            <div class="checklist-values">${renderValues(item.model_values, item.relation.matched_model)}</div>
        `;
        itemDiv.appendChild(modelSide);

        // Relation
        const relationDiv = document.createElement('div');
        relationDiv.className = `checklist-relation ${item.relation.class}`;
        relationDiv.textContent = item.relation.text;
        itemDiv.appendChild(relationDiv);

        // Reference values
        const refSide = document.createElement('div');
        refSide.className = 'checklist-side';
        refSide.innerHTML = `
            <div class="checklist-side-label">Reference</div>
            <div class="checklist-values">${renderValues(item.reference_values, item.relation.matched_ref)}</div>
        `;
        itemDiv.appendChild(refSide);

        container.appendChild(itemDiv);
    });
}

function renderValues(values, matchedIndices) {
    if (!values || values.length === 0) {
        return '<span class="value-empty">No values</span>';
    }

    const matchedSet = new Set(matchedIndices || []);

    return values.map((value, index) => {
        // Indices in evaluation are 1-based
        const isMatched = matchedSet.has(index + 1);
        const className = matchedSet.size > 0
            ? (isMatched ? 'value-matched' : 'value-unmatched')
            : 'value-single';
        return `<div class="value-item ${className}">${escapeHtml(value)}</div>`;
    }).join('');
}

function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

// =============================================
// Document Checklist Visualization Functions
// =============================================

async function loadVizDocData() {
    try {
        const response = await fetch('data/viz_doc_checklist.json');
        if (!response.ok) return;
        vizDocData = await response.json();
        initializeDocVizControls();
    } catch (error) {
        // Data not yet available
    }
}

function initializeDocVizControls() {
    const methodSelect = document.getElementById('doc-method-select');
    const caseSelect = document.getElementById('doc-case-select');

    if (!methodSelect || !caseSelect || !vizDocData) return;

    // Populate method dropdown with friendly names
    vizDocData.methods.forEach(m => {
        const option = document.createElement('option');
        option.value = m.id;

        // Create friendly label
        let label = '';
        if (m.method === 'end_to_end') {
            label = 'End-to-End';
        } else if (m.method === 'chunk_by_chunk') {
            label = 'Chunk-by-Chunk';
        } else if (m.method === 'gavel_agent') {
            label = 'GAVEL-AGENT';
        }

        // Add model name (shortened)
        if (m.model) {
            let shortModel = m.model;
            if (m.model.includes('gpt-4.1')) shortModel = 'GPT-4.1';
            else if (m.model.includes('gpt-oss')) shortModel = 'GPT-oss 20B';
            else if (m.model.includes('Qwen3-32B')) shortModel = 'Qwen3 32B';
            else if (m.model.includes('Qwen3-30B')) shortModel = 'Qwen3 30B-A3B';
            label += ` (${shortModel})`;
        }

        // Add config if present
        if (m.config) {
            label += ` [${m.config}]`;
        }

        option.textContent = label;
        methodSelect.appendChild(option);
    });

    methodSelect.addEventListener('change', updateDocCaseDropdown);
    caseSelect.addEventListener('change', renderDocVisualization);

    updateDocCaseDropdown();
}

function updateDocCaseDropdown() {
    const methodSelect = document.getElementById('doc-method-select');
    const caseSelect = document.getElementById('doc-case-select');
    const methodId = methodSelect.value;

    caseSelect.innerHTML = '';

    if (vizDocData && vizDocData.cases[methodId]) {
        const caseIds = Object.keys(vizDocData.cases[methodId]).sort();
        caseIds.forEach(caseId => {
            const option = document.createElement('option');
            option.value = caseId;
            option.textContent = `Case ${caseId}`;
            caseSelect.appendChild(option);
        });
    }

    renderDocVisualization();
}

function renderDocVisualization() {
    const methodSelect = document.getElementById('doc-method-select');
    const caseSelect = document.getElementById('doc-case-select');
    const summaryContent = document.getElementById('doc-summary-content');
    const checklistContainer = document.getElementById('doc-checklist-container');
    const caseLink = document.getElementById('doc-case-link');

    if (!methodSelect || !caseSelect || !vizDocData) return;

    const methodId = methodSelect.value;
    const caseId = caseSelect.value;

    // Update case link
    if (caseLink && caseId) {
        caseLink.href = `https://clearinghouse.net/case/${caseId}`;
    }

    if (!methodId || !caseId || !vizDocData.cases[methodId] || !vizDocData.cases[methodId][caseId]) {
        summaryContent.textContent = 'Select a method and case to view the summary.';
        checklistContainer.innerHTML = '<p class="checklist-placeholder">Select a method and case to view checklist comparison.</p>';
        return;
    }

    const caseData = vizDocData.cases[methodId][caseId];

    // Render summary
    summaryContent.textContent = caseData.summary || 'No summary available.';

    // Render checklist comparison
    renderDocChecklist(caseData.checklist, checklistContainer);
}

function renderDocChecklist(checklist, container) {
    container.innerHTML = '';

    if (!checklist || Object.keys(checklist).length === 0) {
        container.innerHTML = '<p class="checklist-placeholder">No checklist data available for this case.</p>';
        return;
    }

    // Sort checklist items alphabetically
    const sortedItems = Object.keys(checklist).sort();

    sortedItems.forEach(itemName => {
        const item = checklist[itemName];
        const itemDiv = document.createElement('div');
        itemDiv.className = 'checklist-item';

        // Item name header
        const nameDiv = document.createElement('div');
        nameDiv.className = 'checklist-item-name';
        nameDiv.textContent = itemName;
        itemDiv.appendChild(nameDiv);

        // Model values (from documents)
        const modelSide = document.createElement('div');
        modelSide.className = 'checklist-side';
        modelSide.innerHTML = `
            <div class="checklist-side-label">Model (from Documents)</div>
            <div class="checklist-values">${renderDocValues(item.model_values, item.relation.matched_model)}</div>
        `;
        itemDiv.appendChild(modelSide);

        // Relation
        const relationDiv = document.createElement('div');
        relationDiv.className = `checklist-relation ${item.relation.class}`;
        relationDiv.textContent = item.relation.text;
        itemDiv.appendChild(relationDiv);

        // Reference values (from summary)
        const refSide = document.createElement('div');
        refSide.className = 'checklist-side';
        refSide.innerHTML = `
            <div class="checklist-side-label">Reference (from Summary)</div>
            <div class="checklist-values">${renderDocValues(item.reference_values, item.relation.matched_ref)}</div>
        `;
        itemDiv.appendChild(refSide);

        container.appendChild(itemDiv);
    });
}

function renderDocValues(values, matchedIndices) {
    if (!values || values.length === 0) {
        return '<span class="value-empty">No values</span>';
    }

    const matchedSet = new Set(matchedIndices || []);

    return values.map((value, index) => {
        // Indices in evaluation are 1-based
        const isMatched = matchedSet.has(index + 1);
        const className = matchedSet.size > 0
            ? (isMatched ? 'value-matched' : 'value-unmatched')
            : 'value-single';
        return `<div class="value-item ${className}">${escapeHtml(value)}</div>`;
    }).join('');
}
