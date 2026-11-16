/**
 * Asakta Vision AI - Main Application Logic
 * Consolidating detection, editing, and submission workflows.
 */

let detections = [];
let uploadedFile = null; // Legacy single file
let batchImages = [];    // New Batch Array: [{file, src, detections, master, dims, processed: bool}]
let activeBatchIndex = -1;
let isProcessingBatch = false;
let masterResult = null;
let surveyResult = {};
let imageDimensions = { width: 0, height: 0 };
let activePreviewCropBounds = null;
let currentPreviewBaseSrc = '';
const cropReviewState = {
    sourceCache: new Map(),
    renderToken: 0
};
const SMART_CROP_COMPONENT_LABELS = new Set([
    'V_CROSS_ARM',
    'V_CROSS',
    'TAPPING_ARM',
    'TAPPING_CHANNEL',
    'SIDE_ARM',
    'SIDE_ARM_CHANNEL',
    'BOX_ARM'
]);
const SMART_CROP_MIN_WIDTH = 640;
const SMART_CROP_MIN_HEIGHT = 640;
let CLASS_OPTIONS = [
    "POLE", "MAIN_POLE", "STRUT_POLE",
    "HT_DISC", "HT_PIN", "LT_PIN", "SHACKLE_INSULATOR", "INSULATOR",
    "V_CROSS_ARM", "TAPPING_ARM", "SIDE_ARM", "T_RISING", "TOP_CLEAT", "SPECIAL_CLAMP", "STAY_SET", "AB_SWITCH", "STREET_LIGHT", "BOX_ARM", "DTR",
    "3_WIRE_CROSSARM", "5_WIRE_CROSSARM",
    "CONDUCTOR", "WIRE_BROKEN", "VEGETATION", "OBJECT",
    "V_CROSS", "TAPPING_CHANNEL", "SIDE_ARM_CHANNEL"
];

const CLASS_COLORS = {
    "POLE": "#ffffff",
    "MAIN_POLE": "#ffffff",
    "STRUT_POLE": "#ff7f50",
    "INSULATOR": "#00ff00",
    "HT_PIN": "#10b981",
    "HT_DISC": "#3b82f6",
    "LT_PIN": "#14b8a6",
    "SHACKLE_INSULATOR": "#f59e0b",
    "V_CROSS_ARM": "#f43f5e",
    "V_CROSS": "#f43f5e",
    "T_RISING": "#ff00ff",
    "TAPPING_ARM": "#d946ef",
    "TAPPING_CHANNEL": "#d946ef",
    "SIDE_ARM": "#a855f7",
    "SIDE_ARM_CHANNEL": "#a855f7",
    "3_WIRE_CROSSARM": "#06b6d4",
    "5_WIRE_CROSSARM": "#ec4899",
    "TOP_CLEAT": "#84cc16",
    "SPECIAL_CLAMP": "#64748b",
    "STAY_SET": "#475569",
    "AB_SWITCH": "#06b6d4",
    "STREET_LIGHT": "#f59e0b",
    "BOX_ARM": "#d5740dff",
    "DTR": "#8b5cf6",
    "CONDUCTOR": "#00ffff",
    "WIRE_BROKEN": "#ef4444",
    "VEGETATION": "#fbbf24",
    "OBJECT": "#a8a29e"
};

let expandedGroups = new Set();

function getFakeConfidenceValue(rawConf) {
    return Math.round(rawConf * 100) + '%';
}

function formatDetectionLabel(label) {
    return (label || 'OBJECT').replace(/_/g, ' ');
}

function hideCropReview() {
    cropReviewState.renderToken += 1;
    activePreviewCropBounds = null;
    const card = document.getElementById('cropReviewCard');
    const canvas = document.getElementById('cropReviewCanvas');
    if (card) card.classList.add('hidden');
    if (canvas) {
        const ctx = canvas.getContext('2d');
        ctx.clearRect(0, 0, canvas.width || 1, canvas.height || 1);
    }
}

async function renderCropReview() {
    hideCropReview();
}

// Drawing State
let isDrawMode = false;
let isDrawing = false;
let drawStart = null;
let pendingBbox = null;

// History & Shortcuts
let historyStack = [];
let redoStack = [];
const MAX_HISTORY = 50;
let lastSaveTime = Date.now();

function saveToHistory() {
    if (historyStack.length >= MAX_HISTORY) historyStack.shift();
    historyStack.push(JSON.stringify(detections));
    redoStack = []; // Clear redo on new action
}

function undo() {
    if (historyStack.length > 1) {
        redoStack.push(historyStack.pop());
        detections = JSON.parse(historyStack[historyStack.length - 1]);
        renderResults();
        renderBoxes();
        saveDraft();
        showToast("Undo", "primary");
    }
}

function redo() {
    if (redoStack.length > 0) {
        const state = redoStack.pop();
        historyStack.push(state);
        detections = JSON.parse(state);
        renderResults();
        renderBoxes();
        saveDraft();
        showToast("Redo", "primary");
    }
}

function saveDraft() {
    if (isProcessingBatch) return;
    // In-memory save for active batch image
    if (activeBatchIndex !== -1) {
        batchImages[activeBatchIndex].detections = [...detections];
    }
    // If inspecting a video pole, sync detections back to videoPoleResults
    if (currentTab === 'video' && inspectingVideoPoleId !== null) {
        const pole = videoPoleResults.find(p => p.pole_id === inspectingVideoPoleId);
        if (pole) {
            pole.detections = [...detections];
        }
    }
    // Persist to localStorage immediately
    persistDraftToStorage();
    lastSaveTime = Date.now();
}

// ─── localStorage Draft Persistence ──────────────────────────────────────────
const DRAFT_KEY = 'asakta_worker_draft';

function persistDraftToStorage() {
    if (!batchImages.length) return;
    try {
        const payload = {
            savedAt: Date.now(),
            batchImages: batchImages.map(b => ({
                src: b.src,
                original_src: b.originalSrc || b.src,
                name: b.file?.name || b.name || 'image.jpg',
                uploaded_image_id: b.uploaded_image_id || null,
                detections: b.detections || [],
                master: b.master || null,
                dims: b.dims || {},
                processed: b.processed || false
            })),
            activeBatchIndex,
            detections,
            masterResult,
            imageDimensions
        };
        localStorage.setItem(DRAFT_KEY, JSON.stringify(payload));
    } catch (e) {
        console.warn('[Draft] localStorage write failed', e);
    }
}

function clearDraftStorage() {
    localStorage.removeItem(DRAFT_KEY);
}

function checkAndRestoreDraft() {
    try {
        const raw = localStorage.getItem(DRAFT_KEY);
        if (!raw) return;
        const draft = JSON.parse(raw);
        if (!draft || !draft.batchImages || draft.batchImages.length === 0) return;

        const ageMinutes = Math.round((Date.now() - draft.savedAt) / 60000);
        const imageCount = draft.batchImages.length;
        const processedCount = draft.batchImages.filter(b => b.processed).length;

        // Show restore banner
        const banner = document.getElementById('draftRestoreBanner');
        if (!banner) return;

        document.getElementById('draftInfo').textContent =
            `${imageCount} image${imageCount > 1 ? 's' : ''} · ${processedCount} analysed · saved ${ageMinutes < 1 ? 'just now' : ageMinutes + 'm ago'}`;

        banner.classList.remove('hidden');
        banner.onclick = () => { };

        document.getElementById('btnRestoreDraft').onclick = () => {
            batchImages = draft.batchImages.map(b => ({
                file: { name: b.name },
                name: b.name,
                src: b.src,
                originalSrc: b.original_src || b.src,
                uploaded_image_id: b.uploaded_image_id || null,
                detections: b.detections || [],
                master: b.master || null,
                dims: b.dims || {},
                processed: b.processed || false
            }));
            activeBatchIndex = draft.activeBatchIndex >= 0 ? draft.activeBatchIndex : 0;
            detections = draft.detections || [];
            masterResult = draft.masterResult || null;
            imageDimensions = draft.imageDimensions || {};

            // Restore preview image
            if (batchImages[activeBatchIndex]?.src) {
                const preview = document.getElementById('preview');
                if (preview) currentPreviewBaseSrc = batchImages[activeBatchIndex].originalSrc || batchImages[activeBatchIndex].src;
                const imgContainer = document.getElementById('imageContainer');
                const submitSection = document.getElementById('submitSection');
                if (imgContainer) imgContainer.classList.remove('hidden');
                if (submitSection) submitSection.classList.remove('hidden');
            }

            renderBatchStrip();
            renderResults();
            renderBoxes();

            banner.classList.add('hidden');
            showToast('Draft restored!', 'success');
        };

        document.getElementById('btnDiscardDraft').onclick = () => {
            clearDraftStorage();
            banner.classList.add('hidden');
        };

    } catch (e) {
        console.warn('[Draft] Restore check failed', e);
    }
}

// Warn user before closing tab if work is in progress
window.addEventListener('beforeunload', (e) => {
    const hasWork = batchImages.some(b => b.processed) || detections.length > 0;
    if (hasWork) {
        e.preventDefault();
        e.returnValue = 'You have unsaved annotations. Are you sure you want to leave?';
        return e.returnValue;
    }
});

document.addEventListener('DOMContentLoaded', () => {
    const uploadInput = document.getElementById('upload');
    const previewImg = document.getElementById('preview');
    const dropZone = document.getElementById('dropZone');

    // Check for saved draft on page load
    checkAndRestoreDraft();

    // Auto-save to localStorage every 10 seconds if work is in progress
    setInterval(() => {
        if (batchImages.some(b => b.processed) || detections.length > 0) {
            persistDraftToStorage();
        }
    }, 10000);

    // Keyboard Shortcuts
    document.addEventListener('keydown', (e) => {
        if (e.ctrlKey || e.metaKey) {
            if (e.key === 'z') { e.preventDefault(); e.shiftKey ? redo() : undo(); }
            if (e.key === 's') { e.preventDefault(); saveDraft(); }
        }
        if (e.key === 'Delete' || e.key === 'Backspace') {
            // Logic to delete selected box if applicable
        }
    });

    if (uploadInput) {
        uploadInput.addEventListener('change', handleUpload);
    }

    if (dropZone) {
        dropZone.addEventListener('click', (e) => {
            if (currentTab === 'image' && batchImages.length > 0) return;
            uploadInput.click();
        });
        dropZone.addEventListener('dragover', (e) => {
            e.preventDefault();
            dropZone.classList.add('border-blue-500', 'bg-blue-500/5');
        });
        dropZone.addEventListener('dragleave', () => {
            dropZone.classList.remove('border-blue-500', 'bg-blue-500/5');
        });
        dropZone.addEventListener('drop', (e) => {
            e.preventDefault();
            dropZone.classList.remove('border-blue-500', 'bg-blue-500/5');
            if (e.dataTransfer.files.length) {
                uploadInput.files = e.dataTransfer.files;
                handleUpload({ target: uploadInput });
            }
        });
    }

    const overlay = document.getElementById('detectionOverlay');
    if (overlay) {
        // Unified pointer handling for both mouse and touch
        overlay.addEventListener('mousedown', handleDragStart);
        overlay.addEventListener('touchstart', handleDragStart, { passive: false });
    }
});

function handleUpload(e) {
    const files = Array.from(e.target.files);
    if (!files.length) return;

    if (currentTab === 'tree') {
        handleTreeUpload(files);
        e.target.value = '';
        return;
    }

    files.forEach(file => {
        const originalSrc = URL.createObjectURL(file);
        batchImages.push({
            file: file,
            src: originalSrc,
            originalSrc: originalSrc,
            detections: [],
            master: null,
            dims: { width: 0, height: 0 },
            processed: false,
            uploaded_image_id: null
        });
    });

    document.getElementById('uploadPrompt').classList.add('hidden');
    document.getElementById('batchStripWrapper').classList.remove('hidden');
    document.getElementById('imageContainer').classList.remove('hidden');

    const dz = document.getElementById('dropZone');
    dz.classList.add('py-4', 'border-transparent');
    dz.classList.remove('p-10', 'border-dashed', 'border-white/5', 'hover:border-blue-500/30', 'cursor-pointer');

    renderBatchStrip();

    if (activeBatchIndex === -1) {
        selectBatchImage(batchImages.length - files.length);
    }

    e.target.value = '';
}

function revokeTreePreview(src) {
    if (src && src.startsWith('blob:')) {
        URL.revokeObjectURL(src);
    }
}

function syncTreeControls() {
    const controls = document.getElementById('treeSecondaryControls');
    const primaryStatus = document.getElementById('treePrimaryStatus');
    const removePrimaryBtn = document.getElementById('btnRemoveTreePrimary');
    if (!controls || !primaryStatus || !removePrimaryBtn) return;

    controls.classList.toggle('hidden', currentTab !== 'tree');
    primaryStatus.textContent = treeAnalysisState.primaryFile ? treeAnalysisState.primaryFile.name : 'Required';
    removePrimaryBtn.classList.toggle('hidden', !treeAnalysisState.primaryFile);
}

function handleTreeUpload(files) {
    const file = files[0];
    if (!file) return;

    revokeTreePreview(treeAnalysisState.primarySrc);
    treeAnalysisState.primaryFile = file;
    treeAnalysisState.primarySrc = URL.createObjectURL(file);
    treeAnalysisState.result = null;
    treeAnalysisState.processed = false;
    treeAnalysisState.reportId = '';
    treeAnalysisState.selectedTreeId = '';

    uploadedFile = file;
    detections = [];
    masterResult = null;
    imageDimensions = { width: 0, height: 0 };
    activeBatchIndex = -1;

    document.getElementById('uploadPrompt').classList.add('hidden');
    document.getElementById('batchStripWrapper').classList.add('hidden');
    document.getElementById('imageContainer').classList.remove('hidden');
    currentPreviewBaseSrc = treeAnalysisState.primarySrc;
    document.getElementById('preview').src = treeAnalysisState.primarySrc;
    document.getElementById('masterIdentityCard').classList.add('hidden');
    document.getElementById('submitSection').classList.add('hidden');

    const dz = document.getElementById('dropZone');
    dz.classList.add('py-4', 'border-transparent');
    dz.classList.remove('p-10', 'border-dashed', 'border-white/5', 'hover:border-blue-500/30', 'cursor-pointer');

    syncTreeControls();
    renderBoxes();
    renderTreeResults();
    showToast('Pole context image loaded successfully', 'success');
}

function removeTreePrimary() {
    revokeTreePreview(treeAnalysisState.primarySrc);
    treeAnalysisState.primaryFile = null;
    treeAnalysisState.primarySrc = '';
    treeAnalysisState.result = null;
    treeAnalysisState.processed = false;
    treeAnalysisState.reportId = '';
    treeAnalysisState.selectedTreeId = '';
    uploadedFile = null;
    detections = [];
    masterResult = null;
    const preview = document.getElementById('preview');
    if (preview) preview.src = '';
    document.getElementById('imageContainer').classList.add('hidden');
    document.getElementById('uploadPrompt').classList.remove('hidden');
    syncTreeControls();
    renderBoxes();
    renderTreeResults();
}

function revokeBatchImageUrls(item) {
    if (item?.src && item.src.startsWith('blob:')) {
        URL.revokeObjectURL(item.src);
    }
    if (item?.originalSrc && item.originalSrc !== item.src && item.originalSrc.startsWith('blob:')) {
        URL.revokeObjectURL(item.originalSrc);
    }
}

function clearActiveImageState() {
    detections = [];
    uploadedFile = null;
    masterResult = null;
    imageDimensions = { width: 0, height: 0 };
    currentPreviewBaseSrc = '';

    const preview = document.getElementById('preview');
    if (preview) preview.src = '';

    const overlay = document.getElementById('detectionOverlay');
    if (overlay) overlay.innerHTML = '';

    const resultBox = document.getElementById('resultBox');
    if (resultBox) resultBox.innerHTML = '';

    const masterCard = document.getElementById('masterIdentityCard');
    if (masterCard) masterCard.classList.add('hidden');

    const submitSection = document.getElementById('submitSection');
    if (submitSection) submitSection.classList.add('hidden');

    const finalSubmitBtn = document.getElementById('finalSubmitBtn');
    if (finalSubmitBtn) {
        finalSubmitBtn.disabled = true;
        finalSubmitBtn.innerHTML = '<i class="fa-solid fa-cloud-arrow-up"></i> SUBMIT RESULTS';
    }

    hideCropReview();
}

function renderBatchStrip() {
    const strip = document.getElementById('batchStrip');
    strip.innerHTML = '';

    document.getElementById('batchCount').textContent = `${batchImages.length} Files`;

    batchImages.forEach((item, index) => {
        const thumb = document.createElement('div');
        thumb.className = `batch-thumb ${index === activeBatchIndex ? 'active' : ''} ${item.processed ? 'processed' : ''}`;
        thumb.onclick = () => selectBatchImage(index);

        const img = document.createElement('img');
        img.src = item.src;
        thumb.appendChild(img);

        // Hover Remove Button
        const removeBtn = document.createElement('button');
        removeBtn.className = 'batch-thumb-remove';
        removeBtn.innerHTML = '<i class="fa-solid fa-xmark"></i>';
        removeBtn.onclick = (e) => {
            e.stopPropagation();
            removeBatchImage(index);
        };
        thumb.appendChild(removeBtn);

        strip.appendChild(thumb);
    });

    // Add "Plus" button for adding more views (Max 3)
    if (batchImages.length < 3) {
        const addBtn = document.createElement('div');
        addBtn.className = 'batch-thumb-add group h-[40px] w-[40px] shrink-0 cursor-pointer';
        addBtn.title = "Add another view (Max 3)";
        addBtn.onclick = () => document.getElementById('upload').click();
        addBtn.innerHTML = `
            <div class="w-full h-full rounded-lg border-2 border-dashed border-white/10 flex items-center justify-center group-hover:border-blue-500/50 group-hover:bg-blue-500/5 transition-all">
                <i class="fa-solid fa-plus text-[10px] text-gray-500 group-hover:text-blue-400"></i>
            </div>
        `;
        strip.appendChild(addBtn);
    }
}

function removeBatchImage(index) {
    const item = batchImages[index];
    revokeBatchImageUrls(item);

    batchImages.splice(index, 1);

    if (batchImages.length === 0) {
        activeBatchIndex = -1;
        clearActiveImageState();

        const strip = document.getElementById('batchStrip');
        if (strip) strip.innerHTML = '';
        const batchCount = document.getElementById('batchCount');
        if (batchCount) batchCount.textContent = '0 Files';

        document.getElementById('batchStripWrapper').classList.add('hidden');
        document.getElementById('imageContainer').classList.add('hidden');
        document.getElementById('uploadPrompt').classList.remove('hidden');

        const dz = document.getElementById('dropZone');
        if (dz) {
            dz.classList.remove('py-4', 'border-transparent');
            dz.classList.add('p-10', 'border-dashed', 'border-white/5', 'hover:border-blue-500/30', 'cursor-pointer');
        }

        const uploadInput = document.getElementById('upload');
        if (uploadInput) uploadInput.value = '';

        localStorage.removeItem(DRAFT_KEY);
        renderAnalysisPlaceholder('fa-magnifying-glass', 'Results will appear here', 'RUN IMAGE INFERENCE TO POPULATE THE ANALYSIS PANEL');
        showToast('Image removed', 'info');
        return;
    }

    // Handle index shifting
    if (activeBatchIndex === index) {
        const next = Math.min(index, batchImages.length - 1);
        activeBatchIndex = -1; // Force clean select
        selectBatchImage(next);
    } else if (activeBatchIndex > index) {
        activeBatchIndex--;
    }
    renderBatchStrip();
}

function selectBatchImage(index) {
    if (isProcessingBatch) return;
    if (index < 0 || index >= batchImages.length) return;

    // Save current active state before switching (if needed)
    if (activeBatchIndex !== -1) {
        batchImages[activeBatchIndex].detections = [...detections];
        batchImages[activeBatchIndex].master = masterResult;
        batchImages[activeBatchIndex].dims = { ...imageDimensions };
    }

    activeBatchIndex = index;
    const item = batchImages[index];

    // Load new image state
    uploadedFile = item.file;
    detections = [...item.detections];
    masterResult = item.master;
    imageDimensions = { ...item.dims };

    // Update UI
    currentPreviewBaseSrc = item.originalSrc || item.src;
    document.getElementById('preview').src = currentPreviewBaseSrc;
    renderBatchStrip();
    renderResults();
    renderBoxes();
    renderCropReview();
}

function resetSession(force = false) {
    if (currentTab === 'image') {
        resetImageSession(force);
    } else if (currentTab === 'video') {
        resetVideoSession(force);
    } else {
        resetTreeSession(force);
    }
}

function resetImageSession(force = false) {
    if (!force && batchImages.length > 0) {
        if (!confirm("Clear current batch and all detection results?")) return;
    }

    batchImages.forEach(revokeBatchImageUrls);
    batchImages = [];
    activeBatchIndex = -1;
    if (currentTab === 'image') {
        clearActiveImageState();
    }

    document.getElementById('uploadPrompt').classList.remove('hidden');
    document.getElementById('batchStripWrapper').classList.add('hidden');
    document.getElementById('imageContainer').classList.add('hidden');
    
    const dz = document.getElementById('dropZone');
    if (dz) {
        dz.classList.remove('py-4', 'border-transparent');
        dz.classList.add('p-10', 'border-dashed', 'border-white/5', 'hover:border-blue-500/30', 'cursor-pointer');
    }
    const uploadInput = document.getElementById('upload');
    if (uploadInput) uploadInput.value = '';

    const surveyContainer = document.getElementById('survey-form-container');
    if (surveyContainer) surveyContainer.innerHTML = '';

    localStorage.removeItem(DRAFT_KEY);
    showToast("Session cleared for next survey", "info");
}

function resetVideoSession(force = false) {
    if (!force && videoPoleResults.length > 0) {
        if (!confirm("Clear current video survey and all pole results?")) return;
    }

    uploadedVideoFile = null;
    videoPoleResults = [];
    inspectingVideoPoleId = null;
    if (currentTab === 'video') {
        detections = [];
        masterResult = null;
    }

    const videoUploadPrompt = document.getElementById('videoUploadPrompt');
    if (videoUploadPrompt) videoUploadPrompt.classList.remove('hidden');
    
    const videoContainer = document.getElementById('videoContainer');
    if (videoContainer) videoContainer.classList.add('hidden');
    
    const videoPreview = document.getElementById('videoPreview');
    if (videoPreview) {
        videoPreview.src = '';
        videoPreview.load();
    }

    const videoDropZone = document.getElementById('videoDropZone');
    if (videoDropZone) {
        videoDropZone.classList.remove('py-4', 'border-transparent');
        videoDropZone.classList.add('p-10', 'border-dashed', 'border-white/5', 'hover:border-blue-500/30', 'cursor-pointer');
    }
    const videoUploadInput = document.getElementById('videoUpload');
    if (videoUploadInput) videoUploadInput.value = '';

    const imageContainer = document.getElementById('imageContainer');
    if (imageContainer && currentTab === 'video') {
        imageContainer.classList.add('hidden');
    }

    if (currentTab === 'video') {
        document.getElementById('resultBox').innerHTML = '';
        document.getElementById('masterIdentityCard').classList.add('hidden');
        const submitSection = document.getElementById('submitSection');
        if (submitSection) submitSection.classList.add('hidden');
        hideCropReview();
    }

    showToast("Video session reset", "info");
}

function resetTreeSession(force = false) {
    if (!force && treeAnalysisState.primaryFile) {
        if (!confirm('Clear current pole vegetation analysis result?')) return;
    }

    revokeTreePreview(treeAnalysisState.primarySrc);

    treeAnalysisState = {
        primaryFile: null,
        primarySrc: '',
        result: null,
        processed: false,
        reportId: '',
        selectedTreeId: '',
    };

    if (currentTab === 'tree') {
        detections = [];
        uploadedFile = null;
        masterResult = null;
    }

    document.getElementById('uploadPrompt').classList.remove('hidden');
    document.getElementById('batchStripWrapper').classList.add('hidden');
    document.getElementById('imageContainer').classList.add('hidden');
    document.getElementById('masterIdentityCard').classList.add('hidden');
    document.getElementById('submitSection').classList.add('hidden');

    const preview = document.getElementById('preview');
    if (preview) preview.src = '';
    const overlay = document.getElementById('detectionOverlay');
    if (overlay) overlay.innerHTML = '';

    const dz = document.getElementById('dropZone');
    if (dz) {
        dz.classList.remove('py-4', 'border-transparent');
        dz.classList.add('p-10', 'border-dashed', 'border-white/5', 'hover:border-blue-500/30', 'cursor-pointer');
    }

    const uploadInput = document.getElementById('upload');
    if (uploadInput) uploadInput.value = '';
    const poleIdInput = document.getElementById('treePoleId');
    const latInput = document.getElementById('treeLatitude');
    const lonInput = document.getElementById('treeLongitude');
    if (poleIdInput) poleIdInput.value = '';
    if (latInput) latInput.value = '';
    if (lonInput) lonInput.value = '';

    syncTreeControls();
    renderAnalysisPlaceholder('fa-tree', 'Pole vegetation analysis ready', 'UPLOAD ONE POLE CONTEXT IMAGE TO MAP TREE RISKS');
    showToast('Pole vegetation session reset', 'info');
}

async function resizeImage(file, maxWidth, maxHeight) {
    return new Promise((resolve) => {
        const reader = new FileReader();
        reader.readAsDataURL(file);
        reader.onload = (event) => {
            const img = new Image();
            img.src = event.target.result;
            img.onload = () => {
                const canvas = document.createElement('canvas');
                let width = img.width;
                let height = img.height;

                if (width > height) {
                    if (width > maxWidth) {
                        height *= maxWidth / width;
                        width = maxWidth;
                    }
                } else {
                    if (height > maxHeight) {
                        width *= maxHeight / height;
                        height = maxHeight;
                    }
                }

                canvas.width = width;
                canvas.height = height;
                const ctx = canvas.getContext('2d');
                ctx.drawImage(img, 0, 0, width, height);

                canvas.toBlob((blob) => {
                    resolve(blob);
                }, 'image/jpeg', 0.85);
            };
        };
    });
}

async function processImage() {
    if (batchImages.length === 0) {
        showToast("Please upload an image first", "warning");
        return;
    }

    const btn = document.getElementById('btnRun');
    const btnText = btn.querySelector('.btn-text');
    const loader = btn.querySelector('.loader');

    btn.disabled = true;
    loader.classList.remove('hidden');
    isProcessingBatch = true;

    try {
        let processedCount = 0;

        for (let i = 0; i < batchImages.length; i++) {
            if (batchImages[i].processed) continue;

            btnText.textContent = `Processing ${i + 1} of ${batchImages.length}...`;

            let file = batchImages[i].file;
            let imageToUpload = file;
            if (file.size > 1024 * 1024) {
                imageToUpload = await resizeImage(file, 1280, 1280);
            }

            const formData = new FormData();
            formData.append("image", imageToUpload, "image.jpg");

            let response;
            try {
                response = await fetch("/predict", {
                    method: "POST",
                    headers: { "ngrok-skip-browser-warning": "69420" },
                    body: formData
                });
            } catch (error) {
                showToast(`Failed on image ${i + 1}: ${error.message || 'Network error'}`, "danger");
                continue;
            }

            if (!response.ok) {
                const errorData = await response.json().catch(() => ({}));
                showToast(`Failed on image ${i + 1}: ${errorData.error || response.status}`, "danger");
                continue;
            }

            const data = await response.json();

            batchImages[i].detections = data.detections.map(d => ({
                ...d,
                label: d.label.toUpperCase(),
                confirmed: false
            }));
            batchImages[i].master = data.master;
            batchImages[i].survey = data.survey_questionnaire || {};
            batchImages[i].dims = { width: data.width, height: data.height };
            batchImages[i].uploaded_image_id = data.uploaded_image_id || batchImages[i].uploaded_image_id || null;
            batchImages[i].inference_mode = data.inference_mode || "accurate";

            if (data.annotated_image) {
                function formatB64(src) {
                    if (!src) return 'https://via.placeholder.com/400x300?text=No+Data';
                    let clean = src.toString().trim();
                    if (clean.startsWith('http') || clean.startsWith('blob:') || clean.startsWith('/static/')) return clean;
                    const parts = clean.split('base64,');
                    clean = parts[parts.length - 1];
                    return 'data:image/jpeg;base64,' + clean;
                }
                batchImages[i].src = formatB64(data.annotated_image);
            }

            batchImages[i].processed = true;
            processedCount++;
            renderBatchStrip();
        }

        if (processedCount > 0) {
            isProcessingBatch = false;
            const currentIdx = activeBatchIndex;
            activeBatchIndex = -1;
            selectBatchImage(currentIdx !== -1 ? currentIdx : 0);

            document.getElementById("imageContainer").classList.remove("hidden");
            document.getElementById("submitSection").classList.remove("hidden");

            saveDraft();
            showToast(`Batch Complete: ${processedCount} images analyzed`, "success");
        } else {
            showToast("All images are already processed", "info");
        }
    } catch (err) {
        showToast(err.message || "Batch processing error", "danger");
    } finally {
        isProcessingBatch = false;
        btn.disabled = false;
        btnText.textContent = "Run Detection";
        loader.classList.add('hidden');
    }
}

// Auto-save to server every 30 seconds
setInterval(() => {
    if (Date.now() - lastSaveTime > 30000) {
        saveDraft();
    }
}, 30000);

// Keyboard Shortcuts
window.addEventListener('keydown', (e) => {
    if (e.ctrlKey && e.key === 'z') {
        e.preventDefault();
        undo();
    }
    if (e.ctrlKey && e.key === 'y') {
        e.preventDefault();
        redo();
    }
    if (e.ctrlKey && e.key === 's') {
        e.preventDefault();
        saveDraft();
        showToast("Draft Saved", "success");
    }
    if (e.key === 'Delete' || e.key === 'Backspace') {
        // Only delete if a box is highlighted/selected
        const highlighted = document.querySelector('.detection-box.highlighted');
        if (highlighted) {
            const idx = parseInt(highlighted.id.split('-')[1]);
            removeDetection(idx);
            showToast("Object Removed", "warning");
        }
    }
});

function renderResults() {
    // Guard: if we are in video mode and NOT inspecting a specific pole,
    // do NOT overwrite the video survey pole cards that renderVideoResults() produced.
    if (currentTab === 'video' && (typeof inspectingVideoPoleId === 'undefined' || inspectingVideoPoleId === null)) {
        hideCropReview();
        return;
    }

    const container = document.getElementById("resultBox");
    const masterCard = document.getElementById("masterIdentityCard");
    container.innerHTML = "";

    // If inspecting a video pole, add a back button to return to the video survey list
    if (currentTab === 'video' && typeof inspectingVideoPoleId !== 'undefined' && inspectingVideoPoleId !== null) {
        const backBtn = document.createElement("button");
        backBtn.className = "w-full mb-4 py-2.5 bg-gray-800/60 hover:bg-gray-700/60 text-gray-300 rounded-xl text-[10px] font-bold uppercase tracking-wider flex items-center justify-center gap-2 border border-white/5 active:scale-95 transition-all";
        backBtn.innerHTML = `<i class="fa-solid fa-arrow-left-long mr-1.5"></i> Back to Video Survey`;
        backBtn.onclick = exitPoleInspection;
        container.appendChild(backBtn);
    }

    // 1. Populate Master Asset Identity Card
    if (masterResult) {
        masterCard.classList.remove('hidden');
        document.getElementById("masterClass").textContent = masterResult.final_class.replace(/_/g, ' ');
        document.getElementById("masterVoltage").textContent = masterResult.voltage === "raw_mode" ? "Raw Predictions" : masterResult.voltage;

        const confEl = document.getElementById("masterConfidence");
        const numericConf = parseFloat(masterResult.confidence);
        const confVal = Number.isFinite(numericConf) ? (numericConf >= 0.66 ? "high" : (numericConf >= 0.33 ? "medium" : "low")) : ((masterResult.confidence || "low").toLowerCase());
        if (confVal === 'high') confEl.className = "text-[8px] font-bold text-emerald-400/80 uppercase tracking-widest";
        else if (confVal === 'medium') confEl.className = "text-[8px] font-bold text-amber-400/80 uppercase tracking-widest";
        else confEl.className = "text-[8px] font-bold text-rose-400/80 uppercase tracking-widest";

        // 1.1 Update Pole Stability Row
        const stabilityRow = document.getElementById("poleStabilityRow");
        const angleEl = document.getElementById("poleLeanAngle");
        const statusBadge = document.getElementById("poleStatusBadge");

        if (masterResult.pole_lean_angle !== undefined) {
            stabilityRow.classList.remove("hidden");
            const angle = parseFloat(masterResult.pole_lean_angle);
            angleEl.textContent = `${angle.toFixed(1)}°`;

            let status = "Raw Pole";
            let badgeClass = "bg-emerald-500/10 text-emerald-400 border border-emerald-500/20";

            if (masterResult.pole_type === 'strut_pole' || masterResult.pole_status === 'strut_pole') {
                status = "Strut Pole";
                badgeClass = "bg-blue-500/10 text-blue-400 border border-blue-500/20";
            } else {
                status = "Pole";
                badgeClass = "bg-emerald-500/10 text-emerald-400 border border-emerald-500/20";
            }

            statusBadge.textContent = status;
            statusBadge.className = `px-2 py-0.5 rounded text-[8px] font-bold uppercase ${badgeClass}`;
        } else {
            stabilityRow.classList.add("hidden");
        }

        // 1.1.5 Compute Max Hardware Counts Across Batch
        const maxCounts = {};
        let processedImagesCount = 0;
        batchImages.forEach(imgData => {
            if (!imgData.processed || !imgData.detections) return;
            processedImagesCount++;
            const currentCounts = {};
            imgData.detections.forEach(d => {
                // Group by main category (e.g. PIN_INSULATOR -> INSULATOR)
                let type = d.label;
                if (type.includes('INSULATOR') || type.includes('PIN') || type.includes('DISC') || type.includes('SHACKLE')) type = 'INSULATOR';
                else if (type.includes('POLE')) type = 'POLE';
                else if (type.includes('CROSSARM') || type === 'V_CROSS_ARM' || type === 'V_CROSS' || type === 'TAPPING_ARM' || type === 'TAPPING_CHANNEL' || type === 'SIDE_ARM' || type === 'SIDE_ARM_CHANNEL' || type === 'T_RISING' || type === 'TOP_CLEAT' || type === 'SPECIAL_CLAMP' || type === 'STAY_SET' || type === 'AB_SWITCH' || type === 'BOX_ARM') type = 'CROSSARM';
                else if (type.includes('CONDUCTOR')) type = 'CONDUCTOR';
                else type = type.split('_')[0];

                currentCounts[type] = (currentCounts[type] || 0) + 1;
            });
            for (const type in currentCounts) {
                if (!maxCounts[type] || currentCounts[type] > maxCounts[type]) {
                    maxCounts[type] = currentCounts[type];
                }
            }
        });

        const batchCountsRow = document.getElementById("batchCountsRow");
        if (Object.keys(maxCounts).length > 0 && processedImagesCount > 1) {
            batchCountsRow.classList.remove("hidden");
            batchCountsRow.innerHTML = `<span class="w-full text-[8px] text-gray-500 font-bold uppercase mb-1">Max Hardware Detected Across ${processedImagesCount} Views</span>`;

            for (const type in maxCounts) {
                let icon = 'fa-tag';
                if (type === 'POLE') icon = 'fa-tower-broadcast';
                else if (type === 'INSULATOR') icon = 'fa-bolt';
                else if (type === 'CROSSARM') icon = 'fa-compass-drafting';
                else if (type === 'CONDUCTOR') icon = 'fa-layer-group';

                batchCountsRow.innerHTML += `
                    <div class="px-2 py-1 bg-white/5 border border-white/10 rounded flex items-center gap-1.5 shadow-sm">
                        <i class="fa-solid ${icon} text-[9px] text-blue-400"></i>
                        <span class="text-[9px] font-bold text-white">${maxCounts[type]} <span class="text-gray-500">${type}S</span></span>
                    </div>
                `;
            }
        } else {
            batchCountsRow.classList.add("hidden");
        }
    } else {
        masterCard.classList.add('hidden');
    }

    // 2. Aggregate detections:
    //    - In video pole inspection mode, only show the current pole's detections (stored in global `detections`).
    //    - In image mode, aggregate across all batch images.
    let allDetectionsList = [];
    if (currentTab === 'video' && inspectingVideoPoleId !== null) {
        // Use only the globally loaded video pole detections — avoid pulling from batchImages
        detections.forEach((d, detIdx) => {
            allDetectionsList.push({ ...d, imgIdx: 0, detIdx });
        });
    } else {
        batchImages.forEach((img, imgIdx) => {
            if (img.detections) {
                img.detections.forEach((d, detIdx) => {
                    allDetectionsList.push({ ...d, imgIdx, detIdx });
                });
            }
        });
    }

    if (allDetectionsList.length === 0) {
        container.innerHTML = `
            <div class="text-center py-20 bg-black/30 rounded-2xl border border-dashed border-gray-800">
                <i class="fa-solid fa-wand-magic-sparkles text-4xl text-gray-700 mb-4 block"></i>
                <p class="text-gray-600 text-sm italic">Waiting for analysis results...</p>
            </div>
        `;
        const submitBtn = document.getElementById('finalSubmitBtn');
        if (submitBtn) {
            submitBtn.classList.add('opacity-50', 'cursor-not-allowed');
            submitBtn.disabled = true;
        }
        hideCropReview();
        return;
    }

    allDetectionsList.sort((a, b) => a.label.localeCompare(b.label));

    const itemsContainer = document.createElement("div");
    itemsContainer.className = "space-y-2 pb-4 max-h-[300px] overflow-y-auto pr-1 custom-scrollbar";

    const allConfirmedList = allDetectionsList.length > 0 && allDetectionsList.every(d => d.confirmed);
    const masterHeader = document.createElement("div");
    masterHeader.className = "flex items-center justify-between p-3 mb-3 border border-gray-800 rounded-xl bg-gray-900/60 sticky top-0 z-10 backdrop-blur-md";
    masterHeader.innerHTML = `
        <div class="flex items-center gap-3">
            <i class="fa-solid fa-list-ul text-blue-400"></i>
            <span class="text-[10px] font-bold text-gray-300 uppercase tracking-widest">ALL DETECTIONS (${allDetectionsList.length})</span>
        </div>
        <button onclick="bulkConfirmAll()" 
                class="px-3 py-1.5 rounded-lg text-[8px] font-bold uppercase tracking-widest transition-all flex items-center gap-1.5 ${allConfirmedList ? 'bg-green-600/20 text-green-400 border border-green-500/30' : 'bg-white/5 text-gray-500 border border-white/10 hover:bg-white/10 hover:text-white'}">
            <i class="fa-solid ${allConfirmedList ? 'fa-check-double' : 'fa-check'}"></i>
            ${allConfirmedList ? 'ALL CHECKED' : 'BULK CHECK ALL'}
        </button>
    `;
    container.appendChild(masterHeader);

    allDetectionsList.forEach((obj) => {
        const label = obj.label;
        const baseColor = CLASS_COLORS[label.toUpperCase()] || "#a8a29e";

        const itemDiv = document.createElement("div");
        itemDiv.className = `flex items-center justify-between p-3 rounded-lg border transition-all result-card-hover ${obj.confirmed ? 'border-green-500/40 bg-green-500/5' : 'bg-white/5 border-white/5'} hover:border-white/20`;

        // Interactive Sync: Glow the box on the image when hovering the result item
        itemDiv.onmouseenter = () => {
            // Auto switch image view if needed
            if (activeBatchIndex !== obj.imgIdx) {
                selectBatchImage(obj.imgIdx);
            }
                        const box = document.getElementById(`box-${obj.detIdx}`);
            if (box) box.classList.add('highlighted');
            const lbl = document.getElementById(`label-${obj.detIdx}`);
            if (lbl) {
                lbl.classList.remove('opacity-0');
                lbl.classList.add('opacity-100');
                lbl.style.transform = 'scale(1.1)';
            }
        };
        itemDiv.onmouseleave = () => {
            const box = document.getElementById(`box-${obj.detIdx}`);
            if (box) box.classList.remove('highlighted');
            const lbl = document.getElementById(`label-${obj.detIdx}`);
            if (lbl) {
                lbl.classList.remove('opacity-100');
                lbl.classList.add('opacity-0');
                lbl.style.transform = '';
            }
        };

        let detailStr = "";
        let metaIcon = "fa-tag";
        const fakeConfStr = getFakeConfidenceValue(obj.confidence);

        if ((obj.label.includes('INS') || obj.label.includes('PIN') || obj.label.includes('DISC') || obj.label.includes('SHACKLE')) && obj.details) {
            detailStr = `${obj.details.type || obj.label} | Conf: ${fakeConfStr}`;
            metaIcon = "fa-bolt";
        } else if ((obj.label.includes('CROSSARM') || obj.label === 'V_CROSS_ARM' || obj.label === 'V_CROSS' || obj.label === 'TAPPING_ARM' || obj.label === 'TAPPING_CHANNEL' || obj.label === 'SIDE_ARM' || obj.label === 'SIDE_ARM_CHANNEL' || obj.label === 'T_RISING' || obj.label === 'TOP_CLEAT' || obj.label === 'SPECIAL_CLAMP' || obj.label === 'STAY_SET' || obj.label === 'AB_SWITCH' || obj.label === 'BOX_ARM') && obj.details) {
            detailStr = `Label: ${obj.details.raw_label || obj.label} | Conf: ${fakeConfStr}`;
            metaIcon = "fa-compass-drafting";
        } else if (obj.label.includes('POLE') && obj.details) {
            const lean = obj.details.lean || 0;
            const isStrut = obj.details.type === 'strut_pole';
            const isExtreme = !isStrut && lean > 10;

            if (isStrut) {
                detailStr = `<span class="text-blue-400">ANGLE: ${lean}°</span> | <span class="text-blue-300">strut pole</span> | <span class="text-white/60">Conf: ${fakeConfStr}</span>`;
                metaIcon = "fa-ruler-combined";
            } else {
                const leanColor = isExtreme ? 'text-rose-400 font-black' : (lean > 5 ? 'text-amber-400' : 'text-emerald-400');
                const abnormalityTag = isExtreme ? `<span class="bg-rose-500/20 text-rose-400 px-2 py-0.5 rounded-full text-[7px] border border-rose-500/30 ml-2 animate-pulse">ABNORMALITY</span>` : "";
                detailStr = `<span class="${leanColor}">LEAN: ${lean}°</span>${abnormalityTag} | MAIN POLE | <span class="text-white/60">Conf: ${fakeConfStr}</span>`;
                metaIcon = "fa-triangle-exclamation";
            }
        } else if (obj.label === 'WIRE_BROKEN') {
            detailStr = `<span class="text-rose-500 font-bold underline">CRITICAL: SNAPPED CONDUCTOR</span> | Conf: ${fakeConfStr}`;
            metaIcon = "fa-scissors";
        } else if (obj.label === 'VEGETATION') {
            detailStr = `<span class="text-amber-500 font-bold">ENCROACHMENT DETECTED</span> | Conf: ${fakeConfStr}`;
            metaIcon = "fa-leaf";
        } else {
            detailStr = `Confidence: ${fakeConfStr}`;
        }

        itemDiv.innerHTML = `
            <div class="flex items-center gap-3">
                <div class="w-1.5 h-1.5 rounded-full" style="background: ${baseColor}"></div>
                <div>
                    <div class="flex items-center gap-2">
                        <p class="text-[10px] font-bold text-gray-200 uppercase tracking-tight">${label} ID-${obj.detIdx + 1}</p>
                        ${batchImages.length > 1 ? `<span class="px-1.5 py-0.5 bg-blue-500/20 text-blue-400 rounded border border-blue-500/30 text-[7px] font-bold">IMAGE ${obj.imgIdx + 1}</span>` : ''}
                    </div>
                    <div class="flex items-center gap-1.5 mt-0.5">
                        <i class="fa-solid ${metaIcon} text-[8px] text-gray-600"></i>
                        <p class="text-[9px] text-gray-500 font-medium uppercase tracking-widest">${detailStr}</p>
                    </div>
                </div>
            </div>
            <div class="flex items-center gap-2">
                ${currentTab === 'video' && inspectingVideoPoleId !== null
                    ? `<button onclick="toggleConfirm(${obj.detIdx})" class="btn ${obj.confirmed ? 'bg-green-600 text-white' : 'btn-outline border-white/5 bg-white/5 hover:border-white/20'} !p-2 !h-8 !w-8 !rounded-lg text-[10px]">
                            <i class="fa-solid ${obj.confirmed ? 'fa-check-double' : 'fa-check'}"></i>
                       </button>
                       <button onclick="removeDetection(${obj.detIdx})" class="p-2 text-gray-600 hover:text-rose-400 transition-colors">
                           <i class="fa-solid fa-trash-can text-[10px]"></i>
                       </button>`
                    : `<button onclick="toggleConfirmGlobal(${obj.imgIdx}, ${obj.detIdx})" class="btn ${obj.confirmed ? 'bg-green-600 text-white' : 'btn-outline border-white/5 bg-white/5 hover:border-white/20'} !p-2 !h-8 !w-8 !rounded-lg text-[10px]">
                           <i class="fa-solid ${obj.confirmed ? 'fa-check-double' : 'fa-check'}"></i>
                       </button>
                       <button onclick="removeDetectionGlobal(${obj.imgIdx}, ${obj.detIdx})" class="p-2 text-gray-600 hover:text-rose-400 transition-colors">
                           <i class="fa-solid fa-trash-can text-[10px]"></i>
                       </button>`
                }
            </div>
        `;
        itemsContainer.appendChild(itemDiv);
    });

    container.appendChild(itemsContainer);

    // Update final submit button state
    const allConfirmed = allDetectionsList.every(d => d.confirmed);
    const submitBtn = document.getElementById('finalSubmitBtn');
    if (submitBtn) {
        if (allConfirmed && allDetectionsList.length > 0) {
            submitBtn.classList.remove('opacity-50', 'cursor-not-allowed');
            submitBtn.disabled = false;
        } else {
            submitBtn.classList.add('opacity-50', 'cursor-not-allowed');
            submitBtn.disabled = true;
        }
    }

    renderCropReview();

    // Trigger auto-survey classification and questionnaire auto-fill
    if (window.PoleSurveyClassifier && typeof window.PoleSurveyClassifier.processDetectionsAndRedirect === 'function') {
        const activeItem = activeBatchIndex !== -1 ? batchImages[activeBatchIndex] : null;
        const activeFileName = activeItem ? (activeItem.name || (activeItem.file ? activeItem.file.name : 'Active Image')) : 'Active Image';
        window.PoleSurveyClassifier.processDetectionsAndRedirect({
            detections: detections,
            image_name: activeFileName
        });
    }
}

function updateLabel(index, val) {
    saveToHistory();
    if (val === "Custom") {
        const customVal = prompt("Enter custom class name:");
        if (customVal) {
            detections[index].label = customVal.toUpperCase();
        }
    } else {
        detections[index].label = val;
    }
    renderResults();
    renderBoxes();
    saveDraft();
}

function toggleConfirmGlobal(imgIdx, detIdx) {
    saveToHistory();
    if (imgIdx !== -1 && batchImages[imgIdx]) {
        batchImages[imgIdx].detections[detIdx].confirmed = !batchImages[imgIdx].detections[detIdx].confirmed;
    }
    if (activeBatchIndex === imgIdx || (currentTab === 'video' && inspectingVideoPoleId !== null)) {
        detections[detIdx].confirmed = !detections[detIdx].confirmed;
    }
    renderResults();
    renderBoxes();
    saveDraft();
    updateUndoRedoButtons();
}

function removeDetectionGlobal(imgIdx, detIdx) {
    saveToHistory();
    if (imgIdx !== -1 && batchImages[imgIdx]) {
        batchImages[imgIdx].detections.splice(detIdx, 1);
    }
    if (activeBatchIndex === imgIdx || (currentTab === 'video' && inspectingVideoPoleId !== null)) {
        detections.splice(detIdx, 1);
    }
    renderResults();
    renderBoxes();
    saveDraft();
    updateUndoRedoButtons();
}

function toggleConfirm(index) {
    toggleConfirmGlobal(activeBatchIndex, index);
}

function removeDetection(index) {
    removeDetectionGlobal(activeBatchIndex, index);
}

// Keep a backup of bulkConfirm if any legacy function calls it
function bulkConfirm(label, state = true) {
    saveToHistory();
    let count = 0;
    detections.forEach(d => {
        if (d.label === label) {
            d.confirmed = state;
            count++;
        }
    });
    renderResults();
    renderBoxes();
    showToast(`${state ? 'Confirmed' : 'Unconfirmed'} all ${label}s (${count} items)`, "success");
}

function bulkConfirmAll() {
    saveToHistory();
    
    if (currentTab === 'video' && inspectingVideoPoleId !== null) {
        const allConfirmed = detections.length > 0 && detections.every(d => d.confirmed);
        const targetState = !allConfirmed;
        
        detections.forEach(d => d.confirmed = targetState);
        
        const pole = videoPoleResults.find(p => p.pole_id === inspectingVideoPoleId);
        if (pole) {
            pole.detections = detections.map(d => ({ ...d }));
        }
    } else {
        const allConfirmed = !batchImages.some(img => img.detections && img.detections.some(d => !d.confirmed));
        const targetState = !allConfirmed;
        
        batchImages.forEach(img => {
            if (img.detections) {
                img.detections.forEach(d => d.confirmed = targetState);
            }
        });
        
        if (batchImages[activeBatchIndex] && batchImages[activeBatchIndex].detections) {
            detections = [...batchImages[activeBatchIndex].detections];
        }
    }
    
    renderResults();
    renderBoxes();
    saveDraft();
    updateUndoRedoButtons();
}

function removeGroup(label) {
    if (confirm(`Are you sure you want to remove all "${label}" detections?`)) {
        detections = detections.filter(d => d.label !== label);
        renderResults();
        renderBoxes();
        saveDraft();
        showToast(`Removed all ${label} items`, "primary");
    }
}

function renderPoleVegetationBoxes(overlay, img) {
    overlay.innerHTML = '';
    const result = treeAnalysisState.result || {};
    const trees = result.trees || [];
    const wireLines = result.wire_lines || [];
    if (!trees.length && !wireLines.length) return;

    const displayWidth = img.clientWidth || img.naturalWidth || 0;
    const displayHeight = img.clientHeight || img.naturalHeight || 0;
    const naturalWidth = img.naturalWidth || displayWidth || 1;
    const naturalHeight = img.naturalHeight || displayHeight || 1;
    if (!displayWidth || !displayHeight) return;

    const scalePoint = (point) => ({ x: (Number(point?.x || 0) / naturalWidth) * displayWidth, y: (Number(point?.y || 0) / naturalHeight) * displayHeight });

    wireLines.forEach((line) => {
        const wire = document.createElementNS('http://www.w3.org/2000/svg', 'line');
        wire.setAttribute('x1', Math.max(0, Math.min(1, Number(line.x1 || 0))) * displayWidth);
        wire.setAttribute('y1', Math.max(0, Math.min(1, Number(line.y1 || 0))) * displayHeight);
        wire.setAttribute('x2', Math.max(0, Math.min(1, Number(line.x2 || 0))) * displayWidth);
        wire.setAttribute('y2', Math.max(0, Math.min(1, Number(line.y2 || 0))) * displayHeight);
        wire.setAttribute('stroke', '#67e8f9');
        wire.setAttribute('stroke-width', '2');
        wire.setAttribute('stroke-dasharray', '7 5');
        wire.setAttribute('stroke-linecap', 'round');
        wire.setAttribute('opacity', '0.95');
        overlay.appendChild(wire);
    });

    trees.forEach((tree, index) => {
        const bbox = tree?.bbox_normalized;
        if (!bbox) return;

        const x1 = Math.max(0, Math.min(1, Number(bbox.x_min || 0))) * displayWidth;
        const y1 = Math.max(0, Math.min(1, Number(bbox.y_min || 0))) * displayHeight;
        const x2 = Math.max(0, Math.min(1, Number(bbox.x_max || 0))) * displayWidth;
        const y2 = Math.max(0, Math.min(1, Number(bbox.y_max || 0))) * displayHeight;
        const w = x2 - x1;
        const h = y2 - y1;
        if (w <= 2 || h <= 2) return;

        const metrics = tree.distance_metrics || {};
        const risk = `${tree.risk_level || 'low'}`.toLowerCase();
        const color = risk === 'critical' ? '#fb7185' : risk === 'high' ? '#fb923c' : risk === 'medium' ? '#facc15' : risk === 'low' ? '#22d3ee' : '#34d399';
        const selected = treeAnalysisState.selectedTreeId && tree.tree_id === treeAnalysisState.selectedTreeId;

        const shape = document.createElementNS('http://www.w3.org/2000/svg', 'rect');
        shape.setAttribute('x', x1);
        shape.setAttribute('y', y1);
        shape.setAttribute('width', w);
        shape.setAttribute('height', h);
        shape.setAttribute('rx', '12');
        shape.setAttribute('stroke', color);
        shape.setAttribute('stroke-width', selected ? '3' : '2');
        shape.setAttribute('fill', color);
        shape.setAttribute('fill-opacity', selected ? '0.18' : '0.10');
        shape.style.pointerEvents = 'auto';
        shape.style.cursor = 'pointer';
        shape.addEventListener('click', () => selectVegetationTree(tree.tree_id));
        overlay.appendChild(shape);

        if (metrics.nearest_tree_point_px && metrics.nearest_wire_point_px) {
            const start = scalePoint(metrics.nearest_tree_point_px);
            const end = scalePoint(metrics.nearest_wire_point_px);
            const clearance = document.createElementNS('http://www.w3.org/2000/svg', 'line');
            clearance.setAttribute('x1', start.x);
            clearance.setAttribute('y1', start.y);
            clearance.setAttribute('x2', end.x);
            clearance.setAttribute('y2', end.y);
            clearance.setAttribute('stroke', selected ? '#ffffff' : color);
            clearance.setAttribute('stroke-width', selected ? '2.5' : '1.5');
            clearance.setAttribute('opacity', '0.85');
            overlay.appendChild(clearance);
        }

        const clearanceText = metrics.estimated_wire_clearance_m === null || metrics.estimated_wire_clearance_m === undefined
            ? 'N/A'
            : `~${metrics.estimated_wire_clearance_m} m`;
        const labelText = document.createElementNS('http://www.w3.org/2000/svg', 'text');
        labelText.textContent = `Tree ${index + 1} ${formatTreeLabel(tree.risk_level)} | ${clearanceText}`;
        labelText.setAttribute('font-size', '12px');
        labelText.setAttribute('font-family', 'Outfit, Inter, sans-serif');
        labelText.setAttribute('font-weight', '700');
        labelText.setAttribute('fill', '#08111f');
        labelText.style.visibility = 'hidden';
        overlay.appendChild(labelText);
        const textBox = labelText.getBBox();
        overlay.removeChild(labelText);
        labelText.style.visibility = 'visible';

        const rectW = textBox.width + 18;
        const rectH = textBox.height + 10;
        const labelY = y1 > rectH + 8 ? y1 - rectH - 6 : y1 + 6;
        const labelX = Math.min(displayWidth - rectW / 2 - 4, Math.max(rectW / 2 + 4, x1 + w / 2));

        const bg = document.createElementNS('http://www.w3.org/2000/svg', 'rect');
        bg.setAttribute('x', labelX - rectW / 2);
        bg.setAttribute('y', labelY);
        bg.setAttribute('width', rectW);
        bg.setAttribute('height', rectH);
        bg.setAttribute('rx', '10');
        bg.setAttribute('fill', color);

        labelText.setAttribute('x', labelX);
        labelText.setAttribute('y', labelY + rectH / 2);
        labelText.setAttribute('text-anchor', 'middle');
        labelText.setAttribute('dominant-baseline', 'middle');

        const labelGroup = document.createElementNS('http://www.w3.org/2000/svg', 'g');
        labelGroup.style.pointerEvents = 'none';
        labelGroup.appendChild(bg);
        labelGroup.appendChild(labelText);
        overlay.appendChild(labelGroup);
    });
}

function renderBoxes() {
    const overlay = document.getElementById('detectionOverlay');
    const img = document.getElementById('preview');
    if (!overlay || !img) return;

    if (currentTab === 'tree' && treeAnalysisState.result) {
        renderPoleVegetationBoxes(overlay, img);
        return;
    }

    overlay.innerHTML = "";
    if (detections.length === 0) return;

    const defs = document.createElementNS("http://www.w3.org/2000/svg", "defs");
    defs.innerHTML = `
        <filter id="glow" x="-20%" y="-20%" width="140%" height="140%">
            <feGaussianBlur stdDeviation="3" result="blur" />
            <feComposite in="SourceGraphic" in2="blur" operator="over" />
        </filter>
    `;
    overlay.appendChild(defs);

    const cropX1 = activePreviewCropBounds ? activePreviewCropBounds[0] : 0;
    const cropY1 = activePreviewCropBounds ? activePreviewCropBounds[1] : 0;
    const cropWidth = activePreviewCropBounds ? Math.max(1, activePreviewCropBounds[2] - activePreviewCropBounds[0]) : imageDimensions.width;
    const cropHeight = activePreviewCropBounds ? Math.max(1, activePreviewCropBounds[3] - activePreviewCropBounds[1]) : imageDimensions.height;

    const displayWidth = img.clientWidth;
    const displayHeight = img.clientHeight;
    const scaleX = displayWidth / cropWidth;
    const scaleY = displayHeight / cropHeight;

    const labelCounts = {};

    detections.forEach((obj, i) => {
        if (!obj.bbox) return;

        const baseLabel = obj.label.toUpperCase();
        labelCounts[baseLabel] = (labelCounts[baseLabel] || 0) + 1;
        const confText = getFakeConfidenceValue(obj.confidence);

        let labelText = `${baseLabel} ${confText}`;
        if (baseLabel.includes('POLE') && obj.details && obj.details.lean !== undefined) {
            labelText += ` | LEAN: ${obj.details.lean}°`;
        }

        const [rawX1, rawY1, rawX2, rawY2] = obj.bbox;
        const clippedX1 = Math.max(rawX1, cropX1);
        const clippedY1 = Math.max(rawY1, cropY1);
        const clippedX2 = Math.min(rawX2, cropX1 + cropWidth);
        const clippedY2 = Math.min(rawY2, cropY1 + cropHeight);
        if (clippedX2 <= clippedX1 || clippedY2 <= clippedY1) return;

        const x = (clippedX1 - cropX1) * scaleX;
        const y = (clippedY1 - cropY1) * scaleY;
        const w = (clippedX2 - clippedX1) * scaleX;
        const h = (clippedY2 - clippedY1) * scaleY;

        const baseColor = CLASS_COLORS[baseLabel] || "#a8a29e";
        const color = obj.manual ? "#f43f5e" : baseColor;

        const shape = document.createElementNS("http://www.w3.org/2000/svg", "rect");
        shape.setAttribute("x", x);
        shape.setAttribute("y", y);
        shape.setAttribute("width", w);
        shape.setAttribute("height", h);

        shape.setAttribute("stroke", color);
        shape.setAttribute("stroke-width", obj.manual ? "2.5" : "1.5");

        if (baseLabel !== "CONDUCTOR") {
            shape.setAttribute("fill", color);
            shape.setAttribute("fill-opacity", "0.15");
        } else {
            shape.setAttribute("fill", "transparent");
            shape.setAttribute("class", "conductor-trace");
        }

        shape.setAttribute("id", `box-${i}`);
        shape.classList.add("detection-box");
        shape.style.pointerEvents = "auto";
        if (obj.manual) shape.classList.add("manual-box");
        overlay.appendChild(shape);

        const labelX = x + (w / 2);
        const labelY = y;

        // --- 3. Draw Pill Label (Background + Text) ---
        const labelGroup = document.createElementNS("http://www.w3.org/2000/svg", "g");
        labelGroup.setAttribute("class", "label-pill opacity-0 transition-opacity duration-200 pointer-events-none");
        labelGroup.setAttribute("id", `label-${i}`);

        const labelTextEl = document.createElementNS("http://www.w3.org/2000/svg", "text");
        labelTextEl.textContent = labelText;
        labelTextEl.setAttribute("font-size", "11px");
        labelTextEl.setAttribute("font-family", "Outfit, Inter, sans-serif");
        labelTextEl.setAttribute("font-weight", "700");

        // Smart contrast: if the background color is white or cyan/yellow, use dark text
        const isLightBg = color === "#ffffff" || color === "#00ffff" || color === "#fbbf24";
        labelTextEl.setAttribute("fill", isLightBg ? "#0f172a" : "#ffffff");

        labelTextEl.setAttribute("text-anchor", "middle");
        labelTextEl.setAttribute("dominant-baseline", "middle");

        // Hide temporarily to measure
        labelTextEl.style.visibility = "hidden";
        overlay.appendChild(labelTextEl);
        const bbox = labelTextEl.getBBox();
        overlay.removeChild(labelTextEl);
        labelTextEl.style.visibility = "visible";

        const paddingH = 8;
        const paddingV = 4;
        const rectW = bbox.width + paddingH * 2;
        const rectH = bbox.height + paddingV * 2;

        const labelRect = document.createElementNS("http://www.w3.org/2000/svg", "rect");
        labelRect.setAttribute("x", labelX - rectW / 2);
        labelRect.setAttribute("y", (labelY - rectH - 5 < 0) ? labelY + 5 : labelY - rectH - 5);
        labelRect.setAttribute("width", rectW);
        labelRect.setAttribute("height", rectH);
        labelRect.setAttribute("rx", "6");
        labelRect.setAttribute("fill", color);
        labelRect.setAttribute("class", "label-bg");

        labelTextEl.setAttribute("x", labelX);
        labelTextEl.setAttribute("y", (labelY - rectH - 5 < 0) ? labelY + 5 + rectH / 2 : labelY - rectH - 5 + rectH / 2);

        labelGroup.appendChild(labelRect);
        labelGroup.appendChild(labelTextEl);
        overlay.appendChild(labelGroup);

        // Hover events on the shape to toggle label visibility
        shape.addEventListener('mouseenter', () => {
            labelGroup.classList.remove('opacity-0');
            labelGroup.classList.add('opacity-100');
        });
        shape.addEventListener('mouseleave', () => {
            labelGroup.classList.remove('opacity-100');
            labelGroup.classList.add('opacity-0');
        });
    });
}

window.addEventListener('resize', renderBoxes);

/**
 * Compresses an image using Canvas before submission.
 * Reduces resolution to max 1280px and quality to 0.7 for optimal server speed.
 */
async function compressImage(src, maxWidth = 1200, quality = 0.75) {
    return new Promise((resolve, reject) => {
        const img = new Image();
        img.src = src;
        img.onload = () => {
            const canvas = document.createElement('canvas');
            let width = img.width;
            let height = img.height;

            if (width > maxWidth) {
                height *= maxWidth / width;
                width = maxWidth;
            }

            canvas.width = width;
            canvas.height = height;
            const ctx = canvas.getContext('2d');
            ctx.drawImage(img, 0, 0, width, height);

            // Convert to JPEG with specified quality
            const b64 = canvas.toDataURL('image/jpeg', quality);
            resolve(b64.split(',')[1]); // Return raw b64 data
        };
        img.onerror = reject;
    });
}

// =========================
// MANUAL ANNOTATION LOGIC
// =========================

function toggleDrawMode() {
    isDrawMode = !isDrawMode;
    const btn = document.getElementById('btnDrawMode');
    const overlay = document.getElementById('detectionOverlay');
    const status = document.getElementById('drawStatus');
    const imageContainer = document.getElementById('imageContainer');

    // Reset state when toggling
    cancelManualDraw();

    if (isDrawMode) {
        btn.classList.add('active');
        overlay.classList.add('draw-mode');
        overlay.classList.remove('pointer-events-none');
        overlay.style.pointerEvents = "auto";
        imageContainer.classList.add('draw-active');
        status.classList.remove('hidden');
        status.classList.add('flex');
        updateDrawStatus("PRESS & DRAG TO DRAW BOX");
        showToast("Draw Mode: Enabled", "primary");
    } else {
        btn.classList.remove('active');
        overlay.classList.remove('draw-mode');
        overlay.classList.add('pointer-events-none');
        overlay.style.pointerEvents = "none";
        imageContainer.classList.remove('draw-active');
        status.classList.add('hidden');
        status.classList.remove('flex');
        showToast("Draw Mode: Disabled", "primary");
    }
}

function updateDrawStatus(text) {
    const statusEl = document.getElementById('drawStatus');
    const label = statusEl.querySelector('span:last-child');
    if (label) {
        label.textContent = text;
        label.style.letterSpacing = "0.05em";
    }
}

function addNewClass() {
    const className = prompt("Enter name for the new object category:");
    if (className && className.trim()) {
        const upperName = className.trim().toUpperCase();
        if (!CLASS_OPTIONS.includes(upperName)) {
            CLASS_OPTIONS.push(upperName);
            showToast(`Added '${upperName}' to category list`, "success");
            renderResults(); // Refresh list to show new option in dropdowns
        } else {
            showToast("Category already exists", "warning");
        }
    }
}

function manageClasses() {
    if (CLASS_OPTIONS.length === 0) {
        showToast("No custom classes to manage", "warning");
        return;
    }

    let listStr = CLASS_OPTIONS.map((c, i) => `${i + 1}. ${c}`).join('\n');
    let selection = prompt(`Select class number to manage:\n\n${listStr}\n\n(Enter number)`);

    if (!selection) return;
    let idx = parseInt(selection) - 1;

    if (idx >= 0 && idx < CLASS_OPTIONS.length) {
        let oldName = CLASS_OPTIONS[idx];
        let action = prompt(`Managing "${oldName}"\nType 'R' to Rename or 'D' to Delete:`).toUpperCase();

        if (action === 'R') {
            let newName = prompt(`Enter new name for ${oldName}:`);
            if (newName && newName.trim()) {
                renameClass(oldName, newName.trim().toUpperCase());
            }
        } else if (action === 'D') {
            deleteClass(oldName);
        }
    }
}

function renameClass(oldName, newName) {
    // 1. Update the options list
    const optIdx = CLASS_OPTIONS.indexOf(oldName);
    if (optIdx !== -1) {
        CLASS_OPTIONS[optIdx] = newName;
    }

    // 2. Update all existing detections using this label
    let updateCount = 0;
    detections.forEach(det => {
        if (det.label === oldName) {
            det.label = newName;
            updateCount++;
        }
    });

    renderResults();
    renderBoxes();
    showToast(`Renamed ${oldName} to ${newName} (${updateCount} items updated)`, "success");
}

function deleteClass(name) {
    if (confirm(`Delete category "${name}"? Existing detections will remain but their category label will be static.`)) {
        CLASS_OPTIONS = CLASS_OPTIONS.filter(c => c !== name);
        renderResults();
        showToast(`Deleted category ${name}`, "warning");
    }
}

// =========================
// POINTER NORMALIZATION
// =========================

function getPointerPos(e) {
    const overlay = document.getElementById('detectionOverlay');
    const rect = overlay.getBoundingClientRect();

    // Support both mouse and touch events
    let clientX, clientY;
    if (e.touches && e.touches.length > 0) {
        clientX = e.touches[0].clientX;
        clientY = e.touches[0].clientY;
    } else {
        clientX = e.clientX;
        clientY = e.clientY;
    }

    return {
        x: clientX - rect.left,
        y: clientY - rect.top,
        rawX: clientX,
        rawY: clientY
    };
}

function handleDragStart(e) {
    if (!isDrawMode || !uploadedFile) return;

    // Prevent accidental triggers and scrolling while drawing
    e.stopPropagation();
    if (e.type === 'touchstart') e.preventDefault();
    if (e.type === 'mousedown' && e.button !== 0) return; // Only left click

    const pos = getPointerPos(e);
    drawStart = { x: pos.x, y: pos.y };
    isDrawing = true;

    // Add global listeners to track movement outside the SVG
    window.addEventListener('mousemove', handleDragMove);
    window.addEventListener('touchmove', handleDragMove, { passive: false });
    window.addEventListener('mouseup', handleDragEnd);
    window.addEventListener('touchend', handleDragEnd);

    // Initial Marker
    const hud = document.getElementById('manualPointHud');
    hud.innerHTML = ""; // Clear old markers
    const marker = document.createElement('div');
    marker.className = 'point-marker';
    const overlay = document.getElementById('detectionOverlay');
    marker.style.left = `${(pos.x / overlay.clientWidth) * 100}%`;
    marker.style.top = `${(pos.y / overlay.clientHeight) * 100}%`;
    marker.id = 'startMarker';
    hud.appendChild(marker);

    updateDrawStatus("RELEASE TO FINISH BOX");
}

function handleDragMove(e) {
    if (!isDrawing || !drawStart) return;
    if (e.type === 'touchmove') e.preventDefault();

    const pos = getPointerPos(e);
    const overlay = document.getElementById('detectionOverlay');

    let ghost = document.getElementById('ghostBox');
    if (!ghost) {
        ghost = document.createElementNS("http://www.w3.org/2000/svg", "rect");
        ghost.setAttribute("id", "ghostBox");
        ghost.setAttribute("class", "ghost-box active");
        overlay.appendChild(ghost);
    }

    const x = Math.min(drawStart.x, pos.x);
    const y = Math.min(drawStart.y, pos.y);
    const w = Math.abs(drawStart.x - pos.x);
    const h = Math.abs(drawStart.y - pos.y);

    ghost.setAttribute("x", x);
    ghost.setAttribute("y", y);
    ghost.setAttribute("width", w);
    ghost.setAttribute("height", h);
}

function handleDragEnd(e) {
    if (!isDrawing) return;

    // Remove global listeners
    window.removeEventListener('mousemove', handleDragMove);
    window.removeEventListener('touchmove', handleDragMove);
    window.removeEventListener('mouseup', handleDragEnd);
    window.removeEventListener('touchend', handleDragEnd);

    const pos = getPointerPos(e.type === 'touchend' ? { touches: e.changedTouches } : e);
    const overlay = document.getElementById('detectionOverlay');
    const rect = overlay.getBoundingClientRect();

    const x1 = Math.min(drawStart.x, pos.x);
    const y1 = Math.min(drawStart.y, pos.y);
    const x2 = Math.max(drawStart.x, pos.x);
    const y2 = Math.max(drawStart.y, pos.y);

    // Minimum size threshold to prevent accidental clicks
    const width = Math.abs(x2 - x1);
    const height = Math.abs(y2 - y1);

    if (width < 15 || height < 15) {
        cancelManualDraw();
        return;
    }

    isDrawing = false;
    pendingBbox = [x1, y1, x2, y2];

    // Add visual "selection" appearance
    const ghost = document.getElementById('ghostBox');
    if (ghost) ghost.classList.add('final-preview');

    showLabelPicker(pos.rawX, pos.rawY);
    updateDrawStatus("SELECT CATEGORY BELOW");
}

function showLabelPicker(clientX, clientY) {
    const modal = document.getElementById('labelPickerModal');
    const backdrop = document.getElementById('labelPickerBackdrop');
    const options = document.getElementById('pickerOptions');
    options.innerHTML = '';

    CLASS_OPTIONS.forEach(opt => {
        const btn = document.createElement('button');
        btn.className = 'label-picker-btn';
        btn.textContent = opt;
        btn.onclick = (e) => {
            e.stopPropagation();
            saveManualDraw(opt);
        };
        options.appendChild(btn);
    });

    backdrop.classList.remove('hidden');
    modal.classList.remove('hidden');

    // RESPONSIVE POSITIONING
    if (window.innerWidth < 640) {
        // Mobile: Center on screen
        modal.style.left = '50%';
        modal.style.top = '50%';
        modal.style.transform = 'translate(-50%, -50%)';
    } else {
        // Desktop: Float near click but stay within viewport
        let left = clientX + 30;
        let top = clientY - 100;

        const modalWidth = 340;
        const modalHeight = 400;

        // Viewport clamping
        if (left + modalWidth > window.innerWidth) left = window.innerWidth - modalWidth - 20;
        if (left < 20) left = 20;

        if (top + modalHeight > window.innerHeight) top = window.innerHeight - modalHeight - 20;
        if (top < 20) top = 20;

        modal.style.left = `${left}px`;
        modal.style.top = `${top}px`;
        modal.style.transform = 'none';
    }
}

function saveManualDraw(labelOverride = null) {
    // Prevent event from bubbling to dropZone if this was called from a button click
    if (window.event) window.event.stopPropagation();

    const customInput = document.getElementById('customLabelInput');
    const custom = customInput.value.trim().toUpperCase();
    const label = labelOverride || custom || "OBJECT";

    if (!pendingBbox) return;

    const overlay = document.getElementById('detectionOverlay');
    const rect = overlay.getBoundingClientRect();

    // Scale back to original image coordinates
    const scaleX = imageDimensions.width / rect.width;
    const scaleY = imageDimensions.height / rect.height;

    const newDet = {
        label: label,
        confidence: 1.0,
        bbox: [
            pendingBbox[0] * scaleX,
            pendingBbox[1] * scaleY,
            pendingBbox[2] * scaleX,
            pendingBbox[3] * scaleY
        ],
        confirmed: true,
        manual: true
    };

    detections.push(newDet);
    saveToHistory();
    saveDraft();

    // Add to CLASS_OPTIONS if new
    if (custom && !CLASS_OPTIONS.includes(custom)) {
        CLASS_OPTIONS.push(custom);
    }

    cancelManualDraw();
    renderResults();
    renderBoxes();
    showToast(`Added manual ${label}`, "success");
}

function cancelManualDraw(e) {
    if (e && e.stopPropagation) e.stopPropagation();

    drawStart = null;
    isDrawing = false;
    pendingBbox = null;

    const ghost = document.getElementById('ghostBox');
    if (ghost) ghost.remove();

    const marker = document.getElementById('startMarker');
    if (marker) marker.remove();

    const modal = document.getElementById('labelPickerModal');
    const backdrop = document.getElementById('labelPickerBackdrop');
    if (modal) modal.classList.add('hidden');
    if (backdrop) backdrop.classList.add('hidden');

    const customInput = document.getElementById('customLabelInput');
    if (customInput) customInput.value = '';

    updateDrawStatus("STEP 1: CLICK TO START");
}

function showToast(msg, type = "primary") {
    const toast = document.createElement("div");
    const colors = {
        success: "bg-emerald-600",
        danger: "bg-rose-600",
        warning: "bg-amber-600",
        primary: "bg-blue-600"
    };

    toast.className = `fixed bottom-8 left-1/2 -translate-x-1/2 ${colors[type]} text-white px-6 py-3 rounded-xl shadow-2xl z-[100] animate-fade-in font-bold flex items-center gap-3`;

    const icons = {
        success: "fa-circle-check",
        danger: "fa-circle-xmark",
        warning: "fa-triangle-exclamation",
        primary: "fa-info-circle"
    };

    toast.innerHTML = `<i class="fa-solid ${icons[type]}"></i> ${msg}`;
    document.body.appendChild(toast);

    setTimeout(() => {
        toast.style.opacity = '0';
        toast.style.transform = 'translate(-50%, 20px)';
        toast.style.transition = 'all 0.3s ease';
        setTimeout(() => toast.remove(), 300);
    }, 4000);
}

// ==========================================
// AR STREAMING LOGIC
// ==========================================
let arStream = null;
let arInterval = null;

async function toggleARMode() {
    if (arStream) {
        stopAR();
    } else {
        await startAR();
    }
}

async function startAR() {
    const arVideo = document.getElementById('arVideo');
    const arContainer = document.getElementById('arContainer');
    const arCanvas = document.getElementById('arCanvas');

    try {
        arStream = await navigator.mediaDevices.getUserMedia({ video: { facingMode: "environment" } });
        if (arVideo) {
            arVideo.srcObject = arStream;
            arContainer.classList.remove('hidden');

            // Hide normal upload UI
            document.getElementById('dropZone').classList.add('hidden');
            document.getElementById('imageContainer').classList.add('hidden');
            document.getElementById('submitSection').classList.add('hidden');

            arVideo.onloadedmetadata = () => {
                arCanvas.width = arVideo.videoWidth;
                arCanvas.height = arVideo.videoHeight;
                arInterval = setInterval(processARFrame, 800); // sample every 800ms
            };
        }
        showToast("AR Mode Active", "success");
    } catch (err) {
        showToast("Camera access denied or unavailable", "danger");
        console.error("AR Start Error:", err);
    }
}

function stopAR() {
    const arContainer = document.getElementById('arContainer');
    const arCanvas = document.getElementById('arCanvas');

    if (arStream) {
        arStream.getTracks().forEach(t => t.stop());
        arStream = null;
    }
    if (arInterval) clearInterval(arInterval);
    arInterval = null;

    if (arContainer && document.getElementById('dropZone')) {
        arContainer.classList.add('hidden');
        document.getElementById('dropZone').classList.remove('hidden');
    }
    if (arCanvas) {
        const ctx = arCanvas.getContext('2d');
        ctx.clearRect(0, 0, arCanvas.width, arCanvas.height);
    }
    showToast("AR Mode Stopped", "info");
}

async function processARFrame() {
    const arVideo = document.getElementById('arVideo');
    const arCanvas = document.getElementById('arCanvas');

    if (!arVideo || !arCanvas || !arStream) return;

    const ctx = arCanvas.getContext('2d');

    const tempCanvas = document.createElement('canvas');
    tempCanvas.width = arVideo.videoWidth;
    tempCanvas.height = arVideo.videoHeight;
    tempCanvas.getContext('2d').drawImage(arVideo, 0, 0);
    const frameBase64 = tempCanvas.toDataURL('image/jpeg', 0.6);

    document.getElementById('arOverlayLoading').classList.remove('hidden');

    try {
        const response = await fetch('/predict_stream', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ image: frameBase64 })
        });

        if (response.ok) {
            const data = await response.json();
            drawARBoxes(ctx, data.detections, data.width, data.height);

            // Sync side-panel
            detections = data.detections.map(d => ({ ...d, label: d.label.toUpperCase(), confirmed: false }));
            masterResult = data.master;
            renderResults();
        }
    } catch (e) {
        console.warn("AR Frame drop", e);
    } finally {
        document.getElementById('arOverlayLoading').classList.add('hidden');
    }
}

function drawARBoxes(ctx, detectionsData, origW, origH) {
    if (!ctx) return;
    const arCanvas = document.getElementById('arCanvas');
    ctx.clearRect(0, 0, arCanvas.width, arCanvas.height);

    const scaleX = arCanvas.width / origW;
    const scaleY = arCanvas.height / origH;

    detectionsData.forEach(d => {
        const color = CLASS_COLORS[d.label.toUpperCase()] || "#00ff00";
        if (d.polygon && d.polygon.length > 0) {
            ctx.beginPath();
            ctx.moveTo(d.polygon[0][0] * scaleX, d.polygon[0][1] * scaleY);
            for (let i = 1; i < d.polygon.length; i++) {
                ctx.lineTo(d.polygon[i][0] * scaleX, d.polygon[i][1] * scaleY);
            }
            ctx.closePath();
            ctx.lineWidth = 3;
            ctx.strokeStyle = color;
            ctx.stroke();
            ctx.fillStyle = color + "33";
            ctx.fill();
        } else if (d.bbox && d.bbox.length === 4) {
            const [x1, y1, x2, y2] = d.bbox;
            ctx.lineWidth = 3;
            ctx.strokeStyle = color;
            ctx.strokeRect(x1 * scaleX, y1 * scaleY, (x2 - x1) * scaleX, (y2 - y1) * scaleY);
        }

        // draw label
        const cx = (d.bbox && !!d.bbox.length ? d.bbox[0] * scaleX : d.polygon[0][0] * scaleX);
        const cy = (d.bbox && !!d.bbox.length ? d.bbox[1] * scaleY : d.polygon[0][1] * scaleY);

        ctx.fillStyle = color;
        ctx.fillRect(cx, cy - 20, ctx.measureText(d.label).width + 10, 20);
        ctx.fillStyle = "#000000";
        ctx.font = "bold 12px Inter";
        ctx.fillText(d.label.toUpperCase(), cx + 5, cy - 5);
    });
}
async function submitAsset() {
    if (!detections || detections.length === 0) {
        showToast("No active results to submit", "warning");
        return;
    }

    const btn = document.getElementById('finalSubmitBtn');
    btn.disabled = true;
    const originalContent = btn.innerHTML;
    btn.innerHTML = `<div class="loader w-4 h-4 border-2 border-white border-t-transparent rounded-full animate-spin"></div> SUBMITTING...`;

    try {
        // Collect all images in the batch to send to Admin
        const imagesToSubmit = batchImages.filter(img => img.processed).map(img => ({
            image_b64: img.src,
            uploaded_image_id: img.uploaded_image_id || null,
            detections: img.detections || [],
            pole_angle: img.master ? img.master.pole_lean_angle : (masterResult ? masterResult.pole_lean_angle : 0.0)
        }));

        if (imagesToSubmit.length === 0) {
            showToast("No processed images to submit", "danger");
            btn.disabled = false;
            btn.innerHTML = originalContent;
            return;
        }

        const payload = {
            images: imagesToSubmit,
            master: masterResult || (batchImages[0] ? batchImages[0].master : {}),
            survey_data: surveyResult || (batchImages[0] ? batchImages[0].survey : {})
        };

        const response = await fetch('/api/save_asset', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(payload)
        });

        const data = await response.json();
        if (data.status === 'success') {
            showToast("✓ Success! Survey Point Submitted & Saved", "success");
            resetImageSession(true);
            const surveyContainer = document.getElementById('survey-form-container');
            if (surveyContainer) surveyContainer.innerHTML = '';
            window.scrollTo({ top: 0, behavior: 'smooth' });
        } else {
            throw new Error(data.message);
        }
    } catch (err) {
        console.error("Submission Error:", err);
        showToast(`Submission failed: ${err.message}`, "danger");
        btn.disabled = false;
        btn.innerHTML = originalContent;
    }
}

async function downloadCurrentResult() {
    if (!detections || detections.length === 0) {
        showToast("No active results to download", "warning");
        return;
    }

    try {
        showToast("Preparing Admin Studio...", "primary");

        const activeImg = activeBatchIndex !== -1 ? batchImages[activeBatchIndex] : null;

        const assetData = {
            id: `TEMP_${Date.now()}`,
            worker_name: "Local Session",
            status: "draft",
            timestamp: new Date().toLocaleString(),
            asset_class: masterResult ? masterResult.final_class : "Unclassified",
            voltage: masterResult ? (masterResult.voltage === "raw_mode" ? "Raw Predictions" : masterResult.voltage) : "Unknown",
            reason: masterResult ? masterResult.reason : "Manual Review",
            images: [{
                image_b64: activeImg ? activeImg.src : null,
                uploaded_image_id: activeImg ? activeImg.uploaded_image_id || null : null,
                detections: detections
            }]
        };

        // Save as draft first so the admin studio can read it
        const response = await fetch('/api/save_draft', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ id: assetData.id, type: 'export', data: JSON.stringify(assetData) })
        });

        if (response.ok) {
            showToast("Opening Admin Studio", "success");
            window.location.href = `/admin/asset/${assetData.id}`;
        }
    } catch (err) {
        console.error("Export Error:", err);
        showToast("Export failed", "danger");
    }
}

// ─── Video Survey Support ──────────────────────────────────────────
let currentTab = 'image'; // 'image' | 'video'
let uploadedVideoFile = null;
let videoPoleResults = [];
let inspectingVideoPoleId = null;
let treeAnalysisState = {
    primaryFile: null,
    primarySrc: '',
    result: null,
    processed: false,
    reportId: '',
    selectedTreeId: '',
};

function updateUploadPromptCopy(mode = 'image') {
    const title = document.getElementById('uploadPromptTitle');
    const subtitle = document.getElementById('uploadPromptSubtitle');
    if (!title || !subtitle) return;

    if (mode === 'tree') {
        title.textContent = 'Select one tree image for species and health analysis';
        subtitle.textContent = 'UPLOAD FULL TREE FIRST. LEAF PHOTO IS OPTIONAL';
    } else {
        title.textContent = 'Select inspection image(s)';
        subtitle.textContent = 'DRAG AND DROP OR CLICK';
    }
}

function renderAnalysisPlaceholder(icon, title, subtitle) {
    hideCropReview();
    const resultBox = document.getElementById('resultBox');
    resultBox.innerHTML = `
        <div class="text-center py-16 px-6 bg-black/30 rounded-2xl border border-dashed border-gray-800">
            <i class="fa-solid ${icon} text-4xl text-gray-700 mb-4 block"></i>
            <p class="text-sm font-semibold text-gray-300">${title}</p>
            <p class="text-[11px] text-gray-500 mt-2 uppercase tracking-wider">${subtitle}</p>
        </div>
    `;
}

function switchInputMode(mode) {
    if (currentTab === mode) {
        if (mode === 'video' && inspectingVideoPoleId !== null) {
            exitPoleInspection();
        }
        return;
    }

    if (currentTab === 'image' && activeBatchIndex !== -1) {
        batchImages[activeBatchIndex].detections = [...detections];
        batchImages[activeBatchIndex].master = masterResult;
        batchImages[activeBatchIndex].dims = { ...imageDimensions };
    }

    currentTab = mode;

    const tabImg = document.getElementById('tabImage');
    const tabVid = document.getElementById('tabVideo');
    const tabTree = document.getElementById('tabTree');
    const dropImg = document.getElementById('dropZone');
    const dropVid = document.getElementById('videoDropZone');
    const imgContainer = document.getElementById('imageContainer');
    const vidContainer = document.getElementById('videoContainer');
    const btnRun = document.getElementById('btnRun');
    const btnText = btnRun.querySelector('.btn-text');
    const submitSection = document.getElementById('submitSection');
    const masterCard = document.getElementById('masterIdentityCard');
    const treeControls = document.getElementById('treeSecondaryControls');

    [tabImg, tabVid, tabTree].forEach((tab) => {
        if (!tab) return;
        tab.classList.remove('text-blue-500', 'border-blue-500');
        tab.classList.add('text-gray-500', 'border-transparent');
    });

    if (mode === 'image') {
        updateUploadPromptCopy('image');
        tabImg.classList.add('text-blue-500', 'border-blue-500');
        tabImg.classList.remove('text-gray-500', 'border-transparent');

        dropImg.classList.remove('hidden');
        dropVid.classList.add('hidden');
        vidContainer.classList.add('hidden');
        treeControls?.classList.add('hidden');

        if (batchImages.length > 0) {
            document.getElementById('batchStripWrapper').classList.remove('hidden');
            imgContainer.classList.remove('hidden');
            document.getElementById('uploadPrompt').classList.add('hidden');
            const prevIdx = activeBatchIndex !== -1 ? activeBatchIndex : 0;
            activeBatchIndex = -1;
            selectBatchImage(prevIdx);
        } else {
            document.getElementById('batchStripWrapper').classList.add('hidden');
            imgContainer.classList.add('hidden');
            document.getElementById('uploadPrompt').classList.remove('hidden');
            const dz = document.getElementById('dropZone');
            if (dz) {
                dz.classList.remove('py-4', 'border-transparent');
                dz.classList.add('p-10', 'border-dashed', 'border-white/5', 'hover:border-blue-500/30', 'cursor-pointer');
            }
            renderAnalysisPlaceholder('fa-magnifying-glass', 'Results will appear here', 'RUN IMAGE INFERENCE TO POPULATE THE ANALYSIS PANEL');
            hideCropReview();
            masterCard.classList.add('hidden');
            if (submitSection) submitSection.classList.add('hidden');
        }
        btnText.textContent = 'Run Interface';
    } else if (mode === 'video') {
        updateUploadPromptCopy('image');
        tabVid.classList.add('text-blue-500', 'border-blue-500');
        tabVid.classList.remove('text-gray-500', 'border-transparent');

        dropImg.classList.add('hidden');
        document.getElementById('batchStripWrapper').classList.add('hidden');
        treeControls?.classList.add('hidden');

        if (uploadedVideoFile) {
            dropVid.classList.remove('hidden');
            vidContainer.classList.remove('hidden');
            document.getElementById('videoUploadPrompt').classList.add('hidden');
            const vdz = document.getElementById('videoDropZone');
            if (vdz) {
                vdz.classList.add('py-4', 'border-transparent');
                vdz.classList.remove('p-10', 'border-dashed', 'border-white/5', 'hover:border-blue-500/30', 'cursor-pointer');
            }
        } else {
            dropVid.classList.remove('hidden');
            vidContainer.classList.add('hidden');
            document.getElementById('videoUploadPrompt').classList.remove('hidden');
            const vdz = document.getElementById('videoDropZone');
            if (vdz) {
                vdz.classList.remove('py-4', 'border-transparent');
                vdz.classList.add('p-10', 'border-dashed', 'border-white/5', 'hover:border-blue-500/30', 'cursor-pointer');
            }
        }

        if (inspectingVideoPoleId !== null) {
            imgContainer.classList.remove('hidden');
            vidContainer.classList.add('hidden');
            dropVid.classList.add('hidden');
            const poleId = inspectingVideoPoleId;
            inspectingVideoPoleId = null;
            inspectPoleFrame(poleId);
        } else {
            imgContainer.classList.add('hidden');
            hideCropReview();
            renderVideoResults();
        }
        btnText.textContent = 'Run Video Survey';
    } else {
        updateUploadPromptCopy('tree');
        tabTree.classList.add('text-blue-500', 'border-blue-500');
        tabTree.classList.remove('text-gray-500', 'border-transparent');

        dropImg.classList.remove('hidden');
        dropVid.classList.add('hidden');
        vidContainer.classList.add('hidden');
        document.getElementById('batchStripWrapper').classList.add('hidden');

        const dz = document.getElementById('dropZone');
        if (dz) {
            dz.classList.remove('hidden');
            dz.classList.remove('p-10', 'border-dashed', 'border-white/5', 'hover:border-blue-500/30', 'cursor-pointer');
            dz.classList.add('py-4', 'border-transparent');
        }

        treeControls?.classList.remove('hidden');
        syncTreeControls();
        if (treeAnalysisState.primarySrc) {
            document.getElementById('uploadPrompt').classList.add('hidden');
            imgContainer.classList.remove('hidden');
            document.getElementById('preview').src = treeAnalysisState.primarySrc;
            currentPreviewBaseSrc = treeAnalysisState.primarySrc;
            detections = [];
            renderBoxes();
            renderTreeResults();
        } else {
            imgContainer.classList.add('hidden');
            document.getElementById('uploadPrompt').classList.remove('hidden');
            hideCropReview();
            renderAnalysisPlaceholder('fa-tree', 'Pole vegetation analysis ready', 'UPLOAD ONE POLE CONTEXT IMAGE TO MAP TREE RISKS');
        }
        masterCard.classList.add('hidden');
        if (submitSection) submitSection.classList.add('hidden');
        btnText.textContent = 'Analyze Vegetation';
    }

    inspectingVideoPoleId = null;
}

function runAnalysis() {
    if (currentTab === 'image') {
        processImage();
    } else if (currentTab === 'video') {
        processVideo();
    } else {
        processTree();
    }
}

function formatTreeLabel(value) {
    const text = `${value || 'Unknown'}`.replace(/_/g, ' ').trim();
    if (!text) return 'Unknown';
    return text.replace(/\w/g, letter => letter.toUpperCase());
}

function escapeHtml(value) {
    return `${value ?? ''}`
        .replace(/&/g, '&amp;')
        .replace(/</g, '&lt;')
        .replace(/>/g, '&gt;')
        .replace(/"/g, '&quot;')
        .replace(/'/g, '&#39;');
}

function formatPlantConfidence(value) {
    return `${Math.round(Number(value || 0) * 100)}%`;
}

function formatEstimatedMeters(value) {
    if (value === null || value === undefined || Number.isNaN(Number(value))) {
        return 'Not available';
    }
    return `~${Number(value).toFixed(2)} m`;
}

function getTreeRiskClasses(level) {
    switch (`${level || ''}`.toLowerCase()) {
        case 'critical':
            return { panel: 'border-rose-500/30 bg-rose-500/10', badge: 'bg-rose-500/15 text-rose-300 border border-rose-500/30' };
        case 'high':
            return { panel: 'border-orange-500/30 bg-orange-500/10', badge: 'bg-orange-500/15 text-orange-300 border border-orange-500/30' };
        case 'medium':
            return { panel: 'border-amber-500/30 bg-amber-500/10', badge: 'bg-amber-500/15 text-amber-300 border border-amber-500/30' };
        case 'low':
            return { panel: 'border-cyan-500/25 bg-cyan-500/10', badge: 'bg-cyan-500/15 text-cyan-300 border border-cyan-500/30' };
        default:
            return { panel: 'border-emerald-500/25 bg-emerald-500/10', badge: 'bg-emerald-500/15 text-emerald-300 border border-emerald-500/30' };
    }
}

function getTreeCardSpecies(tree) {
    const plantnet = tree?.plantnet_result || {};
    return plantnet.common_name || plantnet.scientific_name || plantnet.best_match || 'Species pending';
}

function getTreeContextMeta() {
    return {
        poleId: document.getElementById('treePoleId')?.value.trim() || '',
        latitude: document.getElementById('treeLatitude')?.value.trim() || '',
        longitude: document.getElementById('treeLongitude')?.value.trim() || '',
    };
}

async function selectVegetationTree(treeId) {
    if (!treeAnalysisState.reportId || !treeId) return;
    if (treeAnalysisState.selectedTreeId === treeId && treeAnalysisState.result) return;

    const previousTreeId = treeAnalysisState.selectedTreeId;
    treeAnalysisState.selectedTreeId = treeId;
    renderTreeResults();
    renderBoxes();

    try {
        const response = await fetch('/pole-vegetation/select-tree', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json', 'ngrok-skip-browser-warning': '69420' },
            body: JSON.stringify({ report_id: treeAnalysisState.reportId, user_selected_tree_id: treeId }),
        });
        const payload = await response.json().catch(() => ({}));
        if (!response.ok) throw new Error(payload.error || `Server returned ${response.status}`);
        treeAnalysisState.result = payload;
        treeAnalysisState.selectedTreeId = payload.user_selected_tree_id || treeId;
        treeAnalysisState.reportId = payload.report_id || treeAnalysisState.reportId;
        renderTreeResults();
        renderBoxes();
        showToast(`Selected ${formatTreeLabel(treeId)} for species confirmation`, 'success');
    } catch (err) {
        treeAnalysisState.selectedTreeId = previousTreeId;
        renderTreeResults();
        renderBoxes();
        showToast(err.message || 'Tree selection failed', 'danger');
    }
}

function renderTreeResults() {
    const resultBox = document.getElementById('resultBox');
    const masterCard = document.getElementById('masterIdentityCard');
    const submitSection = document.getElementById('submitSection');
    masterCard.classList.add('hidden');
    submitSection.classList.add('hidden');

    if (!treeAnalysisState.result) {
        renderAnalysisPlaceholder('fa-tree', 'Pole vegetation analysis ready', 'RUN ANALYSIS TO LABEL TREE REGIONS, RISKS AND SPECIES');
        return;
    }

    const data = treeAnalysisState.result;
    const trees = Array.isArray(data.trees) ? data.trees : [];
    const safety = Array.isArray(data.safety_instructions) ? data.safety_instructions : [];
    const selectedTreeId = data.user_selected_tree_id || treeAnalysisState.selectedTreeId;
    treeAnalysisState.selectedTreeId = selectedTreeId || '';

    const summaryCards = [
        { label: 'Overall Risk', value: formatTreeLabel(data.overall_risk || 'unknown') },
        { label: 'Vegetation Regions', value: `${trees.length}` },
    ].map(item => `
        <div class="rounded-xl border border-white/8 bg-white/5 p-3">
            <p class="text-[9px] font-bold uppercase tracking-[0.25em] text-slate-500">${escapeHtml(item.label)}</p>
            <p class="mt-2 text-sm font-semibold text-white">${escapeHtml(item.value)}</p>
        </div>
    `).join('');

    const treeCards = trees.map((tree, index) => {
        const tones = getTreeRiskClasses(tree.risk_level);
        const selected = selectedTreeId === tree.tree_id;
        const metrics = tree.distance_metrics || {};
        return `
            <div class="rounded-2xl border ${tones.panel} p-4 transition-all ${selected ? 'ring-1 ring-blue-400/40 shadow-lg shadow-blue-950/20' : ''}">
                <div class="flex items-start justify-between gap-3">
                    <div>
                        <p class="text-[10px] font-bold uppercase tracking-[0.3em] text-slate-400">Tree ${index + 1}</p>
                        <h3 class="mt-2 text-base font-semibold text-white">${escapeHtml(formatTreeLabel(tree.position || tree.tree_id))}</h3>
                    </div>
                    <span class="px-2.5 py-1 rounded-full text-[9px] font-bold uppercase tracking-widest ${tones.badge}">${escapeHtml(formatTreeLabel(tree.risk_level || 'no_risk'))}</span>
                </div>
                <div class="mt-4 space-y-2 text-sm text-slate-300">
                    <div class="flex justify-between gap-4"><span class="text-slate-500">Estimated pole gap</span><span class="text-right text-white">${escapeHtml(formatEstimatedMeters(metrics.estimated_pole_distance_m))}</span></div>
                    <div class="flex justify-between gap-4"><span class="text-slate-500">Distance confidence</span><span class="text-right text-white">${escapeHtml(formatTreeLabel(metrics.distance_confidence || 'low'))}</span></div>
                    <div class="flex justify-between gap-4"><span class="text-slate-500">Species</span><span class="text-right text-white">${escapeHtml(getTreeCardSpecies(tree))}</span></div>
                    <div class="flex justify-between gap-4"><span class="text-slate-500">Species confidence</span><span class="text-right text-white">${escapeHtml(formatPlantConfidence(tree?.plantnet_result?.confidence))}</span></div>
                    <div class="flex justify-between gap-4"><span class="text-slate-500">Action</span><span class="text-right text-white max-w-[60%]">${escapeHtml(formatTreeLabel(tree.recommended_action || 'monitor'))}</span></div>
                </div>
                <div class="mt-4 rounded-xl border border-white/10 bg-black/20 p-3 space-y-2">
                    <div>
                        <p class="text-[9px] font-bold uppercase tracking-[0.25em] text-slate-500 mb-2">Risk Reason</p>
                        <p class="text-sm leading-6 text-slate-200">${escapeHtml(tree.risk_reason || 'No risk reason available.')}</p>
                    </div>
                    <div>
                        <p class="text-[9px] font-bold uppercase tracking-[0.25em] text-slate-500 mb-2">Distance Note</p>
                        <p class="text-sm leading-6 text-slate-300">${escapeHtml(metrics.distance_note || 'Estimated from the visible pole and wire geometry only.')}</p>
                    </div>
                </div>
                <div class="mt-4 flex items-center justify-between gap-3">
                    <div class="text-[10px] uppercase tracking-widest ${tree.species_detection_required ? 'text-amber-300' : 'text-emerald-300'} font-bold">
                        ${tree.species_detection_required ? 'Species checked on risky crop' : 'Species optional for this region'}
                    </div>
                    <button type="button" onclick="selectVegetationTree('${escapeHtml(tree.tree_id)}')" ${selected ? 'disabled' : ''}
                        class="px-3 py-2 rounded-xl text-[10px] font-bold uppercase tracking-widest transition-all ${selected ? 'bg-blue-600/20 text-blue-300 border border-blue-500/30 cursor-default' : 'bg-white/5 hover:bg-blue-600/20 text-slate-300 hover:text-blue-200 border border-white/10 hover:border-blue-500/30'}">
                        ${selected ? 'Selected Tree' : 'Select This Tree'}
                    </button>
                </div>
            </div>
        `;
    }).join('') || `<div class="rounded-2xl border border-dashed border-white/10 bg-black/20 p-5 text-center text-sm text-slate-500">No vegetation regions were returned for this image.</div>`;

    const treeToTreeCards = (data.tree_to_tree_distances || []).map((entry) => `
        <div class="rounded-xl border border-white/10 bg-white/5 p-3 text-sm text-slate-300">
            <div class="flex justify-between gap-4"><span class="text-slate-500">${escapeHtml(entry.from_tree)} → ${escapeHtml(entry.to_tree)}</span><span class="text-white">${escapeHtml(formatEstimatedMeters(entry.estimated_distance_m))}</span></div>
            <p class="mt-2 text-xs text-slate-500">${escapeHtml(entry.distance_note || 'Estimated using assumed 9 m pole height.')}</p>
        </div>
    `).join('');

    resultBox.innerHTML = `
        <div class="space-y-4 pb-4">
            <div class="rounded-2xl border border-emerald-500/20 bg-gradient-to-br from-emerald-500/15 via-cyan-500/10 to-transparent p-5 shadow-xl shadow-emerald-950/20">
                <p class="text-[10px] font-bold uppercase tracking-[0.3em] text-emerald-300/80">Pole Vegetation Summary</p>
                <h3 class="mt-2 text-xl font-semibold text-white leading-tight">${escapeHtml(data.summary || 'Pole vegetation analysis complete')}</h3>
                <p class="mt-3 text-sm text-slate-300">Distances are estimated from a single image using an assumed 9 m pole height. They are not survey-grade measurements.</p>
            </div>
            <div class="grid grid-cols-2 gap-3">${summaryCards}</div>
            ${data.ask_extra_photo ? `<div class="rounded-2xl border border-amber-500/20 bg-amber-500/10 p-4"><p class="text-[10px] font-bold uppercase tracking-[0.25em] text-amber-300">One Extra Photo Needed</p><p class="mt-2 text-sm text-amber-100/90">${escapeHtml(data.extra_photo_instruction || 'Capture one safer supporting photo.')}</p></div>` : ''}
            <div class="space-y-3">${treeCards}</div>
            ${treeToTreeCards ? `<div class="rounded-2xl border border-white/10 bg-white/5 p-4"><p class="text-[10px] font-bold uppercase tracking-[0.25em] text-cyan-300 mb-3">Estimated Tree-To-Tree Gaps</p><div class="space-y-2">${treeToTreeCards}</div></div>` : ''}
            ${safety.length ? `<div class="rounded-2xl border border-blue-500/20 bg-blue-500/10 p-4"><p class="text-[10px] font-bold uppercase tracking-[0.25em] text-blue-300 mb-2">Safety Instructions</p><ul class="space-y-2 text-sm text-slate-200">${safety.map(item => `<li>${escapeHtml(item)}</li>`).join('')}</ul></div>` : ''}
        </div>
    `;
}

async function processTree() {
    if (!treeAnalysisState.primaryFile) {
        showToast('Please upload a pole context image first', 'warning');
        return;
    }

    const btn = document.getElementById('btnRun');
    const btnText = btn.querySelector('.btn-text');
    const loader = btn.querySelector('.loader');
    const meta = getTreeContextMeta();

    btn.disabled = true;
    loader.classList.remove('hidden');
    btnText.textContent = 'Analyzing Vegetation...';

    try {
        let imageToUpload = treeAnalysisState.primaryFile;
        if (treeAnalysisState.primaryFile.size > 1024 * 1024) {
            imageToUpload = await resizeImage(treeAnalysisState.primaryFile, 1600, 1600);
        }

        const formData = new FormData();
        formData.append('image', imageToUpload, 'pole_vegetation.jpg');
        if (meta.poleId) formData.append('pole_id', meta.poleId);
        if (meta.latitude) formData.append('lat', meta.latitude);
        if (meta.longitude) formData.append('lon', meta.longitude);

        const response = await fetch('/pole-vegetation/analyze', {
            method: 'POST',
            headers: { 'ngrok-skip-browser-warning': '69420' },
            body: formData,
        });

        const payload = await response.json().catch(() => ({}));
        if (!response.ok) throw new Error(payload.error || `Server returned ${response.status}`);

        treeAnalysisState.result = payload;
        treeAnalysisState.processed = true;
        treeAnalysisState.reportId = payload.report_id || '';
        treeAnalysisState.selectedTreeId = payload.user_selected_tree_id || '';
        syncTreeControls();
        renderTreeResults();
        renderBoxes();
        showToast('Pole vegetation analysis complete', 'success');
    } catch (err) {
        showToast(err.message || 'Pole vegetation analysis failed', 'danger');
    } finally {
        btn.disabled = false;
        loader.classList.add('hidden');
        btnText.textContent = 'Analyze Vegetation';
    }
}

document.addEventListener('DOMContentLoaded', () => {
    const removePrimaryBtn = document.getElementById('btnRemoveTreePrimary');
    removePrimaryBtn?.addEventListener('click', removeTreePrimary);
    syncTreeControls();
});

window.selectVegetationTree = selectVegetationTree;

// Attach Video Dropzone events
document.addEventListener('DOMContentLoaded', () => {
    const videoUploadInput = document.getElementById('videoUpload');
    const videoDropZone = document.getElementById('videoDropZone');

    if (videoDropZone && videoUploadInput) {
        videoDropZone.addEventListener('click', (e) => {
            if (videoPoleResults.length > 0) return;
            videoUploadInput.click();
        });
        videoDropZone.addEventListener('dragover', (e) => {
            e.preventDefault();
            videoDropZone.classList.add('border-blue-500', 'bg-blue-500/5');
        });
        videoDropZone.addEventListener('dragleave', () => {
            videoDropZone.classList.remove('border-blue-500', 'bg-blue-500/5');
        });
        videoDropZone.addEventListener('drop', (e) => {
            e.preventDefault();
            videoDropZone.classList.remove('border-blue-500', 'bg-blue-500/5');
            if (e.dataTransfer.files.length) {
                videoUploadInput.files = e.dataTransfer.files;
                handleVideoUpload({ target: videoUploadInput });
            }
        });
        videoUploadInput.addEventListener('change', handleVideoUpload);
    }
});

function handleVideoUpload(e) {
    const file = e.target.files[0];
    if (!file) return;

    uploadedVideoFile = file;

    // Reset UI
    document.getElementById('videoUploadPrompt').classList.add('hidden');
    document.getElementById('videoContainer').classList.remove('hidden');
    document.getElementById('videoPreview').src = URL.createObjectURL(file);

    const dz = document.getElementById('videoDropZone');
    dz.classList.add('py-4', 'border-transparent');
    dz.classList.remove('p-10', 'border-dashed', 'border-white/5', 'hover:border-blue-500/30', 'cursor-pointer');

    showToast("Video loaded successfully", "success");
}

async function processVideo() {
    if (!uploadedVideoFile) {
        showToast("Please upload a video first", "warning");
        return;
    }

    const btn = document.getElementById('btnRun');
    const btnText = btn.querySelector('.btn-text');
    const loader = btn.querySelector('.loader');

    btn.disabled = true;
    loader.classList.remove('hidden');
    btnText.textContent = "Uploading Video...";

    const formData = new FormData();
    formData.append("video", uploadedVideoFile);
    formData.append("target_fps", 3); // Extract at 3 FPS for higher temporal density

    try {
        const response = await fetch("/api/predict_video", {
            method: "POST",
            headers: { "ngrok-skip-browser-warning": "69420" },
            body: formData
        });

        if (!response.ok) {
            const errData = await response.json().catch(() => ({}));
            throw new Error(errData.error || `Server returned ${response.status}`);
        }

        const initialData = await response.json();
        const taskId = initialData.task_id;
        btnText.textContent = "Queued in Background...";

        // Start polling function
        const pollTask = () => {
            return new Promise((resolve, reject) => {
                const interval = setInterval(async () => {
                    try {
                        const statusRes = await fetch(`/api/task_status/${taskId}`, {
                            headers: { "ngrok-skip-browser-warning": "69420" }
                        });
                        if (!statusRes.ok) {
                            clearInterval(interval);
                            reject(new Error(`Failed to check task status`));
                            return;
                        }
                        const statusData = await statusRes.json();
                        if (statusData.status === 'processing') {
                            btnText.textContent = `Processing Survey Video (${statusData.progress}%)...`;
                        } else if (statusData.status === 'queued') {
                            btnText.textContent = "In Queue (Waiting)...";
                        } else if (statusData.status === 'done') {
                            clearInterval(interval);
                            resolve(statusData.results);
                        } else if (statusData.status === 'failed') {
                            clearInterval(interval);
                            reject(new Error(statusData.error || "Video processing task failed."));
                        }
                    } catch (pollErr) {
                        clearInterval(interval);
                        reject(pollErr);
                    }
                }, 1500);
            });
        };

        const results = await pollTask();
        videoPoleResults = results;

        renderVideoResults();
        showToast(`Processed video: ${results.length} poles identified`, "success");

    } catch (err) {
        console.error("Video processing failed:", err);
        showToast(err.message || "Video processing failed", "danger");
    } finally {
        btn.disabled = false;
        btnText.textContent = "Run Video Survey";
        loader.classList.add('hidden');
    }
}

function renderVideoResults() {
    const container = document.getElementById("resultBox");
    const masterCard = document.getElementById("masterIdentityCard");
    masterCard.classList.add('hidden'); // Hide single image master card
    container.innerHTML = "";

    if (videoPoleResults.length === 0) {
        container.innerHTML = `
            <div class="text-center py-20 bg-black/30 rounded-2xl border border-dashed border-gray-800">
                <i class="fa-solid fa-triangle-exclamation text-4xl text-gray-700 mb-4 block"></i>
                <p class="text-gray-600 text-sm">No poles were confidently tracked in this video.</p>
            </div>
        `;
        const submitSection = document.getElementById('submitSection');
        if (submitSection) submitSection.classList.add('hidden');
        return;
    }

    // Sort poles by ID
    const sortedPoles = [...videoPoleResults].sort((a, b) => a.pole_id - b.pole_id);

    sortedPoles.forEach(pole => {
        const card = document.createElement("div");
        card.id = `video-pole-card-${pole.pole_id}`;
        card.className = pole.saved
            ? "p-5 mb-4 bg-green-500/5 border border-green-500/30 rounded-2xl space-y-4 transition-all shadow-md animate-fade-in"
            : "p-5 mb-4 bg-gray-900/40 border border-white/5 rounded-2xl space-y-4 hover:border-blue-500/20 transition-all shadow-md animate-fade-in";

        // Format raw/base64 image
        const imgSource = 'data:image/jpeg;base64,' + pole.annotated_image;

        card.innerHTML = `
            <div class="flex flex-col md:flex-row gap-4">
                <!-- Best Frame Image (Interactive Magnifier Zoom click) -->
                <div onclick="inspectPoleFrame(${pole.pole_id})" class="w-full md:w-1/3 relative rounded-xl overflow-hidden bg-black border border-white/10 shrink-0 aspect-video md:aspect-auto md:h-28 cursor-pointer hover:ring-2 hover:ring-blue-500 transition-all group" title="Click to inspect frame details">
                    <img src="${imgSource}" class="w-full h-full object-cover" />
                    <div class="absolute inset-0 bg-black/40 opacity-0 group-hover:opacity-100 transition-opacity flex items-center justify-center">
                        <i class="fa-solid fa-magnifying-glass-plus text-white text-lg"></i>
                    </div>
                </div>
                
                <!-- Pole Survey Info -->
                <div class="flex-1 space-y-2">
                    <div class="flex items-center justify-between">
                        <span class="px-2.5 py-0.5 bg-blue-500/10 text-blue-400 rounded-full text-[9px] font-bold uppercase tracking-wider">Pole ID #${pole.pole_id}</span>
                        <span class="text-[9px] text-gray-500 font-mono">Timestamp: ${pole.timestamp_sec}s</span>
                    </div>
                    
                    <h3 class="text-xs font-bold text-white uppercase">${pole.pole_type.replace(/_/g, ' ')}</h3>
                    
                    <div class="grid grid-cols-2 gap-x-4 gap-y-1.5 pt-1 border-t border-white/5 text-[9px] text-gray-400 font-medium uppercase tracking-wider">
                        <div><i class="fa-solid fa-compass-drafting text-blue-400 mr-1.5"></i>Channel: <span class="text-white font-bold">${pole.crossarm_shape || 'none'}</span></div>
                        <div><i class="fa-solid fa-bolt text-emerald-400 mr-1.5"></i>Insulators: <span class="text-white font-bold">${pole.insulator_count}</span></div>
                        <div><i class="fa-solid fa-gauge text-purple-400 mr-1.5"></i>Voltage: <span class="text-white font-bold">${pole.master.voltage || 'unknown'}</span></div>
                        <div><i class="fa-solid fa-chart-simple text-amber-400 mr-1.5"></i>Confidence: <span class="text-white font-bold">${Math.round(pole.confidence * 100)}%</span></div>
                    </div>
                </div>
            </div>
            
            <div class="flex gap-2 pt-2 border-t border-white/5">
                ${pole.saved
                    ? `<button id="btn-submit-pole-${pole.pole_id}" disabled
                           class="px-4 py-2 bg-green-600/10 text-green-400 border border-green-500/20 rounded-xl text-[9px] font-bold uppercase tracking-wider flex items-center justify-center gap-1.5 flex-1 font-['Outfit']">
                           <i class="fa-solid fa-circle-check"></i> Pole Survey Saved
                       </button>`
                    : `<button id="btn-submit-pole-${pole.pole_id}" onclick="confirmPoleSurvey(${pole.pole_id})"
                           class="px-4 py-2 bg-blue-600 hover:bg-blue-500 text-white text-[9px] font-bold uppercase tracking-wider rounded-xl transition-all flex items-center justify-center gap-1.5 flex-1 shadow-lg shadow-blue-900/10 border border-blue-500/20 font-['Outfit']">
                           <i class="fa-solid fa-cloud-arrow-up"></i> Confirm & Save
                       </button>`
                }
                <button onclick="deletePoleResult(${pole.pole_id})"
                    class="px-3 py-2 bg-rose-600/10 hover:bg-rose-600/20 text-rose-400 text-[9px] font-bold uppercase tracking-wider rounded-xl transition-all flex items-center justify-center gap-1.5 border border-rose-500/20 shadow-lg shadow-rose-950/20 font-['Outfit']">
                    <i class="fa-solid fa-trash-can"></i> Delete
                </button>
            </div>
        `;
        container.appendChild(card);
    });

    // Configure submitSection for Video Survey mode
    const submitSection = document.getElementById('submitSection');
    if (submitSection) {
        submitSection.classList.remove('hidden');
        const btnContainer = submitSection.querySelector('.grid');
        if (btnContainer) {
            btnContainer.className = "grid grid-cols-1 gap-3";
            const unsavedCount = videoPoleResults.filter(p => !p.saved).length;
            btnContainer.innerHTML = `
                <button id="finalSubmitBtn" onclick="submitWholeVideo()" 
                    class="bg-blue-600 hover:bg-blue-500 text-white py-4 rounded-xl transition-all uppercase font-bold text-[11px] tracking-widest flex items-center justify-center gap-2 shadow-xl shadow-blue-900/20 ${unsavedCount === 0 ? 'opacity-50 cursor-not-allowed' : ''}"
                    ${unsavedCount === 0 ? 'disabled' : ''}>
                    <i class="fa-solid fa-cloud-arrow-up"></i> SUBMIT ALL POLES
                </button>
            `;
        }
    }
}

async function confirmPoleSurvey(poleId) {
    const pole = videoPoleResults.find(p => p.pole_id === poleId);
    if (!pole) return;

    // Fall back to finalSubmitBtn if button specifically for this pole is not in the DOM (e.g. in inspection view)
    const btn = document.getElementById(`btn-submit-pole-${poleId}`) || document.getElementById('finalSubmitBtn');
    if (!btn) return;

    btn.disabled = true;
    const originalHTML = btn.innerHTML;
    btn.innerHTML = `<div class="loader w-3 h-3 border-2 border-white border-t-transparent rounded-full animate-spin"></div> Saving...`;

    try {
        const payload = {
            images: [{
                image_b64: pole.annotated_image,
                uploaded_image_id: pole.uploaded_image_id || null,
                detections: pole.detections.map(d => ({
                    ...d,
                    label: d.label.toUpperCase(),
                    confirmed: true
                })),
                pole_angle: pole.master ? pole.master.pole_lean_angle : 0.0
            }],
            master: pole.master,
            survey_data: pole.survey_questionnaire
        };

        const response = await fetch('/api/save_asset', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(payload)
        });

        const data = await response.json();
        if (data.status === 'success') {
            showToast("Pole submitted successfully!", "success");
            pole.saved = true;

            // Remove the successfully saved pole from results
            videoPoleResults = videoPoleResults.filter(p => p.pole_id !== poleId);

            if (inspectingVideoPoleId === poleId) {
                exitPoleInspection(true); // Exit inspection view silently without double toast
            } else {
                renderVideoResults();
            }
        } else {
            throw new Error(data.message);
        }
    } catch (err) {
        console.error(`Error saving Pole #${poleId}:`, err);
        showToast(`Failed to save Pole #${poleId}: ${err.message}`, "danger");
        btn.disabled = false;
        btn.innerHTML = originalHTML;
    }
}

async function submitWholeVideo() {
    const unsavedPoles = videoPoleResults.filter(p => !p.saved);
    if (unsavedPoles.length === 0) {
        showToast("All poles in this video have already been saved.", "info");
        return;
    }

    const btn = document.getElementById('finalSubmitBtn');
    btn.disabled = true;
    const originalContent = btn.innerHTML;
    btn.innerHTML = `<div class="loader w-4 h-4 border-2 border-white border-t-transparent rounded-full animate-spin"></div> SUBMITTING...`;

    let successCount = 0;
    let failCount = 0;

    for (const pole of unsavedPoles) {
        try {
            const payload = {
                images: [{
                    image_b64: pole.annotated_image,
                    uploaded_image_id: pole.uploaded_image_id || null,
                    detections: pole.detections.map(d => ({
                        ...d,
                        label: d.label.toUpperCase(),
                        confirmed: true
                    })),
                    pole_angle: pole.master ? pole.master.pole_lean_angle : 0.0
                }],
                master: pole.master,
                survey_data: pole.survey_questionnaire
            };

            const response = await fetch('/api/save_asset', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(payload)
            });

            const data = await response.json();
            if (data.status === 'success') {
                pole.saved = true;
                successCount++;
            } else {
                failCount++;
            }
        } catch (err) {
            console.error(`Error saving pole #${pole.pole_id}:`, err);
            failCount++;
        }
    }

    btn.disabled = false;
    btn.innerHTML = originalContent;

    // Filter out the successfully saved poles from results
    videoPoleResults = videoPoleResults.filter(p => !p.saved);
    renderVideoResults();

    if (failCount === 0) {
        showToast(`Successfully saved all ${successCount} poles to admin database!`, "success");
        // Auto-reset video side after a delay
        setTimeout(() => {
            resetVideoSession(true);
            window.scrollTo({ top: 0, behavior: 'smooth' });
        }, 2000);
    } else {
        showToast(`Saved ${successCount} poles, but ${failCount} failed.`, "warning");
    }
}

function inspectPoleFrame(poleId) {
    const pole = videoPoleResults.find(p => p.pole_id === poleId);
    if (!pole) return;

    inspectingVideoPoleId = poleId;

    // Adjust left pane dropzones/containers
    document.getElementById('dropZone').classList.remove('hidden');
    document.getElementById('uploadPrompt').classList.add('hidden');
    document.getElementById('dropZone').classList.add('py-4', 'border-transparent');
    document.getElementById('dropZone').classList.remove('p-10', 'border-dashed', 'border-white/5', 'hover:border-blue-500/30', 'cursor-pointer');

    document.getElementById('videoContainer').classList.add('hidden');
    document.getElementById('videoDropZone').classList.add('hidden');
    document.getElementById('imageContainer').classList.remove('hidden');

    // Set active inspection data globals
    detections = pole.detections.map(d => ({
        ...d,
        label: d.label.toUpperCase(),
        confirmed: d.confirmed || false
    }));
    masterResult = pole.master;
    imageDimensions = { width: pole.width, height: pole.height };

    // Set preview image source to best frame
    const preview = document.getElementById('preview');
    currentPreviewBaseSrc = 'data:image/jpeg;base64,' + pole.annotated_image;
    preview.src = currentPreviewBaseSrc;

    // Setup manual confirm submit action
    const submitSection = document.getElementById('submitSection');
    submitSection.classList.remove('hidden');

    // In video mode, show both Submit and Delete buttons in a grid
    const btnContainer = submitSection.querySelector('.grid');
    if (btnContainer) {
        btnContainer.className = "grid grid-cols-2 gap-3";
        btnContainer.innerHTML = `
            <button id="finalSubmitBtn" class="bg-blue-600 hover:bg-blue-500 text-white py-4 rounded-xl transition-all uppercase font-bold text-[11px] tracking-widest flex items-center justify-center gap-2 shadow-xl shadow-blue-900/20">
                <i class="fa-solid fa-cloud-arrow-up"></i> SUBMIT POLE #${poleId}
            </button>
            <button id="deleteInspectBtn" class="bg-rose-600/10 hover:bg-rose-600/20 text-rose-400 py-4 rounded-xl transition-all uppercase font-bold text-[11px] tracking-widest flex items-center justify-center gap-2 border border-rose-500/20 shadow-xl shadow-rose-950/20">
                <i class="fa-solid fa-trash-can"></i> DELETE
            </button>
        `;

        document.getElementById('finalSubmitBtn').onclick = () => confirmPoleSurvey(poleId);
        document.getElementById('deleteInspectBtn').onclick = () => {
            if (confirm(`Are you sure you want to delete Pole ID #${poleId} from the survey results?`)) {
                videoPoleResults = videoPoleResults.filter(p => p.pole_id !== poleId);
                exitPoleInspection();
                showToast(`Pole ID #${poleId} removed from results`, "warning");
            }
        };
    }

    // Refresh right pane display and SVG boxes
    renderResults();

    if (preview.complete) {
        renderBoxes();
    } else {
        preview.onload = () => {
            renderBoxes();
            preview.onload = null;
        };
    }

    showToast(`Inspecting Pole ID #${poleId} details`, "info");
}

function deletePoleResult(poleId) {
    if (confirm(`Are you sure you want to delete Pole ID #${poleId} from the survey results?`)) {
        videoPoleResults = videoPoleResults.filter(p => p.pole_id !== poleId);
        renderVideoResults();
        showToast(`Pole ID #${poleId} removed from results`, "warning");
    }
}

function exitPoleInspection(silent = false) {
    inspectingVideoPoleId = null;

    // Hide image components
    hideCropReview();
    document.getElementById('imageContainer').classList.add('hidden');
    document.getElementById('dropZone').classList.add('hidden');

    // Restore video survey preview
    document.getElementById('videoDropZone').classList.remove('hidden');
    document.getElementById('videoContainer').classList.remove('hidden');

    // Hide submit footer
    document.getElementById('submitSection').classList.add('hidden');

    // Restore submit section buttons to single-column layout for single image mode
    const submitSection = document.getElementById('submitSection');
    const btnContainer = submitSection.querySelector('.grid');
    if (btnContainer) {
        btnContainer.className = "grid grid-cols-1 gap-3";
        btnContainer.innerHTML = `
            <button id="finalSubmitBtn" onclick="submitAsset()" disabled
                class="bg-blue-600 hover:bg-blue-500 text-white py-4 rounded-xl opacity-50 cursor-not-allowed transition-all uppercase font-bold text-[11px] tracking-widest flex items-center justify-center gap-2 shadow-xl shadow-blue-900/20">
                <i class="fa-solid fa-cloud-arrow-up"></i> SUBMIT RESULTS
            </button>
        `;
    }

    // Re-render the full pole survey cards
    renderVideoResults();
    if (!silent) {
        showToast("Returned to survey overview", "info");
    }
}
