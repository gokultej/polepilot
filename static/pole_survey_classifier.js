/**
 * static/pole_survey_classifier.js
 * Standalone module to auto-classify pole detection outputs, autofill survey questions,
 * dynamically render dark-themed survey forms, and store & display submitted survey point summaries.
 */

window.PoleSurveyClassifier = (function () {
    let isFeatureEnabled = true;
    let lastRenderedState = null;

    // Initialize module & sync config from backend
    async function init() {
        try {
            const resp = await fetch('/api/survey/config');
            const data = await resp.json();
            if (data && typeof data.auto_survey_redirect_enabled === 'boolean') {
                isFeatureEnabled = data.auto_survey_redirect_enabled;
            }
        } catch (err) {
            console.warn('[PoleSurveyClassifier] Could not fetch config:', err);
        }
    }

    // Process detection results and determine survey redirection
    async function processDetectionsAndRedirect(detectionResult) {
        if (!isFeatureEnabled) {
            console.log('[PoleSurveyClassifier] Feature disabled. Skipping auto-redirection.');
            return null;
        }

        // Only process survey questionnaire on Image Inference tab
        if (typeof currentTab !== 'undefined' && currentTab !== 'image') {
            console.log('[PoleSurveyClassifier] Non-image tab active. Skipping survey questionnaire.');
            const targetContainer = document.getElementById('survey-form-container');
            if (targetContainer) targetContainer.style.display = 'none';
            return null;
        }

        try {
            const response = await fetch('/api/survey/classify', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(detectionResult)
            });

            const resData = await response.json();
            const classification = resData.classification;

            if (!classification || !classification.pole_detected) {
                showPoleNotDetectedNotice(classification ? classification.message : 'Pole Not Detected');
                return { redirect: false, status: 'POLE_NOT_DETECTED' };
            }

            console.log(`[PoleSurveyClassifier] Classified Pole: ${classification.title} (${classification.schema_id})`);
            console.log(`[PoleSurveyClassifier] Reason: ${classification.reason}`);

            triggerSurveyRedirection(classification, resData.schema);
            return { redirect: true, classification: classification, schema: resData.schema };

        } catch (err) {
            console.error('[PoleSurveyClassifier] Classification request failed:', err);
            return null;
        }
    }

    function showPoleNotDetectedNotice(message) {
        const existingNotice = document.querySelector('.pole-notice-banner');
        if (existingNotice) existingNotice.remove();

        const masterClassEl = document.getElementById("masterClass");
        if (masterClassEl) {
            masterClassEl.textContent = "POLE NOT DETECTED";
        }

        const alertBox = document.createElement('div');
        alertBox.className = 'pole-notice-banner warning';
        alertBox.innerHTML = `
            <div class="p-3 my-2 bg-rose-500/10 border border-rose-500/20 text-rose-400 rounded-xl text-xs font-mono flex items-center gap-2">
                <i class="fa-solid fa-triangle-exclamation text-rose-400"></i>
                <span><strong>Notice:</strong> ${message}</span>
            </div>`;
        const container = document.getElementById('survey-form-container') || document.body;
        container.prepend(alertBox);
    }

    function triggerSurveyRedirection(classification, schema) {
        const event = new CustomEvent('poleSurveyRedirect', {
            detail: { classification, schema }
        });
        window.dispatchEvent(event);

        // Update Master Identity Card title on dashboard
        const masterClassEl = document.getElementById("masterClass");
        if (masterClassEl && classification && classification.title) {
            masterClassEl.textContent = classification.title.replace("Survey - ", "");
        }

        if (schema && schema.questions) {
            lastRenderedState = { classification, schema };
            renderSurveyForm(classification, schema);
        }
    }

    // Render full 39-question dark-themed survey form matching GIS-YOLO UI theme
    function renderSurveyForm(classification, schema) {
        const targetContainer = document.getElementById('survey-form-container');
        if (!targetContainer) return;

        const autofill = classification.autofill_data || { answers: {}, is_33kv_section_visible: false };
        const answers = autofill.answers || {};
        const is33kvVisible = autofill.is_33kv_section_visible;
        const schemaId = classification.schema_id || '11kv_ht_pole';
        const schemaTitle = classification.title || 'Survey - 11KV HT Pole';

        let html = `
            <div class="survey-card bg-gray-900/80 backdrop-blur-xl border border-white/10 rounded-2xl p-5 shadow-2xl space-y-4 text-left">
                <div class="flex items-center justify-between pb-3 border-b border-white/10">
                    <div class="flex items-center gap-2">
                        <i class="fa-solid fa-clipboard-list text-blue-400 text-sm"></i>
                        <h3 class="text-xs font-bold text-blue-400 uppercase tracking-widest font-mono m-0">
                            ${schemaTitle}
                        </h3>
                    </div>
                    <span class="bg-blue-500/10 text-blue-300 border border-blue-500/20 px-2.5 py-1 rounded-full text-[9px] font-mono font-bold tracking-wider">
                        ✨ Auto-Filled (Image: ${autofill.image_name || 'Active Image'} | Sheds: ${autofill.shed_count || 0})
                    </span>
                </div>
                <p class="text-[10px] text-gray-400 font-mono m-0 leading-relaxed">${classification.reason}</p>

                <form id="pole-survey-form" class="space-y-3 mt-4" onsubmit="PoleSurveyClassifier.handleSurveyFormSubmit(event, '${schemaId}', '${schemaTitle}')">
                    <div class="space-y-3 max-h-[550px] overflow-y-auto pr-1 custom-scrollbar">`;

        schema.questions.forEach(q => {
            const is33kvQuestion = (q.id >= 21 && q.id <= 32);
            const isHidden = is33kvQuestion && !is33kvVisible;
            const value = answers[q.field_name] !== undefined ? answers[q.field_name] : '';

            let controlHtml = '';
            if (q.field_type === 'dropdown') {
                const options = (q.dropdown_options || []).map(opt =>
                    `<option value="${opt}" class="bg-slate-900 text-white" ${String(opt) === String(value) ? 'selected' : ''}>${opt}</option>`
                ).join('');
                controlHtml = `
                    <select name="${q.field_name}" class="w-full bg-slate-900/90 border border-white/10 rounded-xl px-3 py-2 text-xs text-white outline-none focus:border-blue-500/50 transition-all font-mono">
                        ${options}
                    </select>`;
            } else if (q.field_type === 'counter') {
                controlHtml = `
                    <div class="flex items-center gap-2">
                        <button type="button" class="w-8 h-8 rounded-lg bg-white/5 hover:bg-white/10 text-white border border-white/10 flex items-center justify-center font-bold text-sm transition-all" onclick="this.nextElementSibling.stepDown();">-</button>
                        <input type="number" name="${q.field_name}" value="${value || 0}" min="0" class="w-16 bg-slate-900/90 border border-white/10 rounded-lg py-1 text-center text-xs text-white font-mono font-bold outline-none focus:border-blue-500/50"/>
                        <button type="button" class="w-8 h-8 rounded-lg bg-white/5 hover:bg-white/10 text-white border border-white/10 flex items-center justify-center font-bold text-sm transition-all" onclick="this.previousElementSibling.stepUp();">+</button>
                    </div>`;
            } else if (q.field_type === 'textarea') {
                controlHtml = `<textarea name="${q.field_name}" maxlength="${q.max_length || 100}" placeholder="Enter remarks..." class="w-full bg-slate-900/90 border border-white/10 rounded-xl px-3 py-2 text-xs text-white placeholder-gray-600 outline-none focus:border-blue-500/50 transition-all font-mono h-20">${value}</textarea>`;
            } else {
                controlHtml = `<input type="text" name="${q.field_name}" value="${value}" maxlength="${q.max_length || ''}" placeholder="Value..." class="w-full bg-slate-900/90 border border-white/10 rounded-xl px-3 py-2 text-xs text-white placeholder-gray-600 outline-none focus:border-blue-500/50 transition-all font-mono"/>`;
            }

            html += `
                <div class="p-3 bg-black/40 border border-white/5 rounded-xl transition-all hover:border-white/10 question-row" id="q-row-${q.id}" style="display: ${isHidden ? 'none' : 'block'};">
                    <label class="text-[10px] font-bold text-gray-300 uppercase tracking-wider block mb-1.5">
                        <span class="text-blue-400 font-mono">${q.id}.</span> ${q.field_label}
                        ${is33kvQuestion ? '<span class="bg-amber-500/10 text-amber-400 border border-amber-500/20 px-1.5 py-0.5 rounded text-[8px] font-mono ml-2">33KV Item</span>' : ''}
                    </label>
                    ${controlHtml}
                </div>`;
        });

        html += `
                    </div>
                    <div class="pt-3 border-t border-white/10">
                        <button type="submit" id="btn-submit-survey" class="w-full py-3.5 bg-gradient-to-r from-blue-600 to-indigo-600 hover:from-blue-500 hover:to-indigo-500 text-white font-bold text-xs uppercase tracking-widest rounded-xl shadow-lg shadow-blue-900/30 transition-all flex items-center justify-center gap-2 cursor-pointer border border-blue-400/20">
                            <i class="fa-solid fa-cloud-arrow-up"></i> SUBMIT SURVEY POINT
                        </button>
                    </div>
                </form>
            </div>`;

        targetContainer.innerHTML = html;
    }

    // Handles form submission to store survey data in database
    async function handleSurveyFormSubmit(event, schemaId, schemaTitle) {
        event.preventDefault();

        const submitBtn = document.getElementById('btn-submit-survey');
        if (submitBtn) {
            submitBtn.disabled = true;
            submitBtn.innerHTML = `<i class="fa-solid fa-spinner animate-spin"></i> SAVING SURVEY POINT...`;
        }

        const form = document.getElementById('pole-survey-form');
        if (!form) return;

        const formData = new FormData(form);
        const surveyData = {};
        for (let [key, val] of formData.entries()) {
            surveyData[key] = val;
        }

        const locNum = surveyData['pole_location_number'] || 'P-001';

        const activeItem = (typeof activeBatchIndex !== 'undefined' && activeBatchIndex !== -1 && typeof batchImages !== 'undefined') ? batchImages[activeBatchIndex] : null;
        let imageB64 = '';
        if (activeItem && activeItem.file) {
            try {
                imageB64 = await new Promise((resolve) => {
                    const reader = new FileReader();
                    reader.onload = () => resolve(reader.result || '');
                    reader.onerror = () => resolve('');
                    reader.readAsDataURL(activeItem.file);
                });
            } catch (e) {
                console.warn('[PoleSurveyClassifier] Could not read activeItem.file as data URL', e);
            }
        }
        if (!imageB64) {
            const candidateSrc = (activeItem ? activeItem.src : (document.getElementById('preview')?.src || '')) || '';
            if (candidateSrc.startsWith('data:image')) {
                imageB64 = candidateSrc;
            }
        }
        const uploadedImageId = activeItem ? (activeItem.uploaded_image_id || null) : null;
        const currentDetections = typeof detections !== 'undefined' ? detections : [];

        try {
            const response = await fetch('/api/survey/submit', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    schema_id: schemaId,
                    title: schemaTitle,
                    pole_location_number: locNum,
                    survey_data: surveyData,
                    image_b64: imageB64,
                    uploaded_image_id: uploadedImageId,
                    detections: currentDetections
                })
            });

            const res = await response.json();
            if (res && (res.status === 'SUCCESS' || res.status === 'PARTIAL_SUCCESS')) {
                if (typeof showToast === 'function') {
                    showToast(`✓ Survey Point ${locNum} Stored Successfully!`, 'success');
                }
                if (typeof resetImageSession === 'function') {
                    resetImageSession(true);
                }
                const targetContainer = document.getElementById('survey-form-container');
                if (targetContainer) targetContainer.innerHTML = '';
                window.scrollTo({ top: 0, behavior: 'smooth' });
            } else {
                alert(`Error storing survey point: ${res.message || 'Unknown error'}`);
            }
        } catch (err) {
            console.error('[PoleSurveyClassifier] Failed to submit survey point:', err);
            alert('Failed to store survey point. Please check your server connection.');
        } finally {
            if (submitBtn) {
                submitBtn.disabled = false;
                submitBtn.innerHTML = `<i class="fa-solid fa-cloud-arrow-up"></i> SUBMIT SURVEY POINT`;
            }
        }
    }

    // Renders clean Submitted Pole Summary Card after successful submission
    function renderSubmittedPoleSummaryCard(locNum, schemaTitle, surveyData) {
        const targetContainer = document.getElementById('survey-form-container');
        if (!targetContainer) return;

        let summaryRowsHtml = `
            <div class="grid grid-cols-2 gap-2 text-[10px] font-mono">
                <div class="p-2 bg-white/5 rounded-lg border border-white/5"><span class="text-gray-400 block">Status</span><strong class="text-emerald-400 font-bold">${surveyData.pole_erection_status || 'Pole'}</strong></div>
                <div class="p-2 bg-white/5 rounded-lg border border-white/5"><span class="text-gray-400 block">Location No.</span><strong class="text-blue-400 font-bold">${locNum}</strong></div>
                <div class="p-2 bg-white/5 rounded-lg border border-white/5"><span class="text-gray-400 block">Is Pin Point</span><strong class="text-white font-bold">${surveyData.is_pin_point || 'No'}</strong></div>
                <div class="p-2 bg-white/5 rounded-lg border border-white/5"><span class="text-gray-400 block">Channels Erected</span><strong class="text-white font-bold">${surveyData.no_of_channels_erected || '0'}</strong></div>
            </div>
            <div class="p-2.5 bg-white/5 rounded-lg border border-white/5 text-[10px] font-mono">
                <span class="text-gray-400 block mb-0.5">Pole Type</span>
                <strong class="text-amber-300 font-bold leading-tight block">${surveyData.pole_type || 'Er. of 9.1M 280Kg WL w/o base plate'}</strong>
            </div>`;

        let html = `
            <div class="survey-submitted-card bg-gray-900/90 backdrop-blur-xl border border-emerald-500/30 rounded-2xl p-5 shadow-2xl space-y-4 text-left animate-in fade-in zoom-in duration-300">
                <div class="flex items-center justify-between pb-3 border-b border-emerald-500/20">
                    <div class="flex items-center gap-2">
                        <i class="fa-solid fa-circle-check text-emerald-400 text-base"></i>
                        <h3 class="text-xs font-bold text-emerald-400 uppercase tracking-widest font-mono m-0">
                            SURVEY POINT STORED (${locNum})
                        </h3>
                    </div>
                    <span class="bg-emerald-500/20 text-emerald-300 border border-emerald-500/30 px-2.5 py-1 rounded-full text-[9px] font-mono font-bold tracking-wider uppercase">
                        ✓ REGISTERED IN DB
                    </span>
                </div>

                <p class="text-[10px] text-gray-400 font-mono m-0 leading-relaxed">
                    Successfully saved survey record for <strong class="text-white">${schemaTitle}</strong> into local and database registry.
                </p>

                ${summaryRowsHtml}

                <div class="flex flex-col sm:flex-row items-center gap-3 pt-2">
                    <button type="button" onclick="PoleSurveyClassifier.reopenForm()" class="w-full sm:w-1/2 py-3 bg-white/10 hover:bg-white/20 text-white font-bold text-xs uppercase tracking-widest rounded-xl transition-all font-mono border border-white/10 flex items-center justify-center gap-1.5 cursor-pointer">
                        <i class="fa-solid fa-pen-to-square"></i> Edit Answers
                    </button>
                    <button type="button" onclick="PoleSurveyClassifier.prepareNextPole('${locNum}')" class="w-full sm:w-1/2 py-3 bg-gradient-to-r from-blue-600 to-indigo-600 hover:from-blue-500 hover:to-indigo-500 text-white font-bold text-xs uppercase tracking-widest rounded-xl transition-all font-mono border border-blue-400/20 flex items-center justify-center gap-1.5 shadow-lg shadow-blue-900/30 cursor-pointer">
                        <i class="fa-solid fa-plus"></i> Next Pole Survey
                    </button>
                </div>
            </div>`;

        targetContainer.innerHTML = html;
    }

    function reopenForm() {
        if (lastRenderedState) {
            renderSurveyForm(lastRenderedState.classification, lastRenderedState.schema);
        }
    }

    function prepareNextPole(prevLocNum) {
        if (lastRenderedState) {
            const form = document.getElementById('pole-survey-form');
            renderSurveyForm(lastRenderedState.classification, lastRenderedState.schema);

            // Compute next pole location number
            let nextNum = 'P-001';
            const match = (prevLocNum || '').match(/^P-(\d+)$/);
            if (match) {
                const nextVal = parseInt(match[1], 10) + 1;
                nextNum = `P-${String(nextVal).padStart(3, '0')}`;
            }
            const newForm = document.getElementById('pole-survey-form');
            if (newForm) {
                const locInput = newForm.querySelector('[name="pole_location_number"]');
                if (locInput) locInput.value = nextNum;
            }
        }
    }

    // Self-initialize on load
    document.addEventListener('DOMContentLoaded', init);

    return {
        init: init,
        setEnabled: function (flag) { isFeatureEnabled = !!flag; },
        getEnabled: function () { return isFeatureEnabled; },
        processDetectionsAndRedirect: processDetectionsAndRedirect,
        renderSurveyForm: renderSurveyForm,
        handleSurveyFormSubmit: handleSurveyFormSubmit,
        reopenForm: reopenForm,
        prepareNextPole: prepareNextPole
    };
})();
