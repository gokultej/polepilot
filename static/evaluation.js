(() => {
  'use strict';
  const $ = id => document.getElementById(id);
  const state = { bootstrap: null, sample: null, file: null, blobUrl: null, job: null, busy: false, poll: null, selectedDetection: -1, generation: 0, map: null, mapLayer: null, mapTiles: null, mapInitPromise: null, demoLayer: null, demoData: null, ownCount: 0, csrf: document.querySelector('meta[name="csrf-token"]')?.content || '' };
  const nodes = ['sample-list','run-recorded','run-live','run-explanation','global-error','inspection-image','image-coordinate-space','image-placeholder','processing-cover','processing-title','job-state','image-dimensions','detection-overlay','detection-list','provenance','live-status','review-form','review-decision','review-note','save-review','review-feedback','reset-demo','evaluation-upload'];
  const el = Object.fromEntries(nodes.map(id => [id, $(id)]));

  function node(tag, cls, text) { const n = document.createElement(tag); if (cls) n.className = cls; if (text !== undefined) n.textContent = text; return n; }
  function error(message) { el['global-error'].textContent = message || ''; el['global-error'].hidden = !message; }
  function status(message) { el['live-status'].textContent = message; }
  async function api(path, options = {}) {
    const headers = new Headers(options.headers || {});
    if (options.method && options.method !== 'GET') headers.set('X-CSRF-Token', state.csrf);
    let response;
    try { response = await fetch(path, { ...options, headers, credentials: 'same-origin' }); }
    catch { throw new Error('Cannot reach the evaluation backend. Check your connection, then try again.'); }
    let payload;
    try { payload = await response.json(); } catch { throw new Error(`The server returned an unexpected response (${response.status}). Please try again.`); }
    if (!response.ok) throw new Error(payload.error || payload.message || `Request failed (${response.status}).`);
    return payload;
  }
  function jsonPost(data) { return { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(data) }; }
  function safeUrl(value) { try { const u = new URL(value, location.origin); return u.origin === location.origin ? u.href : ''; } catch { return ''; } }
  function prettyKey(value) { return String(value).replace(/_/g, ' ').replace(/\b\w/g, c => c.toUpperCase()); }
  function formatValue(value) {
    if (value === null || value === undefined || value === '') return 'Not available';
    if (typeof value === 'boolean') return value ? 'Yes' : 'No';
    if (typeof value === 'object') return JSON.stringify(value, null, 2);
    return String(value);
  }
  function metadata(target, data) {
    target.replaceChildren();
    if (!data || typeof data !== 'object' || !Object.keys(data).length) { target.append(node('div', 'small-copy', 'No metadata returned for this image.')); return; }
    Object.entries(data).forEach(([key, value]) => { const row = node('div'); row.append(node('dt', '', prettyKey(key)), node('dd', '', formatValue(value))); target.append(row); });
  }
  function updateControls() {
    const ready = Boolean(state.bootstrap && (state.sample || state.file));
    el['run-recorded'].disabled = state.busy || !ready || !!state.file || !state.sample?.recorded;
    el['run-live'].disabled = state.busy || !ready || !state.bootstrap?.live_inference_enabled;
    el['reset-demo'].disabled = state.busy || !state.bootstrap;
    el['evaluation-upload'].disabled = state.busy;
    el['sample-list'].querySelectorAll('button').forEach(b => { b.disabled = state.busy; b.setAttribute('aria-pressed', String(b.dataset.id === state.sample?.id && !state.file)); });
    if (state.busy) el['run-explanation'].textContent = 'Inspection in progress. You can review or export the result when it completes.';
    else if (state.file) el['run-explanation'].textContent = state.bootstrap?.live_inference_enabled ? 'Your image will run through the local models. CPU inference may take a minute.' : 'Live inference is disabled on this instance. Choose a supplied sample with a replay instead.';
    else if (state.sample?.recorded) el['run-explanation'].textContent = 'Replay loads a recorded real-model run immediately. Live inference runs the models again.';
    else if (state.sample) el['run-explanation'].textContent = state.bootstrap?.live_inference_enabled ? 'No recorded run is available for this sample. Use live inference to inspect it.' : 'No replay is available for this sample and live inference is disabled. Choose another sample.';
    else el['run-explanation'].textContent = 'Choose a supplied sample or upload a field image to begin.';
  }
  function showImage(url, alt) {
    el['inspection-image'].src = url;
    el['inspection-image'].alt = alt || 'Selected utility survey field image';
    el['image-coordinate-space'].hidden = false;
    el['image-placeholder'].hidden = true;
  }
  function clearResult() {
    state.job = null; state.selectedDetection = -1;
    el['detection-overlay'].replaceChildren();
    $('evidence-empty').hidden = false; $('evidence-result').hidden = true;
    $('survey-empty').hidden = false; $('survey-result').hidden = true;
    $('detection-count').textContent = '—';
    el['job-state'].textContent = 'READY TO INSPECT'; el['job-state'].className = 'state-badge';
    el['image-dimensions'].textContent = 'ORIGINAL FIELD EVIDENCE';
    el['provenance'].replaceChildren(node('span', 'mono', 'EVIDENCE, NOT A SAFETY CERTIFICATE'), node('p', '', 'Detections support inspection. Confidence is not calibrated accuracy; a qualified reviewer must verify findings.'));
    el['review-note'].value = ''; el['review-decision'].value = 'needs_review'; el['review-feedback'].textContent = '';
    $('review-latitude').value=''; $('review-longitude').value='';
    ['review-note','review-decision','save-review'].forEach(k => { el[k].disabled = true; });
    $('review-latitude').disabled=true; $('review-longitude').disabled=true;
    ['export-json','export-csv'].forEach(k => { $(k).removeAttribute('href'); $(k).setAttribute('aria-disabled', 'true'); $(k).tabIndex = -1; });
    $('step-input').className = 'active'; $('step-inspect').className = ''; $('step-review').className = '';
  }
  function chooseSample(sample) {
    if (state.busy) return;
    if (state.blobUrl) { URL.revokeObjectURL(state.blobUrl); state.blobUrl = null; }
    state.sample = sample; state.file = null; el['evaluation-upload'].value = ''; $('upload-name').hidden = true;
    clearResult(); error(''); showImage(safeUrl(sample.image_url), sample.description || sample.title); updateControls();
    status(`${sample.title} selected. Choose replay or live inference.`);
  }
  function renderSamples(samples) {
    el['sample-list'].replaceChildren();
    $('sample-count').textContent=`${samples.length} samples`;
    $('sample-match-count').textContent=`${samples.length} field images`;
    if (!samples.length) { el['sample-list'].append(node('p', 'small-copy', 'No supplied images are available. Upload a field image to try live inference.')); return; }
    samples.forEach(sample => {
      const button = node('button', 'sample-button'); button.type = 'button'; button.dataset.id = sample.id; button.setAttribute('aria-pressed', 'false');
      const img = node('img'); img.src = safeUrl(sample.image_url); img.alt = ''; img.loading = 'lazy';
      const copy = node('span'); copy.append(node('strong', '', sample.title), node('small', '', sample.recorded ? 'Recorded run available' : 'Live inference sample'));
      button.dataset.search=`${sample.title} ${sample.description || ''}`.toLowerCase();
      button.append(img, copy); button.addEventListener('click', () => chooseSample(sample)); el['sample-list'].append(button);
    });
  }
  $('sample-search').addEventListener('input',()=>{
    const query=$('sample-search').value.trim().toLowerCase();let matches=0;
    el['sample-list'].querySelectorAll('.sample-button').forEach(button=>{const visible=!query||button.dataset.search.includes(query);button.hidden=!visible;if(visible)matches++;});
    $('sample-match-count').textContent=matches?`${matches} matching sample${matches===1?'':'s'}`:'No matching samples. Try a different search.';
  });
  function renderCapabilities(data) {
    const target = $('capability-list'); target.replaceChildren();
    (data.capabilities || []).forEach(item => { const article = node('article', 'capability-item'); const head = node('div'); const tag = node('span', 'tag', prettyKey(item.status || 'Workspace')); if (/demo|available|public/i.test(item.status || '')) tag.classList.add('tag-demo'); head.append(node('h3', '', item.title), tag); article.append(head, node('p', '', item.description)); target.append(article); });
    if (!target.children.length) target.append(node('p','small-copy','See the product overview for the full workflow.'));
    const models = $('model-list'); models.replaceChildren();
    (data.models || []).forEach(item => { const tag = node('span', `model-tag${item.available ? '' : ' unavailable'}`); const dot = node('span', 'status-dot'); dot.setAttribute('aria-hidden','true'); tag.append(dot, document.createTextNode(`${item.name} · ${item.available ? 'present' : 'missing'}`)); models.append(tag); });
    const limits = $('limitations-list'); limits.replaceChildren(); (data.limitations || []).forEach(text => limits.append(node('li', '', text)));
  }
  function switchTab(name, focus = false) {
    document.querySelectorAll('.result-tabs [role=tab]').forEach(button => { const selected = button.dataset.pane === name; button.setAttribute('aria-selected', String(selected)); button.tabIndex = selected ? 0 : -1; if (selected && focus) button.focus(); });
    ['evidence','survey','review'].forEach(k => { $(`pane-${k}`).hidden = k !== name; });
    if (name === 'review' && state.job?.status === 'done') { $('step-inspect').className = 'complete'; $('step-review').className = 'active'; }
  }
  function svgElement(tag, attrs) { const n = document.createElementNS('http://www.w3.org/2000/svg', tag); Object.entries(attrs).forEach(([k,v]) => n.setAttribute(k, String(v))); return n; }
  function drawDetections() {
    const overlay = el['detection-overlay']; overlay.replaceChildren();
    const result = state.job?.result;
    if (!result || !$('show-overlays').checked || !result.width || !result.height) return;
    overlay.setAttribute('viewBox', `0 0 ${result.width} ${result.height}`);
    (result.detections || []).forEach((det, index) => {
      const group = svgElement('g', {}); const highlighted = index === state.selectedDetection;
      let shape, x, y;
      if (Array.isArray(det.polygon) && det.polygon.length > 2 && det.polygon.every(p => Array.isArray(p) && p.length >= 2 && p.slice(0,2).every(Number.isFinite))) {
        const points = det.polygon; shape = svgElement('polygon', { points: points.map(p => `${p[0]},${p[1]}`).join(' '), class: `detection-region${highlighted ? ' is-highlighted' : ''}` }); x = Math.min(...points.map(p=>p[0])); y = Math.min(...points.map(p=>p[1]));
      } else if (Array.isArray(det.bbox) && det.bbox.length === 4 && det.bbox.every(Number.isFinite)) {
        const [x1,y1,x2,y2] = det.bbox; if (x2 <= x1 || y2 <= y1) return;
        shape = svgElement('rect', { x:x1,y:y1,width:x2-x1,height:y2-y1,class:`detection-region${highlighted ? ' is-highlighted' : ''}` }); x=x1; y=y1;
      } else return;
      group.append(shape);
      if (highlighted || state.selectedDetection === -1) { const fontSize = Math.max(11, result.width / 55); const text = svgElement('text', { x:Math.max(4,x+4), y:Math.max(fontSize+5,y-7), class:'overlay-label', style:`font-size:${fontSize}px` }); text.textContent=prettyKey(det.label || 'Object'); group.append(text); }
      overlay.append(group);
    });
  }
  function renderDetections(result) {
    const detections = Array.isArray(result.detections) ? result.detections : [];
    el['detection-list'].replaceChildren(); $('detection-count').textContent = String(detections.length);
    if (!detections.length) { el['detection-list'].append(node('p','no-detections','No objects were detected at the current thresholds. This is a valid model result. Try another view or a clearer image; do not infer asset absence from this alone.')); }
    detections.forEach((det,index) => {
      const button = node('button', 'detection-button'); button.type='button'; button.setAttribute('aria-pressed','false');
      const confidence = Number.isFinite(det.confidence) ? `${(Math.max(0,Math.min(1,det.confidence))*100).toFixed(1)}%` : 'N/A';
      button.append(node('span','detection-name',`${String(index+1).padStart(2,'0')}  ${prettyKey(det.label || 'Object')}`),node('span','detection-confidence',confidence));
      button.addEventListener('click', () => { state.selectedDetection = state.selectedDetection === index ? -1 : index; el['detection-list'].querySelectorAll('button').forEach((b,i)=>b.setAttribute('aria-pressed',String(i===state.selectedDetection))); drawDetections(); });
      el['detection-list'].append(button);
    });
    metadata($('master-data'),result.master); $('evidence-empty').hidden=true; $('evidence-result').hidden=false;
  }
  function renderSurvey(result) {
    const classification=result.classification || {};
    const autofill=classification.autofill_data;
    const summary={
      classification:classification.title || (classification.pole_detected===false?'No pole classification returned':'Classification not supplied'),
      status:classification.status,
      pole_detected:classification.pole_detected,
      reason:classification.reason,
      survey_schema:classification.schema_id
    };
    metadata($('classification-data'),summary);
    const target = $('survey-data'); target.replaceChildren();
    if(classification.feature_summary&&typeof classification.feature_summary==='object'){
      target.append(node('h3','record-subheading','Detected feature summary'));
      const features=node('dl','metadata-list');metadata(features,classification.feature_summary);target.append(features);
    }
    target.append(node('h3','record-subheading','Detection-derived questionnaire'));
    const survey = result.survey_questionnaire;
    if (!survey || typeof survey !== 'object' || !Object.keys(survey).length) target.append(node('p','small-copy','No questionnaire output was returned for this image.'));
    else if (Array.isArray(survey)) survey.forEach((item,index) => { const row=node('div','survey-item'); row.append(node('strong','',item.question || item.label || `Field ${index+1}`),node('p','',formatValue(item.answer ?? item.value ?? item))); target.append(row); });
    else Object.entries(survey).forEach(([key,value])=>{ const row=node('div','survey-item'); row.append(node('strong','',prettyKey(key)),node(typeof value==='object'?'pre':'p','',formatValue(value))); target.append(row); });
    if(autofill?.answers&&typeof autofill.answers==='object'){
      target.append(node('h3','record-subheading','Rule-suggested form values'),node('p','result-help','Unverified suggestions: these fields include rule defaults, such as span distance and pole specifications. They are not measurements from this photograph. Verify or replace every value in the field.'));
      const labels={pole_erection_status:'Pole erection status',pole_location_number:'Suggested pole number (verify uniqueness)',pole_to_pole_distance:'Pole-to-pole distance (unmeasured rule default)',pole_type:'Pole specification (rule suggestion)',no_of_channels_erected:'Channels erected (suggested)',er_of_11kv_polymer_insulators_with_pins:'11 kV polymer pin insulators (suggested count)',remarks:'Remarks'};
      const selected={};Object.entries(labels).forEach(([key,label])=>{if(key in autofill.answers)selected[label]=autofill.answers[key];});
      const values=node('dl','metadata-list');metadata(values,selected);target.append(values);
      const more=node('details','record-details');more.append(node('summary','',`All ${Object.keys(autofill.answers).length} suggested survey fields`));const allValues=node('dl','metadata-list');metadata(allValues,autofill.answers);more.append(allValues);target.append(more);
    }
    const raw=node('details','record-details');raw.append(node('summary','','Raw classification & questionnaire JSON'),node('pre','',JSON.stringify({classification:result.classification,survey_questionnaire:result.survey_questionnaire},null,2)));target.append(raw);
    $('survey-empty').hidden=true; $('survey-result').hidden=false;
  }
  function renderProvenance(result) {
    const p = result.provenance || {}; const isRecorded=p.mode==='recorded';
    el['provenance'].replaceChildren(node('strong','',isRecorded?'Recorded real-model run':'Live model inference'),node('span','mono',p.source || 'FIELD IMAGE ANALYSIS'));
    const meta=node('div','provenance-grid');
    const generated = p.generated_at ? new Date(p.generated_at) : null;
    meta.append(node('span','',`Generated: ${generated && !Number.isNaN(generated.valueOf()) ? generated.toLocaleString() : 'Not supplied'}`),node('span','',`Run time: ${Number.isFinite(p.elapsed_seconds) ? `${p.elapsed_seconds.toFixed(2)} s` : 'Not supplied'}`));
    el['provenance'].append(meta,node('p','',isRecorded?'This result was generated by the project models and saved for reproducible evaluation. Replay does not run new inference.':'This result was generated for your request by the local project models. Verify inferred values against field observations.'));
    if (Array.isArray(p.model_files) && p.model_files.length) el['provenance'].append(node('p','',`Model files: ${p.model_files.join(', ')}`));
  }
  function completeJob(job) {
    state.job=job; state.busy=false; el['processing-cover'].hidden=true;
    if (job.status==='failed') { el['job-state'].textContent='INSPECTION FAILED'; el['job-state'].className='state-badge is-failed'; error(job.error || 'Inference failed. Please try again or choose a recorded sample.'); status('Inspection failed. Choose another image or retry.'); updateControls(); return; }
    if (!job.result) { error('The completed job has no result. Please retry the inspection.'); updateControls(); return; }
    const result=job.result; const resultUrl=safeUrl(result.image_url); if(resultUrl) showImage(resultUrl,'Original image used for this model result');
    el['job-state'].textContent=result.provenance?.mode==='recorded'?'RECORDED RUN / READY FOR REVIEW':'LIVE RUN / READY FOR REVIEW'; el['job-state'].className='state-badge is-done';
    el['image-dimensions'].textContent=result.width && result.height ? `${result.width} × ${result.height} PX`:'FIELD IMAGE';
    renderDetections(result); renderSurvey(result); renderProvenance(result); drawDetections();
    ['review-note','review-decision','save-review'].forEach(k=>el[k].disabled=false);
    $('review-latitude').disabled=false; $('review-longitude').disabled=false;
    if(job.review){el['review-note'].value=job.review.note || ''; el['review-decision'].value=job.review.decision || 'needs_review';$('review-latitude').value=job.review.latitude ?? '';$('review-longitude').value=job.review.longitude ?? '';}
    ['json','csv'].forEach(format=>{const a=$(`export-${format}`);a.href=`/api/evaluation/jobs/${encodeURIComponent(job.id)}/export?format=${format}`;a.removeAttribute('aria-disabled');a.removeAttribute('tabindex');});
    $('step-input').className='complete'; $('step-inspect').className='active'; $('step-review').className='';
    status(`Inspection complete. ${(result.detections || []).length} detections returned. Inspect the evidence, then save a review and export.`); updateControls();
  }
  async function pollJob(id,generation) {
    if(generation!==state.generation)return;
    try {
      const job=await api(`/api/evaluation/jobs/${encodeURIComponent(id)}`);
      if(generation!==state.generation)return;
      if(['done','failed'].includes(job.status)){completeJob(job);return;}
      el['processing-title'].textContent=job.status==='queued'?'Queued for local inference':'Inspecting your field image'; el['job-state'].textContent=job.status==='queued'?'QUEUED':'MODELS RUNNING';
      status(job.status==='queued'?'Your image is queued. Waiting for the inference worker.':'Local models are processing the image. You can leave this page open.');
      state.poll=setTimeout(()=>pollJob(id,generation),1500);
    }catch(exc){state.busy=false;el['processing-cover'].hidden=true;el['job-state'].textContent='CONNECTION INTERRUPTED';error(`${exc.message} Run the inspection again if needed.`);updateControls();status('Unable to retrieve the inspection result.');}
  }
  async function run(mode) {
    if(state.busy)return;
    error('');clearResult();switchTab('evidence');state.busy=true;state.generation++;const generation=state.generation;
    el['processing-cover'].hidden=false;el['processing-title'].textContent=mode==='recorded'?'Loading recorded model evidence':'Preparing local inference';el['job-state'].textContent='PREPARING';updateControls();status(mode==='recorded'?'Loading recorded real-model run…':'Submitting image for live inference…');
    try{
      let options;
      if(state.file){const data=new FormData();data.append('image',state.file);data.append('mode','live');options={method:'POST',body:data};}
      else options=jsonPost({sample_id:state.sample.id,mode});
      const job=await api('/api/evaluation/jobs',options);
      if(generation!==state.generation)return;
      if(!job.id)throw new Error('The server did not return an inspection ID.');
      await pollJob(job.id,generation);
    }catch(exc){state.busy=false;el['processing-cover'].hidden=true;el['job-state'].textContent='INSPECTION NOT STARTED';error(exc.message);updateControls();status('Inspection could not start. Correct the issue and try again.');}
  }
  el['run-recorded'].addEventListener('click',()=>run('recorded'));el['run-live'].addEventListener('click',()=>run('live'));
  el['evaluation-upload'].addEventListener('change',()=>{
    const file=el['evaluation-upload'].files?.[0];if(!file)return;
    const max=(state.bootstrap?.limits?.upload_mb || 10)*1024*1024;
    if(!['image/jpeg','image/png','image/webp'].includes(file.type)){error('Choose a JPG, PNG or WebP image.');el['evaluation-upload'].value='';return;}
    if(file.size>max){error(`The image is too large. Upload a file no larger than ${state.bootstrap?.limits?.upload_mb || 10} MB.`);el['evaluation-upload'].value='';return;}
    if(state.blobUrl)URL.revokeObjectURL(state.blobUrl);state.blobUrl=URL.createObjectURL(file);state.file=file;state.sample=null;clearResult();error('');showImage(state.blobUrl,`Uploaded field image: ${file.name}`);$('upload-name').textContent=file.name;$('upload-name').hidden=false;updateControls();status('Your image is ready. Run live inference to inspect it.');
  });
  $('show-overlays').addEventListener('change',drawDetections);
  document.querySelectorAll('.result-tabs [role=tab]').forEach((button,index,buttons)=>{button.addEventListener('click',()=>switchTab(button.dataset.pane));button.addEventListener('keydown',event=>{let next;if(event.key==='ArrowRight')next=(index+1)%buttons.length;else if(event.key==='ArrowLeft')next=(index+buttons.length-1)%buttons.length;else if(event.key==='Home')next=0;else if(event.key==='End')next=buttons.length-1;else return;event.preventDefault();switchTab(buttons[next].dataset.pane,true);});});
  el['review-form'].addEventListener('submit',async event=>{
    event.preventDefault();if(!state.job?.id || state.job.status!=='done')return;
    const latText=$('review-latitude').value.trim(),lngText=$('review-longitude').value.trim();
    if(Boolean(latText)!==Boolean(lngText)){el['review-feedback'].className='form-feedback is-error';el['review-feedback'].textContent='Enter both latitude and longitude, or leave both empty.';return;}
    const latitude=latText===''?null:Number(latText),longitude=lngText===''?null:Number(lngText);
    if((latitude!==null&&(!Number.isFinite(latitude)||Math.abs(latitude)>90))||(longitude!==null&&(!Number.isFinite(longitude)||Math.abs(longitude)>180))){el['review-feedback'].className='form-feedback is-error';el['review-feedback'].textContent='Latitude must be −90 to 90 and longitude must be −180 to 180.';return;}
    el['save-review'].disabled=true;el['review-feedback'].className='form-feedback';el['review-feedback'].textContent='Saving your review…';
    const reviewedJobId=state.job.id;
    try{const saved=await api(`/api/evaluation/jobs/${encodeURIComponent(reviewedJobId)}/review`,jsonPost({decision:el['review-decision'].value,note:el['review-note'].value,latitude,longitude}));if(state.job?.id===reviewedJobId){state.job.review=saved.review;el['review-feedback'].textContent='Review saved. Your exports now include this decision and note.';$('step-review').className='complete';status('Review saved to your isolated demo session. Export JSON, CSV or GeoJSON to take the evidence with you.');}refreshMap();}
    catch(exc){if(state.job?.id===reviewedJobId){el['review-feedback'].textContent=exc.message;el['review-feedback'].classList.add('is-error');}}finally{el['save-review'].disabled=state.job?.status!=='done';}
  });
  el['reset-demo'].addEventListener('click',async()=>{
    if(state.busy)return;el['reset-demo'].disabled=true;error('');
    try{await api('/api/evaluation/reset',{method:'POST'});state.generation++;if(state.poll)clearTimeout(state.poll);clearResult();switchTab('evidence');$('sample-search').value='';renderSamples(state.bootstrap.samples || []);if(state.bootstrap.samples?.length)chooseSample(state.bootstrap.samples[0]);else{state.sample=null;state.file=null;el['evaluation-upload'].value='';$('upload-name').hidden=true;el['image-coordinate-space'].hidden=true;el['image-placeholder'].hidden=false;}refreshMap();status('Your demo session was reset. Start a new inspection.');}
    catch(exc){error(exc.message);}finally{updateControls();}
  });
  el['inspection-image'].addEventListener('error',()=>{if(el['inspection-image'].getAttribute('src'))error('The selected image could not be loaded. Try another sample or reload the page.');});
  async function refreshMap(){
    if(!state.map)return;
    try{
      const geojson=await api('/api/evaluation/assets.geojson');
      if(state.mapLayer)state.map.removeLayer(state.mapLayer);
      state.mapLayer=window.L.geoJSON(geojson,{pointToLayer:(feature,latlng)=>window.L.circleMarker(latlng,{radius:9,color:'#fff',weight:3,fillColor:'#226a9f',fillOpacity:1}),onEachFeature:(feature,layer)=>{const content=node('div','map-popup');content.append(node('small','popup-layer-label','YOUR INSPECTION'),node('strong','',feature.properties?.label || 'Inspected asset'),node('p','',prettyKey(feature.properties?.decision || 'needs_review')),node('small','','Reviewer-entered coordinates'));layer.bindPopup(content);}}).addTo(state.map);
      state.ownCount=Array.isArray(geojson.features)?geojson.features.length:0;
      updateMapSummary();fitMapLayers();
    }catch(exc){$('map-feedback').textContent=`Could not refresh asset locations: ${exc.message} Coordinate entry and exports remain available.`;}
  }
  function updateMapSummary(){
    const demoShown=Boolean(state.demoLayer&&state.map?.hasLayer(state.demoLayer));
    const count=state.ownCount;
    $('map-feedback').textContent=demoShown?`${state.demoData.features.length} fictional demo poles displayed around Vijayawada–Guntur. ${count} of your located inspections. Demo poles are excluded from your export.`:count?`${count} of your located inspection${count===1?'':'s'}. Coordinates are reviewer-entered.`:'No personal locations yet. Run a sample inspection below, open Review, enter verified GPS coordinates and save. Or turn on Show demo poles to explore the fictional AP sample.';
    $('map-demo-banner').hidden=!demoShown;
  }
  function fitMapLayers(){
    if(!state.map)return;
    const bounds=window.L.latLngBounds([]);
    [state.mapLayer,state.demoLayer].forEach(layer=>{if(layer&&state.map.hasLayer(layer)&&layer.getBounds().isValid())bounds.extend(layer.getBounds());});
    if(bounds.isValid())state.map.fitBounds(bounds,{padding:[35,35],maxZoom:14});else state.map.setView([22.5,79],4);
  }
  async function applyDemoLayer(){
    if(!state.map)return;
    if($('show-demo-poles').checked){
      if(!state.demoData){
        state.demoData=await api('/static/demo_poles.geojson');
        if(!Array.isArray(state.demoData.features))throw new Error('Fictional demo layer is unavailable.');
      }
      if(!state.demoLayer){
        const colors={demo_reviewed:'#789936',needs_review:'#d59b3f',inspection_planned:'#82918c'};
        state.demoLayer=window.L.geoJSON(state.demoData,{pointToLayer:(feature,latlng)=>window.L.circleMarker(latlng,{radius:7,color:'#203c29',weight:2,fillColor:colors[feature.properties?.status]||'#82918c',fillOpacity:1}),onEachFeature:(feature,layer)=>{
          const p=feature.properties||{},content=node('div','map-popup demo-popup');
          content.append(node('small','popup-layer-label','FICTIONAL AP DEMO POLE'),node('strong','',p.id||p.label),node('p','',p.area),node('p','',`Status: ${prettyKey(p.status||'illustrative')}`),node('p','',`Components: ${formatValue(p.components)}`),node('p','',p.review_note),node('small','popup-fictional-warning','Synthetic presentation data. No real survey was performed; equipment and status are illustrative.'));
          layer.bindPopup(content);layer.bindTooltip(p.id||p.label||'Fictional demo pole');
        }});
      }
      state.demoLayer.addTo(state.map);
      $('demo-pole-count').textContent=String(state.demoData.features.length);
      [['demo-reviewed-count','demo_reviewed'],['demo-review-count','needs_review'],['demo-planned-count','inspection_planned']].forEach(([id,status])=>{$(id).textContent=String(state.demoData.features.filter(f=>f.properties?.status===status).length);});
    }else if(state.demoLayer){state.map.removeLayer(state.demoLayer);}
    updateMapSummary();fitMapLayers();
  }
  function loadMapLibrary(){
    if(window.L&&document.getElementById('leaflet-css')?.sheet)return Promise.resolve();
    return Promise.all([
      new Promise((resolve,reject)=>{
        document.getElementById('leaflet-css')?.remove();
        const css=document.createElement('link');css.id='leaflet-css';css.rel='stylesheet';css.href='/static/vendor/leaflet/leaflet.css';
        const timeout=setTimeout(()=>{css.remove();reject(new Error('Map styles loading timed out.'));},12000);
        css.onload=()=>{clearTimeout(timeout);resolve();};css.onerror=()=>{clearTimeout(timeout);css.remove();reject(new Error('Map styles could not be loaded.'));};document.head.append(css);
      }),
      new Promise((resolve,reject)=>{
        if(window.L){resolve();return;}
        const script=document.createElement('script');script.src='/static/vendor/leaflet/leaflet.js';
        const timeout=setTimeout(()=>{script.remove();reject(new Error('Map library loading timed out.'));},12000);
        script.onload=()=>{clearTimeout(timeout);window.L?resolve():reject(new Error('Map library unavailable.'));};script.onerror=()=>{clearTimeout(timeout);script.remove();reject(new Error('Map library could not be loaded.'));};document.head.append(script);
      })
    ]);
  }
  async function initMap(retry=false){
    if(state.mapInitPromise)return state.mapInitPromise;
    const button=$('load-map');button.disabled=true;button.hidden=true;
    state.mapInitPromise=(async()=>{
      try{
        if(!state.map){
          $('map-feedback').textContent='Opening your map and loading OpenStreetMap tiles…';
          await loadMapLibrary();$('map-placeholder').hidden=true;
          const demoSelected=$('show-demo-poles').checked;
          state.map=window.L.map('asset-map',{scrollWheelZoom:false}).setView(demoSelected?[16.43,80.54]:[22.5,79],demoSelected?10:4);
          state.mapTiles=window.L.tileLayer('https://tile.openstreetmap.org/{z}/{x}/{y}.png',{maxZoom:19,attribution:'&copy; <a href="https://www.openstreetmap.org/copyright" target="_blank" rel="noopener noreferrer">OpenStreetMap</a> contributors'}).on('tileerror',()=>{$('map-feedback').textContent='Map tiles are unavailable. You can still save GPS coordinates in Review and export GeoJSON.';button.hidden=false;button.disabled=false;button.textContent='Retry map tiles ↻';});
        }else if(retry){state.mapTiles?.redraw();}
        state.map.invalidateSize();await applyDemoLayer();await refreshMap();
        if(state.mapTiles&&!state.map.hasLayer(state.mapTiles))state.mapTiles.addTo(state.map);
      }catch(exc){
        if(state.map){state.map.remove();state.map=null;state.mapLayer=null;state.mapTiles=null;}
        $('map-placeholder').hidden=false;$('map-feedback').textContent=`${exc.message} You can still enter coordinates in Review and export GeoJSON.`;button.hidden=false;button.disabled=false;button.textContent='Retry map ↻';
      }finally{state.mapInitPromise=null;}
    })();
    return state.mapInitPromise;
  }
  $('load-map').addEventListener('click',()=>initMap(true));
  $('show-demo-poles').addEventListener('change',async()=>{try{if(!state.map)await initMap();else await applyDemoLayer();}catch(exc){$('map-feedback').textContent=`Could not display fictional demo poles: ${exc.message}`;}});
  if('IntersectionObserver' in window){
    const observer=new IntersectionObserver(entries=>{entries.forEach(entry=>{if(entry.isIntersecting){initMap();observer.unobserve(entry.target);}});},{threshold:0.1});
    observer.observe($('asset-map'));
  }else{
    const loadWhenVisible=()=>{const rect=$('asset-map').getBoundingClientRect();if(rect.top<innerHeight&&rect.bottom>0){initMap();window.removeEventListener('scroll',loadWhenVisible);window.removeEventListener('resize',loadWhenVisible);}};
    window.addEventListener('scroll',loadWhenVisible,{passive:true});window.addEventListener('resize',loadWhenVisible);loadWhenVisible();
  }
  window.addEventListener('resize',()=>{if(state.map)state.map.invalidateSize();});
  async function boot(){
    try{const data=await api('/api/evaluation/bootstrap');state.bootstrap=data;state.csrf=data.csrf_token || state.csrf;renderSamples(data.samples || []);renderCapabilities(data);if(data.samples?.length)chooseSample(data.samples[0]);else updateControls();status(data.live_inference_enabled?'Evaluation backend connected. Sample replay and live inference are available.':'Evaluation backend connected. Live inference is disabled; use a supplied recorded run.');}
    catch(exc){error(exc.message);el['sample-list'].replaceChildren(node('p','small-copy','Samples are unavailable. Reload the page to reconnect.'));$('capability-list').replaceChildren(node('p','small-copy','Feature availability could not be loaded.'));status('Evaluation backend is unavailable. Reload the page to retry.');}
  }
  boot();
})();
