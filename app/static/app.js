/* ════════════════════════════════════════════════════
   DeepGuard — Frontend Logic
   ════════════════════════════════════════════════════ */

const API = '';  // same-origin; adjust if running separately

// ── Utility ──────────────────────────────────────────
const $ = id => document.getElementById(id);

function showPage(name) {
  document.querySelectorAll('.page').forEach(p => p.classList.remove('active'));
  $('page-' + name).classList.add('active');
  document.querySelectorAll('.nav-btn').forEach(b => b.classList.remove('active'));
  const map = { home: 'navHomeBtn', image: 'navImageBtn', video: 'navVideoBtn' };
  if (map[name]) $(map[name]).classList.add('active');
  window.scrollTo({ top: 0, behavior: 'smooth' });
}

// ── Navbar scroll effect ──────────────────────────────
window.addEventListener('scroll', () => {
  document.getElementById('navbar').classList.toggle('scrolled', window.scrollY > 10);
});

// ── Navigation wiring ────────────────────────────────
$('navHomeBtn').addEventListener('click', () => showPage('home'));
$('navImageBtn').addEventListener('click', () => showPage('image'));
$('navVideoBtn').addEventListener('click', () => showPage('video'));
$('nav-brand-btn').addEventListener('click', () => showPage('home'));
$('heroImageBtn').addEventListener('click', () => showPage('image'));
$('heroVideoBtn').addEventListener('click', () => showPage('video'));

// ── Status check ─────────────────────────────────────
async function checkStatus() {
  try {
    const r = await fetch(`${API}/api/status`);
    const d = await r.json();
    const dot = $('statusDot'), txt = $('statusText');
    if (d.status === 'ok') {
      dot.className = 'status-dot ok';
      txt.textContent = 'Models ready';
    } else {
      dot.className = 'status-dot warn';
      txt.textContent = 'Degraded';
    }
  } catch {
    $('statusDot').className = 'status-dot err';
    $('statusText').textContent = 'Offline';
  }
}
checkStatus();

// ══════════════════════════════════════════════════════
//  IMAGE DETECTION
// ══════════════════════════════════════════════════════
let imageFile = null;

function setupDropZone(zoneId, inputId, browseId, onFile) {
  const zone   = $(zoneId);
  const input  = $(inputId);
  const browse = $(browseId);

  browse.addEventListener('click', e => { e.stopPropagation(); input.click(); });
  zone.addEventListener('click', () => input.click());

  input.addEventListener('change', () => {
    if (input.files[0]) onFile(input.files[0]);
  });

  zone.addEventListener('dragover', e => {
    e.preventDefault(); zone.classList.add('drag-over');
  });
  zone.addEventListener('dragleave', () => zone.classList.remove('drag-over'));
  zone.addEventListener('drop', e => {
    e.preventDefault(); zone.classList.remove('drag-over');
    if (e.dataTransfer.files[0]) onFile(e.dataTransfer.files[0]);
  });
}

function onImageSelected(file) {
  imageFile = file;
  const url = URL.createObjectURL(file);
  $('imagePreview').src = url;
  $('imagePreviewRow').style.display = 'flex';
  $('imageAnalyzeRow').style.display = 'flex';
  $('imageResults').style.display = 'none';
  $('imageError').style.display = 'none';
  $('imageExplainRow').style.display = 'none';
}

setupDropZone('imageDropZone', 'imageFileInput', 'imageBrowseBtn', onImageSelected);

$('imageAnalyzeBtn').addEventListener('click', async () => {
  if (!imageFile) return;

  const btn = $('imageAnalyzeBtn');
  $('imageAnalyzeBtnText').textContent = 'Analyzing…';
  $('imageAnalyzeSpinner').style.display = 'inline-block';
  btn.disabled = true;
  $('imageResults').style.display = 'none';
  $('imageError').style.display = 'none';

  const fd = new FormData();
  fd.append('file', imageFile);

  try {
    const res = await fetch(`${API}/api/detect/image`, { method: 'POST', body: fd });

    if (!res.ok) {
      const err = await res.json();
      throw new Error(err.detail || 'Server error');
    }

    const d = await res.json();
    renderImageResult(d);

  } catch (e) {
    $('imageError').textContent = '⚠️ ' + e.message;
    $('imageError').style.display = 'block';
  } finally {
    $('imageAnalyzeBtnText').textContent = '🔍 Analyze Image';
    $('imageAnalyzeSpinner').style.display = 'none';
    btn.disabled = false;
  }
});

function renderImageResult(d) {
  const isFake = d.prediction === 'DEEPFAKE';

  // Verdict
  $('imageVerdict').textContent = isFake ? '🚨 DEEPFAKE DETECTED' : '✅ AUTHENTIC — REAL';
  $('imageVerdict').className = 'result-verdict ' + (isFake ? 'verdict-fake' : 'verdict-real');
  $('imageConfidence').textContent = `Confidence: ${d.confidence}%`;

  // Bars — animate after paint
  requestAnimationFrame(() => {
    $('imageRealBar').style.width = d.real_probability + '%';
    $('imageFakeBar').style.width = d.fake_probability + '%';
  });
  $('imageRealVal').textContent = d.real_probability + '%';
  $('imageFakeVal').textContent = d.fake_probability + '%';

  // Warning banner (no face detected → fallback)
  let warningEl = $('imageWarningBanner');
  if (!warningEl) {
    warningEl = document.createElement('div');
    warningEl.id = 'imageWarningBanner';
    warningEl.className = 'warning-box';
    $('imageResultCard').after(warningEl);
  }
  if (d.warning) {
    warningEl.textContent = '⚠️ ' + d.warning;
    warningEl.style.display = 'block';
  } else {
    warningEl.style.display = 'none';
  }

  $('imageResults').style.display = 'block';

  // Explainability — label changes if no face was detected
  if (d.face_crop_b64) {
    $('imageFaceCrop').src = 'data:image/jpeg;base64,' + d.face_crop_b64;
    // Update the label to reflect what was cropped
    const faceBoxLabel = $('imageExplainRow').querySelector('.preview-box:first-child .box-label');
    if (faceBoxLabel) {
      faceBoxLabel.textContent = d.face_detected ? 'Detected Face' : 'Analyzed Region (no face found)';
    }
    if (d.heatmap_b64) {
      $('imageHeatmap').src = 'data:image/jpeg;base64,' + d.heatmap_b64;
    }
    $('imageExplainRow').style.display = 'flex';
  }
}

// ══════════════════════════════════════════════════════
//  VIDEO DETECTION
// ══════════════════════════════════════════════════════
let videoFile = null;

function onVideoSelected(file) {
  videoFile = file;
  const url = URL.createObjectURL(file);
  $('videoPreview').src = url;
  $('videoPreviewRow').style.display = 'flex';
  $('videoAnalyzeRow').style.display = 'flex';
  $('videoResults').style.display = 'none';
  $('videoError').style.display = 'none';
}

setupDropZone('videoDropZone', 'videoFileInput', 'videoBrowseBtn', onVideoSelected);

$('videoAnalyzeBtn').addEventListener('click', async () => {
  if (!videoFile) return;

  const btn = $('videoAnalyzeBtn');
  $('videoAnalyzeBtnText').textContent = 'Analyzing…';
  $('videoAnalyzeSpinner').style.display = 'inline-block';
  btn.disabled = true;
  $('videoResults').style.display = 'none';
  $('videoError').style.display = 'none';

  const fd = new FormData();
  fd.append('file', videoFile);

  try {
    const res = await fetch(`${API}/api/detect/video`, { method: 'POST', body: fd });

    if (!res.ok) {
      const err = await res.json();
      throw new Error(err.detail || 'Server error');
    }

    const d = await res.json();
    renderVideoResult(d);

  } catch (e) {
    $('videoError').textContent = '⚠️ ' + e.message;
    $('videoError').style.display = 'block';
  } finally {
    $('videoAnalyzeBtnText').textContent = '🔍 Analyze Video';
    $('videoAnalyzeSpinner').style.display = 'none';
    btn.disabled = false;
  }
});

function renderVideoResult(d) {
  const isFake = d.final_prediction === 'DEEPFAKE';

  $('videoVerdict').textContent = isFake ? '🚨 DEEPFAKE DETECTED' : '✅ AUTHENTIC — REAL';
  $('videoVerdict').className = 'result-verdict ' + (isFake ? 'verdict-fake' : 'verdict-real');
  $('videoConfidence').textContent = `Confidence: ${d.confidence}%`;

  // Stats chips
  $('videoStats').innerHTML = `
    <div class="stat-chip">
      <span class="stat-num">${d.frames_analyzed}</span>
      <span class="stat-label">Frames Analyzed</span>
    </div>
    <div class="stat-chip">
      <span class="stat-num" style="color:${d.suspicious_frames > 0 ? 'var(--fake)' : 'var(--real)'}">${d.suspicious_frames}</span>
      <span class="stat-label">Suspicious Frames</span>
    </div>
    <div class="stat-chip">
      <span class="stat-num">${d.mean_fake_probability}%</span>
      <span class="stat-label">Avg Fake Probability</span>
    </div>
  `;

  // Prob bars
  requestAnimationFrame(() => {
    $('videoRealBar').style.width = d.mean_real_probability + '%';
    $('videoFakeBar').style.width = d.mean_fake_probability + '%';
  });
  $('videoRealVal').textContent = d.mean_real_probability + '%';
  $('videoFakeVal').textContent = d.mean_fake_probability + '%';

  // Frame table
  const tbody = $('frameTableBody');
  tbody.innerHTML = '';
  d.frame_results.forEach(f => {
    const isFk = f.prediction === 'DEEPFAKE';
    const tr = document.createElement('tr');
    tr.innerHTML = `
      <td>${f.frame_num}</td>
      <td>${f.timestamp}s</td>
      <td><span class="${isFk ? 'tag-fake' : 'tag-real'}">${f.prediction}</span></td>
      <td>${f.real_probability}%</td>
      <td>${f.fake_probability}%</td>
    `;
    tbody.appendChild(tr);
  });

  // Suspicious frame thumbnails
  const suspicious = d.frame_results.filter(f => f.prediction === 'DEEPFAKE' && f.face_crop_b64);
  const grid = $('suspiciousGrid');
  grid.innerHTML = '';
  if (suspicious.length > 0) {
    suspicious.slice(0, 8).forEach(f => {
      const card = document.createElement('div');
      card.className = 'suspicious-card';
      card.innerHTML = `
        <img src="data:image/jpeg;base64,${f.face_crop_b64}" alt="Frame ${f.frame_num}" />
        <div class="sc-label">Frame ${f.frame_num} · ${f.fake_probability}% fake</div>
      `;
      grid.appendChild(card);
    });
    $('suspiciousFramesSection').style.display = 'block';
  } else {
    $('suspiciousFramesSection').style.display = 'none';
  }

  $('videoResults').style.display = 'block';
}

// ── Init ─────────────────────────────────────────────
showPage('home');
