/* ── State ──────────────────────────────────────────────────────────────── */
const state = {
  challengeType: null,
  challengeId: null,
  capturedBlob: null,
  stream: null,
};

/* ── DOM refs ───────────────────────────────────────────────────────────── */
const $ = id => document.getElementById(id);

const stepChallenge = $('step-challenge');
const stepCamera    = $('step-camera');
const stepResult    = $('step-result');
const loading       = $('loading');
const loadingText   = $('loading-text');

const btnChallenge  = $('btn-challenge');
const btnCapture    = $('btn-capture');
const btnRetake     = $('btn-retake');
const btnSubmit     = $('btn-submit');
const btnRestart    = $('btn-restart');

const video         = $('video');
const canvas        = $('canvas');
const preview       = $('preview');

/* ── Helpers ────────────────────────────────────────────────────────────── */
function showLoading(text = 'Processing...') {
  loadingText.textContent = text;
  loading.classList.remove('hidden');
}

function hideLoading() {
  loading.classList.add('hidden');
}

function show(el)  { el.classList.remove('hidden'); }
function hide(el)  { el.classList.add('hidden'); }

async function apiFetch(path, options = {}) {
  const res = await fetch(path, options);
  const data = await res.json();
  if (!res.ok) {
    const msg = data?.error?.message || data?.detail || 'API error';
    throw new Error(msg);
  }
  return data;
}

/* ── Step 1: Get challenge ──────────────────────────────────────────────── */
btnChallenge.addEventListener('click', async () => {
  btnChallenge.disabled = true;
  showLoading('Getting your challenge...');

  try {
    const data = await apiFetch('/api/v1/verify/challenge', { method: 'POST' });

    state.challengeType = data.challenge_type;
    state.challengeId   = data.challenge_id;

    // Parse emoji from instruction (first char(s) before space)
    const parts = data.instruction.trim().split(' ');
    const emoji = parts[0];
    const instruction = parts.slice(1).join(' ');

    $('challenge-emoji').textContent       = emoji;
    $('challenge-instruction').textContent = instruction;
    $('challenge-id').textContent          = data.challenge_id.slice(0, 8) + '…';

    show($('challenge-display'));

    // Transition to camera step
    setTimeout(() => {
      show(stepCamera);
      $('camera-instruction').textContent = data.instruction;
      startCamera();
      stepCamera.scrollIntoView({ behavior: 'smooth' });
    }, 800);

  } catch (err) {
    alert('Could not get challenge: ' + err.message);
    btnChallenge.disabled = false;
  } finally {
    hideLoading();
  }
});

/* ── Step 2: Camera ─────────────────────────────────────────────────────── */
async function startCamera() {
  try {
    state.stream = await navigator.mediaDevices.getUserMedia({
      video: { facingMode: 'user', width: { ideal: 1280 }, height: { ideal: 960 } },
      audio: false,
    });
    video.srcObject = state.stream;
    show(video);
    hide(preview);
    show(btnCapture);
    hide(btnRetake);
    hide(btnSubmit);
    state.capturedBlob = null;
  } catch (err) {
    alert('Camera access denied. Please allow camera permissions and reload.');
  }
}

btnCapture.addEventListener('click', () => {
  // Draw current video frame to canvas
  canvas.width  = video.videoWidth;
  canvas.height = video.videoHeight;
  const ctx = canvas.getContext('2d');
  // Mirror horizontally to match the CSS mirror on video
  ctx.translate(canvas.width, 0);
  ctx.scale(-1, 1);
  ctx.drawImage(video, 0, 0);

  canvas.toBlob(blob => {
    state.capturedBlob = blob;

    // Show preview instead of live video
    const url = URL.createObjectURL(blob);
    preview.src = url;
    hide(video);
    show(preview);

    hide(btnCapture);
    show(btnRetake);
    show(btnSubmit);
  }, 'image/jpeg', 0.92);
});

btnRetake.addEventListener('click', () => {
  show(video);
  hide(preview);
  show(btnCapture);
  hide(btnRetake);
  hide(btnSubmit);
  state.capturedBlob = null;
});

/* ── Step 3: Submit ─────────────────────────────────────────────────────── */
btnSubmit.addEventListener('click', async () => {
  if (!state.capturedBlob) return;

  btnSubmit.disabled = true;
  showLoading('Analyzing your photo…');

  try {
    const form = new FormData();
    form.append('file', state.capturedBlob, 'selfie.jpg');

    const data = await apiFetch(
      `/api/v1/verify/submit/${state.challengeType}`,
      { method: 'POST', body: form }
    );

    // Stop camera stream
    if (state.stream) {
      state.stream.getTracks().forEach(t => t.stop());
      state.stream = null;
    }

    renderResult(data);
    show(stepResult);
    stepResult.scrollIntoView({ behavior: 'smooth' });

  } catch (err) {
    alert('Verification failed: ' + err.message);
    btnSubmit.disabled = false;
  } finally {
    hideLoading();
  }
});

/* ── Result rendering ───────────────────────────────────────────────────── */
function renderResult(data) {
  const banner = $('result-banner');
  const details = $('result-details');

  banner.textContent = data.message;
  banner.className   = 'result-banner ' + (data.verified ? 'pass' : 'fail');

  const angles = data.pose_angles || {};
  const d      = data.details    || {};

  const items = [
    { label: 'Verified',        value: data.verified ? '✅ Yes' : '❌ No',           cls: data.verified ? 'pass' : 'fail' },
    { label: 'Liveness Score',  value: pct(data.liveness_score),                      cls: data.liveness_passed ? 'pass' : 'fail' },
    { label: 'Confidence',      value: pct(data.confidence),                          cls: '' },
    { label: 'Challenge',       value: data.challenge_type.replace(/_/g, ' '),        cls: '' },
    { label: 'Pose Matched',    value: data.pose_matched ? '✅ Yes' : '❌ No',        cls: data.pose_matched ? 'pass' : 'fail' },
    { label: 'Face Detected',   value: data.face_detected ? '✅ Yes' : '❌ No',       cls: data.face_detected ? 'pass' : 'fail' },
    { label: 'Yaw',             value: angles.yaw  != null ? angles.yaw.toFixed(1)  + '°' : '—', cls: '' },
    { label: 'Pitch',           value: angles.pitch != null ? angles.pitch.toFixed(1) + '°' : '—', cls: '' },
    { label: 'Smile Score',     value: d.smile_score != null ? pct(d.smile_score) : '—',  cls: '' },
    { label: 'Liveness',        value: data.liveness_passed ? 'Passed' : 'Failed',   cls: data.liveness_passed ? 'pass' : 'fail' },
  ];

  details.innerHTML = items.map(i => `
    <div class="detail-item">
      <div class="detail-label">${i.label}</div>
      <div class="detail-value ${i.cls}">${i.value}</div>
    </div>
  `).join('');
}

function pct(v) {
  return v != null ? (v * 100).toFixed(1) + '%' : '—';
}

/* ── Restart ────────────────────────────────────────────────────────────── */
btnRestart.addEventListener('click', () => {
  // Reset state
  state.challengeType = null;
  state.challengeId   = null;
  state.capturedBlob  = null;
  if (state.stream) {
    state.stream.getTracks().forEach(t => t.stop());
    state.stream = null;
  }

  // Reset UI
  hide($('challenge-display'));
  hide(stepCamera);
  hide(stepResult);

  $('result-banner').textContent = '';
  $('result-details').innerHTML  = '';
  preview.src = '';

  btnChallenge.disabled = false;
  btnSubmit.disabled    = false;

  window.scrollTo({ top: 0, behavior: 'smooth' });
});
