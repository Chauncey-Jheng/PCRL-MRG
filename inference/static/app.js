const REQUIRED_COUNT = 24;

const dropzone = document.getElementById('dropzone');
const fileInput = document.getElementById('fileInput');
const thumbs = document.getElementById('thumbs');
const countStatus = document.getElementById('countStatus');
const generateBtn = document.getElementById('generateBtn');
const clearBtn = document.getElementById('clearBtn');
const demoBtn = document.getElementById('demoBtn');
const demoHint = document.getElementById('demoHint');
const resultPanel = document.getElementById('resultPanel');
const statusBadge = document.getElementById('statusBadge');
const reportText = document.getElementById('reportText');
const groundTruthCard = document.getElementById('groundTruthCard');
const groundTruthText = document.getElementById('groundTruthText');
const healthBadge = document.getElementById('healthBadge');

let selectedFiles = [];
let mode = 'upload'; // 'upload' | 'demo' — decides which endpoint generateBtn hits

function renderThumbs(urls) {
  thumbs.innerHTML = '';
  urls.forEach((url) => {
    const img = document.createElement('img');
    img.src = url;
    thumbs.appendChild(img);
  });
}

function updateGenerateEnabled() {
  const ready = mode === 'demo' || selectedFiles.length === REQUIRED_COUNT;
  generateBtn.disabled = ready ? false : true;
}

function setFiles(fileList) {
  selectedFiles = Array.from(fileList);
  mode = 'upload';
  generateBtn.textContent = '生成报告';
  const urls = selectedFiles.map((f) => URL.createObjectURL(f));
  renderThumbs(urls);

  const n = selectedFiles.length;
  countStatus.textContent = `已选择 ${n} / ${REQUIRED_COUNT} 张`;
  countStatus.className = 'mode-status' + (n === REQUIRED_COUNT ? ' ready' : n > 0 ? ' error' : '');
  clearBtn.disabled = n === 0;
  demoHint.textContent = n > REQUIRED_COUNT
    ? `已超出：请正好选择 ${REQUIRED_COUNT} 张（当前 ${n} 张）。`
    : '';
  groundTruthCard.hidden = true;
  resultPanel.hidden = true;
  updateGenerateEnabled();
}

dropzone.addEventListener('click', () => fileInput.click());
dropzone.addEventListener('keydown', (e) => {
  if (e.key === 'Enter' || e.key === ' ') fileInput.click();
});
fileInput.addEventListener('change', (e) => setFiles(e.target.files));

['dragenter', 'dragover'].forEach((evt) =>
  dropzone.addEventListener(evt, (e) => {
    e.preventDefault();
    dropzone.classList.add('dragover');
  })
);
['dragleave', 'drop'].forEach((evt) =>
  dropzone.addEventListener(evt, (e) => {
    e.preventDefault();
    dropzone.classList.remove('dragover');
  })
);
dropzone.addEventListener('drop', (e) => {
  const files = Array.from(e.dataTransfer.files).filter((f) => f.type.startsWith('image/'));
  if (files.length) setFiles(files);
});

clearBtn.addEventListener('click', () => setFiles([]));

demoBtn.addEventListener('click', async () => {
  demoBtn.disabled = true;
  demoHint.textContent = '正在加载示例CT…';
  try {
    const resp = await fetch('/api/demo');
    const data = await resp.json();
    if (!data.available) {
      demoHint.textContent = '示例CT尚未部署在本机（需管理员运行 prepare_demo_sample.py）。';
      return;
    }
    mode = 'demo';
    selectedFiles = [];
    renderThumbs(data.images);
    countStatus.textContent = `示例CT：已加载 ${data.images.length} / 24 张`;
    countStatus.className = 'mode-status ready';
    demoHint.textContent = '点击下方按钮，用该示例CT直接生成报告。';
    resultPanel.hidden = true;
    clearBtn.disabled = true;
    generateBtn.textContent = '用示例CT生成报告';

    if (data.ground_truth && (data.ground_truth.findings || data.ground_truth.impression)) {
      groundTruthCard.hidden = false;
      groundTruthText.textContent =
        [data.ground_truth.findings, data.ground_truth.impression].filter(Boolean).join('\n');
    } else {
      groundTruthCard.hidden = true;
    }
    updateGenerateEnabled();
  } catch (err) {
    demoHint.textContent = `加载示例CT失败：${err.message || err}`;
  } finally {
    demoBtn.disabled = false;
  }
});

function setBusy(label) {
  generateBtn.disabled = true;
  demoBtn.disabled = true;
  clearBtn.disabled = true;
  resultPanel.hidden = false;
  statusBadge.innerHTML = `<span class="spinner"></span>${label}`;
  statusBadge.className = 'mode-status';
  reportText.textContent = '模型生成中，脑CT报告通常需要 10~60 秒，请稍候…';
}

function setIdleAfterGenerate() {
  demoBtn.disabled = false;
  clearBtn.disabled = mode === 'demo' || selectedFiles.length === 0;
  updateGenerateEnabled();
}

async function runRequest(url, options) {
  setBusy('正在生成报告…');
  try {
    const resp = await fetch(url, options);
    const data = await resp.json();
    if (!resp.ok) throw new Error(data.detail || `请求失败（${resp.status}）`);
    statusBadge.textContent = '生成完成';
    statusBadge.className = 'mode-status ready';
    reportText.textContent = data.report || '（模型未返回内容）';
  } catch (err) {
    statusBadge.textContent = '生成失败';
    statusBadge.className = 'mode-status error';
    reportText.textContent = String(err.message || err);
  } finally {
    setIdleAfterGenerate();
  }
}

generateBtn.addEventListener('click', () => {
  if (mode === 'demo') {
    runRequest('/api/demo/generate', { method: 'POST' });
    return;
  }
  if (selectedFiles.length !== REQUIRED_COUNT) return;
  const formData = new FormData();
  selectedFiles.forEach((f) => formData.append('images', f));
  runRequest('/api/generate', { method: 'POST', body: formData });
});

async function checkHealth() {
  try {
    const resp = await fetch('/api/health');
    const data = await resp.json();
    if (data.status === 'ready') {
      healthBadge.textContent = `模型已就绪（视觉token数=${data.visual_token_count}）`;
    } else if (data.status === 'loading') {
      healthBadge.textContent = '模型正在加载中（首次启动可能需要 1~2 分钟）…';
      setTimeout(checkHealth, 4000);
    } else {
      healthBadge.textContent = `模型加载失败：${data.detail || '未知错误'}`;
    }
  } catch (err) {
    healthBadge.textContent = '无法连接推理服务';
    setTimeout(checkHealth, 5000);
  }
}

checkHealth();
updateGenerateEnabled();
