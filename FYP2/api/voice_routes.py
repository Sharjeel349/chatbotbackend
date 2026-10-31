import logging
from typing import Optional

from fastapi import APIRouter, HTTPException, Query
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import FileResponse, HTMLResponse

from FYP2.models.voice_models import VoiceListResponse, VoiceTestRequest, VoiceTestResponse
from FYP2.services import tts_voice_service

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/voices", tags=["Voices"])


@router.get("", response_model=VoiceListResponse)
def get_available_voices():
    """
    Get the complete list of available system voices for text-to-speech.
    Includes recommended voices (e.g. Lekha for Roman Urdu/Hindi, Rishi for South Asian English,
    Samantha for US English, Daniel for UK English), categorized and tagged with gender & locale.
    """
    try:
        return tts_voice_service.get_available_voices_response()
    except Exception as exc:
        logger.exception("Error getting voices: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc))


@router.get("/preview")
async def preview_voice_audio(
    voice: str = Query("Lekha", description="Voice identifier or name (e.g., Lekha, Samantha, Daniel, Rishi)"),
    text: Optional[str] = Query(None, description="Optional custom text. Defaults to the University Advisor intro."),
):
    """
    Stream or download the audio preview directly as audio/wav.
    Ideal for frontend usage with HTML5 <audio> tag or `new Audio(url).play()`.
    """
    try:
        file_path, _, _ = await run_in_threadpool(
            tts_voice_service.synthesize_sample,
            voice,
            text,
        )
        return FileResponse(
            path=file_path,
            media_type="audio/wav",
            filename=file_path.name,
            headers={
                "Cache-Control": "public, max-age=86400",
                "Accept-Ranges": "bytes",
            },
        )
    except Exception as exc:
        logger.exception("Error generating audio preview for voice '%s': %s", voice, exc)
        raise HTTPException(status_code=500, detail=f"Audio generation failed: {exc}")


@router.post("/test", response_model=VoiceTestResponse)
async def test_voice(request: VoiceTestRequest):
    """
    Generate and return audio URL and metadata for testing a specific voice.
    Returns JSON containing the static audio URL, voice name, and timing information.
    """
    try:
        _, audio_url, duration_ms = await run_in_threadpool(
            tts_voice_service.synthesize_sample,
            request.voice,
            request.text,
        )
        return VoiceTestResponse(
            status="success",
            voice=request.voice,
            text=request.text or tts_voice_service.DEFAULT_SAMPLE_TEXT,
            audio_url=audio_url,
            duration_ms=duration_ms,
        )
    except Exception as exc:
        logger.exception("Error testing voice '%s': %s", request.voice, exc)
        raise HTTPException(status_code=500, detail=str(exc))


@router.get("/tester", response_model=None, response_class=HTMLResponse)
def get_voice_tester_ui():
    """
    Interactive web UI to test and audition all available voices with one click.
    """
    html_content = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>University AI Advisor - Voice Selector & Tester</title>
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
  <link href="https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet">
  <style>
    :root {
      --bg: #090d16;
      --card-bg: rgba(22, 30, 49, 0.75);
      --card-border: rgba(255, 255, 255, 0.1);
      --primary: #4f46e5;
      --primary-hover: #4338ca;
      --accent: #06b6d4;
      --accent-glow: rgba(6, 182, 212, 0.25);
      --text-main: #f8fafc;
      --text-muted: #94a3b8;
      --success: #10b981;
      --warning: #f59e0b;
    }

    * {
      box-sizing: border-box;
      margin: 0;
      padding: 0;
    }

    body {
      font-family: 'Plus Jakarta Sans', sans-serif;
      background: radial-gradient(circle at 50% 0%, #1e1b4b 0%, var(--bg) 60%);
      color: var(--text-main);
      min-height: 100vh;
      display: flex;
      flex-direction: column;
      align-items: center;
      padding: 40px 20px;
    }

    .container {
      width: 100%;
      max-width: 820px;
    }

    .header {
      text-align: center;
      margin-bottom: 32px;
    }

    .badge {
      display: inline-flex;
      align-items: center;
      gap: 6px;
      padding: 6px 14px;
      border-radius: 9999px;
      background: rgba(99, 102, 241, 0.15);
      border: 1px solid rgba(99, 102, 241, 0.3);
      color: #a5b4fc;
      font-size: 0.82rem;
      font-weight: 600;
      letter-spacing: 0.05em;
      text-transform: uppercase;
      margin-bottom: 12px;
    }

    h1 {
      font-size: 2.25rem;
      font-weight: 800;
      letter-spacing: -0.025em;
      background: linear-gradient(135deg, #ffffff 30%, #94a3b8 100%);
      -webkit-background-clip: text;
      -webkit-text-fill-color: transparent;
      margin-bottom: 8px;
    }

    p.subtitle {
      color: var(--text-muted);
      font-size: 1rem;
    }

    .card {
      background: var(--card-bg);
      backdrop-filter: blur(16px);
      -webkit-backdrop-filter: blur(16px);
      border: 1px solid var(--card-border);
      border-radius: 20px;
      padding: 28px;
      box-shadow: 0 20px 40px -15px rgba(0, 0, 0, 0.5);
      margin-bottom: 24px;
    }

    .form-group {
      margin-bottom: 22px;
    }

    label {
      display: flex;
      justify-content: space-between;
      align-items: center;
      font-size: 0.88rem;
      font-weight: 600;
      color: #cbd5e1;
      margin-bottom: 8px;
    }

    .label-hint {
      font-size: 0.78rem;
      color: var(--accent);
      font-weight: 500;
    }

    select, textarea, input {
      width: 100%;
      background: rgba(15, 23, 42, 0.75);
      border: 1px solid rgba(255, 255, 255, 0.12);
      border-radius: 12px;
      padding: 14px 16px;
      color: var(--text-main);
      font-family: inherit;
      font-size: 0.95rem;
      transition: all 0.2s ease;
      outline: none;
    }

    select:focus, textarea:focus {
      border-color: var(--accent);
      box-shadow: 0 0 0 3px var(--accent-glow);
    }

    textarea {
      resize: vertical;
      min-height: 85px;
      line-height: 1.5;
    }

    .quick-chips {
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
      margin-top: 10px;
    }

    .chip {
      background: rgba(255, 255, 255, 0.05);
      border: 1px solid rgba(255, 255, 255, 0.1);
      border-radius: 8px;
      padding: 6px 12px;
      font-size: 0.8rem;
      color: #cbd5e1;
      cursor: pointer;
      transition: all 0.2s ease;
      display: inline-flex;
      align-items: center;
      gap: 6px;
    }

    .chip:hover, .chip.active {
      background: rgba(6, 182, 212, 0.15);
      border-color: var(--accent);
      color: #38bdf8;
    }

    .chip.recommended {
      border-color: rgba(245, 158, 11, 0.4);
      background: rgba(245, 158, 11, 0.08);
      color: #fcd34d;
    }

    .actions {
      display: flex;
      gap: 12px;
      margin-top: 8px;
    }

    .btn {
      flex: 1;
      display: inline-flex;
      align-items: center;
      justify-content: center;
      gap: 10px;
      padding: 14px 24px;
      border-radius: 12px;
      font-size: 1rem;
      font-weight: 600;
      cursor: pointer;
      transition: all 0.2s ease;
      border: none;
    }

    .btn-primary {
      background: linear-gradient(135deg, #4f46e5 0%, #06b6d4 100%);
      color: white;
      box-shadow: 0 8px 20px -4px rgba(79, 70, 229, 0.4);
    }

    .btn-primary:hover:not(:disabled) {
      transform: translateY(-1px);
      box-shadow: 0 12px 24px -4px rgba(6, 182, 212, 0.4);
    }

    .btn-primary:disabled {
      opacity: 0.6;
      cursor: not-allowed;
    }

    .btn-secondary {
      background: rgba(255, 255, 255, 0.08);
      color: #cbd5e1;
      border: 1px solid rgba(255, 255, 255, 0.12);
    }

    .btn-secondary:hover {
      background: rgba(255, 255, 255, 0.12);
    }

    .audio-player-wrapper {
      margin-top: 20px;
      padding-top: 20px;
      border-top: 1px solid rgba(255, 255, 255, 0.08);
      display: none;
    }

    .audio-player-wrapper.active {
      display: block;
    }

    audio {
      width: 100%;
      height: 44px;
      border-radius: 8px;
    }

    .status-bar {
      display: flex;
      align-items: center;
      gap: 10px;
      margin-top: 12px;
      font-size: 0.85rem;
      color: var(--text-muted);
    }

    .status-dot {
      width: 8px;
      height: 8px;
      border-radius: 50%;
      background: var(--text-muted);
    }

    .status-dot.playing {
      background: var(--success);
      box-shadow: 0 0 10px var(--success);
      animation: pulse 1.5s infinite;
    }

    @keyframes pulse {
      0%, 100% { opacity: 1; transform: scale(1); }
      50% { opacity: 0.4; transform: scale(1.3); }
    }

    .api-preview {
      background: rgba(10, 15, 28, 0.9);
      border: 1px solid rgba(255, 255, 255, 0.08);
      border-radius: 12px;
      padding: 16px;
      margin-top: 20px;
      font-family: 'JetBrains Mono', monospace;
      font-size: 0.82rem;
      color: #94a3b8;
    }

    .api-title {
      color: #cbd5e1;
      font-weight: 600;
      margin-bottom: 8px;
      display: flex;
      justify-content: space-between;
    }

    .code-box {
      color: #38bdf8;
      word-break: break-all;
    }

    .spinner {
      width: 18px;
      height: 18px;
      border: 2px solid rgba(255, 255, 255, 0.3);
      border-top-color: white;
      border-radius: 50%;
      animation: spin 0.8s linear infinite;
      display: none;
    }

    @keyframes spin {
      to { transform: rotate(360deg); }
    }
  </style>
</head>
<body>
  <div class="container">
    <div class="header">
      <div class="badge">🎙️ Speech Synthesis Engine</div>
      <h1>Advisor Voice Selector & Tester</h1>
      <p class="subtitle">Select and audition available voices for the University Academic AI Advisor</p>
    </div>

    <div class="card">
      <div class="form-group">
        <label for="voiceSelect">
          <span>Select Voice</span>
          <span class="label-hint" id="voiceCountHint">Loading voices...</span>
        </label>
        <select id="voiceSelect">
          <option value="">Loading available voices...</option>
        </select>
      </div>

      <div class="form-group">
        <label>Recommended for University Advisor</label>
        <div class="quick-chips" id="recommendedChips">
          <!-- Dynamically populated -->
        </div>
      </div>

      <div class="form-group">
        <label for="sampleText">
          <span>Test Speech Text</span>
          <span class="label-hint" style="cursor: pointer;" onclick="resetText()">Reset to Default</span>
        </label>
        <textarea id="sampleText"></textarea>
      </div>

      <div class="actions">
        <button class="btn btn-primary" id="playBtn" onclick="testSelectedVoice()">
          <div class="spinner" id="playSpinner"></div>
          <span id="playIcon">▶</span>
          <span id="playBtnText">Play Voice Sample</span>
        </button>
      </div>

      <div class="audio-player-wrapper" id="playerWrapper">
        <audio id="audioPlayer" controls></audio>
        <div class="status-bar">
          <div class="status-dot" id="statusDot"></div>
          <span id="statusMessage">Ready</span>
        </div>
      </div>

      <div class="api-preview">
        <div class="api-title">
          <span>API Integration Endpoint</span>
          <span style="color: #64748b;">GET & POST</span>
        </div>
        <div class="code-box" id="apiEndpointBox">/api/voices/preview?voice=Lekha</div>
      </div>
    </div>
  </div>

  <script>
    const DEFAULT_TEXT = "Hi, I'm your university advisor. How can I help you? Assalamu alaikum, mein ap ka university advisor hon, mein ap ki kaisy madad kr sakta ho";
    let voicesData = null;
    const voiceSelect = document.getElementById('voiceSelect');
    const recommendedChips = document.getElementById('recommendedChips');
    const sampleText = document.getElementById('sampleText');
    const playBtn = document.getElementById('playBtn');
    const playBtnText = document.getElementById('playBtnText');
    const playIcon = document.getElementById('playIcon');
    const playSpinner = document.getElementById('playSpinner');
    const audioPlayer = document.getElementById('audioPlayer');
    const playerWrapper = document.getElementById('playerWrapper');
    const statusDot = document.getElementById('statusDot');
    const statusMessage = document.getElementById('statusMessage');
    const apiEndpointBox = document.getElementById('apiEndpointBox');
    const voiceCountHint = document.getElementById('voiceCountHint');

    sampleText.value = DEFAULT_TEXT;

    function resetText() {
      sampleText.value = DEFAULT_TEXT;
      updateApiBox();
    }

    async function loadVoices() {
      try {
        const res = await fetch('/api/voices');
        voicesData = await res.json();
        voiceCountHint.textContent = `${voicesData.total} voices available`;

        voiceSelect.innerHTML = '';

        // Group into Recommended and All
        const optgroupRec = document.createElement('optgroup');
        optgroupRec.label = '★ Recommended for Advisor';
        voicesData.recommended_voices.forEach(v => {
          const opt = document.createElement('option');
          opt.value = v.id || v.name;
          opt.textContent = `${v.display_name} (${v.gender})`;
          if (v.name === voicesData.default_voice || v.id === voicesData.default_voice) {
            opt.selected = true;
          }
          optgroupRec.appendChild(opt);
        });
        voiceSelect.appendChild(optgroupRec);

        const optgroupAll = document.createElement('optgroup');
        optgroupAll.label = 'Other System Voices';
        voicesData.all_voices.forEach(v => {
          if (!v.is_recommended) {
            const opt = document.createElement('option');
            opt.value = v.id || v.name;
            opt.textContent = `${v.name} (${v.language} - ${v.gender})`;
            optgroupAll.appendChild(opt);
          }
        });
        voiceSelect.appendChild(optgroupAll);

        // Render Quick Chips
        recommendedChips.innerHTML = '';
        voicesData.recommended_voices.forEach(v => {
          const chip = document.createElement('div');
          chip.className = 'chip recommended';
          chip.textContent = `${v.name} (${v.gender})`;
          chip.onclick = () => {
            voiceSelect.value = v.id || v.name;
            updateApiBox();
            testSelectedVoice();
          };
          recommendedChips.appendChild(chip);
        });

        updateApiBox();
      } catch (err) {
        voiceCountHint.textContent = 'Failed to load voices';
        console.error('Error loading voices:', err);
      }
    }

    function updateApiBox() {
      const selected = voiceSelect.value || 'Lekha';
      apiEndpointBox.textContent = `/api/voices/preview?voice=${encodeURIComponent(selected)}`;
    }

    voiceSelect.addEventListener('change', updateApiBox);

    async function testSelectedVoice() {
      const selectedVoice = voiceSelect.value || 'Lekha';
      const text = sampleText.value.trim() || DEFAULT_TEXT;

      playBtn.disabled = true;
      playSpinner.style.display = 'inline-block';
      playIcon.style.display = 'none';
      playBtnText.textContent = 'Generating Speech...';
      playerWrapper.classList.add('active');
      statusDot.className = 'status-dot';
      statusMessage.textContent = 'Generating sample with ' + selectedVoice + '...';

      try {
        const previewUrl = `/api/voices/preview?voice=${encodeURIComponent(selectedVoice)}&text=${encodeURIComponent(text)}`;
        audioPlayer.src = previewUrl;
        await audioPlayer.play();

        statusDot.className = 'status-dot playing';
        statusMessage.textContent = 'Playing voice sample (' + selectedVoice + ')...';
      } catch (err) {
        console.error('Playback error:', err);
        statusDot.className = 'status-dot';
        statusMessage.textContent = 'Error during audio playback: ' + err.message;
      } finally {
        playBtn.disabled = false;
        playSpinner.style.display = 'none';
        playIcon.style.display = 'inline-block';
        playBtnText.textContent = 'Play Voice Sample';
      }
    }

    audioPlayer.onended = () => {
      statusDot.className = 'status-dot';
      statusMessage.textContent = 'Playback completed.';
    };

    audioPlayer.onerror = () => {
      statusDot.className = 'status-dot';
      statusMessage.textContent = 'Failed to stream audio file.';
    };

    loadVoices();
  </script>
</body>
</html>
"""
    return HTMLResponse(content=html_content)
