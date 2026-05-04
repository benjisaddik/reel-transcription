import hashlib
import os
import re
import subprocess
import tempfile
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()


def _patch_speechbrain_lazy():
    try:
        from speechbrain.utils import importutils
    except ImportError:
        return
    orig = importutils.LazyModule.__getattr__
    def safe(self, attr):
        if attr.startswith("__") and attr.endswith("__"):
            raise AttributeError(attr)
        return orig(self, attr)
    importutils.LazyModule.__getattr__ = safe
_patch_speechbrain_lazy()

import imageio_ffmpeg
import streamlit as st
import yt_dlp
from faster_whisper import WhisperModel

FFMPEG_EXE = imageio_ffmpeg.get_ffmpeg_exe()

MODEL_SIZE = os.environ.get("WHISPER_MODEL", "small")
HF_TOKEN = os.environ.get("HF_TOKEN")
COOKIES_FILE = os.environ.get("IG_COOKIES_FILE")
COOKIES_BROWSER = os.environ.get("IG_COOKIES_BROWSER")

st.set_page_config(
    page_title="Transcribe Reels",
    page_icon="◼",
    layout="centered",
    initial_sidebar_state="collapsed",
)

CSS = """
<style>
  @import url('https://fonts.googleapis.com/css2?family=Archivo+Black&family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500;700&display=swap');

  /* Hide default Streamlit chrome */
  #MainMenu, footer, header[data-testid="stHeader"] { visibility: hidden; height: 0; }
  .stDeployButton { display: none !important; }
  [data-testid="stToolbar"] { display: none !important; }

  /* Hide Streamlit's Material Symbols icons (clash with brutalist + my
     universal font-family rule breaks the icon-ligature rendering) */
  [data-testid*="Icon"],
  [data-testid="stIconMaterial"],
  span[class*="material-symbols"],
  span[class*="material-icons"],
  .material-symbols-outlined,
  .material-symbols-rounded,
  .material-symbols-sharp,
  details > summary > span[aria-hidden="true"] {
    display: none !important;
  }

  /* Nuke all border-radius on every element — strict brutalist */
  .block-container *,
  .stApp *,
  [data-baseweb="popover"] * {
    border-radius: 0 !important;
  }

  html, body, [class*="st-"], .stMarkdown, p, span, div {
    font-family: 'Inter', system-ui, sans-serif !important;
    color: #000;
  }

  .block-container {
    padding-top: 3rem !important;
    padding-bottom: 5rem !important;
    max-width: 920px !important;
  }

  /* ── Top band ─────────────────────────────────────────── */
  .top-band {
    display: flex;
    justify-content: space-between;
    align-items: center;
    border-bottom: 3px solid #000;
    padding-bottom: 0.75rem;
    margin-bottom: 3rem;
  }
  .top-band span {
    font-size: 0.7rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.15em;
  }

  /* ── Hero ────────────────────────────────────────────── */
  .hero-title {
    font-family: 'Archivo Black', system-ui, sans-serif !important;
    font-size: 7.5rem;
    line-height: 0.85;
    letter-spacing: -0.04em;
    text-transform: uppercase;
    margin: 0 0 0.5rem 0;
    color: #000;
  }
  .hero-sub {
    font-size: 1.05rem;
    line-height: 1.4;
    font-weight: 500;
    max-width: 44ch;
    margin: 1.5rem 0 3rem 0;
    color: #000;
  }

  /* ── Status grid ─────────────────────────────────────── */
  .status-grid {
    display: grid;
    grid-template-columns: 1fr 1fr 1fr;
    border: 3px solid #000;
    margin-bottom: 2.5rem;
  }
  .status-cell {
    padding: 0.75rem 1.25rem;
    border-right: 3px solid #000;
  }
  .status-cell:last-child { border-right: 0; }
  .status-cell.on { background: #FDE047; }
  .status-label {
    font-size: 0.65rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    color: #737373;
  }
  .status-cell.on .status-label { color: #000; }
  .status-value {
    font-size: 0.95rem;
    font-weight: 700;
    color: #000;
  }

  /* ── URL input ──────────────────────────────────────── */
  div[data-testid="stTextInput"] {
    gap: 0 !important;
  }
  div[data-testid="stTextInput"] > label,
  .stTextInput label {
    background: #000 !important;
    color: #fff !important;
    padding: 0.55rem 1.25rem !important;
    border: 3px solid #000 !important;
    border-bottom: 0 !important;
    font-weight: 700 !important;
    text-transform: uppercase !important;
    letter-spacing: 0.15em !important;
    font-size: 0.7rem !important;
    margin: 0 !important;
    display: block !important;
    width: 100% !important;
    box-sizing: border-box !important;
  }
  div[data-testid="stTextInput"] > label *,
  .stTextInput label *,
  .stTextInput label p {
    color: #fff !important;
    background: transparent !important;
    font-weight: 700 !important;
    font-size: 0.7rem !important;
    letter-spacing: 0.15em !important;
    margin: 0 !important;
  }
  div[data-testid="stTextInput"] > div {
    margin-top: 0 !important;
    padding-top: 0 !important;
  }
  .stTextInput input {
    border: 3px solid #000 !important;
    border-radius: 0 !important;
    padding: 1rem 1.25rem !important;
    font-size: 1.05rem !important;
    font-weight: 500 !important;
    background: #fff !important;
    color: #000 !important;
    font-family: 'Inter', sans-serif !important;
    margin-top: 0 !important;
  }
  .stTextInput input:focus {
    background: #FFFBE6 !important;
    box-shadow: none !important;
    outline: 0 !important;
  }
  div[data-baseweb="input"] {
    border: 0 !important;
    background: transparent !important;
    margin-top: 0 !important;
  }

  /* ── Hide rounded help-tooltip icons (clash with brutalist) ── */
  [data-testid="stTooltipIcon"],
  [data-testid="stTooltipHoverTarget"] {
    display: none !important;
  }

  /* ── Buttons ────────────────────────────────────────── */
  .stButton > button,
  .stButton > button * {
    background: #000 !important;
    color: #fff !important;
    border-radius: 0 !important;
    font-weight: 700 !important;
    font-size: 0.95rem !important;
    text-transform: uppercase !important;
    letter-spacing: 0.08em !important;
    box-shadow: none !important;
  }
  .stButton > button {
    border: 3px solid #000 !important;
    padding: 0.85rem 2.5rem !important;
    width: 100%;
    transition: background 80ms ease, color 80ms ease;
  }
  .stButton > button * { background: transparent !important; }
  .stButton > button:hover,
  .stButton > button:hover * {
    background: #FDE047 !important;
    color: #000 !important;
  }
  .stButton > button:hover { border-color: #000 !important; }
  .stButton > button:hover * { background: transparent !important; }
  .stButton > button:active, .stButton > button:focus {
    box-shadow: none !important;
    outline: 0 !important;
  }

  /* ── Checkbox ──────────────────────────────────────── */
  .stCheckbox label p {
    font-weight: 700 !important;
    text-transform: uppercase !important;
    font-size: 0.85rem !important;
    letter-spacing: 0.05em !important;
    color: #000 !important;
  }
  .stCheckbox {
    padding-top: 0.85rem;
  }

  /* ── Result header ─────────────────────────────────── */
  .result-header {
    border-top: 3px solid #000;
    padding-top: 0.6rem;
    display: flex;
    justify-content: space-between;
    align-items: baseline;
    margin: 4rem 0 1rem 0;
  }
  .result-label {
    font-size: 0.7rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.15em;
    color: #000;
  }
  .result-meta {
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.8rem;
    color: #000;
  }

  /* ── Transcript table ──────────────────────────────── */
  .transcript-table {
    border: 3px solid #000;
    margin-bottom: 2rem;
  }
  .seg-row {
    display: flex;
    align-items: stretch;
    border-bottom: 3px solid #000;
  }
  .seg-row:last-child { border-bottom: 0; }
  .speaker-block {
    width: 150px;
    padding: 0.85rem 0.85rem;
    font-weight: 700;
    font-size: 0.8rem;
    letter-spacing: 0.08em;
    display: flex;
    align-items: center;
    flex-shrink: 0;
  }
  .time-block {
    width: 130px;
    padding: 0.85rem 0.85rem;
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.75rem;
    background: #F5F5F5;
    border-left: 3px solid #000;
    border-right: 3px solid #000;
    display: flex;
    align-items: center;
    flex-shrink: 0;
  }
  .text-block {
    flex: 1;
    padding: 0.85rem 1.1rem;
    font-size: 1rem;
    line-height: 1.45;
    color: #000;
  }

  /* ── Download button — secondary ──────────────────── */
  .stDownloadButton > button,
  .stDownloadButton > button * {
    background: transparent !important;
    color: #000 !important;
    border-radius: 0 !important;
    font-weight: 700 !important;
    font-size: 0.85rem !important;
    text-transform: uppercase !important;
    letter-spacing: 0.08em !important;
  }
  .stDownloadButton > button {
    border: 3px solid #000 !important;
    padding: 0.75rem 1.5rem !important;
    width: auto !important;
  }
  .stDownloadButton > button:hover,
  .stDownloadButton > button:hover * {
    background: #000 !important;
    color: #fff !important;
  }

  /* ── Alerts ─────────────────────────────────────── */
  div[data-testid="stAlert"] {
    border: 3px solid #000 !important;
    border-radius: 0 !important;
    background: #FDE047 !important;
    box-shadow: none !important;
  }
  div[data-testid="stAlert"] * {
    color: #000 !important;
    font-weight: 600 !important;
  }

  /* ── Spinner / status ──────────────────────────── */
  div[data-testid="stSpinner"] > div > div {
    border-color: #000 transparent transparent transparent !important;
  }
  div[data-testid="stSpinner"] p {
    color: #000 !important;
    font-weight: 600 !important;
    text-transform: uppercase !important;
    letter-spacing: 0.05em !important;
    font-size: 0.85rem !important;
  }

  /* ── Expander ─────────────────────────────────── */
  div[data-testid="stExpander"] {
    border: 3px solid #000 !important;
    border-radius: 0 !important;
  }
  div[data-testid="stExpander"] summary {
    font-weight: 700 !important;
    text-transform: uppercase !important;
    letter-spacing: 0.08em !important;
    font-size: 0.8rem !important;
  }

  /* ── Number input (batch cap) ─────────────────── */
  div[data-testid="stNumberInput"] {
    gap: 0 !important;
  }
  div[data-testid="stNumberInput"] > label,
  .stNumberInput label {
    background: #000 !important;
    color: #fff !important;
    padding: 0.55rem 1.25rem !important;
    border: 3px solid #000 !important;
    border-bottom: 0 !important;
    font-weight: 700 !important;
    text-transform: uppercase !important;
    letter-spacing: 0.15em !important;
    font-size: 0.7rem !important;
    margin: 0 !important;
    display: block !important;
    width: 100% !important;
    box-sizing: border-box !important;
  }
  div[data-testid="stNumberInput"] > label *,
  .stNumberInput label * {
    color: #fff !important;
    background: transparent !important;
    font-weight: 700 !important;
    font-size: 0.7rem !important;
    letter-spacing: 0.15em !important;
    margin: 0 !important;
  }
  .stNumberInput input {
    border: 3px solid #000 !important;
    border-radius: 0 !important;
    padding: 0.85rem 1.25rem !important;
    font-size: 1rem !important;
    font-weight: 600 !important;
    background: #fff !important;
    color: #000 !important;
    font-family: 'JetBrains Mono', monospace !important;
  }
  .stNumberInput button {
    border: 3px solid #000 !important;
    border-left: 0 !important;
    border-radius: 0 !important;
    background: #fff !important;
    color: #000 !important;
  }
  .stNumberInput button:hover {
    background: #FDE047 !important;
  }

  /* ── Reel cards (batch results) ────────────────── */
  .reel-card {
    border: 3px solid #000;
    margin-bottom: 1rem;
  }
  .reel-card-header {
    background: #000;
    color: #fff;
    padding: 0.6rem 1.1rem;
    display: flex;
    justify-content: space-between;
    align-items: center;
    font-weight: 700;
    text-transform: uppercase;
    font-size: 0.78rem;
    letter-spacing: 0.08em;
    gap: 1rem;
  }
  .reel-card-header * {
    color: #fff !important;
    background: transparent !important;
  }
  .reel-card-header .id {
    font-family: 'JetBrains Mono', monospace !important;
  }
  .reel-card-body {
    padding: 1rem 1.25rem;
    font-size: 0.98rem;
    line-height: 1.55;
    color: #000;
  }
  .reel-card.processing .reel-card-header {
    background: #FDE047;
  }
  .reel-card.processing .reel-card-header * {
    color: #000 !important;
  }
  .reel-card.processing .reel-card-body {
    color: #737373;
    font-style: italic;
  }
  .reel-card.error .reel-card-header {
    background: #FDE047;
  }
  .reel-card.error .reel-card-header * {
    color: #000 !important;
  }
  .reel-card.error .reel-card-body {
    background: #FFFBE6;
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.85rem;
  }

  /* ── Pills (browser picker) ─────────────────────── */
  [data-testid="stPills"] > label,
  .stPills label {
    background: #000 !important;
    color: #fff !important;
    padding: 0.55rem 1.25rem !important;
    border: 3px solid #000 !important;
    border-bottom: 0 !important;
    font-weight: 700 !important;
    text-transform: uppercase !important;
    letter-spacing: 0.15em !important;
    font-size: 0.7rem !important;
    margin: 0 !important;
    display: block !important;
    width: 100% !important;
    box-sizing: border-box !important;
  }
  [data-testid="stPills"] > label *,
  .stPills label * {
    color: #fff !important;
    background: transparent !important;
    font-weight: 700 !important;
    font-size: 0.7rem !important;
    letter-spacing: 0.15em !important;
    margin: 0 !important;
  }
  [data-testid="stPills"] [role="group"],
  [data-testid="stPills"] [role="radiogroup"] {
    display: flex !important;
    flex-wrap: wrap !important;
    gap: 0 !important;
    border-left: 3px solid #000 !important;
    border-top: 3px solid #000 !important;
    background: #fff !important;
  }
  [data-testid="stPills"] button {
    border: 0 !important;
    border-right: 3px solid #000 !important;
    border-bottom: 3px solid #000 !important;
    border-radius: 0 !important;
    background: #fff !important;
    color: #000 !important;
    font-weight: 700 !important;
    text-transform: uppercase !important;
    letter-spacing: 0.08em !important;
    font-size: 0.78rem !important;
    padding: 0.7rem 1.1rem !important;
    margin: 0 !important;
    box-shadow: none !important;
    flex: 1 0 auto !important;
    transition: background 80ms ease, color 80ms ease;
  }
  [data-testid="stPills"] button:hover {
    background: #FFFBE6 !important;
  }
  [data-testid="stPills"] button[aria-pressed="true"],
  [data-testid="stPills"] button[aria-checked="true"],
  [data-testid="stPills"] button[kind*="selected"] {
    background: #000 !important;
    color: #fff !important;
  }
  [data-testid="stPills"] button * {
    color: inherit !important;
    background: transparent !important;
    font-weight: inherit !important;
    font-size: inherit !important;
    letter-spacing: inherit !important;
  }
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)


SPEAKER_PALETTE = [
    ("#FFFFFF", "#000000"),  # white on black
    ("#000000", "#FDE047"),  # black on yellow
    ("#000000", "#A3E635"),  # black on lime
    ("#000000", "#F472B6"),  # black on pink
    ("#FFFFFF", "#1D4ED8"),  # white on royal blue
    ("#000000", "#FB923C"),  # black on orange
]


def speaker_colors(label: str) -> tuple[str, str]:
    h = int(hashlib.md5(label.encode()).hexdigest(), 16)
    return SPEAKER_PALETTE[h % len(SPEAKER_PALETTE)]


@st.cache_resource
def load_whisper():
    return WhisperModel(MODEL_SIZE, device="auto", compute_type="auto")


@st.cache_resource
def load_diarizer():
    if not HF_TOKEN:
        return None
    import torch
    from pyannote.audio import Pipeline

    _orig_load = torch.load
    def _load(*args, **kwargs):
        kwargs.setdefault("weights_only", False)
        return _orig_load(*args, **kwargs)
    torch.load = _load
    try:
        pipe = Pipeline.from_pretrained(
            "pyannote/speaker-diarization-3.1",
            use_auth_token=HF_TOKEN,
        )
    finally:
        torch.load = _orig_load

    if torch.cuda.is_available():
        pipe.to(torch.device("cuda"))
    elif torch.backends.mps.is_available():
        pipe.to(torch.device("mps"))
    return pipe


SINGLE_REEL_RE = re.compile(r"instagram\.com/(reel|p|tv)/[^/?#]+")


def is_single_reel(url: str) -> bool:
    return bool(SINGLE_REEL_RE.search(url))


def list_profile_reels(profile_url: str, browser=None, limit: int = 5):
    """Return [(shortcode, url), ...] for the most-recent N reels on a profile."""
    ydl_opts = {
        "extract_flat": True,
        "playlistend": limit,
        "quiet": True,
        "no_warnings": True,
    }
    if browser:
        ydl_opts["cookiesfrombrowser"] = (browser,)
    elif COOKIES_FILE:
        ydl_opts["cookiefile"] = COOKIES_FILE

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(profile_url, download=False)
    entries = info.get("entries") or []
    out = []
    for e in entries:
        url = e.get("url") or e.get("webpage_url")
        if not url:
            continue
        if not url.startswith("http"):
            url = f"https://www.instagram.com/p/{e.get('id')}/"
        out.append((e.get("id", "?"), url))
    return out


def download_audio(url: str, out_dir: Path, browser=None) -> Path:
    ydl_opts = {
        "format": "bestaudio/best",
        "outtmpl": str(out_dir / "audio.%(ext)s"),
        "quiet": True,
        "no_warnings": True,
    }
    if browser:
        ydl_opts["cookiesfrombrowser"] = (browser,)
    elif COOKIES_FILE:
        ydl_opts["cookiefile"] = COOKIES_FILE

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=True)
    src_path = Path(ydl.prepare_filename(info))

    # Convert to 16kHz mono WAV with the bundled ffmpeg — pyannote uses
    # libsndfile which doesn't read m4a/aac.
    wav_path = src_path.with_suffix(".wav")
    if src_path.resolve() != wav_path.resolve():
        result = subprocess.run(
            [FFMPEG_EXE, "-y", "-loglevel", "error",
             "-i", str(src_path), "-ar", "16000", "-ac", "1", str(wav_path)],
            capture_output=True,
        )
        if result.returncode != 0:
            raise RuntimeError(f"ffmpeg conversion failed: {result.stderr.decode(errors='replace')}")
        src_path.unlink(missing_ok=True)
    return wav_path


def transcribe(audio_path: Path):
    model = load_whisper()
    segments, info = model.transcribe(str(audio_path), beam_size=5)
    return list(segments), info


def assign_speakers(segments, diarization):
    turns = list(diarization.itertracks(yield_label=True))
    out = []
    for seg in segments:
        best_speaker, best_overlap = None, 0.0
        for turn, _, speaker in turns:
            overlap = max(0.0, min(seg.end, turn.end) - max(seg.start, turn.start))
            if overlap > best_overlap:
                best_overlap = overlap
                best_speaker = speaker
        out.append({
            "start": seg.start,
            "end": seg.end,
            "text": seg.text.strip(),
            "speaker": best_speaker or "?",
        })
    return out


def fmt_time(seconds: float) -> str:
    m, s = divmod(int(seconds), 60)
    return f"{m:02d}:{s:02d}"


# ── TOP BAND ───────────────────────────────────────────────
st.markdown(
    '<div class="top-band">'
    '<span>Reels / Transcribe / v.02</span>'
    '</div>',
    unsafe_allow_html=True,
)

# ── HERO ──────────────────────────────────────────────────
st.markdown('<h1 class="hero-title">Read<br>the reel.</h1>', unsafe_allow_html=True)
st.markdown(
    '<p class="hero-sub">Paste an Instagram URL. Get the words back. Local Whisper, '
    'optional speaker labels, no API keys.</p>',
    unsafe_allow_html=True,
)

# ── STATUS GRID (filled in after controls render) ────────
status_slot = st.empty()

# ── INPUT ─────────────────────────────────────────────────
url = st.text_input(
    "Reel URL or profile URL",
    placeholder="https://www.instagram.com/reel/... or /<username>/",
)

is_batch = bool(url.strip()) and not is_single_reel(url)

BROWSERS = ["PUBLIC", "CHROME", "FIREFOX", "SAFARI", "EDGE", "BRAVE", "OPERA"]
_env_browser = (COOKIES_BROWSER or "").upper().strip()
default_browser = _env_browser if _env_browser in BROWSERS else "PUBLIC"

browser_choice = st.pills(
    "Browser",
    BROWSERS,
    default=default_browser,
    selection_mode="single",
)
if browser_choice is None:
    browser_choice = "PUBLIC"

if is_batch:
    max_reels = st.number_input(
        "Max reels (most recent first)",
        min_value=1,
        max_value=20,
        value=5,
        step=1,
    )
else:
    max_reels = 1

diarize_enabled = st.checkbox(
    "Identify speakers",
    value=bool(HF_TOKEN) and not is_batch,
    disabled=(not HF_TOKEN) or is_batch,
)

selected_browser = None if browser_choice == "PUBLIC" else browser_choice.lower()

go_label = "▸ Transcribe batch" if is_batch else "▸ Transcribe"
go = st.button(go_label)

# ── Render status grid using selected values ─────────────
if selected_browser:
    cookies_value = f"BROWSER · {selected_browser.upper()}"
    cookies_on = True
elif COOKIES_FILE:
    cookies_value = "FILE"
    cookies_on = True
else:
    cookies_value = "PUBLIC"
    cookies_on = False

diar_active = diarize_enabled and bool(HF_TOKEN)

if is_batch:
    mode_label = "Mode"
    mode_value = f"BATCH · {max_reels} REELS"
    mode_on = True
else:
    mode_label = "Speakers"
    mode_value = "DIARISATION ON" if diar_active else ("OFF" if HF_TOKEN else "NO HF TOKEN")
    mode_on = diar_active

status_slot.markdown(
    f"""
    <div class="status-grid">
      <div class="status-cell {'on' if cookies_on else ''}">
        <div class="status-label">Cookies</div>
        <div class="status-value">{cookies_value}</div>
      </div>
      <div class="status-cell {'on' if mode_on else ''}">
        <div class="status-label">{mode_label}</div>
        <div class="status-value">{mode_value}</div>
      </div>
      <div class="status-cell">
        <div class="status-label">Model</div>
        <div class="status-value">WHISPER · {MODEL_SIZE.upper()}</div>
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# ── RUN ───────────────────────────────────────────────────
if go and not url.strip():
    st.error("PASTE A REEL URL OR PROFILE URL ABOVE.")
    st.stop()

if go and is_batch:
    with tempfile.TemporaryDirectory() as td:
        tmp_dir = Path(td)

        with st.spinner(f"FETCHING UP TO {max_reels} REELS…"):
            try:
                reels = list_profile_reels(url, browser=selected_browser, limit=int(max_reels))
            except Exception as e:
                st.error(f"COULDN'T FETCH PROFILE — {e}")
                st.stop()

        if not reels:
            st.error("NO REELS FOUND ON THIS PAGE.")
            st.stop()

        st.markdown(
            f'<div class="result-header">'
            f'<span class="result-label">Batch / {len(reels)} reels</span>'
            f'<span class="result-meta">{url}</span>'
            f'</div>',
            unsafe_allow_html=True,
        )

        all_results = []
        for i, (reel_id, reel_url) in enumerate(reels):
            slot = st.empty()
            slot.markdown(
                f'<div class="reel-card processing">'
                f'  <div class="reel-card-header">'
                f'    <span class="id">{reel_id}</span>'
                f'    <span>{i + 1} / {len(reels)} · PROCESSING…</span>'
                f'  </div>'
                f'  <div class="reel-card-body">Fetching audio and transcribing…</div>'
                f'</div>',
                unsafe_allow_html=True,
            )
            try:
                audio_path = download_audio(reel_url, tmp_dir, browser=selected_browser)
                segments, info = transcribe(audio_path)
                text = " ".join(s.text.strip() for s in segments)
                all_results.append({"id": reel_id, "url": reel_url, "info": info, "text": text})
                slot.markdown(
                    f'<div class="reel-card">'
                    f'  <div class="reel-card-header">'
                    f'    <span class="id">{reel_id}</span>'
                    f'    <span>{info.language.upper()} · {info.duration:.1f}s</span>'
                    f'  </div>'
                    f'  <div class="reel-card-body">{text}</div>'
                    f'</div>',
                    unsafe_allow_html=True,
                )
                audio_path.unlink(missing_ok=True)
            except Exception as e:
                slot.markdown(
                    f'<div class="reel-card error">'
                    f'  <div class="reel-card-header">'
                    f'    <span class="id">{reel_id}</span>'
                    f'    <span>FAILED</span>'
                    f'  </div>'
                    f'  <div class="reel-card-body">{e}</div>'
                    f'</div>',
                    unsafe_allow_html=True,
                )

        if all_results:
            combined = "\n\n".join(
                f"=== {r['id']} ({r['info'].language.upper()} · {r['info'].duration:.1f}s) ==="
                f"\n{r['url']}\n\n{r['text']}"
                for r in all_results
            )
            st.download_button(
                "↓ Download all .txt",
                combined,
                file_name="batch_transcripts.txt",
                mime="text/plain",
            )
    st.stop()

if go:
    with tempfile.TemporaryDirectory() as td:
        tmp_dir = Path(td)

        with st.spinner("FETCHING AUDIO…"):
            try:
                audio_path = download_audio(url, tmp_dir, browser=selected_browser)
            except Exception as e:
                st.error(f"DOWNLOAD FAILED — {e}")
                st.stop()

        with st.spinner("TRANSCRIBING…"):
            segments, info = transcribe(audio_path)

        diarization = None
        if diarize_enabled:
            with st.spinner("IDENTIFYING SPEAKERS…"):
                try:
                    diarization = load_diarizer()(str(audio_path))
                except Exception as e:
                    st.warning(f"Diarization failed, falling back to plain transcript: {e}")

    # ── result header ──
    st.markdown(
        f'<div class="result-header">'
        f'<span class="result-label">Result / Transcript</span>'
        f'<span class="result-meta">{info.language.upper()} · {info.language_probability:.0%} · {info.duration:.1f}s</span>'
        f'</div>',
        unsafe_allow_html=True,
    )

    full_text = " ".join(s.text.strip() for s in segments)

    if diarization is not None:
        labelled = assign_speakers(segments, diarization)
        rows = []
        for seg in labelled:
            fg, bg = speaker_colors(seg["speaker"])
            rows.append(
                f'<div class="seg-row">'
                f'  <div class="speaker-block" style="color:{fg};background:{bg};">{seg["speaker"]}</div>'
                f'  <div class="time-block">{fmt_time(seg["start"])}→{fmt_time(seg["end"])}</div>'
                f'  <div class="text-block">{seg["text"]}</div>'
                f'</div>'
            )
        st.markdown(
            '<div class="transcript-table">' + "".join(rows) + "</div>",
            unsafe_allow_html=True,
        )

        export = "\n".join(
            f"[{fmt_time(s['start'])}→{fmt_time(s['end'])}] {s['speaker']}: {s['text']}"
            for s in labelled
        )

        with st.expander("PLAIN TEXT"):
            st.text(full_text)

    else:
        rows = []
        for s in segments:
            rows.append(
                f'<div class="seg-row">'
                f'  <div class="time-block" style="border-left:0;">{fmt_time(s.start)}→{fmt_time(s.end)}</div>'
                f'  <div class="text-block">{s.text.strip()}</div>'
                f'</div>'
            )
        st.markdown(
            '<div class="transcript-table">' + "".join(rows) + "</div>",
            unsafe_allow_html=True,
        )
        export = full_text

    st.download_button(
        "↓ Download .txt",
        export,
        file_name="transcript.txt",
        mime="text/plain",
    )
