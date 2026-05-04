# Reel Transcription

Transcribe Instagram Reels locally — paste a URL, get the words back, with optional speaker identification. No transcription API keys required.

![Streamlit](https://img.shields.io/badge/Streamlit-1.50-FF4B4B?logo=streamlit&logoColor=white)
![Python](https://img.shields.io/badge/Python-3.9+-3776AB?logo=python&logoColor=white)

## What it does

- Downloads audio from Instagram Reels (public, or private with browser cookies)
- Transcribes locally with [faster-whisper](https://github.com/SYSTRAN/faster-whisper) — no API calls, no per-minute cost
- Optionally identifies speakers with [pyannote.audio](https://github.com/pyannote/pyannote-audio)
- Renders results in a Streamlit UI with timestamped, speaker-labelled segments

## Requirements

- Python 3.9+
- ~2 GB free disk for model weights (downloaded automatically on first run)

You don't need to install ffmpeg system-wide — the project bundles it via `imageio-ffmpeg`.

## Setup

```bash
git clone https://github.com/ChestnutManju/reel-transcription.git
cd reel-transcription
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run

```bash
streamlit run transcribe.py
```

Open <http://localhost:8501>.

The first transcription will take longer because Whisper's model weights (~460 MB for the default `small` model) and pyannote's diarization weights (~100 MB) download into your Hugging Face cache. Subsequent runs are fast.

## Speaker diarization (optional)

To label "who said what", you need a free Hugging Face token and acceptance of two model licenses.

1. Create a token at <https://huggingface.co/settings/tokens> (read access is enough).
2. Open each link below and click **Agree and access repository**:
   - <https://huggingface.co/pyannote/speaker-diarization-3.1>
   - <https://huggingface.co/pyannote/segmentation-3.0>
3. Create a `.env` file in the project root:
   ```
   HF_TOKEN=hf_your_token_here
   ```

Without a token, transcription still works — speakers just aren't identified.

## Private reels (optional)

Public reels work without auth. For private or login-gated reels, the app reads cookies from a browser where you're logged into Instagram.

In the UI, click the browser you use under **BROWSER** — that browser must be installed and logged into Instagram. Or set a default in `.env`:

```
IG_COOKIES_BROWSER=chrome   # or firefox, safari, edge, brave, opera
```

Or use a Netscape-format cookies file (e.g. exported via the [Get cookies.txt LOCALLY](https://chromewebstore.google.com/detail/get-cookiestxt-locally/cclelndahbckbenkjhflpdbgdldlbecc) extension):

```
IG_COOKIES_FILE=/path/to/cookies.txt
```

## Configuration

All configuration is via environment variables (loaded from `.env` if present):

| Variable             | Purpose                                                   | Default |
| -------------------- | --------------------------------------------------------- | ------- |
| `WHISPER_MODEL`      | Whisper model size: `tiny`, `base`, `small`, `medium`, `large-v3` | `small` |
| `HF_TOKEN`           | Hugging Face token, required for diarization              | unset   |
| `IG_COOKIES_BROWSER` | Default browser for Instagram cookies                     | unset   |
| `IG_COOKIES_FILE`    | Path to a Netscape cookies file                           | unset   |

Bigger Whisper models give better accuracy at the cost of speed and RAM. `small` is a good default for short reels; `large-v3` is roughly 3 GB and noticeably slower without a GPU.

## How it works

1. **Download** — `yt-dlp` fetches the best audio stream from the Reel URL.
2. **Convert** — the bundled `ffmpeg` re-encodes the audio to 16 kHz mono WAV (required by pyannote's `libsndfile` reader).
3. **Transcribe** — `faster-whisper` runs the chosen Whisper model locally and returns segments with timestamps and language detection.
4. **Diarize** (optional) — pyannote's `speaker-diarization-3.1` pipeline finds speaker turns; each Whisper segment is then assigned to the speaker with the most overlap.
5. **Render** — segments display in a Streamlit table with per-speaker color blocks.

## Tech stack

- [Streamlit](https://streamlit.io) — UI
- [yt-dlp](https://github.com/yt-dlp/yt-dlp) — audio downloader
- [faster-whisper](https://github.com/SYSTRAN/faster-whisper) — local Whisper inference via CTranslate2
- [pyannote.audio](https://github.com/pyannote/pyannote-audio) — speaker diarization
- [imageio-ffmpeg](https://github.com/imageio/imageio-ffmpeg) — bundled ffmpeg binary

## License

MIT
