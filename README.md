# MoneyPrinterTurbo - Enhanced Fork

This is an enhanced version of [MoneyPrinterTurbo](https://github.com/harry0703/MoneyPrinterTurbo) with significant improvements to subtitle highlighting and TTS capabilities. Full credit goes to the original author and contributors.

## What's Different in This Fork

### Enhanced Subtitle System
- **Word-by-word highlighting**: Each word lights up exactly when spoken, making videos more engaging
- **Real-time synchronization**: Perfect timing with TTS word boundaries
- **Multi-line support**: Works with wrapped text and complex subtitle layouts
- **Customizable colors**: Configure highlight colors through the web interface

### Better Video-Text Matching
- **Semantic search**: Analyzes script content to find relevant video clips instead of random selection
- **Text similarity**: Matches video content to script meaning for better relevance
- **Thumbnail analysis**: Optional video thumbnail similarity for sources like Pexels 

### Open-Source TTS with Voice Cloning
This fork includes **Chatterbox TTS** - a completely free alternative to Azure TTS that runs locally on your machine.

**Key advantages:**
- **No API costs**: Completely free to use, no rate limits
- **Voice cloning**: Clone any voice using 10-60 seconds of reference audio
- **Word-level timing**: Perfect subtitle synchronization with WhisperX integration
- **Automatic speed control**: Configurable speech pacing via environment variables



## Example Videos

See the enhanced features in action:

**Full-Length Video Example**

[![MoneyPrinterTurbo Example Video](https://img.youtube.com/vi/yXc07ROgj80/maxresdefault.jpg)](https://www.youtube.com/watch?v=yXc07ROgj80)

**YouTube Shorts Example**  

[![MoneyPrinterTurbo Shorts Example](https://img.youtube.com/vi/JBAuXpVHt40/maxresdefault.jpg)](https://www.youtube.com/shorts/JBAuXpVHt40)

**Chatterbox TTS Generated Video**  

[![MoneyPrinterTurbo Chatterbox Example](https://img.youtube.com/vi/ZAttF-cVce8/maxresdefault.jpg)](https://youtube.com/shorts/ZAttF-cVce8?feature=share)

> **Features Showcased**: Natural voice synthesis • Word-level subtitle highlighting • Timing synchronization • Open-source TTS quality

## 🖼️ Screenshots - Video Generation Setup

For complete tranparency and some reprodceability, please see below settings used to generate videos shown above

<div align="center">
<img src="docs/ui_config_1.png" alt="Main Interface" width="800"/>

<img src="docs/ui_config_2.png" alt="Voice Settings" width="800"/>
</div>



## Show Me The Prompt

Here's the exact prompt system we use for generating engaging YouTube content:

<details>
<summary><strong>Complete Video Generation Prompt For LLMs of your choice(Click to expand)</strong></summary>

```
ROLE: You are an expert YouTube scriptwriter and content strategist specializing in creating engaging, science-backed content for a broad audience.

OBJECTIVE: Generate a complete text-based content package for a 5-minute YouTube video. The goal is to select a single, highly engaging topic and create all the necessary assets to produce the video, optimized for audience retention and YouTube's algorithm.

TOPIC SELECTION CRITERIA:
• Trending & Relevant: The topic must have high current interest and search volume
• Broad Appeal: Relatable to a wide audience (productivity, health, personal finance, psychology)
• Science-Based: Grounded in widely accepted, mainstream scientific consensus
• Safe & Non-Controversial: Focus on foundational, actionable knowledge

REQUIRED DELIVERABLES:

1. Video Title Options (3x)
   Goal: Create three distinct, clickable YouTube titles optimized for high CTR
   Style Example: "Rewire Your Anxious Brain in 3 Simple Steps"

2. Full Video Script
   Length: 800-900 words (~5-minute speaking time)
   Format: Single paragraph with proper punctuation for TTS optimization
   Tone: Authoritative yet encouraging, digestible for general audience
   TTS Optimization: End sentences with definitive punctuation for natural breaks

3. Pexels Video Search Keywords
   Structure: Keywords organized by script concepts for visual variety
   Output: Single line separated by commas
   Example: brain animation, neural network, person thinking, scrolling on phone

4. YouTube Description & Hashtags
   Description: SEO-optimized summary (2-3 lines) with clear call-to-action
   Hashtags: 10-15 relevant hashtags for maximum discoverability
```
</details>

##  Installation

**Quick Start For Windows (Recommended):**

```bash
# 1. Clone
git clone https://github.com/Asad-Ismail/MoneyPrinterTurbo-Extended.git
cd MoneyPrinterTurbo

# 2. One-click setup + launch
run.bat
```

The web interface opens at `http://localhost:8501`

**What `run.bat` does**

- detects Python and creates `.venv` automatically
- installs `requirements.txt` on first run
- creates `config.toml` from `config.example.toml` if needed
- applies safe defaults for this PC: `cpu-safe`, `ja`, `ja-JP`, `voice clone off`
- launches the Streamlit WebUI

**What happens on the second run**

- existing `.venv` is reused
- dependency install is skipped unless you delete `.venv`
- your existing `config.toml` is kept and only missing safe defaults are filled

**Manual / advanced setup**

```bash
# Linux / macOS Web Interface
./webui.sh

# Optional: conda-based environment
conda env create -f environment.yml
conda activate MoneyPrinterTurbo
pip install -r requirements.txt

# Existing environments: refresh edge_tts after pulling changes
pip install -U edge_tts==7.2.8

# Optional: Install Chatterbox only when you want local TTS experimentation
git clone https://github.com/resemble-ai/chatterbox.git
cd chatterbox && pip install -e . && cd ..

# Optional: CUDA / GPU setup
pip install -r requirements-cuda.txt
source ./setup_cuda_env.sh

# Optional: Customize speech speed when using Chatterbox
export CHATTERBOX_CFG_WEIGHT=0.1  # Very slow
export CHATTERBOX_CFG_WEIGHT=0.2  # Slow (default)
export CHATTERBOX_CFG_WEIGHT=0.3  # Normal speed
```

**Windows helpers**

- `run.bat`: recommended entrypoint for setup + launch
- `webui.bat`: compatibility launcher, now delegates to `run.bat`

If you need a proxy for outbound services, add it to the `[proxy]` section in `config.toml`.
The same proxy settings are used for Pexels/Pixabay requests and `azure-tts-v1` (`edge_tts`).

## New Runtime Basics

- Use a **preset** first: `shorts-basic`, `youtube-explainer`, `news-summary`
- Save your current UI setup as a **profile** to reuse the same settings later
- Default mode is `cpu-safe`, which keeps Chatterbox and voice cloning off unless you explicitly enable higher-load settings
- LLM providers now support `OpenAI`, `Groq`, and `OpenRouter` through the same UI fields

## Japanese Quickstart

- Set the UI language to `ja` and the script language to `ja-JP` for Japanese-first generation
- In Japanese mode, the app defaults to `ja-JP-NanamiNeural` when your saved voice is still the old English default
- Subtitle density checks become stricter for Japanese so long lines are flagged earlier
- `Groq` and `OpenRouter` work for Japanese script generation through the same LLM settings panel
- Keep `cpu-safe` enabled for the first run; voice cloning remains optional and off by default

## 🔧 Troubleshooting

<details>
<summary><strong>Common Issues & Solutions (Click to expand)</strong></summary>

**Chatterbox TTS issues:**
- **Garbled audio**: Text automatically preprocessed and chunked for clarity
- **CUDA errors**: System automatically falls back to CPU mode
- **Force CPU mode**: `export CHATTERBOX_DEVICE=cpu`
- **Voice cloning problems**: Ensure audio is clear and single-speaker
- **Speed control**: Use `CHATTERBOX_CFG_WEIGHT` environment variable

**CUDA/cuDNN compatibility issues:**
- **Error**: `libcudnn_ops_infer.so.8: cannot open shared object file`
- **Cause**: Missing cuDNN 8.x libraries required by some packages
- **Solution**: Automatically handled by startup scripts (`setup_cuda_env.sh`)
- **Manual fix**: `pip install nvidia-cudnn-cu12==8.9.2.26`

**MoviePy TextClip issues:**
- **Error**: `got an unexpected keyword argument 'align'`
- **Cause**: Newer MoviePy versions removed the `align` parameter
- **Solution**: Remove or comment out `align` parameter in `TextClip` calls

**General issues:**
- If `run.bat` says Python was not found, install Python 3.11+ and retry
- If the first launch fails during pip install, rerun `run.bat` after fixing your network or proxy settings
- Check that all dependencies are installed correctly
- Ensure your Python environment is activated
- For GPU issues, CPU mode provides a reliable fallback
- If `azure-tts-v1` fails with `403 Invalid response status`, upgrade `edge_tts` with `pip install -U edge_tts==7.2.8`
- If your network requires a proxy, set `[proxy].https` or `[proxy].http` in `config.toml`

**Advanced CUDA Setup:**
The project includes automatic CUDA environment configuration:
- `setup_cuda_env.sh` - Shared CUDA environment setup
- `webui.sh` - Web interface with CUDA support

If you encounter CUDA library issues, the startup scripts automatically:
1. Add cuDNN library paths to `LD_LIBRARY_PATH` (Linux) 
2. Set optimal CUDA memory allocation settings

</details>

## Contributions and Support 

If you found this project useful please give it a star and consider contributing to it or open an issue if you have an idea that can make it more useful.

## Original Project Credits

This fork maintains full compatibility with the original MoneyPrinterTurbo while adding new features. Check out the [original repository](https://github.com/harry0703/MoneyPrinterTurbo) for the base project documentation and additional features.
