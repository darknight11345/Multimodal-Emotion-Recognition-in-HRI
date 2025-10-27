# Real-Time Audio-Video Emotion Recognition

This repository contains scripts and notebooks (`.py` and `.ipynb`) to analyze audio and video streams in real-time, detect the dominant expressed emotion, and display it through an interactive interface.

## Repository Overview
- **train_audio** (`.py` / `.ipynb`): Code to train the model for analyzing audio inputs.  
- **train_video** (`.py` / `.ipynb`): Code to train the model for analyzing video inputs.  

Once the models are trained, they can be used in the respective **realtime_audio** and **realtime_video** scripts/notebooks to process features captured from the user’s microphone and webcam in real-time.  

- **GUI + latefusion** (`.py` / `.ipynb`): Combines audio and video predictions, applies a late fusion strategy, and displays the predicted emotion dynamically in a GUI. This multimodal approach provides more stable and accurate predictions compared to single-modality models.

---

## Getting Started / Usage

### 1. Clone the Repository
```bash
git clone https://github.com/yourusername/your-repo-name.git
cd your-repo-name
