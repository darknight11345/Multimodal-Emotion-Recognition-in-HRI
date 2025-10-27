# Real-Time Audio-Video Emotion Recognition

This repository contains scripts and notebooks (`.py` and `.ipynb`) to analyze audio and video streams in real-time, detect the dominant expressed emotion, and display it through an interactive interface.

## Repository Overview
- **train_audio** ( `.ipynb`): Code to train the model for analyzing audio inputs.  
- **train_video** (`.py`): Code to train the model for analyzing video inputs.  

Once the models are trained, they can be used in the respective **realtime_audio** and **realtime_video** scripts/notebooks to process features captured from the user’s microphone and webcam in real-time.  

- **GUI + latefusion** (`.py`): Combines audio and video predictions, applies a late fusion strategy, and displays the predicted emotion dynamically in a GUI. This multimodal approach provides more stable and accurate predictions compared to single-modality models.

---

## Getting Started / Usage

### 1. Clone the Repository
```bash
git clone https://github.com/darknight11345/Multimodal-Emotion-Recognition-in-HRI.git
````
### 2. Install Dependencies
Make sure you have Python 3.8+ installed, then install the required packages:
```bash
pip install -r requirements.txt
```
###3. Train the Models
Audio Model:
```bash
python train_audio.py
```
Video Model:
```bash
python train_video.py
```
##4. Run Real-Time Single Modality Prediction
Audio only:
```bash
python realtime_audio.py
```
Video only:
```bash
python realtime_video.py
```
###5. Run Real-Time Multimodal Prediction with GUI
```bash
python GUI_latefusion.py
```
The GUI will open, capture live webcam and microphone inputs, and display dynamic emotion predictions with progress bars and labels.

- Notes

The fusion performance reported in this project (approx. 0.76 accuracy, 0.74 F1-score) is estimated based on independently trained audio and video models.

Future updates will include evaluation on a synchronized multimodal dataset for more accurate benchmarking.
