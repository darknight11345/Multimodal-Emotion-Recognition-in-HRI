# fused_gui.py
import threading
import time
from io import BytesIO

import cv2
import numpy as np
import sounddevice as sd
import soundfile as sf
import librosa
import noisereduce as nr
from pydub import AudioSegment, effects
import tkinter as tk
from tkinter import ttk

from keras.models import model_from_json

# -------------------------------
# Settings
# -------------------------------
AUDIO_RATE = 16000           # standard sample rate for speech models
AUDIO_RECORD_SECONDS = 2.0   # record length used for each audio inference
AUDIO_HOP = 512
AUDIO_FRAME = 2048

VIDEO_TARGET_SIZE = (48, 48)
FUSION_WEIGHTS = (0.6, 0.4)  # (video_weight, audio_weight)
audio_threshold = 0.01


# -------------------------------
# Load pretrained models
# -------------------------------
# Update the paths if needed
VIDEO_JSON_PATH = r"D:\ulm\Course Cogsys\Sem 4\AI for Auto\Realtime_video\emotiondetector.json"
VIDEO_WEIGHTS_PATH = r"D:\ulm\Course Cogsys\Sem 4\AI for Auto\Realtime_video\emotiondetector.h5"
AUDIO_JSON_PATH = r'D:\ulm\Course Cogsys\Sem 4\AI for Auto\Model\model17082025.json'
AUDIO_WEIGHTS_PATH = r'D:\ulm\Course Cogsys\Sem 4\AI for Auto\Model\model17082025.weights.h5'


face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")



print("Loading models...")
with open(VIDEO_JSON_PATH, "r") as f:
    video_json = f.read()
video_model = model_from_json(video_json)
video_model.load_weights(VIDEO_WEIGHTS_PATH)

with open(AUDIO_JSON_PATH, "r") as f:
    audio_json = f.read()
audio_model = model_from_json(audio_json)
audio_model.load_weights(AUDIO_WEIGHTS_PATH)
audio_model.compile(loss='categorical_crossentropy', optimizer='RMSProp', metrics=['categorical_accuracy'])
print("Models loaded.")

# -------------------------------
# Labels
# -------------------------------
video_labels = {0: 'angry', 1: 'disgust', 2: 'fear', 3: 'happy', 4: 'neutral', 5: 'sad', 6: 'surprise'}
audio_labels = {0: 'neutral', 1: 'calm', 2: 'happy', 3: 'sad', 4: 'angry', 5: 'fearful', 6: 'disgust', 7: 'surprise'}
# use list of video labels (order) for GUI bars
labels = [video_labels[i] for i in range(len(video_labels))]

def detect_face(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
    if len(faces) > 0:
        # pick the first detected face
        x, y, w, h = faces[0]
        return True, (x, y, w, h)
    return False, None


# -------------------------------
# Audio preprocessing
# -------------------------------
def preprocess_audio_array(arr: np.ndarray, sr: int = AUDIO_RATE,
                           frame_length=AUDIO_FRAME, hop_length=AUDIO_HOP):
    """
    arr: 1-D float32 numpy array in range [-1, 1]
    returns: model-ready array shape (1, timesteps, features) per your audio model
    """
    # 1) normalize amplitude using pydub (optional)
    try:
        # pydub expects integer PCM; convert floats to int16 temporarily
        pcm16 = (arr * 32767).astype(np.int16)
        seg = AudioSegment(pcm16.tobytes(), frame_rate=sr, sample_width=2, channels=1)
        seg = effects.normalize(seg, headroom=5.0)
        arr2 = np.array(seg.get_array_of_samples()).astype('float32') / 32767.0
    except Exception:
        # fallback if pydub fails - just use input
        arr2 = arr

    # 2) reduce noise
    try:
        arr_nr = nr.reduce_noise(y=arr2, sr=sr)
    except Exception:
        arr_nr = arr2

    # 3) compute features
    f1 = librosa.feature.rms(y=arr_nr, frame_length=frame_length, hop_length=hop_length, center=True).T
    f2 = librosa.feature.zero_crossing_rate(arr_nr, frame_length=frame_length, hop_length=hop_length, center=True).T
    f3 = librosa.feature.mfcc(y=arr_nr, sr=sr, n_mfcc=13, hop_length=hop_length).T

    # ensure shapes are compatible
    try:
        X = np.concatenate((f1, f2, f3), axis=1)
    except Exception as e:
        # if concatenation fails because of small audio, pad with zeros
        min_len = min(f1.shape[0], f2.shape[0], f3.shape[0]) if (f1.size and f2.size and f3.size) else 0
        if min_len == 0:
            # produce a single-frame zero array with expected feature size
            feat_dim = (1 + 1 + 13)
            X = np.zeros((1, feat_dim), dtype=np.float32)
        else:
            f1s = f1[:min_len]
            f2s = f2[:min_len]
            f3s = f3[:min_len]
            X = np.concatenate((f1s, f2s, f3s), axis=1)

    X = np.expand_dims(X, axis=0)  # (1, timesteps, features)
    # Silence detection based on average RMS energy
    mean_rms = np.mean(f1)  # f1 is already frame-wise RMS values
    silent = mean_rms < audio_threshold
    return X.astype('float32'),silent

# -------------------------------
# Fusion mapping
# -------------------------------
def map_audio_to_video_labels(pred_audio_raw):
    """
    pred_audio_raw: shape (1,8) audio model probabilities
    returns: shape (1,7) mapped probabilities aligned to video label order:
      [angry, disgust, fear, happy, neutral, sad, surprise]
    mapping used from your earlier code but corrected for label names/ordering
    """
    a = pred_audio_raw[0]
    mapped = np.array([
        a[4],                    # angry
        a[6],                    # disgust
        a[5],                    # fearful -> fear
        a[2],                    # happy
        a[0] + 0.3 * a[1],       # neutral + small calm contribution
        a[3],                    # sad
        a[7]                     # surprise
    ], dtype='float32').reshape(1, 7)
    # normalize to sum 1 (if nonzero)
    s = mapped.sum()
    if s > 0:
        mapped = mapped / s
    return mapped

def late_fusion(pred_video, pred_audio_mapped,silent_detect, w_video=FUSION_WEIGHTS[0], w_audio=FUSION_WEIGHTS[1]):
    if silent_detect:
        w_audio=0.05
        w_video=0.95
    else:
        w_audio=0.9
        w_video=0.1

    fused = w_video * pred_video + w_audio * pred_audio_mapped
    print(pred_video,pred_audio_mapped,fused)
    
    # normalize
    s = fused.sum(axis=1, keepdims=True)
    s[s == 0] = 1.0
    fused = fused / s
    return fused

# -------------------------------
# Audio capture thread (non-blocking)
# -------------------------------
class AudioWorker(threading.Thread):
    def __init__(self):
        super().__init__(daemon=True)
        self.latest_audio_mapped = np.zeros((1, len(labels)), dtype='float32')
        self.audio_silence_detection = 0
        self.running = True

    def run(self):
        while self.running:
            try:
                recording = sd.rec(int(AUDIO_RECORD_SECONDS * AUDIO_RATE), samplerate=AUDIO_RATE,
                                   channels=1, dtype='float32')
                sd.wait()
                audio = recording.flatten()
                # preprocess & predict
                X,silent  = preprocess_audio_array(audio, sr=AUDIO_RATE)
                pred_audio_raw = audio_model.predict(X, verbose=0)  # (1,8)
                mapped = map_audio_to_video_labels(pred_audio_raw)  # (1,7)
                self.latest_audio_mapped = mapped
                self.audio_silence_detection = silent
            except Exception as e:
                print("Audio worker error:", e)
                time.sleep(0.1)

    def stop(self):
        self.running = False

# -------------------------------
# Video + GUI
# -------------------------------
class FusionApp:
    def __init__(self, root):
        self.root = root
        root.title("Emotion Recognition - Fused Output")
        root.geometry("600x420")
        self.progress_bars = {}
        for i, label in enumerate(labels):
            tk.Label(root, text=label, font=("Arial", 11)).grid(row=i, column=0, padx=8, pady=4, sticky="w")
            pb = ttk.Progressbar(root, orient="horizontal", length=420, mode="determinate", maximum=100)
            pb.grid(row=i, column=1, padx=8, pady=4)
            self.progress_bars[label] = pb

        self.emotion_var = tk.StringVar(value="Fused Emotion: -")
        tk.Label(root, textvariable=self.emotion_var, font=("Arial", 14), fg="green").grid(row=len(labels), column=0, columnspan=2, pady=10)

        # Video capture
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            raise RuntimeError("Cannot open camera")

        # latest predictions
        self.latest_video_pred = np.zeros((1, len(labels)), dtype='float32')
        self.audio_worker = AudioWorker()
        self.audio_worker.start()

        # Start update loop
        self._stop = False
        self.update_loop()

        # Bind close
        root.protocol("WM_DELETE_WINDOW", self.on_close)

    def on_close(self):
        self._stop = True
        self.audio_worker.stop()
        # release camera
        try:
            self.cap.release()
        except Exception:
            pass
        self.root.after(100, self.root.destroy)

    def update_loop(self):
        if self._stop:
            return

        ret, frame = self.cap.read()
        if ret:
            # preprocess face
            # detect face first
            face_detected,face_coords  = detect_face(frame)
            if face_detected:
                try:
                    x, y, w, h = face_coords
                    # Draw bounding box on the frame
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)   
                    # Crop and preprocess face for video model
                    face_roi = frame[y:y+h, x:x+w]  
                    face = cv2.resize(face_roi, VIDEO_TARGET_SIZE)
                    face_gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY).astype('float32') / 255.0
                    face_in = np.expand_dims(face_gray, axis=(0, -1))  # (1,48,48,1)
                    pred_video = video_model.predict(face_in, verbose=0)  # (1,7)
                    self.latest_video_pred = pred_video.astype('float32')
                except Exception as e:
                    # fallback: keep previous
                    print("Video preprocessing/pred error:", e)
                    pred_video = self.latest_video_pred
            else:
                pred_video = np.zeros((1, len(labels)), dtype='float32')
            # fuse with latest audio
            audio_mapped = self.audio_worker.latest_audio_mapped
            silent_detect=self.audio_worker.audio_silence_detection
            # determine fused prediction
            if face_detected:
                # normal fusion (consider audio silence)
                fused = late_fusion(self.latest_video_pred, audio_mapped, silent_detect)
            elif not face_detected and not silent_detect:
                fused = late_fusion(self.latest_video_pred,audio_mapped, silent_detect)
            else:
                # both missing: unknown
                fused = np.zeros((1, len(labels)), dtype='float32')

            # update GUI bars
            for i, label in enumerate(labels):
                value = float(fused[0][i]) * 100.0
                self.progress_bars[label]["value"] = value
            if fused.sum() > 0:
                final_label = labels[int(np.argmax(fused))]
                self.emotion_var.set(f"Fused Emotion: {final_label} ({(100*np.max(fused)): .1f}%)")
            else:
                final_label = "Unknown"
                self.emotion_var.set("Fused Emotion: Unknown")
            # show a small OpenCV window with the camera and fused label
            disp = frame.copy()
            cv2.putText(disp, f"{final_label}", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
            
            cv2.imshow("Camera (press q to quit)", disp)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                self.on_close()
                return

        # schedule next update
        self.root.after(50, self.update_loop)  # ~20 FPS GUI updates

# -------------------------------
# Run app
# -------------------------------
if __name__ == "__main__":
    root = tk.Tk()
    app = FusionApp(root)

    try:
        root.mainloop()
    finally:
        try:
            cv2.destroyAllWindows()
        except Exception:
            pass
        print("App closed.")
