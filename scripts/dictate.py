# /// script
# requires-python = ">=3.10"
# dependencies = ["transformers==5.18.0", "torch", "numpy", "sounddevice", "keyboard", "pyperclip"]
# ///
"""دیکته‌ی فارسی در هر برنامه: کلید را نگه دار، حرف بزن، رها کن."""
import threading, time
import numpy as np, sounddevice as sd, keyboard, pyperclip, torch
from transformers import AutoModelForRNNT, AutoProcessor

MODEL = "mehdi-hf/nemotron-asr-streaming-farsi"
HOTKEY = "f9"        # کلیدی که در برنامه‌هایتان کاربردی ندارد
SR = 16000
MAX_SECONDS = 30     # حداکثر طول هر ضبط

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Loading model on {device}...")
processor = AutoProcessor.from_pretrained(MODEL)
model = AutoModelForRNNT.from_pretrained(MODEL).to(device).eval()
processor.set_num_lookahead_tokens(13)   # دقیق‌ترین حالت

def transcribe(audio: np.ndarray) -> str:
    inputs = processor(audio, sampling_rate=SR, language="fa-IR").to(device, dtype=model.dtype)
    with torch.inference_mode():
        out = model.generate(**inputs, return_dict_in_generate=True)
    return processor.batch_decode(out.sequences, skip_special_tokens=True)[0].strip()

transcribe(np.zeros(SR, np.float32))     # گرم کردن مدل تا اولین بار کند نباشد

frames, recording = [], threading.Event()
def on_audio(indata, n, t, status):
    if recording.is_set():
        frames.append(indata[:, 0].copy())
sd.InputStream(samplerate=SR, channels=1, dtype="float32", callback=on_audio).start()

def paste(text: str) -> None:
    try:
        old = pyperclip.paste()
    except Exception:
        old = None
    pyperclip.copy(text)
    time.sleep(0.05)
    keyboard.send("ctrl+v")
    time.sleep(0.3)
    if old is not None:
        pyperclip.copy(old)              # کلیپ‌بورد قبلی برمی‌گردد

print(f"Ready. Hold {HOTKEY.upper()} and speak Persian.")
while True:
    keyboard.wait(HOTKEY)
    frames.clear(); recording.set(); t0 = time.time()
    while keyboard.is_pressed(HOTKEY) and time.time() - t0 < MAX_SECONDS:
        time.sleep(0.02)
    recording.clear()
    if not frames:
        continue
    audio = np.concatenate(frames)
    if len(audio) < SR * 0.3:            # کمتر از ۰.۳ ثانیه: نادیده بگیر
        continue
    t = time.time()
    text = transcribe(audio)
    print(f"[{time.time() - t:.1f}s] {text}")
    if text:
        paste(text + " ")
