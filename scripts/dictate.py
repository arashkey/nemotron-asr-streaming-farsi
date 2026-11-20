# /// script
# requires-python = ">=3.10"
# dependencies = ["transformers==5.18.0", "torch", "numpy", "sounddevice", "keyboard", "pyperclip", "soxr"]
# ///
"""دیکته‌ی فارسی در هر برنامه‌ی ویندوز.

    uv run dictate.py                       # F9 را نگه دارید، حرف بزنید، رها کنید
    uv run dictate.py --hotkey f8           # کلید دیگر
    uv run dictate.py --lookahead 0.3       # سریع‌تر، کمی کم‌دقت‌تر
    uv run dictate.py --device cpu          # اجبار به CPU
    uv run dictate.py --threads 8           # تعداد هسته‌های CPU برای مدل

خروج: Ctrl+Alt+Q
"""

import argparse
import os
import sys
import threading
import time
import warnings
import winsound

import numpy as np

warnings.filterwarnings("ignore", message=".*max_length.*")
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

SR = 16000
DEFAULT_MODEL = "mehdi-hf/nemotron-asr-streaming-farsi"
LOOKAHEAD = {"1.1": 13, "0.6": 6, "0.3": 3}


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--hotkey", default="f9", help="کلید نگه‌داشتن برای ضبط")
    ap.add_argument("--quit-hotkey", default="ctrl+alt+q", help="کلید خروج")
    ap.add_argument("--max-seconds", type=float, default=30.0, help="حداکثر طول هر ضبط")
    ap.add_argument("--lookahead", default="1.1", choices=list(LOOKAHEAD))
    ap.add_argument("--device", default="auto", help="auto, cpu یا xpu (گرافیک Intel)")
    ap.add_argument("--threads", type=int, default=0, help="تعداد thread برای CPU (۰ = پیش‌فرض)")
    ap.add_argument("--model", default=DEFAULT_MODEL, help="شناسه‌ی Hugging Face یا پوشه‌ی محلی")
    ap.add_argument("--no-beep", action="store_true", help="بدون بوق شروع و پایان")
    return ap.parse_args()


def beep(freq: int, ms: int, enabled: bool) -> None:
    if enabled:
        threading.Thread(target=winsound.Beep, args=(freq, ms), daemon=True).start()


class Recorder:
    """میکروفون همیشه باز است و فقط هنگام نگه‌داشتن کلید صدا نگه داشته می‌شود،
    تا اولین هجای جمله از دست نرود."""

    def __init__(self):
        import sounddevice as sd

        info = sd.query_devices(kind="input")
        self.name = info["name"]
        self.in_sr = int(info["default_samplerate"])
        self.frames: list[np.ndarray] = []
        self.lock = threading.Lock()
        self.active = False
        self.stream = sd.InputStream(samplerate=self.in_sr, channels=1, dtype="float32", callback=self._callback)
        self.stream.start()

    def _callback(self, indata, frames, time_info, status) -> None:
        if self.active:
            with self.lock:
                self.frames.append(indata[:, 0].copy())

    def start(self) -> None:
        with self.lock:
            self.frames = []
        self.active = True

    def stop(self) -> np.ndarray:
        self.active = False
        with self.lock:
            audio = np.concatenate(self.frames) if self.frames else np.zeros(0, np.float32)
            self.frames = []
        if len(audio) and self.in_sr != SR:
            import soxr
            audio = soxr.resample(audio, self.in_sr, SR)
        return audio.astype(np.float32)


def transcribe(model, processor, audio: np.ndarray) -> str:
    import torch

    inputs = processor(audio, sampling_rate=SR, language="fa-IR").to(model.device, dtype=model.dtype)
    with torch.inference_mode():
        out = model.generate(**inputs, return_dict_in_generate=True)
    return processor.batch_decode(out.sequences, skip_special_tokens=True)[0].strip()


def load_model(name: str, device_pref: str, lookahead: int, threads: int):
    import torch
    from transformers import AutoModelForRNNT, AutoProcessor
    from transformers.utils import logging as hf_logging

    hf_logging.set_verbosity_error()
    if threads > 0:
        torch.set_num_threads(threads)

    if device_pref == "auto":
        candidates = []
        if hasattr(torch, "xpu") and torch.xpu.is_available():
            candidates.append("xpu")
        candidates.append("cpu")
    else:
        candidates = [device_pref]

    print(f"Loading {name} ...", flush=True)
    processor = AutoProcessor.from_pretrained(name)
    processor.set_num_lookahead_tokens(lookahead)
    model = AutoModelForRNNT.from_pretrained(name).eval()

    for dev in candidates:
        try:
            model = model.to(dev)
            transcribe(model, processor, np.zeros(SR, np.float32))  # گرم کردن
            t = time.perf_counter()
            transcribe(model, processor, np.zeros(SR * 3, np.float32))
            print(f"Device: {dev} (3 s of audio in {time.perf_counter() - t:.2f} s)", flush=True)
            return model, processor
        except Exception as e:
            print(f"{dev} failed ({type(e).__name__}: {e}); trying next device", flush=True)
    sys.exit("No device could run the model.")


def paste(text: str) -> None:
    """چسباندن از طریق کلیپ‌بورد؛ برای فارسی و نیم‌فاصله مطمئن‌تر از شبیه‌سازی تایپ است."""
    import keyboard
    import pyperclip

    try:
        old = pyperclip.paste()
    except Exception:
        old = None
    pyperclip.copy(text)
    time.sleep(0.05)
    keyboard.send("ctrl+v")
    time.sleep(0.3)
    if old is not None:
        try:
            pyperclip.copy(old)
        except Exception:
            pass


def main() -> None:
    import keyboard

    a = parse_args()
    sound = not a.no_beep
    model, processor = load_model(a.model, a.device, LOOKAHEAD[a.lookahead], a.threads)
    rec = Recorder()
    keyboard.add_hotkey(a.quit_hotkey, lambda: os._exit(0))

    print(f"Microphone: {rec.name} ({rec.in_sr} Hz)")
    print(f"Ready. Hold {a.hotkey.upper()} and speak Persian. Quit: {a.quit_hotkey}\n", flush=True)

    while True:
        keyboard.wait(a.hotkey)
        rec.start()
        beep(880, 80, sound)
        t0 = time.time()
        while keyboard.is_pressed(a.hotkey) and time.time() - t0 < a.max_seconds:
            time.sleep(0.02)
        audio = rec.stop()
        beep(440, 80, sound)

        if len(audio) >= SR * 0.3:  # کوتاه‌تر از ۰.۳ ثانیه نادیده گرفته می‌شود
            t = time.perf_counter()
            try:
                text = transcribe(model, processor, audio)
            except Exception as e:
                print(f"error: {e}", flush=True)
                text = ""
            print(f"[{len(audio) / SR:.1f} s audio -> {time.perf_counter() - t:.2f} s] {text}", flush=True)
            if text:
                paste(text + " ")

        while keyboard.is_pressed(a.hotkey):  # اگر به سقف زمان رسید، تا رها شدن کلید صبر کن
            time.sleep(0.05)


if __name__ == "__main__":
    main()
