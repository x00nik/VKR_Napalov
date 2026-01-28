"""
Модуль для детекции дипфейков в аудио.
Интеграция модели ResNet50 из проекта:
https://github.com/Mardvey-UMA/Neural-Network-for-AudioDeepFake-Detection
"""
from __future__ import annotations

import logging
import os
from typing import Dict, Optional

import librosa
import numpy as np
import soundfile as sf
import tensorflow as tf

logger = logging.getLogger(__name__)

SAMPLE_RATE = 16000
MODEL_INPUT_SIZE = 224
DEFAULT_MODEL_PATH = "models_weights/mardvey_resnet50.h5"
DEFAULT_THRESHOLD = 0.5

try:
    import torch
except Exception:  # pragma: no cover - torch optional for inference inputs
    torch = None


class AudioPreprocessor:
    """Подготовка аудио для ResNet50 (mel-спектрограмма)."""

    def __init__(self, sample_rate: int = SAMPLE_RATE, n_mels: int = 128):
        self.sample_rate = sample_rate
        self.n_mels = n_mels

    def load_audio(self, audio_path: str, max_duration: Optional[float] = None) -> np.ndarray:
        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"Аудио файл не найден: {audio_path}")

        waveform, sr = sf.read(audio_path, dtype="float32", always_2d=False)

        if waveform.size == 0:
            raise ValueError("Аудио файл пустой (0 сэмплов). Проверьте запись.")

        if waveform.ndim == 2:
            waveform = np.mean(waveform, axis=1)

        if sr != self.sample_rate:
            waveform = librosa.resample(
                waveform, orig_sr=sr, target_sr=self.sample_rate
            )

        if max_duration is not None:
            max_samples = int(max_duration * self.sample_rate)
            waveform = waveform[:max_samples]

        if waveform.size == 0:
            raise ValueError("После обработки аудио стало пустым")

        return waveform.astype(np.float32)

    def prepare_for_resnet(self, waveform: np.ndarray) -> np.ndarray:
        mel = librosa.feature.melspectrogram(
            y=waveform,
            sr=self.sample_rate,
            n_mels=self.n_mels,
            n_fft=1024,
            hop_length=320,
            power=2.0,
        )
        mel_db = librosa.power_to_db(mel, ref=np.max)

        mel_min = float(mel_db.min())
        mel_max = float(mel_db.max())
        mel_norm = (mel_db - mel_min) / (mel_max - mel_min + 1e-8)

        mel_img = mel_norm[..., np.newaxis]
        mel_img = tf.image.resize(
            mel_img, (MODEL_INPUT_SIZE, MODEL_INPUT_SIZE), method="bilinear"
        ).numpy()
        mel_img = np.repeat(mel_img, 3, axis=2)

        return mel_img[np.newaxis, ...].astype(np.float32)


class DeepfakeDetector:
    """Детектор на основе ResNet50 (Mardvey-UMA)."""

    def __init__(
        self,
        model_path: str = DEFAULT_MODEL_PATH,
        threshold: float = DEFAULT_THRESHOLD,
    ):
        self.model_path = model_path
        self.threshold = threshold
        self.preprocessor = AudioPreprocessor(SAMPLE_RATE)
        self.model = self._load_model(model_path)

    def _load_model(self, model_path: str) -> tf.keras.Model:
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"Не найден файл модели ResNet50: {model_path}"
            )

        logger.info("Загрузка ResNet50 модели...")
        model = tf.keras.models.load_model(model_path)
        logger.info("✓ ResNet50 модель загружена")
        return model

    def _extract_deepfake_probability(self, raw_pred: np.ndarray) -> float:
        pred = np.asarray(raw_pred)
        if pred.ndim == 0:
            prob = float(pred)
        else:
            pred = pred.reshape(pred.shape[0], -1)
            if pred.shape[1] == 1:
                prob = float(pred[0, 0])
            else:
                probs = pred
                is_prob = (
                    np.all(probs >= 0.0)
                    and np.all(probs <= 1.0)
                    and np.allclose(np.sum(probs, axis=1), 1.0, atol=1e-2)
                )
                if not is_prob:
                    probs = tf.nn.softmax(pred, axis=1).numpy()
                prob = float(probs[0, 1])
        return float(np.clip(prob, 0.0, 1.0))

    def analyze_waveform(self, waveform: np.ndarray) -> Dict:
        if torch is not None and isinstance(waveform, torch.Tensor):
            waveform = waveform.detach().cpu().numpy()

        if waveform.ndim != 1:
            waveform = waveform.reshape(-1)

        x = self.preprocessor.prepare_for_resnet(waveform)
        pred = self.model.predict(x, verbose=0)
        deepfake_prob = self._extract_deepfake_probability(pred)

        is_suspicious = deepfake_prob >= self.threshold
        confidence = max(deepfake_prob, 1.0 - deepfake_prob)

        return {
            "is_suspicious": is_suspicious,
            "is_deepfake": is_suspicious,
            "deepfake_probability": float(deepfake_prob),
            "confidence": float(confidence),
            "models_used": 1,
            "model_predictions": {"resnet50": int(is_suspicious)},
            "model_probabilities": {"resnet50": float(deepfake_prob)},
            "method": "mardvey_resnet50",
        }

    def analyze(self, audio_path: str) -> Dict:
        waveform = self.preprocessor.load_audio(audio_path)
        return self.analyze_waveform(waveform)

    def quick_check(self, audio_path: str, threshold: Optional[float] = None) -> Dict:
        if threshold is not None:
            self.threshold = threshold
        waveform = self.preprocessor.load_audio(audio_path, max_duration=5.0)
        result = self.analyze_waveform(waveform)
        result["method"] = "mardvey_resnet50_quick"
        result["duration_analyzed"] = 5.0
        return result



