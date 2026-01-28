#!/usr/bin/env python3
# train_no_numpy_cuda.py
# Полный пайплайн без NumPy, с поддержкой CUDA, AMP и HuggingFace Trainer fp16.
# Требует: torch, torchaudio, transformers, sklearn
# import transformers
# transformers.utils.import_utils._torch_load_safe_check = lambda *args, **kwargs: None

import os
import argparse
import random
import math
import multiprocessing
from typing import List, Tuple, Dict, Any, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
from torch.utils.data import Dataset, DataLoader

from transformers import (
    Wav2Vec2Processor,
    Wav2Vec2ForSequenceClassification,
    Trainer,
    TrainingArguments,
)
from dataclasses import dataclass
from typing import Dict, List, Union
from sklearn.metrics import accuracy_score

# -------------------------
# Константы / По умолчанию
# -------------------------
DEFAULT_ASV_ROOT = "./LA"
SAMPLE_RATE = 16000
RANDOM_SEED = 42
torch.manual_seed(RANDOM_SEED)
random.seed(RANDOM_SEED)

# -------------------------
# Утилиты для протоколов
# -------------------------
def read_protocol(proto_file: str, audio_dir: str) -> Tuple[List[str], List[int]]:
    paths, labels = [], []
    with open(proto_file, "r", encoding="utf-8") as f:
        for ln in f:
            parts = ln.strip().split()
            if len(parts) < 2:
                continue
            file_id = parts[1]
            key = parts[-1]
            p = os.path.join(audio_dir, file_id + ".flac")
            if os.path.exists(p):
                paths.append(p)
                labels.append(0 if key == "bonafide" else 1)
    return paths, labels

# -------------------------
# Dataset (torch/torchaudio)
# -------------------------
class AudioDatasetTorch(Dataset):
    def __init__(self, paths: List[str], labels: List[int], model_type: str="cnn", max_len_frames: int = 300):
        assert model_type in ("cnn","lstm","wav2vec2")
        self.paths = paths
        self.labels = labels
        self.model_type = model_type
        self.max_len_frames = max_len_frames

        if model_type == "cnn":
            self.mel_transform = torchaudio.transforms.MelSpectrogram(
                sample_rate=SAMPLE_RATE, n_mels=128, n_fft=1024, hop_length=320
            )
        elif model_type == "lstm":
            self.mfcc_transform = torchaudio.transforms.MFCC(
                sample_rate=SAMPLE_RATE, n_mfcc=40, melkwargs={"n_fft":1024, "n_mels":128, "hop_length":320}
            )
        # wav2vec2 will use raw waveform

    def __len__(self):
        return len(self.paths)

    def _load_wave(self, path: str) -> torch.Tensor:
        # Быстрая загрузка без дополнительных проверок
        waveform, sr = torchaudio.load(path, normalize=False)  # waveform shape (channels, samples)
        
        # Оптимизированная обработка
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)  # Быстрее чем torch.mean
        
        if sr != SAMPLE_RATE:
            waveform = torchaudio.functional.resample(
                waveform, 
                orig_freq=sr, 
                new_freq=SAMPLE_RATE,
                resampling_method="sinc_interp_kaiser"  # Быстрый метод
            )
        
        return waveform.squeeze(0)

    def __getitem__(self, idx: int):
        path = self.paths[idx]
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        waveform = self._load_wave(path)  # (samples,)

        if self.model_type == "cnn":
            mel = self.mel_transform(waveform)  # (n_mels, time)
            mel_db = 10.0 * torch.log10(mel + 1e-10)
            # resize to 128x128 using interpolate: expects (B,C,H,W)
            x = mel_db.unsqueeze(0).unsqueeze(0)  # (1,1,n_mels,time)
            x = F.interpolate(x, size=(128,128), mode='bilinear', align_corners=False)
            x = x.squeeze(0)  # (1,128,128)
            return x.float(), label
        elif self.model_type == "lstm":
            mfcc = self.mfcc_transform(waveform)  # (n_mfcc, time)
            mfcc = mfcc.transpose(0,1)  # (time, n_mfcc)
            t = mfcc.shape[0]
            if t < self.max_len_frames:
                pad = torch.zeros(self.max_len_frames - t, mfcc.shape[1], dtype=torch.float32)
                mfcc = torch.cat([mfcc, pad], dim=0)
            else:
                mfcc = mfcc[:self.max_len_frames, :]
            return mfcc.float(), label
        else:
            # wav2vec2: return raw waveform and label (collate will handle)
            return waveform.float(), label

# collate for wav2vec2 when using DataLoader (if needed)
def collate_wav2vec(batch: List[Tuple[torch.Tensor, torch.Tensor]], processor: Wav2Vec2Processor):
    waves = [item[0] for item in batch]
    labels = torch.stack([item[1] for item in batch])
    inputs = processor(waves, sampling_rate=SAMPLE_RATE, return_tensors="pt", padding=True)
    inputs["labels"] = labels
    return inputs

# -------------------------
# Модели: CNN, LSTM
# -------------------------
class CNNClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2)
        # 128x128 -> after two pools => 32x32
        self.fc1 = nn.Linear(32 * 32 * 32, 128)
        self.fc2 = nn.Linear(128, 2)
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)

class LSTMClassifier(nn.Module):
    def __init__(self, input_dim=40, hidden_dim=128, num_layers=2, num_classes=2):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, num_classes)
    def forward(self, x):
        _, (hn, _) = self.lstm(x)
        hn = hn[-1]
        return self.fc(hn)

# -------------------------
# Train / Eval utils (AMP aware)
# -------------------------
def train_epoch_amp(model: nn.Module, dataloader: DataLoader, optimizer: torch.optim.Optimizer,
                    criterion: nn.Module, scaler: torch.amp.GradScaler, device: torch.device) -> float:
    model.train()
    total_loss = 0.0
    batches = 0
    for batch in dataloader:
        X, y = batch
        X = X.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        optimizer.zero_grad()
        with torch.cuda.amp.autocast(enabled=(device.type=="cuda")):
            logits = model(X)
            loss = criterion(logits, y)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        total_loss += loss.item()
        batches += 1
    return total_loss / max(1, batches)

def eval_model(model: nn.Module, dataloader: DataLoader, device: torch.device) -> float:
    model.eval()
    ys, ps = [], []
    with torch.no_grad():
        for X, y in dataloader:
            X = X.to(device)
            logits = model(X)
            preds = torch.argmax(logits, dim=1).cpu()
            if isinstance(y, torch.Tensor):
                ys.extend(y.cpu().numpy())
            else:
                ys.extend(list(y))
            ps.extend(preds.numpy())
    return accuracy_score(ys, ps)

# -------------------------
# Wav2Vec2 Wrap Dataset (на уровне модуля для pickle на Windows)
# -------------------------
class Wav2Vec2WrapDataset(Dataset):
    """Wrapper для AudioDataset - преобразует в формат для Trainer"""
    def __init__(self, audio_dataset: Dataset):
        self.audio_dataset = audio_dataset
    
    def __len__(self):
        return len(self.audio_dataset)
    
    def __getitem__(self, idx):
        waveform, label = self.audio_dataset[idx]
        # Возвращаем сырой waveform - processor будет применен в data_collator
        return {
            "input_values": waveform,
            "labels": label.item() if isinstance(label, torch.Tensor) else label
        }

# -------------------------
# Wav2Vec2 Data Collator
# -------------------------
@dataclass
class DataCollatorCTCWithPadding:
    processor: Wav2Vec2Processor
    padding: Union[bool, str] = True
    max_length: int = 80000  # 5 секунд при 16kHz (для скорости!)
    _first_call: bool = True
    
    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        if self._first_call:
            print(f"[DataCollator] Первый батч: {len(features)} сэмплов...")
            print(f"[DataCollator] Ограничение длины: {self.max_length/SAMPLE_RATE:.1f} сек")
            self._first_call = False
        
        # Извлекаем waveforms и labels
        input_values = [feature["input_values"] for feature in features]
        label_features = [feature["labels"] for feature in features]
        
        # Обрезаем слишком длинные аудио (КРИТИЧНО для скорости!)
        input_values_trimmed = []
        for wav in input_values:
            if isinstance(wav, torch.Tensor):
                if len(wav) > self.max_length:
                    wav = wav[:self.max_length]  # Обрезаем
                input_values_trimmed.append(wav.cpu().numpy())
            else:
                if len(wav) > self.max_length:
                    wav = wav[:self.max_length]
                input_values_trimmed.append(wav)
        
        # Обрабатываем через processor (теперь быстрее!)
        batch = self.processor(
            input_values_trimmed,
            sampling_rate=SAMPLE_RATE,
            padding=self.padding,
            max_length=self.max_length,
            truncation=True,
            return_tensors="pt",
        )
        
        # Добавляем labels
        batch["labels"] = torch.tensor(label_features, dtype=torch.long)
        
        return batch

# -------------------------
# Wav2Vec2 Trainer wrapper
# -------------------------
def make_w2v_trainer(train_dataset: Dataset, dev_dataset: Dataset, processor: Wav2Vec2Processor,
                     fp16: bool, output_dir: str) -> Trainer:
    model = Wav2Vec2ForSequenceClassification.from_pretrained(
        "wav2vec2-base", 
        num_labels=2,
        ignore_mismatched_sizes=True
    )

    # Wrap datasets: используем глобальный класс (для pickle на Windows)
    wrapped_train = Wav2Vec2WrapDataset(train_dataset)
    wrapped_dev = Wav2Vec2WrapDataset(dev_dataset)

    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=8,  # МАЛЕНЬКИЙ для Windows CPU!
        per_device_eval_batch_size=8,
        gradient_accumulation_steps=4,  # Компенсируем маленький batch
        num_train_epochs=3,
        logging_dir=f"{output_dir}/logs",
        logging_steps=50,
        save_steps=500,
        save_total_limit=2,
        fp16=fp16,
        dataloader_num_workers=0,  # ДЛЯ WINDOWS ЛУЧШЕ 0!
        dataloader_pin_memory=True,
        dataloader_prefetch_factor=None,
        gradient_checkpointing=False,
        remove_unused_columns=False,
        tf32=True,
        dataloader_persistent_workers=False,
        torch_compile=False,
        logging_first_step=True,
        report_to="none",
        max_grad_norm=1.0,
    )

    def compute_metrics(p):
        preds = p.predictions.argmax(-1)
        return {"accuracy": (preds == p.label_ids).mean()}

    # Создаем data_collator с ограничением длины для скорости
    data_collator = DataCollatorCTCWithPadding(
        processor=processor, 
        padding=True,
        max_length=80000  # 5 секунд макс (для CPU на Windows)
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=wrapped_train,
        eval_dataset=wrapped_dev,
        tokenizer=processor,
        data_collator=data_collator,
        compute_metrics=compute_metrics
    )
    return trainer

# -------------------------
# Checkpoint helpers
# -------------------------
def save_checkpoint(save_dir: str, name: str, model: nn.Module, optimizer: torch.optim.Optimizer,
                    scaler: Optional[torch.amp.GradScaler], epoch: int, is_dataparallel: bool=False):
    os.makedirs(save_dir, exist_ok=True)
    state = {
        "epoch": epoch,
        "optimizer": optimizer.state_dict()
    }
    state["model_state"] = model.module.state_dict() if is_dataparallel else model.state_dict()
    if scaler is not None:
        state["scaler"] = scaler.state_dict()
    torch.save(state, os.path.join(save_dir, name))
    print(f"[checkpoint] saved {name}")

def load_checkpoint(path: str, model: nn.Module, optimizer: Optional[torch.optim.Optimizer]=None,
                    scaler: Optional[torch.amp.GradScaler]=None, device: torch.device=torch.device("cpu")) -> int:
    ckpt = torch.load(path, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    if optimizer is not None and "optimizer" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer"])
    if scaler is not None and "scaler" in ckpt:
        scaler.load_state_dict(ckpt["scaler"])
    epoch = ckpt.get("epoch", 0)
    print(f"[checkpoint] loaded {path}, epoch={epoch}")
    return epoch

# -------------------------
# Ensemble helper
# -------------------------
def majority_vote(preds: List[int]) -> int:
    return max(set(preds), key=preds.count)

# -------------------------
# Main script
# -------------------------
def main(args):
    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    print("Device:", device)
    if device.type == "cuda":
        print("CUDA available. Devices count:", torch.cuda.device_count())
        for i in range(torch.cuda.device_count()):
            try:
                print(i, torch.cuda.get_device_name(i))
            except Exception:
                pass
        # Оптимизации для максимальной загрузки GPU
        torch.backends.cudnn.benchmark = True  # Автоподбор алгоритмов
        torch.backends.cuda.matmul.allow_tf32 = True  # TF32 для ускорения
        torch.backends.cudnn.allow_tf32 = True
        print("✓ CUDA оптимизации включены (TF32, cudnn.benchmark)")

    # build protocol paths
    train_dir = os.path.join(args.asv_root, "ASVspoof2019_LA_train", "flac")
    dev_dir = os.path.join(args.asv_root, "ASVspoof2019_LA_dev", "flac")
    eval_dir = os.path.join(args.asv_root, "ASVspoof2019_LA_eval", "flac")
    proto_dir = os.path.join(args.asv_root, "ASVspoof2019_LA_cm_protocols")
    train_proto = os.path.join(proto_dir, "ASVspoof2019.LA.cm.train.trn.txt")
    dev_proto = os.path.join(proto_dir, "ASVspoof2019.LA.cm.dev.trl.txt")
    eval_proto = os.path.join(proto_dir, "ASVspoof2019.LA.cm.eval.trl.txt")

    train_paths, train_labels = read_protocol(train_proto, train_dir)
    dev_paths, dev_labels = read_protocol(dev_proto, dev_dir)
    eval_paths, eval_labels = read_protocol(eval_proto, eval_dir)
    print(f"Datasets: train={len(train_paths)} dev={len(dev_paths)} eval={len(eval_paths)}")

    # Datasets & loaders
    cnn_train_ds = AudioDatasetTorch(train_paths, train_labels, model_type="cnn")
    cnn_dev_ds = AudioDatasetTorch(dev_paths, dev_labels, model_type="cnn")
    lstm_train_ds = AudioDatasetTorch(train_paths, train_labels, model_type="lstm")
    lstm_dev_ds = AudioDatasetTorch(dev_paths, dev_labels, model_type="lstm")
    w2v_train_ds = AudioDatasetTorch(train_paths, train_labels, model_type="wav2vec2")
    w2v_dev_ds = AudioDatasetTorch(dev_paths, dev_labels, model_type="wav2vec2")

    cnn_train_loader = DataLoader(cnn_train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)
    cnn_dev_loader = DataLoader(cnn_dev_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)
    lstm_train_loader = DataLoader(lstm_train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)
    lstm_dev_loader = DataLoader(lstm_dev_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)
    # wav2vec2 uses Trainer, no dataloader here (Trainer wraps dataset)

    # models
    cnn = CNNClassifier().to(device)
    lstm = LSTMClassifier().to(device)

    # optionally DataParallel
    is_dp = False
    if args.data_parallel and torch.cuda.device_count() > 1 and device.type == "cuda":
        print("Using DataParallel across GPUs")
        cnn = torch.nn.DataParallel(cnn)
        lstm = torch.nn.DataParallel(lstm)
        is_dp = True

    # optimizers and criterion
    opt_cnn = torch.optim.Adam(cnn.parameters(), lr=args.lr)
    opt_lstm = torch.optim.Adam(lstm.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss().to(device)
    scaler_cnn = torch.amp.GradScaler('cuda') if device.type == "cuda" else None
    scaler_lstm = torch.amp.GradScaler('cuda') if device.type == "cuda" else None

    start_epoch = 0
    if args.resume and os.path.exists(args.resume):
        # try to load checkpoint for cnn/lstm if paths provided
        try:
            start_epoch = load_checkpoint(args.resume, cnn, opt_cnn, scaler_cnn, device)
        except Exception as e:
            print("Could not load checkpoint:", e)


    print("Training CNN...")
    for epoch in range(start_epoch, args.epochs):
        loss_c = train_epoch_amp(cnn, cnn_train_loader, opt_cnn, criterion, scaler_cnn if scaler_cnn else torch.amp.GradScaler('cuda'), device)
        acc_c = eval_model(cnn, cnn_dev_loader, device)
        print(f"[CNN] Epoch {epoch+1}/{args.epochs} loss={loss_c:.4f} dev_acc={acc_c:.4f}")
        save_checkpoint(args.save_dir, f"cnn_ep{epoch+1}.pt", cnn, opt_cnn, scaler_cnn, epoch+1, is_dataparallel=is_dp)

    # Train LSTM
    print("Training LSTM...")
    for epoch in range(args.epochs):
        loss_l = train_epoch_amp(lstm, lstm_train_loader, opt_lstm, criterion, scaler_lstm if scaler_lstm else torch.amp.GradScaler('cuda'), device)
        acc_l = eval_model(lstm, lstm_dev_loader, device)
        print(f"[LSTM] Epoch {epoch+1}/{args.epochs} loss={loss_l:.4f} dev_acc={acc_l:.4f}")
        save_checkpoint(args.save_dir, f"lstm_ep{epoch+1}.pt", lstm, opt_lstm, scaler_lstm, epoch+1, is_dataparallel=is_dp)

        # Fine-tune Wav2Vec2 via Trainer (fp16 if CUDA)
    print("Preparing Wav2Vec2 trainer...")
    processor = Wav2Vec2Processor.from_pretrained("wav2vec2-base")
    trainer = make_w2v_trainer(w2v_train_ds, w2v_dev_ds, processor, fp16=(device.type=="cuda"), output_dir=os.path.join(args.save_dir, "w2v"))
    print("Fine-tuning Wav2Vec2...")
    print("⚠️  ВАЖНО: Первый батч может обрабатываться 1-2 минуты (это нормально!)")
    print("    Processor загружает и обрабатывает аудио, CUDA инициализируется...")
    print("    После первого батча скорость резко увеличится!")
    print()
    trainer.train()
    w2v_model = trainer.model.to(device)

    # Evaluate ensemble on eval set
    print("Evaluating ensemble on eval set...")
    cnn.eval(); lstm.eval(); w2v_model.eval()
    y_true, y_pred = [], []

    for path, label in zip(eval_paths, eval_labels):
        waveform, sr = torchaudio.load(path)
        if sr != SAMPLE_RATE:
            waveform = torchaudio.functional.resample(waveform, orig_freq=sr, new_freq=SAMPLE_RATE)
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
        waveform = waveform.squeeze(0)

        # CNN
        mel = torchaudio.transforms.MelSpectrogram(sample_rate=SAMPLE_RATE, n_mels=128, n_fft=1024, hop_length=320)(waveform)
        mel_db = 10.0 * torch.log10(mel + 1e-10)
        x_c = mel_db.unsqueeze(0).unsqueeze(0)
        x_c = F.interpolate(x_c, size=(128,128), mode='bilinear', align_corners=False).to(device)
        with torch.no_grad():
            p_c = torch.argmax(cnn(x_c), dim=1).item()

        # LSTM
        mfcc = torchaudio.transforms.MFCC(sample_rate=SAMPLE_RATE, n_mfcc=40, melkwargs={"n_fft":1024,"n_mels":128,"hop_length":320})(waveform)
        mfcc = mfcc.transpose(0,1)
        if mfcc.shape[0] < 300:
            pad = torch.zeros(300 - mfcc.shape[0], mfcc.shape[1])
            mfcc = torch.cat([mfcc, pad], dim=0)
        else:
            mfcc = mfcc[:300,:]
        x_l = mfcc.unsqueeze(0).to(device)
        with torch.no_grad():
            p_l = torch.argmax(lstm(x_l), dim=1).item()

        # Wav2Vec2
        inputs = processor(waveform, sampling_rate=SAMPLE_RATE, return_tensors="pt", padding=True)
        inputs = {k: v.to(device) for k,v in inputs.items()}
        with torch.no_grad():
            logits = w2v_model(**inputs).logits
        p_w = torch.argmax(logits, dim=-1).item()

        final = majority_vote([p_c, p_l, p_w])
        y_true.append(label)
        y_pred.append(final)

    acc = accuracy_score(y_true, y_pred)
    print("Ensemble eval accuracy:", acc)

# -------------------------
# CLI
# -------------------------
if __name__ == "__main__":
    # Необходимо для Windows multiprocessing
    multiprocessing.freeze_support()
    
    parser = argparse.ArgumentParser(description="Train audio deepfake detectors (no numpy) with CUDA+AMP support")
    parser.add_argument("--asv-root", type=str, default=DEFAULT_ASV_ROOT, help="path to ASVspoof2019 LA root")
    parser.add_argument("--batch-size", type=int, default=16, help="batch size for CNN/LSTM")
    parser.add_argument("--epochs", type=int, default=3, help="epochs for CNN/LSTM")
    parser.add_argument("--lr",
                        type=float, default=1e-3, help="learning rate")
    parser.add_argument("--num-workers", type=int, default=4, help="dataloader num_workers")
    parser.add_argument("--save-dir", type=str, default="./checkpoints", help="directory to save checkpoints")
    parser.add_argument("--resume", type=str, default="", help="path to checkpoint to resume cnn/lstm")
    parser.add_argument("--device", type=str, default="", help="device string (cuda or cpu). default auto")
    parser.add_argument("--data-parallel", action="store_true", help="use DataParallel if multiple GPUs available")
    args = parser.parse_args()
    main(args)
