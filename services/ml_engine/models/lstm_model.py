"""
LSTM sequence model for price movement prediction.
Input: past 60 candles with technical features.
Output: probability distribution over [UP, FLAT, DOWN].
"""

import logging
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import mlflow
import mlflow.pytorch
from shared.config.settings import get_settings

log = logging.getLogger(__name__)
settings = get_settings()

SEQUENCE_LEN = 60
INPUT_FEATURES = [
    "open", "high", "low", "close", "volume",
    "rsi", "macd", "macd_hist", "adx", "atr_pct",
    "bb_pct", "ema_9", "ema_20", "ema_50", "volume_ratio",
    "candle_pattern_score", "cmf", "stochrsi_k",
]
N_CLASSES = 3


class SequenceDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class LSTMPredictor(nn.Module):
    def __init__(self, input_size: int, hidden_size: int = 128, num_layers: int = 2, dropout: float = 0.3):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout,
        )
        self.attention = nn.MultiheadAttention(embed_dim=hidden_size, num_heads=4, batch_first=True)
        self.norm = nn.LayerNorm(hidden_size)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, N_CLASSES),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        lstm_out, _ = self.lstm(x)
        attn_out, _ = self.attention(lstm_out, lstm_out, lstm_out)
        out = self.norm(lstm_out + attn_out)
        context = out[:, -1, :]
        return self.classifier(context)


class LSTMModelTrainer:
    def __init__(self):
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._model: LSTMPredictor | None = None
        mlflow.set_tracking_uri(settings.ml.mlflow_tracking_uri)

    def build_sequences(self, df) -> tuple[np.ndarray, np.ndarray]:
        import pandas as pd
        features = df[INPUT_FEATURES].values
        future_returns = df["close"].pct_change().shift(-1).values
        labels = np.where(future_returns > 0.005, 2, np.where(future_returns < -0.005, 0, 1))
        X, y = [], []
        for i in range(SEQUENCE_LEN, len(features) - 1):
            X.append(features[i - SEQUENCE_LEN: i])
            y.append(labels[i])
        return np.array(X, dtype=np.float32), np.array(y, dtype=np.int64)

    def train(self, df, epochs: int = 50, batch_size: int = 64, lr: float = 1e-3) -> dict:
        X, y = self.build_sequences(df)
        split = int(len(X) * 0.85)
        train_ds = SequenceDataset(X[:split], y[:split])
        val_ds = SequenceDataset(X[split:], y[split:])
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=False)
        val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

        self._model = LSTMPredictor(input_size=len(INPUT_FEATURES)).to(self._device)
        optimizer = torch.optim.AdamW(self._model.parameters(), lr=lr, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
        criterion = nn.CrossEntropyLoss()

        best_val_acc = 0.0
        with mlflow.start_run(run_name="lstm_training"):
            mlflow.log_params({"epochs": epochs, "batch_size": batch_size, "lr": lr, "seq_len": SEQUENCE_LEN})
            for epoch in range(epochs):
                self._model.train()
                for Xb, yb in train_loader:
                    Xb, yb = Xb.to(self._device), yb.to(self._device)
                    optimizer.zero_grad()
                    loss = criterion(self._model(Xb), yb)
                    loss.backward()
                    nn.utils.clip_grad_norm_(self._model.parameters(), 1.0)
                    optimizer.step()
                scheduler.step()

                val_acc = self._evaluate(val_loader)
                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    mlflow.pytorch.log_model(self._model, "lstm_best")
                mlflow.log_metrics({"val_accuracy": val_acc}, step=epoch)
                log.info("LSTM epoch=%d val_acc=%.4f", epoch + 1, val_acc)

        return {"best_val_accuracy": best_val_acc}

    def _evaluate(self, loader: DataLoader) -> float:
        self._model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for Xb, yb in loader:
                Xb, yb = Xb.to(self._device), yb.to(self._device)
                preds = self._model(Xb).argmax(dim=1)
                correct += (preds == yb).sum().item()
                total += len(yb)
        return correct / total if total else 0.0

    def predict_proba(self, sequence: np.ndarray) -> dict[str, float]:
        if self._model is None:
            raise RuntimeError("LSTM model not loaded.")
        self._model.eval()
        x = torch.tensor(sequence[np.newaxis], dtype=torch.float32).to(self._device)
        with torch.no_grad():
            logits = self._model(x)
            probs = torch.softmax(logits, dim=1).squeeze().cpu().tolist()
        return {"DOWN": probs[0], "FLAT": probs[1], "UP": probs[2]}
