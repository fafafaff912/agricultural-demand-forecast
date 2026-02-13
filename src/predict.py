"""
Прогноз на будущие месяцы лучшей обученной моделью.

Запуск:
    python predict.py --months 12
    python predict.py --months 6 --model nbeats
"""

import argparse
import json
import os
import pickle
import sys
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn


# === Модели (копия из train.py для автономности) ===

class NBeatsBlock(nn.Module):
    def __init__(self, input_size, theta_size, hidden_size=256, n_layers=4):
        super().__init__()
        layers = [nn.Linear(input_size, hidden_size), nn.ReLU()]
        for _ in range(n_layers - 1):
            layers += [nn.Linear(hidden_size, hidden_size), nn.ReLU()]
        self.fc = nn.Sequential(*layers)
        self.theta_b = nn.Linear(hidden_size, input_size)
        self.theta_f = nn.Linear(hidden_size, theta_size)

    def forward(self, x):
        h = self.fc(x)
        return self.theta_b(h), self.theta_f(h)


class NBeats(nn.Module):
    def __init__(self, input_size, output_size, n_stacks=2, n_blocks=3, hidden=256):
        super().__init__()
        self.blocks = nn.ModuleList()
        for _ in range(n_stacks):
            for _ in range(n_blocks):
                self.blocks.append(NBeatsBlock(input_size, output_size, hidden))

    def forward(self, x):
        residual = x
        forecast = torch.zeros(x.size(0), self.blocks[0].theta_f.out_features,
                               device=x.device)
        for block in self.blocks:
            backcast, block_forecast = block(residual)
            residual = residual - backcast
            forecast = forecast + block_forecast
        return forecast


class SimpleTemporalAttention(nn.Module):
    def __init__(self, input_size, output_size, d_model=128, n_heads=4, n_layers=2):
        super().__init__()
        self.input_proj = nn.Linear(1, d_model)
        self.pos_encoding = nn.Parameter(torch.randn(1, input_size, d_model) * 0.1)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_model * 2,
            dropout=0.1, batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.gate = nn.Sequential(nn.Linear(d_model, d_model), nn.Sigmoid())
        self.output_proj = nn.Sequential(
            nn.Linear(d_model * input_size, d_model), nn.ReLU(),
            nn.Dropout(0.1), nn.Linear(d_model, output_size),
        )

    def forward(self, x):
        x = x.unsqueeze(-1)
        x = self.input_proj(x) + self.pos_encoding
        x = self.transformer(x)
        x = x * self.gate(x)
        x = x.flatten(1)
        return self.output_proj(x)


# === Прогноз ===

def predict_future(args):
    """Прогноз на будущие месяцы."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Загрузка данных
    metrics_path = "results/metrics.json"
    model_path = "results/models.pth"

    if not os.path.exists(metrics_path) or not os.path.exists(model_path):
        print("Модели не найдены. Сначала запустите: python train.py")
        sys.exit(1)

    with open(metrics_path, encoding="utf-8") as f:
        results = json.load(f)

    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    scaler = checkpoint["scaler"]
    lookback = checkpoint["lookback"]
    horizon = checkpoint["horizon"]
    product = checkpoint["product"]
    country = checkpoint["country"]

    # Исторические данные
    test_actual = np.array(results["test_actual"])

    # Загрузка данных для получения последнего окна
    data_dir = "data"
    cache_file = os.path.join(data_dir, f"faostat_{product}_{country}.csv")
    if os.path.exists(cache_file):
        df = pd.read_csv(cache_file, parse_dates=["date"])
    else:
        print("Данные не найдены. Запустите train.py сначала.")
        sys.exit(1)

    all_values = df["value"].values.astype(np.float32)
    all_scaled = scaler.transform(all_values.reshape(-1, 1)).flatten()

    # Выбор модели
    model_choice = args.model.lower()

    if model_choice == "nbeats":
        model = NBeats(lookback, args.months, n_stacks=2, n_blocks=3, hidden=256).to(device)
        # Если horizon совпадает, загружаем веса
        if args.months == horizon:
            model = NBeats(lookback, horizon, n_stacks=2, n_blocks=3, hidden=256).to(device)
            model.load_state_dict(checkpoint["nbeats_state"])
        model_name = "N-BEATS"
    elif model_choice == "tft":
        model = SimpleTemporalAttention(lookback, args.months, 128, 4, 2).to(device)
        if args.months == horizon:
            model = SimpleTemporalAttention(lookback, horizon, 128, 4, 2).to(device)
            model.load_state_dict(checkpoint["tft_state"])
        model_name = "TFT"
    elif model_choice == "prophet":
        model_name = "Prophet"
    else:
        model_choice = "nbeats"
        model_name = "N-BEATS"
        model = NBeats(lookback, horizon, n_stacks=2, n_blocks=3, hidden=256).to(device)
        model.load_state_dict(checkpoint["nbeats_state"])

    print(f"\n{'='*50}")
    print(f"ПРОГНОЗ: {product} ({country})")
    print(f"Модель: {model_name}")
    print(f"Горизонт: {args.months} мес.")
    print(f"{'='*50}")

    # Прогноз
    if model_choice == "prophet":
        prophet_path = "results/prophet_model.pkl"
        if os.path.exists(prophet_path):
            with open(prophet_path, "rb") as f:
                prophet_model = pickle.load(f)
            future = prophet_model.make_future_dataframe(periods=args.months, freq="MS")
            pred = prophet_model.predict(future)
            forecast = pred["yhat"].values[-args.months:]
            forecast = np.maximum(forecast, 0)

            future_dates = pd.date_range(
                df["date"].iloc[-1] + pd.DateOffset(months=1),
                periods=args.months, freq="MS",
            )
        else:
            print("Prophet модель не найдена.")
            sys.exit(1)
    else:
        # Нейросеть
        model.eval()
        with torch.no_grad():
            if args.months <= horizon:
                # Используем обученную модель напрямую
                x_input = torch.FloatTensor(all_scaled[-lookback:]).unsqueeze(0).to(device)
                pred_scaled = model(x_input).cpu().numpy().flatten()[:args.months]
            else:
                # Авторегрессия для длинных горизонтов
                current_input = all_scaled[-lookback:].copy()
                predictions = []
                steps = (args.months + horizon - 1) // horizon

                for _ in range(steps):
                    x = torch.FloatTensor(current_input[-lookback:]).unsqueeze(0).to(device)
                    pred = model(x).cpu().numpy().flatten()
                    predictions.extend(pred.tolist())
                    current_input = np.concatenate([current_input, pred])

                pred_scaled = np.array(predictions[:args.months])

            forecast = scaler.inverse_transform(pred_scaled.reshape(-1, 1)).flatten()
            forecast = np.maximum(forecast, 0)

        future_dates = pd.date_range(
            df["date"].iloc[-1] + pd.DateOffset(months=1),
            periods=args.months, freq="MS",
        )

    # Вывод прогноза
    print(f"\n{'Месяц':<15} {'Прогноз':>12}")
    print("-" * 30)
    for date, val in zip(future_dates, forecast):
        print(f"{date.strftime('%Y-%m'):<15} {val:>12.2f}")

    # Визуализация
    fig, ax = plt.subplots(figsize=(14, 6))

    # Исторические данные (последние 3 года)
    n_hist = 36
    hist_dates = df["date"].values[-n_hist:]
    hist_vals = all_values[-n_hist:]
    ax.plot(hist_dates, hist_vals, "ko-", label="Исторические данные",
            linewidth=1.5, markersize=4)

    # Прогноз
    ax.plot(future_dates, forecast, "s--", color="#2ecc71",
            label=f"Прогноз ({model_name})", linewidth=2, markersize=6)

    # Доверительный интервал (±10%)
    lower = forecast * 0.9
    upper = forecast * 1.1
    ax.fill_between(future_dates, lower, upper, alpha=0.2, color="#2ecc71",
                    label="±10% интервал")

    ax.axvline(df["date"].iloc[-1], color="red", linestyle=":", alpha=0.5,
               label="Начало прогноза")

    ax.set_xlabel("Дата", fontsize=12)
    ax.set_ylabel("Объём (млн тонн)", fontsize=12)
    ax.set_title(f"Прогноз: {product} ({country}) — {model_name}",
                 fontsize=14, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)

    plt.tight_layout()
    save_path = "results/future_forecast.png"
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"\nГрафик: {save_path}")
    plt.show()

    # Сохранение прогноза
    forecast_df = pd.DataFrame({
        "date": future_dates,
        "forecast": forecast,
    })
    forecast_df.to_csv("results/future_forecast.csv", index=False)
    print(f"Прогноз: results/future_forecast.csv")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Прогноз на будущее")
    parser.add_argument("--months", type=int, default=12, help="Горизонт (мес.)")
    parser.add_argument("--model", type=str, default="nbeats",
                        choices=["nbeats", "tft", "prophet"])
    args = parser.parse_args()
    predict_future(args)
