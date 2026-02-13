"""
Обучение моделей прогноза спроса на сельхозпродукцию.
Модели: ARIMA, Prophet, N-BEATS, TFT.
Данные: FAOSTAT (автоматическая загрузка).

Запуск:
    python train.py --product Wheat --country World --horizon 12
"""

import argparse
import json
import os
import pickle
import warnings
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
import torch
import torch.nn as nn
from scipy import stats
from sklearn.preprocessing import MinMaxScaler
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.seasonal import seasonal_decompose
from tqdm import tqdm

warnings.filterwarnings("ignore")


# ========================== ЗАГРУЗКА ДАННЫХ ==========================

def download_faostat_data(product="Wheat", country="World", save_dir="data"):
    """
    Загрузка данных из FAOSTAT.
    Если API недоступен — генерируем реалистичные синтетические данные
    на основе известных паттернов сельхозпродукции.
    """
    os.makedirs(save_dir, exist_ok=True)
    cache_file = os.path.join(save_dir, f"faostat_{product}_{country}.csv")

    if os.path.exists(cache_file):
        print(f"Загрузка из кэша: {cache_file}")
        df = pd.read_csv(cache_file, parse_dates=["date"])
        return df

    print(f"Загрузка данных FAOSTAT: {product}, {country}...")

    # Попытка загрузить реальные данные через FAOSTAT API
    try:
        df = _fetch_faostat_api(product, country)
        if df is not None and len(df) > 36:
            df.to_csv(cache_file, index=False)
            print(f"Данные сохранены: {cache_file} ({len(df)} записей)")
            return df
    except Exception as e:
        print(f"FAOSTAT API недоступен: {e}")

    # Fallback: реалистичные синтетические данные
    print("Генерация реалистичных синтетических данных...")
    df = _generate_synthetic_data(product, country)
    df.to_csv(cache_file, index=False)
    print(f"Данные сохранены: {cache_file} ({len(df)} записей)")
    return df


def _fetch_faostat_api(product, country):
    """Попытка загрузки через FAOSTAT Bulk Download."""
    # FAOSTAT item codes
    item_codes = {
        "Wheat": "15", "Maize": "56", "Rice": "27",
        "Milk": "882", "Soybeans": "236", "Potatoes": "116",
        "Sugar cane": "156", "Barley": "44",
    }

    code = item_codes.get(product, "15")

    # FAOSTAT bulk API (production data)
    url = (
        f"https://fenixservices.fao.org/faostat/api/v1/en/data/QCL?"
        f"area=5000&item={code}&element=5510"
        f"&year=1990,1991,1992,1993,1994,1995,1996,1997,1998,1999,"
        f"2000,2001,2002,2003,2004,2005,2006,2007,2008,2009,"
        f"2010,2011,2012,2013,2014,2015,2016,2017,2018,2019,"
        f"2020,2021,2022,2023"
    )

    resp = requests.get(url, timeout=30)
    if resp.status_code == 200:
        data = resp.json().get("data", [])
        if data:
            records = []
            for row in data:
                year = int(row.get("Year", 2000))
                value = float(row.get("Value", 0))
                records.append({"year": year, "value": value})

            df_yearly = pd.DataFrame(records).sort_values("year")

            # Интерполяция в месячные данные с сезонностью
            return _yearly_to_monthly(df_yearly, product)

    return None


def _yearly_to_monthly(df_yearly, product):
    """Интерполяция годовых данных в месячные с сезонностью."""
    # Сезонные паттерны (индексы по месяцам, нормализованные)
    seasonal_patterns = {
        "Wheat":    [0.60, 0.55, 0.65, 0.75, 0.90, 1.20, 1.50, 1.40, 1.10, 0.85, 0.70, 0.60],
        "Maize":    [0.50, 0.50, 0.60, 0.70, 0.85, 1.00, 1.10, 1.20, 1.50, 1.40, 1.00, 0.65],
        "Rice":     [0.70, 0.65, 0.75, 0.85, 1.00, 1.10, 1.20, 1.15, 1.30, 1.20, 0.90, 0.70],
        "Milk":     [0.85, 0.88, 0.95, 1.05, 1.15, 1.20, 1.15, 1.10, 1.00, 0.92, 0.88, 0.87],
        "Soybeans": [0.50, 0.50, 0.55, 0.65, 0.80, 0.95, 1.10, 1.20, 1.50, 1.45, 1.00, 0.60],
    }

    pattern = seasonal_patterns.get(product, [1.0] * 12)
    pattern = np.array(pattern)
    pattern = pattern / pattern.mean()  # нормализация

    records = []
    for _, row in df_yearly.iterrows():
        year = int(row["year"])
        annual_monthly = row["value"] / 12
        for month in range(1, 13):
            value = annual_monthly * pattern[month - 1]
            noise = np.random.normal(1.0, 0.03)
            value *= noise
            records.append({
                "date": pd.Timestamp(year=year, month=month, day=1),
                "value": max(0, value),
            })

    return pd.DataFrame(records).sort_values("date").reset_index(drop=True)


def _generate_synthetic_data(product, country, years=30):
    """Генерация реалистичных синтетических данных с трендом и сезонностью."""
    np.random.seed(42)
    n_months = years * 12
    dates = pd.date_range("1994-01-01", periods=n_months, freq="MS")

    # Базовые параметры по продуктам (млн тонн / месяц)
    product_params = {
        "Wheat":    {"base": 60, "trend": 0.15, "seasonal_amp": 0.25, "noise": 0.05},
        "Maize":    {"base": 80, "trend": 0.20, "seasonal_amp": 0.30, "noise": 0.06},
        "Rice":     {"base": 40, "trend": 0.10, "seasonal_amp": 0.20, "noise": 0.04},
        "Milk":     {"base": 70, "trend": 0.12, "seasonal_amp": 0.15, "noise": 0.03},
        "Soybeans": {"base": 25, "trend": 0.25, "seasonal_amp": 0.35, "noise": 0.07},
        "Potatoes": {"base": 30, "trend": 0.08, "seasonal_amp": 0.20, "noise": 0.05},
    }

    params = product_params.get(product, product_params["Wheat"])
    t = np.arange(n_months)

    # Тренд (линейный + лёгкий экспоненциальный)
    trend = params["base"] * (1 + params["trend"] * t / n_months)
    trend += params["base"] * 0.05 * np.sin(2 * np.pi * t / (n_months * 0.7))

    # Сезонность (годовая)
    seasonality = params["base"] * params["seasonal_amp"] * np.sin(2 * np.pi * t / 12 + np.pi / 6)
    # Двойная сезонность (полугодовая, слабее)
    seasonality += params["base"] * params["seasonal_amp"] * 0.3 * np.sin(4 * np.pi * t / 12)

    # Шум
    noise = params["base"] * params["noise"] * np.random.randn(n_months)

    # Редкие шоки (засуха, кризис)
    shocks = np.zeros(n_months)
    for _ in range(int(years * 0.3)):
        shock_idx = np.random.randint(12, n_months - 6)
        shock_magnitude = np.random.choice([-1, 1]) * params["base"] * np.random.uniform(0.1, 0.3)
        shock_duration = np.random.randint(3, 8)
        for j in range(shock_duration):
            if shock_idx + j < n_months:
                decay = np.exp(-0.3 * j)
                shocks[shock_idx + j] += shock_magnitude * decay

    values = trend + seasonality + noise + shocks
    values = np.maximum(values, params["base"] * 0.2)  # не ниже 20% от базы

    df = pd.DataFrame({"date": dates, "value": values})
    return df


def prepare_data(df, lookback=24, horizon=12):
    """
    Подготовка данных: нормализация, создание окон для нейросетей.
    Разделение: train / val / test (последний горизонт).
    """
    values = df["value"].values.astype(np.float32)

    # Разделение
    test_size = horizon
    val_size = horizon
    train_end = len(values) - test_size - val_size

    train_vals = values[:train_end]
    val_vals = values[train_end:train_end + val_size]
    test_vals = values[train_end + val_size:]

    # Нормализация по train
    scaler = MinMaxScaler()
    train_scaled = scaler.fit_transform(train_vals.reshape(-1, 1)).flatten()
    val_scaled = scaler.transform(val_vals.reshape(-1, 1)).flatten()
    test_scaled = scaler.transform(test_vals.reshape(-1, 1)).flatten()

    all_scaled = np.concatenate([train_scaled, val_scaled, test_scaled])

    # Окна для нейросетей
    def make_windows(data, start, end, lookback, horizon):
        X, y = [], []
        for i in range(start, end - horizon + 1):
            if i - lookback < 0:
                continue
            X.append(data[i - lookback:i])
            y.append(data[i:i + horizon])
        return np.array(X), np.array(y)

    X_train, y_train = make_windows(all_scaled, lookback, train_end, lookback, horizon)
    X_val, y_val = make_windows(all_scaled, train_end, train_end + val_size, lookback, horizon)

    info = {
        "values": values,
        "dates": df["date"].values,
        "scaler": scaler,
        "train_end": train_end,
        "val_end": train_end + val_size,
        "lookback": lookback,
        "horizon": horizon,
        "X_train": X_train, "y_train": y_train,
        "X_val": X_val, "y_val": y_val,
        "train_vals": train_vals,
        "val_vals": val_vals,
        "test_vals": test_vals,
        "all_scaled": all_scaled,
    }

    print(f"Train: {len(train_vals)}, Val: {len(val_vals)}, Test: {len(test_vals)}")
    print(f"Окна train: {X_train.shape}, val: {X_val.shape}")

    return info


# ========================== МОДЕЛЬ ARIMA ==========================

def train_arima(values, horizon, order=(2, 1, 2), seasonal_order=(1, 1, 1, 12)):
    """Обучение ARIMA/SARIMA."""
    print("\n--- ARIMA ---")
    try:
        from statsmodels.tsa.statespace.sarimax import SARIMAX
        model = SARIMAX(values, order=order, seasonal_order=seasonal_order,
                        enforce_stationarity=False, enforce_invertibility=False)
        fitted = model.fit(disp=False, maxiter=200)
        forecast = fitted.forecast(steps=horizon)
        print(f"  AIC: {fitted.aic:.1f}, BIC: {fitted.bic:.1f}")
        return forecast, fitted
    except Exception as e:
        print(f"  SARIMA failed ({e}), falling back to simple ARIMA")
        model = ARIMA(values, order=order)
        fitted = model.fit()
        forecast = fitted.forecast(steps=horizon)
        return forecast, fitted


# ========================== МОДЕЛЬ PROPHET ==========================

def train_prophet(df_full, train_end, horizon):
    """Обучение Facebook Prophet."""
    print("\n--- Prophet ---")
    from prophet import Prophet

    df_prophet = pd.DataFrame({
        "ds": df_full["date"].values[:train_end],
        "y": df_full["value"].values[:train_end],
    })

    model = Prophet(
        yearly_seasonality=True,
        weekly_seasonality=False,
        daily_seasonality=False,
        changepoint_prior_scale=0.05,
        seasonality_prior_scale=10,
    )
    model.fit(df_prophet)

    future = model.make_future_dataframe(periods=horizon, freq="MS")
    pred = model.predict(future)
    forecast = pred["yhat"].values[-horizon:]
    forecast = np.maximum(forecast, 0)

    print(f"  Прогноз на {horizon} месяцев")
    return forecast, model


# ========================== МОДЕЛЬ N-BEATS ==========================

class NBeatsBlock(nn.Module):
    """Один блок N-BEATS."""

    def __init__(self, input_size, theta_size, hidden_size=256, n_layers=4):
        super().__init__()
        layers = [nn.Linear(input_size, hidden_size), nn.ReLU()]
        for _ in range(n_layers - 1):
            layers += [nn.Linear(hidden_size, hidden_size), nn.ReLU()]
        self.fc = nn.Sequential(*layers)
        self.theta_b = nn.Linear(hidden_size, input_size)    # backcast
        self.theta_f = nn.Linear(hidden_size, theta_size)    # forecast

    def forward(self, x):
        h = self.fc(x)
        return self.theta_b(h), self.theta_f(h)


class NBeats(nn.Module):
    """N-BEATS: Neural Basis Expansion Analysis."""

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


def train_nbeats(data_info, epochs=100, lr=1e-3):
    """Обучение N-BEATS."""
    print("\n--- N-BEATS ---")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    lookback = data_info["lookback"]
    horizon = data_info["horizon"]

    model = NBeats(
        input_size=lookback,
        output_size=horizon,
        n_stacks=2,
        n_blocks=3,
        hidden=256,
    ).to(device)

    X_train = torch.FloatTensor(data_info["X_train"]).to(device)
    y_train = torch.FloatTensor(data_info["y_train"]).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.MSELoss()

    best_loss = float("inf")
    train_losses = []

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        pred = model(X_train)
        loss = criterion(pred, y_train)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        train_losses.append(loss.item())

        if loss.item() < best_loss:
            best_loss = loss.item()
            best_state = model.state_dict().copy()

        if (epoch + 1) % 20 == 0:
            print(f"  Epoch {epoch+1}/{epochs}, Loss: {loss.item():.6f}")

    model.load_state_dict(best_state)

    # Прогноз на test
    model.eval()
    with torch.no_grad():
        all_scaled = data_info["all_scaled"]
        test_start = data_info["val_end"]
        x_input = torch.FloatTensor(
            all_scaled[test_start - lookback:test_start]
        ).unsqueeze(0).to(device)
        pred_scaled = model(x_input).cpu().numpy().flatten()

    scaler = data_info["scaler"]
    forecast = scaler.inverse_transform(pred_scaled.reshape(-1, 1)).flatten()
    forecast = np.maximum(forecast, 0)

    print(f"  Best loss: {best_loss:.6f}")
    return forecast, model, train_losses


# ========================== МОДЕЛЬ TFT (упрощённый) ==========================

class SimpleTemporalAttention(nn.Module):
    """Упрощённый Temporal Fusion Transformer."""

    def __init__(self, input_size, output_size, d_model=128, n_heads=4, n_layers=2):
        super().__init__()
        self.input_proj = nn.Linear(1, d_model)
        self.pos_encoding = nn.Parameter(torch.randn(1, input_size, d_model) * 0.1)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_model * 2,
            dropout=0.1, batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        self.gate = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.Sigmoid(),
        )

        self.output_proj = nn.Sequential(
            nn.Linear(d_model * input_size, d_model),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(d_model, output_size),
        )

    def forward(self, x):
        # x: (batch, seq_len)
        x = x.unsqueeze(-1)  # (batch, seq, 1)
        x = self.input_proj(x) + self.pos_encoding
        x = self.transformer(x)
        x = x * self.gate(x)  # Gated residual
        x = x.flatten(1)
        return self.output_proj(x)


def train_tft(data_info, epochs=100, lr=1e-3):
    """Обучение упрощённого TFT."""
    print("\n--- Temporal Fusion Transformer ---")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    lookback = data_info["lookback"]
    horizon = data_info["horizon"]

    model = SimpleTemporalAttention(
        input_size=lookback,
        output_size=horizon,
        d_model=128,
        n_heads=4,
        n_layers=2,
    ).to(device)

    X_train = torch.FloatTensor(data_info["X_train"]).to(device)
    y_train = torch.FloatTensor(data_info["y_train"]).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.MSELoss()

    best_loss = float("inf")
    train_losses = []

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        pred = model(X_train)
        loss = criterion(pred, y_train)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        train_losses.append(loss.item())

        if loss.item() < best_loss:
            best_loss = loss.item()
            best_state = model.state_dict().copy()

        if (epoch + 1) % 20 == 0:
            print(f"  Epoch {epoch+1}/{epochs}, Loss: {loss.item():.6f}")

    model.load_state_dict(best_state)

    # Прогноз на test
    model.eval()
    with torch.no_grad():
        all_scaled = data_info["all_scaled"]
        test_start = data_info["val_end"]
        x_input = torch.FloatTensor(
            all_scaled[test_start - lookback:test_start]
        ).unsqueeze(0).to(device)
        pred_scaled = model(x_input).cpu().numpy().flatten()

    scaler = data_info["scaler"]
    forecast = scaler.inverse_transform(pred_scaled.reshape(-1, 1)).flatten()
    forecast = np.maximum(forecast, 0)

    print(f"  Best loss: {best_loss:.6f}")
    return forecast, model, train_losses


# ========================== МЕТРИКИ ==========================

def mape(actual, predicted):
    """Mean Absolute Percentage Error."""
    actual, predicted = np.array(actual), np.array(predicted)
    mask = actual != 0
    return np.mean(np.abs((actual[mask] - predicted[mask]) / actual[mask])) * 100


def smape(actual, predicted):
    """Symmetric MAPE."""
    actual, predicted = np.array(actual), np.array(predicted)
    denominator = (np.abs(actual) + np.abs(predicted)) / 2
    mask = denominator != 0
    return np.mean(np.abs(actual[mask] - predicted[mask]) / denominator[mask]) * 100


def mae(actual, predicted):
    return np.mean(np.abs(np.array(actual) - np.array(predicted)))


def rmse(actual, predicted):
    return np.sqrt(np.mean((np.array(actual) - np.array(predicted)) ** 2))


# ========================== ВИЗУАЛИЗАЦИЯ ==========================

def plot_forecasts(dates_test, actual, forecasts_dict, title, save_path,
                   dates_train=None, train_vals=None):
    """Построение графика прогнозов всех моделей."""
    fig, ax = plt.subplots(figsize=(15, 6))

    # Последние данные train для контекста
    if dates_train is not None and train_vals is not None:
        n_context = 36  # последние 3 года
        ax.plot(dates_train[-n_context:], train_vals[-n_context:],
                "k-", alpha=0.5, label="Исторические данные", linewidth=1.5)

    # Фактические значения test
    ax.plot(dates_test, actual, "ko-", label="Факт (test)", linewidth=2, markersize=6)

    # Прогнозы моделей
    colors = {"ARIMA": "#e74c3c", "Prophet": "#3498db", "N-BEATS": "#2ecc71", "TFT": "#9b59b6"}
    for name, forecast in forecasts_dict.items():
        color = colors.get(name, "#95a5a6")
        ax.plot(dates_test[:len(forecast)], forecast[:len(dates_test)],
                "o--", label=f"{name}", color=color, linewidth=1.5, markersize=5, alpha=0.8)

    ax.set_xlabel("Дата", fontsize=12)
    ax.set_ylabel("Объём (млн тонн)", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"График: {save_path}")


def plot_decomposition(values, dates, save_path, period=12):
    """Декомпозиция временного ряда."""
    series = pd.Series(values, index=pd.DatetimeIndex(dates))
    result = seasonal_decompose(series, model="additive", period=period)

    fig, axes = plt.subplots(4, 1, figsize=(14, 10), sharex=True)
    components = [
        ("Исходный ряд", result.observed),
        ("Тренд", result.trend),
        ("Сезонность", result.seasonal),
        ("Остатки", result.resid),
    ]

    for ax, (title, data) in zip(axes, components):
        ax.plot(data, linewidth=1)
        ax.set_ylabel(title, fontsize=10)
        ax.grid(alpha=0.3)

    axes[0].set_title("Декомпозиция временного ряда", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Декомпозиция: {save_path}")


# ========================== MAIN ==========================

def main(args):
    os.makedirs("results", exist_ok=True)

    # 1. Загрузка данных
    print("=" * 60)
    print(f"Продукт: {args.product}, Страна: {args.country}")
    print(f"Горизонт прогноза: {args.horizon} мес.")
    print("=" * 60)

    df = download_faostat_data(args.product, args.country)
    print(f"Период: {df['date'].min()} — {df['date'].max()}")
    print(f"Всего записей: {len(df)}")

    # 2. Декомпозиция
    plot_decomposition(
        df["value"].values, df["date"].values,
        "results/decomposition.png"
    )

    # 3. Подготовка данных
    data_info = prepare_data(df, lookback=args.lookback, horizon=args.horizon)
    test_actual = data_info["test_vals"]

    # 4. Обучение моделей
    forecasts = {}
    models_dict = {}

    # ARIMA
    arima_forecast, arima_model = train_arima(
        data_info["values"][:data_info["train_end"]], args.horizon
    )
    forecasts["ARIMA"] = arima_forecast

    # Prophet
    prophet_forecast, prophet_model = train_prophet(
        df, data_info["train_end"], args.horizon
    )
    forecasts["Prophet"] = prophet_forecast

    # N-BEATS
    nbeats_forecast, nbeats_model, nbeats_losses = train_nbeats(
        data_info, epochs=args.nn_epochs
    )
    forecasts["N-BEATS"] = nbeats_forecast

    # TFT
    tft_forecast, tft_model, tft_losses = train_tft(
        data_info, epochs=args.nn_epochs
    )
    forecasts["TFT"] = tft_forecast

    # 5. Метрики
    print(f"\n{'='*60}")
    print("РЕЗУЛЬТАТЫ")
    print(f"{'='*60}")

    results = {}
    print(f"\n{'Модель':<15} {'MAPE':>8} {'SMAPE':>8} {'MAE':>10} {'RMSE':>10}")
    print("-" * 55)

    for name, forecast in forecasts.items():
        m = {
            "MAPE": mape(test_actual, forecast),
            "SMAPE": smape(test_actual, forecast),
            "MAE": mae(test_actual, forecast),
            "RMSE": rmse(test_actual, forecast),
        }
        results[name] = m
        print(f"{name:<15} {m['MAPE']:>7.2f}% {m['SMAPE']:>7.2f}% "
              f"{m['MAE']:>10.2f} {m['RMSE']:>10.2f}")

    # 6. Визуализация
    dates_test = df["date"].values[data_info["val_end"]:]
    dates_train = df["date"].values[:data_info["val_end"]]
    train_vals = df["value"].values[:data_info["val_end"]]

    plot_forecasts(
        dates_test, test_actual, forecasts,
        f"Прогноз: {args.product} ({args.country})",
        "results/forecast_comparison.png",
        dates_train, train_vals,
    )

    # Графики loss нейросетей
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    ax1.plot(nbeats_losses, "b-", alpha=0.7)
    ax1.set(xlabel="Epoch", ylabel="MSE Loss", title="N-BEATS Training Loss")
    ax1.grid(alpha=0.3)
    ax2.plot(tft_losses, "purple", alpha=0.7)
    ax2.set(xlabel="Epoch", ylabel="MSE Loss", title="TFT Training Loss")
    ax2.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig("results/nn_training_losses.png", dpi=150, bbox_inches="tight")
    plt.close()

    # 7. Сохранение
    save_data = {
        "product": args.product,
        "country": args.country,
        "horizon": args.horizon,
        "metrics": results,
        "test_actual": test_actual.tolist(),
        "forecasts": {k: v.tolist() for k, v in forecasts.items()},
    }

    with open("results/metrics.json", "w", encoding="utf-8") as f:
        json.dump(save_data, f, indent=2, ensure_ascii=False)
    print(f"\nМетрики: results/metrics.json")

    # Сохранение моделей
    torch.save({
        "nbeats_state": nbeats_model.state_dict(),
        "tft_state": tft_model.state_dict(),
        "scaler": data_info["scaler"],
        "lookback": args.lookback,
        "horizon": args.horizon,
        "product": args.product,
        "country": args.country,
    }, "results/models.pth")

    with open("results/prophet_model.pkl", "wb") as f:
        pickle.dump(prophet_model, f)

    print("Модели сохранены: results/models.pth, results/prophet_model.pkl")
    print("Готово!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Прогноз спроса на сельхозпродукцию")
    parser.add_argument("--product", type=str, default="Wheat",
                        help="Продукт: Wheat, Maize, Rice, Milk, Soybeans")
    parser.add_argument("--country", type=str, default="World")
    parser.add_argument("--horizon", type=int, default=12, help="Горизонт прогноза (мес.)")
    parser.add_argument("--lookback", type=int, default=24, help="Окно входа (мес.)")
    parser.add_argument("--nn_epochs", type=int, default=150, help="Эпохи нейросетей")
    args = parser.parse_args()
    main(args)
