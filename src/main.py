#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Основной скрипт для прогнозирования временного ряда производства молока.
Модели: ARIMA (baseline) и N-BEATS.
Данные: встроенный MilkDataset из библиотеки darts.
Визуализация сохраняется в reports/figures/.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from darts import TimeSeries
from darts.datasets import MilkDataset
from darts.models import NBEATSModel, ARIMA
from darts.metrics import mape
from darts.dataprocessing.transformers import Scaler
import torch

# Создание папки для сохранения графиков
os.makedirs("reports/figures", exist_ok=True)

def smape(actual: TimeSeries, forecast: TimeSeries) -> float:
    """Вычисление SMAPE между двумя рядами."""
    actual_vals = actual.values().flatten()
    forecast_vals = forecast.values().flatten()
    return 100 * np.mean(2 * np.abs(forecast_vals - actual_vals) /
                         (np.abs(actual_vals) + np.abs(forecast_vals) + 1e-8))

def main():
    # 1. Загрузка данных
    series = MilkDataset().load().with_name("Milk production")
    
    # 2. Предобработка (интерполяция пропусков, если есть)
    if series.has_nans():
        series = series.interpolate(method="linear")
    
    # 3. Разделение на train/test (последние 12 месяцев)
    train, test = series.split_before(series.time_index[-12])
    print(f"Train size: {len(train)} months, Test size: {len(test)} months")
    
    # 4. Масштабирование
    scaler = Scaler()
    train_scaled = scaler.fit_transform(train)
    test_scaled = scaler.transform(test)
    
    # 5. Baseline ARIMA
    print("\nОбучение ARIMA...")
    arima = ARIMA()
    arima.fit(train_scaled)
    arima_forecast_scaled = arima.predict(len(test))
    arima_forecast = scaler.inverse_transform(arima_forecast_scaled)
    
    # 6. N-BEATS
    print("Обучение N-BEATS...")
    nbeats = NBEATSModel(
        input_chunk_length=24,
        output_chunk_length=12,
        n_epochs=50,
        batch_size=32,
        random_state=42,
        pl_trainer_kwargs={"accelerator": "auto", "devices": "auto"}
    )
    nbeats.fit(train_scaled, verbose=True)
    nbeats_forecast_scaled = nbeats.predict(len(test))
    nbeats_forecast = scaler.inverse_transform(nbeats_forecast_scaled)
    
    # 7. Метрики
    mape_arima = mape(test, arima_forecast)
    smape_arima = smape(test, arima_forecast)
    mape_nbeats = mape(test, nbeats_forecast)
    smape_nbeats = smape(test, nbeats_forecast)
    
    print("\n=== Результаты ===")
    print(f"ARIMA   - MAPE: {mape_arima:.2f}%, SMAPE: {smape_arima:.2f}%")
    print(f"N-BEATS - MAPE: {mape_nbeats:.2f}%, SMAPE: {smape_nbeats:.2f}%")
    
    # 8. Визуализация и сохранение
    plt.figure(figsize=(12, 6))
    series.plot(label="Факт")
    arima_forecast.plot(label=f"ARIMA (MAPE={mape_arima:.1f}%)")
    nbeats_forecast.plot(label=f"N-BEATS (MAPE={mape_nbeats:.1f}%)")
    plt.title("Прогноз производства молока на 12 месяцев")
    plt.legend()
    plt.grid(True)
    plt.savefig("reports/figures/forecast_comparison.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("График сохранён в reports/figures/forecast_comparison.png")

if __name__ == "__main__":
    main()
