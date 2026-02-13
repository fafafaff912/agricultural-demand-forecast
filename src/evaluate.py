"""
Оценка и сравнение моделей: MAPE, SMAPE, таблица, графики.

Запуск:
    python evaluate.py
"""

import json
import os

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


def load_results(path="results/metrics.json"):
    """Загрузка результатов из JSON."""
    if not os.path.exists(path):
        print(f"Файл не найден: {path}")
        print("Сначала запустите: python train.py")
        return None

    with open(path, encoding="utf-8") as f:
        return json.load(f)


def print_comparison_table(results):
    """Печать таблицы сравнения моделей."""
    metrics = results["metrics"]

    print(f"\n{'='*65}")
    print(f"СРАВНЕНИЕ МОДЕЛЕЙ — {results['product']} ({results['country']})")
    print(f"Горизонт: {results['horizon']} мес.")
    print(f"{'='*65}")
    print(f"\n{'Модель':<15} {'MAPE ↓':>8} {'SMAPE ↓':>8} {'MAE ↓':>10} {'RMSE ↓':>10}")
    print("-" * 55)

    # Сортировка по SMAPE
    sorted_models = sorted(metrics.items(), key=lambda x: x[1]["SMAPE"])

    for name, m in sorted_models:
        medal = "🥇" if name == sorted_models[0][0] else "  "
        print(f"{medal}{name:<13} {m['MAPE']:>7.2f}% {m['SMAPE']:>7.2f}% "
              f"{m['MAE']:>10.2f} {m['RMSE']:>10.2f}")

    best_name = sorted_models[0][0]
    worst_name = sorted_models[-1][0]
    improvement = (
        (metrics[worst_name]["SMAPE"] - metrics[best_name]["SMAPE"])
        / metrics[worst_name]["SMAPE"] * 100
    )
    print(f"\n✓ Лучшая модель: {best_name}")
    print(f"  Улучшение SMAPE vs {worst_name}: {improvement:.1f}%")


def plot_metrics_comparison(results, save_path="results/metrics_comparison.png"):
    """Столбчатые диаграммы MAPE и SMAPE по моделям."""
    metrics = results["metrics"]
    models = list(metrics.keys())

    mape_vals = [metrics[m]["MAPE"] for m in models]
    smape_vals = [metrics[m]["SMAPE"] for m in models]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    colors = ["#e74c3c", "#3498db", "#2ecc71", "#9b59b6"][:len(models)]

    # MAPE
    bars1 = ax1.bar(models, mape_vals, color=colors, edgecolor="white", linewidth=1.5)
    ax1.set_ylabel("MAPE (%)", fontsize=12)
    ax1.set_title("MAPE по моделям (↓ лучше)", fontsize=13, fontweight="bold")
    ax1.grid(axis="y", alpha=0.3)
    for bar, val in zip(bars1, mape_vals):
        ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                 f"{val:.1f}%", ha="center", fontweight="bold")

    # SMAPE
    bars2 = ax2.bar(models, smape_vals, color=colors, edgecolor="white", linewidth=1.5)
    ax2.set_ylabel("SMAPE (%)", fontsize=12)
    ax2.set_title("SMAPE по моделям (↓ лучше)", fontsize=13, fontweight="bold")
    ax2.grid(axis="y", alpha=0.3)
    for bar, val in zip(bars2, smape_vals):
        ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                 f"{val:.1f}%", ha="center", fontweight="bold")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\nГрафик сравнения: {save_path}")


def plot_forecast_detail(results, save_path="results/forecast_detail.png"):
    """Детальный график прогнозов с ошибками."""
    actual = np.array(results["test_actual"])
    forecasts = results["forecasts"]
    months = np.arange(1, len(actual) + 1)

    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    colors = {"ARIMA": "#e74c3c", "Prophet": "#3498db", "N-BEATS": "#2ecc71", "TFT": "#9b59b6"}

    for ax, (name, forecast) in zip(axes.flatten(), forecasts.items()):
        forecast = np.array(forecast)
        error = actual - forecast

        ax.plot(months, actual, "ko-", label="Факт", ms=5, linewidth=2)
        ax.plot(months, forecast, "o--", label=f"{name}", color=colors.get(name, "gray"),
                ms=5, linewidth=1.5)
        ax.fill_between(months, actual, forecast, alpha=0.2, color=colors.get(name, "gray"))

        mape_val = np.mean(np.abs(error / actual)) * 100
        ax.set_title(f"{name} (MAPE: {mape_val:.1f}%)", fontsize=12, fontweight="bold")
        ax.set_xlabel("Месяц")
        ax.set_ylabel("Объём")
        ax.legend()
        ax.grid(alpha=0.3)

    plt.suptitle(
        f"Детальное сравнение прогнозов — {results['product']}",
        fontsize=14, fontweight="bold",
    )
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Детальный график: {save_path}")


def plot_error_distribution(results, save_path="results/error_distribution.png"):
    """Распределение ошибок прогнозов."""
    actual = np.array(results["test_actual"])
    forecasts = results["forecasts"]

    fig, ax = plt.subplots(figsize=(12, 5))
    colors = {"ARIMA": "#e74c3c", "Prophet": "#3498db", "N-BEATS": "#2ecc71", "TFT": "#9b59b6"}

    for name, forecast in forecasts.items():
        forecast = np.array(forecast)
        pct_error = (actual - forecast) / actual * 100
        ax.hist(pct_error, bins=15, alpha=0.5, label=name,
                color=colors.get(name, "gray"), edgecolor="white")

    ax.axvline(0, color="black", linestyle="--", alpha=0.5)
    ax.set_xlabel("Ошибка (%)", fontsize=12)
    ax.set_ylabel("Частота", fontsize=12)
    ax.set_title("Распределение ошибок прогнозов", fontsize=13, fontweight="bold")
    ax.legend()
    ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Распределение ошибок: {save_path}")


def main():
    results = load_results()
    if results is None:
        return

    print_comparison_table(results)
    plot_metrics_comparison(results)
    plot_forecast_detail(results)
    plot_error_distribution(results)

    # Сохранение текстового отчёта
    metrics = results["metrics"]
    with open("results/report.txt", "w", encoding="utf-8") as f:
        f.write(f"ОТЧЁТ: Прогноз спроса на {results['product']}\n")
        f.write(f"Страна: {results['country']}\n")
        f.write(f"Горизонт: {results['horizon']} мес.\n")
        f.write("=" * 55 + "\n\n")
        f.write(f"{'Модель':<15} {'MAPE':>8} {'SMAPE':>8} {'MAE':>10} {'RMSE':>10}\n")
        f.write("-" * 55 + "\n")
        for name, m in sorted(metrics.items(), key=lambda x: x[1]["SMAPE"]):
            f.write(f"{name:<15} {m['MAPE']:>7.2f}% {m['SMAPE']:>7.2f}% "
                    f"{m['MAE']:>10.2f} {m['RMSE']:>10.2f}\n")

    print(f"\nОтчёт сохранён: results/report.txt")


if __name__ == "__main__":
    main()
