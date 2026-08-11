import matplotlib.pyplot as plt


def plot_model_comparison(actual, forecasts, output_path):
    fig, ax = plt.subplots(figsize=(15, 7))
    actual.plot(ax=ax, label="Actual", linewidth=2)
    for name, forecast in forecasts.items():
        forecast.plot(ax=ax, label=name)
    ax.set_title("24-Hour Appliance Energy Forecast Comparison")
    ax.set_xlabel("Date")
    ax.set_ylabel("Appliance Energy Use")
    ax.legend()
    plt.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
