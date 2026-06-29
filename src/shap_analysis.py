import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parents[1]))
from config import FIGURE_DIR

import shap
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def analyze_shap(model, X_sample: pd.DataFrame, feature_names: list, output_prefix: str = "shap"):
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_sample)

    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values, X_sample, feature_names=feature_names, show=False)
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / f"{output_prefix}_summary.png", dpi=150, bbox_inches="tight")
    plt.close()

    plt.figure(figsize=(10, 8))
    shap.bar_plot(
        np.abs(shap_values).mean(axis=0),
        feature_names=feature_names,
        show=False,
    )
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / f"{output_prefix}_importance.png", dpi=150, bbox_inches="tight")
    plt.close()

    print(f"SHAP plots saved to {FIGURE_DIR}")
