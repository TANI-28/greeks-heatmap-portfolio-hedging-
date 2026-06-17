from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def save_heatmap(
    matrix: np.ndarray,
    spot_shocks: np.ndarray,
    vol_shocks: np.ndarray,
    title: str,
    output_path: Path,
    cbar_label: str,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 7))
    vmax = float(np.nanmax(np.abs(matrix)))
    im = ax.imshow(
        matrix,
        origin="lower",
        aspect="auto",
        cmap="RdBu_r",
        vmin=-vmax,
        vmax=vmax,
        extent=[
            float(spot_shocks.min() * 100),
            float(spot_shocks.max() * 100),
            float(vol_shocks.min() * 100),
            float(vol_shocks.max() * 100),
        ],
    )
    ax.set_title(title)
    ax.set_xlabel("Spot Shock (%)")
    ax.set_ylabel("Vol Shock (vol points)")
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label(cbar_label)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)

