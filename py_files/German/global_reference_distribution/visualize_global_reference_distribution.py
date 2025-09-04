import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
import arviz as az
import pandas as pd

# --- Load and flatten data ---
file_path = "home_directorycsln/data/output/reference_joint_activation_distribution.npy"
data = np.load(file_path).flatten()

# --- Style / figure ---
az.style.use("arviz-doc")
fig, ax = plt.subplots(figsize=(12, 6))

# --- KDE (Scott’s rule) ---
kde = gaussian_kde(data)
x = np.linspace(data.min(), data.max(), 4000)  # denser grid for smooth edges
y = kde(x)

# --- Decile edges (0..1 in 0.1 steps) ---
xq = np.quantile(data, np.linspace(0, 1, 11))

# --- Progressive colors (10 shades) ---
colors = plt.cm.Greens(np.linspace(0.35, 0.95, 10))

# --- Fix x-limit before computing pixel->data conversion (keep your cap) ---
ax.set_xlim(left=float(x.min()), right=0.000075)

# --- Compute a constant gap in data units from a pixel target ---
gap_px = 6  # << adjust this for more/less space (in pixels)
fig.canvas.draw()  # ensure we have a renderer
# Axes width in pixels
bbox = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
axes_width_px = bbox.width * fig.dpi
# Convert pixel gap to data-units gap along x
x_min, x_max = ax.get_xlim()
gap_abs = (x_max - x_min) * (gap_px / max(axes_width_px, 1))

# --- Fill each decile band with a constant gap ---
for i in range(len(xq) - 1):
    # Shrink band by half-gap on each side (skip if band would invert)
    left_raw, right_raw = xq[i], xq[i + 1]
    if right_raw - left_raw <= gap_abs:
        continue  # extremely narrow band; nothing to draw with the chosen gap
    left = left_raw + gap_abs / 2
    right = right_raw - gap_abs / 2

    # Build mask (close the last band)
    if i < len(xq) - 2:
        mask = (x >= left) & (x < right)
    else:
        mask = (x >= left) & (x <= right)

    ax.fill_between(x[mask], y[mask], 0, color=colors[i], alpha=0.7, linewidth=0)

# --- No outline line on top ---
ax.set_xlabel("")
ax.set_ylabel("")
ax.set_title("")
fig.tight_layout()

# --- Save decile thresholds to CSV ---
decile_df = pd.DataFrame({
    'decile': [f'D{i}' for i in range(11)],  # D0, D1, D2, ..., D10
    'percentile': np.arange(0, 110, 10),     # 0%, 10%, 20%, ..., 100%
    'threshold': xq
})

csv_path = "home_directorycsln/data/output/reference_joint_activation_decile_thresholds.csv"
decile_df.to_csv(csv_path, index=False)
print(f"Saved decile thresholds to {csv_path}")

# --- Save as vector PDF (transparent background) ---
out_path = "home_directorycsln/data/output/plots/reference_joint_activation_distribution.pdf"
fig.savefig(out_path, format="pdf", bbox_inches="tight", transparent=True)
print(f"Saved plot to {out_path}")

plt.show()