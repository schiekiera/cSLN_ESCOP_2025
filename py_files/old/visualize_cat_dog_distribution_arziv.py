import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
import arviz as az
import pandas as pd

# --- Load and flatten data ---
file_path = "home_directorycsln/data/output/cat_dog_joint_activation_distribution.npy"

data = np.load(file_path)

# add first and second row together and divide by 2
data = (data[0] + data[1]) / 2

# load thresholds from csv
thresholds_path = "home_directorycsln/data/output/reference_joint_activation_decile_thresholds.csv"
thresholds = pd.read_csv(thresholds_path)

# get decile thresholds
decile_thresholds = thresholds['threshold'].values



# --- Style / figure ---
az.style.use("arviz-doc")
fig, ax = plt.subplots(figsize=(12, 6))

# --- KDE (Scott’s rule) ---
kde = gaussian_kde(data)
x = np.linspace(data.min(), data.max(), 4000)  # denser grid for smooth edges
y = kde(x)

# --- Decile edges from CSV (0%-100%) ---
# Use thresholds loaded from CSV instead of computing from this data
xq = np.asarray(decile_thresholds, dtype=float)
xq = np.sort(xq)

# --- Fix x-limit before computing pixel->data conversion (keep your cap) ---
x_max = 0.00025
ax.set_xlim(left=float(x.min()), right=x_max)

# Use reference thresholds for both analysis and visualization
reference_thresholds = xq  # Keep original thresholds
data_clipped = data[(data >= x.min()) & (data <= x_max)]

# Use reference thresholds for band creation (this is the comparison you want!)
xq = reference_thresholds

print(f"\n=== REFERENCE THRESHOLD ANALYSIS ===\n")
print(f"Using reference thresholds to analyze cat-dog data distribution:")
for i, thresh in enumerate(xq):
    print(f"  D{i} ({i*10}%): {thresh:.2e}")
print()

# --- Progressive colors (10 shades, green gradient) ---
colors = plt.cm.Greens(np.linspace(0.2, 0.95, 10))


# --- Compute a constant gap in data units from a pixel target ---
gap_px = 2  # << small gap between bands for visual separation
fig.canvas.draw()  # ensure we have a renderer
# Axes width in pixels
bbox = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
axes_width_px = bbox.width * fig.dpi
# Convert pixel gap to data-units gap along x
x_min, x_max = ax.get_xlim()
gap_abs = (x_max - x_min) * (gap_px / max(axes_width_px, 1))

# --- No base fill - rely on decile bands for visualization ---

# Count how many cat-dog data points fall into each REFERENCE decile band
print(f"\n=== CAT-DOG vs REFERENCE DECILE ANALYSIS ===\n")
data_total = len(data_clipped)
print(f"Total cat-dog data points in range: {data_total:,}")
print(f"Cat-dog data range: {data_clipped.min():.2e} to {data_clipped.max():.2e}\n")

# Analyze against reference thresholds
num_bands = len(xq) - 1
for i in range(num_bands):
    left_thresh, right_thresh = xq[i], xq[i + 1]
    
    # Count cat-dog data points in this reference decile band
    if i < num_bands - 1:
        data_mask = (data_clipped >= left_thresh) & (data_clipped < right_thresh)
    else:
        data_mask = (data_clipped >= left_thresh) & (data_clipped <= right_thresh)
    
    data_count = np.sum(data_mask)
    percentage = (data_count / data_total) * 100 if data_total > 0 else 0
    
    print(f"Reference Band D{i}-D{i+1} ({i*10}%-{(i+1)*10}%): [{left_thresh:.2e}, {right_thresh:.2e}]")
    print(f"  Cat-dog points in this reference band: {data_count:,} ({percentage:.1f}%)")
    print()

# Now create empirical deciles for better visualization while keeping reference analysis
print(f"=== CREATING EMPIRICAL BANDS FOR VISUALIZATION ===\n")
xq_empirical = np.percentile(data_clipped, np.arange(0, 101, 10))  # 0%, 10%, 20%, ..., 100%
print("Using empirical deciles for visual banding (but analysis above uses reference thresholds)")

# --- Fill with empirical deciles for better visual gradient ---
num_bands = len(xq_empirical) - 1
band_colors = plt.cm.Greens(np.linspace(0.2, 0.95, num_bands))

for i in range(num_bands):
    # Get empirical band boundaries for visualization
    left_raw, right_raw = xq_empirical[i], xq_empirical[i + 1]
    
    # Apply small gaps for visual separation
    left = left_raw + gap_abs / 2 if i > 0 else left_raw
    right = right_raw - gap_abs / 2 if i < num_bands - 1 else right_raw

    # Build mask for KDE plotting
    if i < num_bands - 1:
        mask = (x >= left) & (x < right)
    else:
        mask = (x >= left) & (x <= right)

    # Fill band if it contains KDE points
    if np.any(mask):
        ax.fill_between(x[mask], y[mask], 0, color=band_colors[i], alpha=1.0, linewidth=0)

# --- Draw reference threshold lines (from external CSV) ---
for d in reference_thresholds[1:-1]:  # skip first and last
    if x.min() < d < x_max:  # only draw lines within our x-range
        y_at_d = float(np.interp(d, x, y))
        if y_at_d > 0:
            ax.vlines(d, 0, y_at_d, colors="white", linestyles="-", linewidth=1.5, alpha=0.9)

# --- No outline line on top ---
ax.set_xlabel("")
ax.set_ylabel("")
ax.set_title("")
fig.tight_layout()

# --- Save as vector PDF (transparent background) ---
out_path = "home_directorycsln/data/output/plots/cat_dog_joint_activation_distribution.pdf"
fig.savefig(out_path, format="pdf", bbox_inches="tight", transparent=True)
print(f"Saved plot to {out_path}")

plt.show()
plt.show()