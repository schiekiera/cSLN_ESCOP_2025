import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load and flatten data
file_path = "home_directorycsln/data/output/reference_joint_activation_distribution.npy"
data = np.load(file_path).flatten()

# Set style (no grid to avoid white bands)
sns.set(style="white")

# Create plot
plt.figure(figsize=(12, 6))

# KDE only (no histogram)
line_color = "black"
ax = sns.kdeplot(data, fill=False, color=line_color, linewidth=2)

# Extract KDE curve data
kde_x = ax.lines[0].get_xdata()
kde_y = ax.lines[0].get_ydata()

# Calculate deciles from the original data
data_deciles = np.percentile(data, np.arange(10, 100, 10))
data_min = np.percentile(data, 0)
data_max = np.percentile(data, 100)

# Respect the requested x-maximum
x_max = 0.000075
mask_within_xlim = kde_x <= x_max
kde_x = kde_x[mask_within_xlim]
kde_y = kde_y[mask_within_xlim]

# Create 10 equal-width bands along the KDE x-range for perfect alignment
kde_min = kde_x.min()
kde_max = min(kde_x.max(), x_max)
band_boundaries = np.linspace(kde_min, kde_max, 11)  # 11 points create 10 bands

# Fill each decile band with increasing intensity
for i in range(10):
    band_left = band_boundaries[i]
    band_right = band_boundaries[i + 1]
    
    # Create mask for this specific band
    band_mask = (kde_x >= band_left) & (kde_x <= band_right)
    if not np.any(band_mask):
        continue

    # Alpha increases with the decile index (0.2 to 0.8)
    alpha = 0.2 + (i / 9.0) * 0.6
    ax.fill_between(
        kde_x[band_mask],
        0,
        kde_y[band_mask],
        color="blue",
        alpha=alpha,
        linewidth=0,
    )

# Draw decile markers at the actual band boundaries (perfect alignment)
for i in range(1, 10):  # Skip first and last boundaries (edges)
    boundary = band_boundaries[i]
    if boundary <= x_max:
        y_at_boundary = np.interp(boundary, kde_x, kde_y)
        ax.vlines(boundary, 0, y_at_boundary, colors=line_color, linestyles="-", linewidth=1.5)

# Also draw the original data deciles as reference (optional - can be removed)
# for d in data_deciles:
#     if d <= x_max:
#         y_at_d = np.interp(d, kde_x, kde_y)
#         ax.vlines(d, 0, y_at_d, colors="red", linestyles=":", linewidth=1, alpha=0.5)

# Remove labels and title
plt.xlabel("")
plt.ylabel("")
plt.title("")

# Clean layout
plt.tight_layout()

# Ensure no gridlines
ax.grid(False)

# Set the x-axis limit to a maximum of 0.000075.
plt.xlim(left=0, right=0.000075)

# Save as vector PDF with no extra borders, background transparent
plt.savefig("home_directorycsln/data/output/reference_joint_activation_distribution.pdf", format='pdf', bbox_inches='tight', transparent=True)

print("Saved reference joint activation distribution to home_directorycsln/data/output/reference_joint_activation_distribution.pdf")

# Optionally show it
plt.show()