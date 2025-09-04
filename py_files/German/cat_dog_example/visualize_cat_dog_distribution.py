import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
import arviz as az
import pandas as pd

# Load data
data_path = "home_directorycsln/data/output/cat_dog_joint_activation_distribution.npy"
data = np.load(data_path)
data = (data[0] + data[1]) / 2  # Average first two rows

# Load external thresholds
thresh_path = "home_directorycsln/data/output/reference_joint_activation_decile_thresholds.csv"
thresholds_df = pd.read_csv(thresh_path)
thresholds = np.sort(thresholds_df['threshold'].values)

# Create KDE
kde = gaussian_kde(data)
x_min, x_max = data.min(), 0.00025
x = np.linspace(x_min, x_max, 4000)
y = kde(x)

# Setup plot
az.style.use("arviz-doc")
fig, ax = plt.subplots(figsize=(12, 6))

# Create green color gradient
n_bands = len(thresholds) - 1
colors = plt.cm.Greens(np.linspace(0.3, 0.95, n_bands))

print(f"Data range: {data.min():.2e} to {data.max():.2e}")
print(f"Plot range: {x_min:.2e} to {x_max:.2e}")
print(f"Number of threshold bands: {n_bands}")

# Fill each band
for i in range(n_bands):
    left_thresh = thresholds[i] 
    right_thresh = thresholds[i + 1]
    
    print(f"\nBand {i}: [{left_thresh:.2e}, {right_thresh:.2e}]")
    
    # Only fill if band overlaps with plot range
    if right_thresh < x_min or left_thresh > x_max:
        print(f"  Skipped - outside plot range")
        continue
        
    # Clip to plot range
    left_clipped = max(left_thresh, x_min)
    right_clipped = min(right_thresh, x_max)
    
    print(f"  Clipped: [{left_clipped:.2e}, {right_clipped:.2e}]")
    
    # Create mask for this band
    if i == n_bands - 1:  # Last band includes right boundary
        mask = (x >= left_clipped) & (x <= right_clipped)
    else:
        mask = (x >= left_clipped) & (x < right_clipped)
    
    points_in_band = np.sum(mask)
    if points_in_band > 0:
        ax.fill_between(x[mask], y[mask], 0, color=colors[i], alpha=1.0, linewidth=0)
        print(f"  Filled with {points_in_band} points")
        
        # Count actual data points in this band
        if i == n_bands - 1:
            data_mask = (data >= left_thresh) & (data <= right_thresh)
        else:
            data_mask = (data >= left_thresh) & (data < right_thresh)
        data_count = np.sum(data_mask)
        data_pct = (data_count / len(data)) * 100
        print(f"  Contains {data_count:,} data points ({data_pct:.1f}%)")
    else:
        print(f"  No points to fill")

# Draw threshold lines
for thresh in thresholds[1:-1]:  # Skip first and last
    if x_min <= thresh <= x_max:
        y_at_thresh = np.interp(thresh, x, y)
        if y_at_thresh > 0:
            ax.vlines(thresh, 0, y_at_thresh, colors='white', linewidth=1.5, alpha=0.9)

# Clean up plot
ax.set_xlim(x_min, x_max)
ax.set_xlabel("")
ax.set_ylabel("")
ax.set_title("")
fig.tight_layout()

# Save plot
output_path = "home_directorycsln/data/output/plots/cat_dog_joint_activation_distribution_simple.pdf"
fig.savefig(output_path, format="pdf", bbox_inches="tight", transparent=True)
print(f"\nSaved plot to {output_path}")

# Create band analysis summary for CSV export
band_analysis = []
for i in range(n_bands):
    left_thresh = thresholds[i] 
    right_thresh = thresholds[i + 1]
    
    # Count actual data points in this band
    if i == n_bands - 1:
        data_mask = (data >= left_thresh) & (data <= right_thresh)
    else:
        data_mask = (data >= left_thresh) & (data < right_thresh)
    
    data_count = np.sum(data_mask)
    data_pct = (data_count / len(data)) * 100
    
    band_analysis.append({
        'band_id': f'D{i}-D{i+1}',
        'percentile_range': f'{i*10}%-{(i+1)*10}%',
        'threshold_min': left_thresh,
        'threshold_max': right_thresh,
        'data_points': data_count,
        'percentage': data_pct,
        'in_plot_range': (right_thresh >= x_min and left_thresh <= x_max)
    })

# Convert to DataFrame and save
analysis_df = pd.DataFrame(band_analysis)
csv_path = "home_directorycsln/data/output/cat_dog_vs_reference_band_analysis.csv"
analysis_df.to_csv(csv_path, index=False, float_format='%.2e')
print(f"Saved band analysis to {csv_path}")

# Also save raw data with band assignments
data_with_bands = []
for i, value in enumerate(data):
    # Find which band this data point belongs to
    band_id = -1
    for j in range(n_bands):
        if j == n_bands - 1:  # Last band includes right boundary
            if thresholds[j] <= value <= thresholds[j + 1]:
                band_id = j
                break
        else:
            if thresholds[j] <= value < thresholds[j + 1]:
                band_id = j
                break
    
    data_with_bands.append({
        'data_index': i,
        'activation_value': value,
        'reference_band': f'D{band_id}-D{band_id+1}' if band_id >= 0 else 'outside_range',
        'percentile_band': f'{band_id*10}%-{(band_id+1)*10}%' if band_id >= 0 else 'outside_range'
    })

data_df = pd.DataFrame(data_with_bands)
data_csv_path = "home_directorycsln/data/output/cat_dog_data_with_reference_bands.csv"
data_df.to_csv(data_csv_path, index=False, float_format='%.2e')
print(f"Saved individual data points with band assignments to {data_csv_path}")

plt.show()
