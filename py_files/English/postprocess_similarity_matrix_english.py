import torch
import numpy as np
import os

# --- File paths ---
vocab_path = "home_directorycsln/data/input/fasttext_english_cosine_vocab.txt"
matrix_path = "home_directorycsln/data/input/fasttext_english_cosine.npy"
output_dir = "home_directorycsln/data/input/DM_matrices"

# define language
language = "english"

os.makedirs(output_dir, exist_ok=True)

# --- Load cosine similarity matrix (SM) ---
print("Loading cosine similarity matrix...")
sm = np.load(matrix_path).astype(np.float32)  # Shape: (N, N)
print(f"Matrix shape: {sm.shape}")
N = sm.shape[0]

# --- Convert to torch and move to GPU ---
sm_gpu = torch.from_numpy(sm).to("cuda")

# --- Step 5: Row-normalize SM to get SMNORM (diagonal is already 0) ---
print("Normalizing rows of SM...")
row_sums = sm_gpu.sum(dim=1, keepdim=True)
smnorm_gpu = sm_gpu / row_sums
smnorm_gpu[torch.isnan(smnorm_gpu)] = 0.0  # Replace NaNs from zero division

# --- Step 6: Compute DM = (2 * SMNORM + I) / 3 ---
print("Computing DM matrix...")
identity_gpu = torch.eye(N, device="cuda")
dm_gpu = (2 * smnorm_gpu + identity_gpu) / 3

# --- Step 7: Compute DM^k for k in [1,2,3,4,5] ---
print("Computing DM^k for k in [1,2, 3, 4, 5] and saving...")

dm_power = dm_gpu.clone()
for k in range(1, 6):
    dm_power = dm_power @ dm_gpu
    output_path = os.path.join(output_dir, f"{language}_fasttext_dm_power_k{k}.npy")
    np.save(output_path, dm_power.cpu().numpy().astype(np.float32))
    print(f"Saved DM^{k} to {output_path}")

print("All DM^k matrices saved successfully.")