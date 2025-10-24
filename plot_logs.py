import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# --- Configuration ---
name = "full_config_small_k"
csv_path = f"results/{name}/logs/progress.csv"  # change this to your actual CSV path

# --- Load data ---
df = pd.read_csv(csv_path)

# --- Convert relevant columns to numpy arrays ---
steps = df["step"].to_numpy()
total_loss = df["total"].to_numpy()
length_loss = df["length"].to_numpy()
curvature_loss = df["curvature"].to_numpy()
boundary_loss = df["boundary"].to_numpy()
collision_loss = df["collision"].to_numpy()
volume_ratio = df["volume_ratio"].to_numpy()

# --- Find indices where volume_ratio changes ---
change_indices = np.where(np.diff(volume_ratio) != 0)[0] + 1  # +1 → mark new value positions

# --- Plot ---
plt.figure(figsize=(15, 7))

# Plot total and individual losses
plt.plot(steps, total_loss, label="Total Loss", color="black", linewidth=2)
plt.plot(steps, length_loss, label="Length")
plt.plot(steps, curvature_loss, label="Curvature")
plt.plot(steps, boundary_loss, label="Boundary")
plt.plot(steps, collision_loss, label="Collision")

# Log scale
plt.yscale("log")
plt.ylim(bottom=1e-7)

# Add vertical lines for fibre_r_mean changes
for i in change_indices:
    plt.axvline(x=steps[i], color="grey", linestyle="--", alpha=0.5)
    plt.text(
        steps[i], plt.ylim()[1], f"{volume_ratio[i]:.3f}",
        rotation=90, va="bottom", ha="center", fontsize=8, color="grey"
    )

# Labels and formatting
plt.xlabel("Step")
plt.ylabel("Loss (log scale)")
plt.title("Loss Components over Steps")
plt.legend()
plt.grid(True, which="both", linestyle="--", linewidth=0.5)
plt.tight_layout()
plt.savefig(f"{name}_total_loss_plot.png", dpi=400)
