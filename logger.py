import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

class Logger:
    def __init__(self, job_name, file_name):
        self.job_name = job_name
        self.file_name = file_name
        os.makedirs(f"results/{self.job_name}/logs", exist_ok=True)
        with open(f"results/{self.job_name}/logs/{self.file_name}.csv", "a") as f:
            f.write(
                "step,time,length,curvature,boundary,overlap,total,"
                "n_fibres,fibre_r_mean,domain_size_x,domain_size_y,"
                "domain_size_z,volume_ratio,phase\n"
            )

    def log(self, simulation, time):
        loss = simulation.optimizer.losses
        n_fibres = simulation.rve.fibre_coords.shape[0]
        fibre_r_mean = simulation.rve.fibre_r.mean().item()
        domain_size = simulation.rve.domain_size.cpu().numpy()
        if simulation.optimizer.phase == 'overlap': phase = 0
        elif simulation.optimizer.phase == 'joint': phase = 1
        with open(f"results/{self.job_name}/logs/{self.file_name}.csv", "a") as f:
            fibre_to_volume_ratio = simulation.rve.get_fibre_to_volume_ratio()
            f.write(
                f"{simulation.current_step},{time:.6f},{loss['length']:.6f},{loss['curvature']:.6f},"
                f"{loss['boundary']:.6f},{loss['overlap']:.12f},{sum(loss.values()):.6f},{n_fibres},"
                f"{fibre_r_mean:.3f},{domain_size[0]:.3f},{domain_size[1]:.3f},"
                f"{domain_size[2]:.3f},{fibre_to_volume_ratio:.6f},{phase}\n"
            )

    def plot(self, save_figs):
        '''Plots the current contents of the saved log file'''
        csv_path = f"results/{self.job_name}/logs/{self.file_name}.csv"
        root = f"results/{self.job_name}/figs"
        if save_figs:
            os.makedirs(root, exist_ok=True)
        # --- Load data ---
        df = pd.read_csv(csv_path)

        # --- Convert relevant columns to numpy arrays ---
        steps = df["step"].to_numpy()
        total_loss = df["total"].to_numpy()
        length_loss = df["length"].to_numpy()
        curvature_loss = df["curvature"].to_numpy()
        boundary_loss = df["boundary"].to_numpy()
        overlap_loss = df["overlap"].to_numpy()
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
        plt.plot(steps, overlap_loss, label="Overlap")

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
        if save_figs:
            plt.savefig(f"{root}/losses.png", dpi=400)
        else:
            plt.show()