import torch
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import numpy as np
from rve import RVE
import os
import utils

def eval(name, step, device, save_figs):
    if step is None: step = utils.latest_rve(f"results/{name}/rve")
    rve = RVE.eval(name, step, device)
    root = f"results/{name}/figs/{step}"
    if save_figs:
        os.makedirs(root, exist_ok=True)
    
    # Vectors between consecutive points
    vecs = rve.fibre_coords[:, 1:] - rve.fibre_coords[:, :-1]  # (n_fibres, n_points-1, 3)
    lengths = torch.norm(vecs, dim=2).cpu().numpy().flatten()  # (n_fibres, n_points-1)
    # Angles to axis in xz plane
    angles_xz = (torch.atan2(vecs[:,:,0], vecs[:,:,2]) * 180.0 / torch.pi).cpu().numpy().flatten() # (n_fibres, n_points-1)
    # Angles to axis in yz plane
    angles_yz = (torch.atan2(vecs[:,:,1], vecs[:,:,2]) * 180.0 / torch.pi).cpu().numpy().flatten() # (n_fibres, n_points-1)
    # 3D angles to z axis
    angles_3d = (torch.atan2(torch.norm(vecs[:,:,:2], dim=2), vecs[:,:,2]) * 180.0 / torch.pi).cpu().numpy()

    # Plot histograms
    fig, ax = plt.subplots(1, 3, figsize=(18, 6), constrained_layout=True)
    bins = 100
    ax[0].hist(angles_xz, bins=bins, weights=lengths, edgecolor='black', density=True)
    ax[0].set_xlabel("xz angles (degrees)")
    ax[0].set_ylabel("Frequency")
    ax[1].hist(angles_yz, bins=bins, weights=lengths, edgecolor='black', density=True)
    ax[1].set_xlabel("yz angles (degrees)")
    ax[1].set_ylabel("Frequency")
    ax[2].hist(angles_3d.flatten(), bins=bins, weights=lengths, edgecolor='black', density=True)
    ax[2].set_xlabel("3D angles to z (degrees)")
    ax[2].set_ylabel("Frequency")
    plt.suptitle(f"Fibre Angle Distributions") # for volume ratio {rve.get_fibre_to_volume_ratio():.3f}")
    if save_figs:
        plt.savefig(f"{root}/angle_hist.png")
    else:
        plt.show()

    # Plot trajectories
    fig, ax = plt.subplots(figsize=(6, 6))
    fibres_np = rve.fibre_coords.cpu().numpy()
    for i in range(fibres_np.shape[0]):
        x = fibres_np[i, :, 0]
        y = fibres_np[i, :, 1]
        angle = angles_3d[i]  # (resolution-1,)
        # Create list of segments [(x0,y0)-(x1,y1), (x1,y1)-(x2,y2), ...]
        points = np.array([x, y]).T.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)
        # Create a LineCollection with colormap based on angle
        lc = LineCollection(segments, cmap='jet', norm=plt.Normalize(angles_3d.min(), angles_3d.max()))
        lc.set_array(angle)
        lc.set_linewidth(2)
        ax.add_collection(lc)
        # Add scatter points (segment endpoints)
        #ax.scatter(x, y, s=10, edgecolor='k', linewidth=0.3, alpha=0.8)
    # Format plot
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_title("XY-plane Projection Colored by 3D Angle to Z-axis")
    ax.axis("equal")
    # Add colorbar
    cbar = plt.colorbar(lc, ax=ax)
    cbar.set_label("Angle to Z-axis (degrees)")
    if save_figs:
        plt.savefig(f"{root}/trajectories.png")
    else:
        plt.show()

    # Calculate and plot tortuosity
    real_lengths = vecs.norm(dim=2).sum(dim=1)
    end_to_end = torch.norm(rve.fibre_coords[:, -1] - rve.fibre_coords[:, 0], dim=1)
    ratios = (real_lengths / end_to_end).detach().cpu().numpy()
    # Plot histogram
    plt.figure(figsize=(6, 4))
    plt.hist(ratios, bins=50, color='steelblue', edgecolor='black')
    plt.xlabel("Length Ratio (Actual / End-to-End)")
    plt.ylabel("Number of Fibres")
    plt.title("Fibre Tortuosity Distribution")
    plt.grid(alpha=0.3)
    if save_figs:
        plt.savefig(f"{root}/tortuosity.png")
    else:
        plt.show()

def main():
    name = 'airtex'
    step = None
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)
    eval(name, step, device, save_figs=False)

if __name__ == "__main__":
    main()