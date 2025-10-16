import torch
import matplotlib.pyplot as plt
import utils
import os
import math

def main():
    save_figs = True
    name = "std_radii"
    step = 358901
    if step is None:
        path = utils.latest_rve(f"results/{name}/rve")
    else:
        path = f"results/{name}/rve/{step}.pt"
    model = torch.load(path)
    fibre_coords = model["fibre_coords"]  # (n_fibres, n_points, 3)
    domain_size = model["domain_size"]
    fibre_r = model["fibre_r"]
    step = model["step"]
    volume_ratio = torch.pow(fibre_r, 2).sum() * math.pi / (domain_size[0] * domain_size[1])

    # Vectors between consecutive points
    vecs = fibre_coords[:, 1:] - fibre_coords[:, :-1]  # (n_fibres, n_points-1, 3)
    # Angles to axis in xz plane
    angles_xz = (torch.atan2(vecs[:,:,0], vecs[:,:,2]) * 180.0 / torch.pi).detach().cpu().numpy().flatten() # (n_fibres, n_points-1)
    # Angles to axis in yz plane
    angles_yz = (torch.atan2(vecs[:,:,1], vecs[:,:,2]) * 180.0 / torch.pi).detach().cpu().numpy().flatten() # (n_fibres, n_points-1)
    # 3D angles to z axis
    angles_3d = (torch.atan2(torch.norm(vecs[:,:,:2], dim=2), vecs[:,:,2]) * 180.0 / torch.pi).detach().cpu().numpy().flatten()
    # Plot histograms
    fig, ax = plt.subplots(1, 3, figsize=(18, 6), constrained_layout=True)
    bins = 100
    ax[0].hist(angles_xz, bins=bins, edgecolor='black', density=True)
    ax[0].set_xlabel("xz angles (degrees)")
    ax[0].set_ylabel("Frequency")
    ax[1].hist(angles_yz, bins=bins, edgecolor='black', density=True)
    ax[1].set_xlabel("yz angles (degrees)")
    ax[1].set_ylabel("Frequency")
    ax[2].hist(angles_3d, bins=bins, edgecolor='black', density=True)
    ax[2].set_xlabel("3D angles to z (degrees)")
    ax[2].set_ylabel("Frequency")
    plt.suptitle(f"Fibre Angle Distributions for volume ratio {volume_ratio:.3f}")
    if save_figs:
        os.makedirs(f"figs/{name}", exist_ok=True)
        plt.savefig(f"figs/{name}/angle_hist_xy_{step}.png")
    else:
        plt.show()

if __name__ == "__main__":
    main()