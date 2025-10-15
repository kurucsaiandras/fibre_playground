import torch
import matplotlib.pyplot as plt
import utils

def main():
    save_figs = False
    name = "test_run_radii"
    step = None
    if step is None:
        path = utils.latest_rve(f"results/{name}/rve")
    else:
        path = f"results/{name}/rve/{step}.pt"
    model = torch.load(path)
    fibre_coords = model["fibre_coords"]  # (n_fibres, n_points, 3)

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
    bins = 50
    ax[0].hist(angles_xz, bins=bins, edgecolor='black')
    ax[0].set_xlabel("xz angles (degrees)")
    ax[0].set_ylabel("Frequency")
    ax[1].hist(angles_yz, bins=bins, edgecolor='black')
    ax[1].set_xlabel("yz angles (degrees)")
    ax[1].set_ylabel("Frequency")
    ax[2].hist(angles_3d, bins=bins, edgecolor='black')
    ax[2].set_xlabel("3D angles to z (degrees)")
    ax[2].set_ylabel("Frequency")
    plt.suptitle("Fibre Angle Distributions")
    if save_figs:
        plt.savefig("figs/test_run_radii/angle_hist_xy.png")
    else:
        plt.show()

if __name__ == "__main__":
    main()