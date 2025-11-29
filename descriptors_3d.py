import torch
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import numpy as np
from rve import RVE
import os
import utils
from torchkbnufft import KbNufftAdjoint

def fft(fibre_coords, domain_size):
    device = fibre_coords.device
    # --- Segment directions and structure tensors ---
    v = fibre_coords[:, 1:] - fibre_coords[:, :-1]  # (n_fibres, n_points-1, 3)
    v = torch.cat([v, v[:, -1:, :]], dim=1)   # pad last
    v = v.reshape(-1, 3)                        # (N, 3)
    x = fibre_coords.reshape(-1, 3)             # (N, 3)

    if False: # cut central square if cylindrical domain
        radius = min(domain_size[:2]) / 2
        center = domain_size[:2] / 2
        side = min(domain_size[:2]) / torch.sqrt(torch.tensor(2.0))
        lower_bound = center - side / 2
        upper_bound = center + side / 2
        mask_x = (x[:, 0] >= lower_bound[0]) & (x[:, 0] <= upper_bound[0])
        mask_y = (x[:, 1] >= lower_bound[1]) & (x[:, 1] <= upper_bound[1])
        mask = mask_x & mask_y
        x = x[mask]
        v = v[mask]
        # shift x to start from (0,0)
        x[:, 0] -= lower_bound[0]
        x[:, 1] -= lower_bound[1]

    v = v - v.mean(dim=0, keepdim=True)
    x = x.T.unsqueeze(0) / domain_size[0]
    size = 400
    nufft = KbNufftAdjoint(im_size=(size, size, size)).to(device)

    vx_cplx = torch.stack([v[:,0], torch.zeros_like(v[:,0])], dim=-1).unsqueeze(0).unsqueeze(0)
    vy_cplx = torch.stack([v[:,1], torch.zeros_like(v[:,1])], dim=-1).unsqueeze(0).unsqueeze(0)
    vz_cplx = torch.stack([v[:,2], torch.zeros_like(v[:,2])], dim=-1).unsqueeze(0).unsqueeze(0)

    # Create (3, N) input for each vector component
    vx_f = nufft(vx_cplx, x)
    vy_f = nufft(vy_cplx, x)
    vz_f = nufft(vz_cplx, x)

    # Stack components → (64,64,64,3,2)
    F = torch.stack([vx_f, vy_f, vz_f], dim=-2).squeeze(0).squeeze(0)

    # Compute power spectrum
    P = (F[...,0]**2 + F[...,1]**2).sum(dim=-1)
    return P.cpu().numpy()

def structure_bins(fibre_coords, domain_size, radii, border_type, chunk_size=100):
    """
    Args:
        fibre_coords: (n_fibres, n_points, 3)
        domain_size: (3,)
        radii: list or 1D tensor of floats
        border_type: 'rect' or 'cyl'
        chunk_size: number of grid points processed per chunk

    Returns:
        vectors: (n_radii, nx, ny, nz, 3)
        std_dev: (n_radii, nx, ny, nz)
    """
    radii = torch.as_tensor(radii, device=fibre_coords.device, dtype=torch.float32)

    # --- Segment directions and structure tensors ---
    vecs = fibre_coords[:, 1:] - fibre_coords[:, :-1]  # (n_fibres, n_points-1, 3)
    vecs = torch.cat([vecs, vecs[:, -1:, :]], dim=1)   # pad last
    vecs = vecs.reshape(-1, 3)                         # (N, 3)
    fibre_points = fibre_coords.reshape(-1, 3)         # (N, 3)

    # --- Create dense grid with spacing = smallest radius ---
    min_r = torch.min(radii)
    grid_x = torch.arange(0, domain_size[0], min_r, device=fibre_coords.device)
    grid_y = torch.arange(0, domain_size[1], min_r, device=fibre_coords.device)
    grid_z = torch.arange(0, domain_size[2], min_r, device=fibre_coords.device)
    grid = torch.stack(torch.meshgrid(grid_x, grid_y, grid_z, indexing="ij"), dim=-1)  # (nx, ny, nz, 3)
    G = grid.reshape(-1, 3)  # (M, 3)

    n_radii = len(radii)
    M = G.shape[0]
    nx, ny, nz = grid.shape[:3]

    # --- Allocate accumulator ---
    summed_vectors = torch.zeros(n_radii, M, 3, device=fibre_coords.device)
    summed_stddev = torch.zeros(n_radii, M, device=fibre_coords.device)

    # --- Process in chunks to limit memory use ---
    for start in range(0, M, chunk_size):
        end = min(start + chunk_size, M)
        G_chunk = G[start:end]  # (chunk_size, 3)

        # distances: (chunk_size, N)
        dists = torch.cdist(G_chunk, fibre_points)  

        # masks for all radii: (n_radii, chunk_size, N)
        masks = (dists.unsqueeze(0) < radii[:, None, None])

        # add vectors that are in range
        # chunk_sum: (n_radii, chunk_size, 3)
        chunk_sum = torch.einsum('rkn,nd->rkd', masks.float(), vecs)
        # add stddev of angle of vectors to the sum
        vecs_norm = vecs / (vecs.norm(dim=-1, keepdim=True) + 1e-8)  # (N, 3)
        # Compute dot product: (radii, chunk_size, N)
        dots = torch.einsum("rkd,nd->rkn", chunk_sum, vecs_norm)
        # Norms
        chunk_norms = chunk_sum.norm(dim=-1, keepdim=True)  # (radii, chunk_size, 1)
        # vecs_norm are already normalized

        cos = dots / (chunk_norms + 1e-8)  # since vecs_norm is unit length
        cos = torch.clamp(cos, -1.0, 1.0)
        angles = torch.acos(cos) * (180.0 / torch.pi)  # (n_radii, chunk_size, N)
        angles_in_range = torch.where(masks, angles, torch.zeros_like(angles))  # (n_radii, chunk_size, N)
        # compute stddev
        mean_angles = torch.sqrt(torch.sum(angles_in_range**2, dim=-1) / (masks.sum(dim=-1) + 1e-8))  # (n_radii, chunk_size)

        if border_type == 'rect':
            # G_chunk: (chunk_size, 3)
            # radii: (n_radii,)
            gx, gy, gz = G_chunk[:, 0], G_chunk[:, 1], G_chunk[:, 2]
            r = radii[:, None]  # (n_radii, 1)

            mask_x = (gx >= r) & (gx <= (domain_size[0] - r))
            mask_y = (gy >= r) & (gy <= (domain_size[1] - r))
            mask_z = (gz >= r) & (gz <= (domain_size[2] - r))

            border_mask = mask_x & mask_y & mask_z  # (n_radii, chunk_size)
            # Broadcast to (n_radii, chunk_size, 3, 3)
            border_mask_3d = border_mask[:, :, None]
            chunk_sum = torch.where(border_mask_3d, chunk_sum, torch.zeros_like(chunk_sum))
            mean_angles = torch.where(border_mask, mean_angles, torch.zeros_like(mean_angles))

        elif border_type == 'cyl':
            gx, gy, gz = G_chunk[:, 0], G_chunk[:, 1], G_chunk[:, 2]
            r = radii[:, None]  # (n_radii, 1)
            center_x = domain_size[0] / 2.0
            center_y = domain_size[1] / 2.0
            dist_to_center = torch.sqrt((gx - center_x)**2 + (gy - center_y)**2)  # (chunk_size,)
            dist_to_center = dist_to_center[None, :]  # (1, chunk_size)
            mask_z = (gz >= r) & (gz <= (domain_size[2] - r))

            r = radii[:, None]  # (n_radii, 1)
            max_radius = min(center_x, center_y) - r  # (n_radii, 1)
            border_mask = (dist_to_center <= max_radius) & mask_z  # (n_radii, chunk_size)

            border_mask_3d = border_mask[:, :, None]
            chunk_sum = torch.where(border_mask_3d, chunk_sum, torch.zeros_like(chunk_sum))
            mean_angles = torch.where(border_mask, mean_angles, torch.zeros_like(mean_angles))

        summed_vectors[:, start:end] = chunk_sum
        summed_stddev[:, start:end] = mean_angles

        del dists, masks, chunk_sum  # free memory

    vectors = summed_vectors.reshape(n_radii, nx, ny, nz, 3)
    std_dev = summed_stddev.reshape(n_radii, nx, ny, nz)

    return vectors, std_dev

def mds(fibre_coords, max_samples, step, weight):
    fibre_coords = fibre_coords[:, ::step, :]
    # first select a subset of fibres if too many
    n_fibres, n_points, _ = fibre_coords.shape
    if n_fibres > max_samples:
        indices = torch.randperm(n_fibres)[:max_samples]
        fibre_coords = fibre_coords[indices]
        n_fibres = max_samples
    vecs = fibre_coords[:, 1:] - fibre_coords[:, :-1]  # (n_fibres, n_points-1, 3)
    vecs = torch.cat([vecs, vecs[:, -1:, :]], dim=1)  # (n_fibres, n_points, 3)
    # compute pairwise distances between fibres based on the average distance between their segments
    # Distances
    diff = fibre_coords[:, None, :, :] - fibre_coords[None, :, :, :]
    dists = diff.norm(dim=-1).mean(dim=-1)

    # Angles (segment-wise)
    norms = vecs / (vecs.norm(dim=-1, keepdim=True) + 1e-8)
    cos_angles = torch.einsum('ikd,jkd->ijk', norms, norms)
    angles = torch.acos(torch.clamp(cos_angles, -1.0, 1.0)).mean(dim=-1)
    # combine distances and angles into a single distance matrix as euclidean distance
    dists = dists / dists.max()
    angles = angles / angles.max() * weight
    combined_dists = dists**2 + angles**2
    # compute MDS
    n = combined_dists.shape[0]
    H = torch.eye(n, device=fibre_coords.device) - torch.ones((n, n), device=fibre_coords.device) / n
    B = -0.5 * H @ combined_dists @ H
    # eigen decomposition
    eigvals, eigvecs = torch.linalg.eigh(B)
    # take the top 2 eigenvectors
    idx = torch.argsort(eigvals, descending=True)[:2]
    L = torch.diag(torch.sqrt(torch.clamp(eigvals[idx], min=0)))
    V = eigvecs[:, idx]
    Y = V @ L  # (n_fibres * n_points, 2)
    return Y.cpu().numpy(), dists.flatten().cpu().numpy(), angles.flatten().cpu().numpy()

def eval(name, step, device, save_figs):
    if step is None: step = utils.latest_rve(f"results/{name}/rve")
    rve = RVE.eval(name, step, device)
    # convert fibre coords to float
    rve.fibre_coords = rve.fibre_coords.float()
    #rve.fibre_coords = rve.fibre_coords[:, ::20, :]  # downsample for speed
    root = f"results/{name}/figs/{step}"
    if save_figs:
        os.makedirs(root, exist_ok=True)

    # FFT power spectrum
    P = fft(rve.fibre_coords, rve.domain_size)
    plt.figure(figsize=(6, 5))
    plt.imshow(np.log1p(P[:, :, P.shape[2]//2].T), origin='lower', cmap='inferno')
    plt.colorbar(label='Log Power Spectrum')
    plt.title('FFT Power Spectrum Slice (Z mid-plane)')
    plt.xlabel('kx')
    plt.ylabel('ky')
    if save_figs:
        plt.savefig(f"{root}/fft_power_spectrum.png", dpi=400)
    else:
        plt.show()
        # FFT power spectrum
    plt.figure(figsize=(6, 5))
    plt.imshow(np.log1p(P[:, P.shape[1]//2, :].T), origin='lower', cmap='inferno')
    plt.colorbar(label='Log Power Spectrum')
    plt.title('FFT Power Spectrum Vertical Slice')
    plt.xlabel('kx')
    plt.ylabel('kz')
    if save_figs:
        plt.savefig(f"{root}/fft_power_spectrum_vertical.png", dpi=400)
    else:
        plt.show()

    # orientation bins
    if False:
        radii = [rve.domain_size[0] * 0.1]
        vectors, stddev = structure_bins(rve.fibre_coords, rve.domain_size, radii, border_type='cyl')
        # absolute angle in xy plane
        angles_xy_bins = torch.atan2(vectors[:,:,:,:,1], vectors[:,:,:,:,0]) * (180.0 / torch.pi)  # (n_radii, nx, ny, nz)
        
        # plot a slice of angles_xy
        slice_idx = angles_xy_bins.shape[2] // 2
        plt.figure(figsize=(6, 5))
        plt.imshow(angles_xy_bins[len(radii)//2, :, :, slice_idx].cpu().numpy().T, origin='lower', cmap='hsv', vmin=-180, vmax=180)
        plt.colorbar(label='Orientation in XY plane (degrees)')
        plt.title('Structure Tensor Orientation Slice (Z mid-plane)')
        plt.xlabel('X')
        plt.ylabel('Y')
        if save_figs:
            plt.savefig(f"{root}/orientation_bins_slice.png", dpi=400)
        else:
            plt.show()

    
    # multi-dimensional scaling plot
    mds_points, dists, angles = mds(rve.fibre_coords, 25000, 10, 4.4)  # (n_fibres * n_points, 2)
    #dists, angles = mds(rve.fibre_coords, 1000, 2)  # (n_fibres * n_points, 2)
    plt.figure(figsize=(8, 8))
    plt.scatter(mds_points[:, 0], mds_points[:, 1], s=1, alpha=0.5)
    plt.title("MDS Projection of Fibre Segments")
    plt.xlabel("MDS Dimension 1")
    plt.ylabel("MDS Dimension 2")
    if save_figs:
        plt.savefig(f"{root}/mds.png", dpi=400)
    else:
        plt.show()
    # also plot distances against angles
    if False:
        plt.figure(figsize=(6, 6))
        plt.scatter(dists, angles, s=0.5, alpha=0.1)
        plt.title("Pairwise Distances vs Angles between Fibre Segments")
        plt.xlabel("Scaled Distance")
        plt.ylabel("Scaled Angle")
        if save_figs:
            plt.savefig(f"{root}/distances_vs_angles.png", dpi=400)
        else:
            plt.show()

    # Vectors between consecutive points
    vecs = rve.fibre_coords[:, 1:] - rve.fibre_coords[:, :-1]  # (n_fibres, n_points-1, 3)
    lengths = torch.norm(vecs, dim=2).cpu().numpy().flatten()  # (n_fibres, n_points-1)
    # Angles to axis in xz plane
    angles_xz = (torch.atan2(vecs[:,:,0], vecs[:,:,2]) * 180.0 / torch.pi).cpu().numpy().flatten() # (n_fibres, n_points-1)
    # Angles to axis in yz plane
    angles_yz = (torch.atan2(vecs[:,:,1], vecs[:,:,2]) * 180.0 / torch.pi).cpu().numpy().flatten() # (n_fibres, n_points-1)
    # 3D angles to z axis
    angles_3d = (torch.atan2(torch.norm(vecs[:,:,:2], dim=2), vecs[:,:,2]) * 180.0 / torch.pi).cpu().numpy()
    # absolute angle in xy plane
    angles_xy = (torch.atan2(vecs[:,:,1], vecs[:,:,0]) * 180.0 / torch.pi).cpu().numpy() # (n_fibres, n_points-1)

    # Plot histograms for inclination
    fig, ax = plt.subplots(2, 2, figsize=(18, 12), constrained_layout=True)
    bins = 100
    ax[0][0].hist(angles_xz, bins=bins, weights=lengths, edgecolor='black', density=True)
    ax[0][0].set_xlabel(f"xz angles (degrees), mean={np.average(angles_xz, weights=lengths):.2f}, std={np.sqrt(np.cov(angles_xz, aweights=lengths)):.2f}")
    ax[0][0].set_ylabel("Frequency")
    ax[0][1].hist(angles_yz, bins=bins, weights=lengths, edgecolor='black', density=True)
    ax[0][1].set_xlabel(f"yz angles (degrees), mean={np.average(angles_yz, weights=lengths):.2f}, std={np.sqrt(np.cov(angles_yz, aweights=lengths)):.2f}")
    ax[0][1].set_ylabel("Frequency")
    ax[1][0].hist(angles_3d.flatten(), bins=bins, weights=lengths, edgecolor='black', density=True)
    ax[1][0].set_xlabel(f"3D angles to z (degrees), mean={np.average(angles_3d.flatten(), weights=lengths):.2f}, std={np.sqrt(np.cov(angles_3d.flatten(), aweights=lengths)):.2f}")
    ax[1][0].set_ylabel("Frequency")
    ax[1][1].hist(angles_xy.flatten(), bins=bins, weights=lengths, edgecolor='black', density=True)
    ax[1][1].set_xlabel(f"Orientation in xy plane (degrees), mean={np.average(angles_xy.flatten(), weights=lengths):.2f}")
    ax[1][1].set_ylabel("Frequency")
    plt.suptitle(f"Fibre Angle Distributions")
    if save_figs:
        plt.savefig(f"{root}/angles_hist.png", dpi=400)
    else:
        plt.show()

    # Plot trajectories colored with angle to z axis
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
        lc = LineCollection(
            segments,
            cmap='jet',
            norm=plt.Normalize(angles_3d.min(), angles_3d.max()),
            linewidths=0.5,       # thinner lines
            alpha=0.8             # semi-transparent (0.0–1.0)
        )
        lc.set_array(angle)
        #lc.set_linewidth(2)
        ax.add_collection(lc)
        # Add scatter points (segment endpoints)
        #ax.scatter(x, y, s=10, edgecolor='k', linewidth=0.3, alpha=0.8)
    # Format plot
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_title("XY-plane Projection Colored by 3D Angle to Z-axis")
    ax.axis("equal")
    ax.set_aspect('equal', adjustable='box')
    # Add colorbar
    cbar = plt.colorbar(lc, ax=ax)
    cbar.set_label("Angle to Z-axis (degrees)")
    if save_figs:
        plt.savefig(f"{root}/trajectories_angles.png", dpi=400)
    else:
        plt.show()
    
    # Plot trajectories colored with orientation in xy plane
    fig, ax = plt.subplots(figsize=(6, 6))
    for i in range(fibres_np.shape[0]):
        x = fibres_np[i, :, 0]
        y = fibres_np[i, :, 1]
        angle = angles_xy[i]  # (resolution-1,)
        # Create list of segments [(x0,y0)-(x1,y1), (x1,y1)-(x2,y2), ...]
        points = np.array([x, y]).T.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)
        # Create a LineCollection with colormap based on angle
        lc = LineCollection(
            segments,
            cmap='hsv',
            norm=plt.Normalize(-180, 180),
            linewidths=0.5,       # thinner lines
            alpha=0.8             # semi-transparent (0.0–1.0)
        )
        lc.set_array(angle)
        #lc.set_linewidth(2)
        ax.add_collection(lc)
    # Format plot
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_title("XY-plane Projection Colored by Orientation")
    ax.axis("equal")
    ax.set_aspect('equal', adjustable='box')
    # Add colorbar
    cbar = plt.colorbar(lc, ax=ax)
    cbar.set_label("Orientation in XY-plane (degrees)")
    if save_figs:
        plt.savefig(f"{root}/trajectories_orientations.png", dpi=400)
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
    plt.title(f"Fibre Tortuosity (Mean = {ratios.mean():.3f})")
    plt.grid(alpha=0.3)
    if save_figs:
        plt.savefig(f"{root}/tortuosity.png")
    else:
        plt.show()
    # Plot zoomed in histogram
    plt.figure(figsize=(6, 4))
    plt.hist(ratios, bins=50, range=(1.0, 1.1), color='steelblue', edgecolor='black')
    plt.xlabel("Length Ratio (Actual / End-to-End)")
    plt.ylabel("Number of Fibres")
    plt.title(f"Fibre Tortuosity (Mean = {ratios.mean():.3f})")
    plt.grid(alpha=0.3)
    if save_figs:
        plt.savefig(f"{root}/tortuosity_zoomed.png")
    else:
        plt.show()

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", type=str, default="big_sparse_noisy_init_angstd_10", help="Configuration file name (without .yaml)")
    parser.add_argument("--step", type=str, default=None, help="Job name for saving results")
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #device = 'cpu'
    print("Using device:", device)
    eval(args.name, args.step, device, save_figs=True)

if __name__ == "__main__":
    main()