import torch
import numpy as np
from scipy.spatial import Voronoi
from shapely.geometry import Polygon, box
import matplotlib.pyplot as plt
import os
import utils
from rve import RVE
import math

def voronoi_polygon_areas(points, domain_size, offsets):
    """
    Compute the areas of Voronoi polygons given a set of 2D points, treating the points as if they are in a periodic domain.
    
    Args:
        points (torch.Tensor): A tensor of shape (N, 2) representing N 2D points.
         domain_size (torch.Tensor): (3,) domain size.
         offsets (torch.Tensor): (K, 2) periodic offsets (e.g., 9x2 for periodic domain).
    Returns:
        np array: A tensor of shape (N,) containing the area of each Voronoi polygon.
    """
    # Create periodic copies of the points
    all_points = (points.unsqueeze(0) + offsets.unsqueeze(1)).reshape(-1, 2).cpu().numpy()
    
    vor = Voronoi(all_points)
    
    areas = np.zeros(len(points))

    # Domain box for clipping
    clip_box = box(0, 0, domain_size[0], domain_size[1]) if len(offsets) == 1 else None
    
    for i in range(len(points)):
        region_index = vor.point_region[i]
        vertices = vor.regions[region_index]
        
        if -1 in vertices:  # Infinite region
            areas[i] = np.inf
            continue
        
        polygon = Polygon(vor.vertices[vertices])

        # Clip polygon to domain if needed
        if clip_box is not None:
            polygon = polygon.intersection(clip_box)
        
        areas[i] = polygon.area
    
    return np.array(areas)

def is_inside_domain(points, domain_size, type='rect'):
    """
    Check if points are inside the given domain.

    Args:
        points (torch.Tensor): (..., 2) point coordinates.
        domain_size (torch.Tensor): (2,) width and height of the study area.

    Returns:
        torch.Tensor: (N,) boolean tensor indicating if each point is inside the domain.
    """
    if type == 'rect':
        inside_x = (points[..., 0] >= 0) & (points[..., 0] <= domain_size[0])
        inside_y = (points[..., 1] >= 0) & (points[..., 1] <= domain_size[1])
        return inside_x & inside_y
    if type == 'cyl':
        center = domain_size / 2
        radius = min(domain_size) / 2
        dists = torch.sqrt((points[..., 0] - center[0])**2 + (points[..., 1] - center[1])**2)
        return dists <= radius

def get_weights(c, dists, domain_size, n_samples=512):
    """
    Compute edge-correction weights for Ripley's K function (circumference-based, vectorized).

    Args:
        c (torch.Tensor): (chunk_size, 2) query points (chunk of total set).
        dists (torch.Tensor): (chunk_size, N) pairwise distances between c and all points.
        domain_size (torch.Tensor): (2,) width and height of the study area.
        n_samples (int): number of samples along the circumference for estimation.

    Returns:
        torch.Tensor: (chunk_size, N) correction weights for each point pair (i,j).
    """
    device = c.device

    # Uniformly sample points on unit circle
    angles = torch.linspace(0, 2 * torch.pi, n_samples, device=device)
    unit_circle = torch.stack([torch.cos(angles), torch.sin(angles)], dim=1)  # (n_samples, 2)
    # Broadcast circle samples for all (chunk_size, N)
    # samples[i, j, k, :] = position on circumference of radius dists[i,j] around c[i]
    samples = c[:, None, None, :] + dists[:, :, None, None] * unit_circle[None, None, :, :]  # (chunk_size, N, n_samples, 2)
    # Check whether circumference points are inside domain
    inside = is_inside_domain(samples, domain_size)  # (chunk_size, N, n_samples)
    # Compute fraction of circumference inside domain
    inside_frac = inside.float().mean(dim=-1)  # (chunk_size, N)
    # Avoid division by zero: if circle entirely outside domain, set weight = 0
    weights = torch.where(inside_frac > 0, 1.0 / inside_frac, torch.zeros_like(inside_frac))
    # Set self-pairs (distance = 0) to weight = 0
    weights = torch.where(dists == 0, torch.zeros_like(weights), weights)

    return weights

def ripleys_k(points: torch.Tensor, hs: torch.Tensor, domain_size: torch.Tensor, offsets: torch.Tensor, chunk_size=10):
    """
    Compute Ripley's K function in chunks to manage memory usage.

    Args:
        points (torch.Tensor): (N, 2) point coordinates.
        hs (torch.Tensor): (M,) distance thresholds.
        domain_size (torch.Tensor): (2,) domain size.
        offsets (torch.Tensor): (K, 2) periodic offsets (e.g., 9x2 for periodic domain).
        chunk_size (int): Number of points to process in each chunk.

    Returns:
        torch.Tensor: (M,) k values for each distance in hs.
    """
    device = points.device
    N = points.shape[0]
    K = offsets.shape[0]
    hs = hs.to(device)
    offsets = offsets.to(device)
    counts = torch.zeros(len(hs), device=device)
    for start in range(0, N, chunk_size):
        end = min(start + chunk_size, N)
        c = points[start:end]  # (chunk_size, 2)
        # Pairwise difference matrix (chunk_size, N, 2)
        diff = c[:, None, :] - points[None, :, :]  # (chunk_size, N, 2)
        # Apply all offsets: (K, chunk_size, N, 2)
        shifted = diff[None, :, :, :] + offsets[:, None, None, :]
        # Compute distances for all offset copies
        dist = torch.sqrt(torch.sum(shifted ** 2, dim=-1))  # (K, chunk_size, N)
        eps = 1e-6 # to avoid counting self-pairs
        if K > 1: # Apply edge correction weights only when not applying PBC
            # Compare distances with thresholds
            within_h = (dist[..., None] < hs) & (dist[..., None] > eps)  # (K, chunk_size, N, M)
            counts += within_h.sum(dim=(0, 1, 2))  # sum over K, chunk_size, N
        elif K == 1:
            dist = dist.squeeze(0)  # (chunk_size, N)
            weights = get_weights(c, dist, domain_size)  # (chunk_size, N)
            within_h = (dist[..., None] < hs) & (dist[..., None] > eps)  # (chunk_size, N, M)
            weighted_counts = within_h.float() * weights[..., None]
            counts += weighted_counts.sum(dim=(0, 1))  # sum over chunk_size, N
            #c_idx, p_idx, h_idx = torch.where(within_h & (weights[..., None] == 0))
            #if len(c_idx) > 0:
            #    plt.scatter(c[c_idx, 0].cpu(), c[c_idx, 1].cpu(), color='r')
            #    plt.scatter(points[p_idx, 0].cpu(), points[p_idx, 1].cpu(), color='b')
            #    # circle from c to points
            #    for i in range(len(c_idx)):
            #        circle = plt.Circle((c[c_idx[i], 0].cpu(), c[c_idx[i], 1].cpu()), dist[c_idx[i], p_idx[i]].cpu(), color='b', fill=False, linestyle='--')
            #        plt.gca().add_artist(circle)
    # Normalize
    area = domain_size[0] * domain_size[1]
    k_values = (area / (N * (N - 1))) * counts
    return k_values

def slice_density(points, fibre_r_mean, domain_size, resolution, chunk_size=1000):
    """
    Compute 2D density of points in a horizontal slice of the domain.

    Args:
        points (torch.Tensor): (N, 2) point coordinates.
        fibre_r_mean (float): mean fibre radius.
        domain_size (torch.Tensor): (2,) domain size.
        resolution (int): number of bins along each axis.
        chunk_size (int): number of points to process in each chunk.

    Returns:
        torch.Tensor: (resolution, resolution) density map.
    """
    device = points.device
    N = points.shape[0]
    density = torch.zeros((resolution, resolution), device=device)
    grid_x = torch.linspace(0, domain_size[0], resolution, device=device)
    grid_y = torch.linspace(0, domain_size[1], resolution, device=device)
    grid = torch.stack(torch.meshgrid(grid_x, grid_y, indexing='ij'), dim=-1)  # (resolution, resolution, 2)
    radius = domain_size[0] / resolution * 5.0 # assuming square domain
    for start in range(0, N, chunk_size):
        end = min(start + chunk_size, N)
        c = points[start:end, :2]  # (chunk_size, 2)
        # Compute pairwise distances between c and grid points
        diff = c[:, None, None, :] - grid[None, :, :, :]  # (chunk_size, resolution, resolution, 2)
        dist = torch.sqrt(torch.sum(diff ** 2, dim=-1))  # (chunk_size, resolution, resolution)
        # Increment density where distance is less than radius/2
        within_range = dist < radius
        density += within_range.sum(dim=0)
    # Normalize density
    s = fibre_r_mean**2 / radius**2
    density *= s
    return density

def eval(name, step, device, save_figs):
    if step is None: step = utils.latest_rve(f"results/{name}/rve")
    rve = RVE.eval(name, step, device)
    fibre_r_mean = rve.fibre_r.mean()
    root = f"results/{name}/figs/{step}"
    if save_figs:
        os.makedirs(root, exist_ok=True)
    overall_density = rve.get_fibre_to_volume_ratio()
    # Get a horizontal slice halfway up the domain
    z = 0.5 * rve.domain_size[2]
    slice_points = utils.get_slice(rve.fibre_coords, z)
    if False: # cut central square if cylindrical domain
        radius = min(rve.domain_size[:2]) / 2 + fibre_r_mean
        overall_density = torch.pow(rve.fibre_r, 2).sum() / radius**2
        center = rve.domain_size[:2] / 2
        side = min(rve.domain_size[:2]) / torch.sqrt(torch.tensor(2.0))
        lower_bound = center - side / 2
        upper_bound = center + side / 2
        mask_x = (slice_points[:, 0] >= lower_bound[0]) & (slice_points[:, 0] <= upper_bound[0])
        mask_y = (slice_points[:, 1] >= lower_bound[1]) & (slice_points[:, 1] <= upper_bound[1])
        mask = mask_x & mask_y
        slice_points = slice_points[mask]
        # shift slice_points to start from (0,0)
        slice_points[:, 0] -= lower_bound[0]
        slice_points[:, 1] -= lower_bound[1]
        rve.domain_size = torch.tensor([side, side, rve.domain_size[2]], device=rve.domain_size.device)
    #dists_from_center = torch.sqrt((slice_points[:, 0] - rve.domain_size[0]/2)**2 + (slice_points[:, 1] - rve.domain_size[1]/2)**2)
    #slice_points = slice_points[dists_from_center < min(rve.domain_size[0], rve.domain_size[1])/2]
    print(f"Number of intersection points at z={z}: {len(slice_points)}")
    
    if len(slice_points) == 0:
        print("No intersection points found.")
        return
    
    if rve.apply_pbc:
        offsets = (torch.cartesian_prod(torch.tensor([-1, 0, 1]), torch.tensor([-1, 0, 1])).to(rve.device) * rve.domain_size[:-1])
        offsets = torch.cat([offsets[4:], offsets[:4]], dim=0)
    else:
        offsets = torch.tensor([[0, 0]], device=rve.device)

    # Compute Voronoi polygon areas
    areas = voronoi_polygon_areas(slice_points, rve.domain_size, offsets)
    finite_areas = areas[np.isfinite(areas)]
    
    # Plot histogram of areas
    plt.figure()
    plt.hist(finite_areas, bins=30, edgecolor='black')
    plt.title(f"Voronoi Polygon Areas (mean={np.mean(finite_areas):.2f}, std={np.std(finite_areas):.2f})")
    plt.xlabel("Area")
    plt.ylabel("Frequency")
    if save_figs:
        plt.savefig(f"{root}/voronoi_areas_hist.png")
    else:
        plt.show()
    
    # Compute Ripley's K function
    # plot slice points
    plt.figure()
    plt.scatter(slice_points[:, 0].cpu(), slice_points[:, 1].cpu(), s=1)
    plt.title(f"Fibre Cross-Sections at z={z:.2f}")
    plt.xlabel("X")
    plt.ylabel("Y")
    #plt.xlim(0, rve.domain_size[0].cpu())
    #plt.ylim(0, rve.domain_size[1].cpu())
    plt.gca().set_aspect('equal', adjustable='box')
    hs = torch.linspace(0.1, 30, 200).to(rve.device)*fibre_r_mean
    # Divide hs by fibre radius for plotting
    hs_norm = (hs / fibre_r_mean).cpu().numpy()
    k_values = ripleys_k(slice_points, hs, rve.domain_size[:2], offsets)
    if save_figs:
        plt.savefig(f"{root}/slice_points.png")
    else:
        plt.show()
    
    # Plot Ripley's K function
    csr = (torch.pi * hs*hs).cpu().numpy()
    plt.figure()
    plt.plot(hs_norm, k_values.cpu().numpy(), label="Ripley's K")
    plt.plot(hs_norm, csr, 'r--', label="Theoretical K (CSR)")
    plt.title("Ripley's K Function")
    plt.xlabel("Distance divided by fibre radius")
    plt.ylabel("K(h)")
    plt.grid()
    plt.legend()
    if save_figs:
        plt.savefig(f"{root}/ripleys_k.png")
    else:
        plt.show()

    # Plot L(h)
    L_values = torch.sqrt(k_values / torch.pi) - hs
    plt.figure()
    plt.plot(hs_norm, L_values.cpu().numpy(), label="L(h)")
    plt.axhline(0, color='r', linestyle='--', label="CSR Line") # Complete Spatial Randomness line
    plt.title("Diff. between Ripley's K and theoretical Poisson")
    plt.xlabel("Distance devided by fibre radius")
    plt.ylabel("L(h)")
    plt.grid()
    plt.legend()
    if save_figs:
        plt.savefig(f"{root}/ripleys_l.png")
    else:
        plt.show()

    # Plot Pair Distribution Function
    rdf_values = torch.gradient(k_values, spacing=(hs,))[0] / (2 * torch.pi * hs)
    plt.figure()
    plt.plot(hs_norm, rdf_values.cpu().numpy(), label="G(h)")
    plt.axhline(1, color='r', linestyle='--', label="CSR Line") # Complete Spatial Randomness line
    plt.title("Pair Distribution Function")
    plt.xlabel("Distance divided by fibre radius")
    plt.ylabel("G(h)")
    plt.grid()
    plt.legend()
    if save_figs:
        plt.savefig(f"{root}/pair_distrib.png")
    else:
        plt.show()

    # Compute and plot density map
    density_map = slice_density(slice_points, fibre_r_mean, rve.domain_size[:2], resolution=200)
    plt.figure()
    plt.imshow(density_map.cpu().numpy().T, extent=(0, rve.domain_size[0].cpu(), 0, rve.domain_size[1].cpu()), origin='lower', cmap='jet', interpolation='nearest')
    plt.colorbar(label='Density')
    plt.title(f"2D Density Map (overall density={overall_density:.4f})")
    plt.xlabel('X')
    plt.ylabel('Y')
    if save_figs:
        plt.savefig(f"{root}/density_map.png")
    else:
        plt.show()

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", type=str, default="airtex_ds", help="Configuration file name (without .yaml)")
    parser.add_argument("--step", type=str, default=None, help="Job name for saving results")
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)
    eval(args.name, args.step, device, save_figs=True)

if __name__ == "__main__":
    main()