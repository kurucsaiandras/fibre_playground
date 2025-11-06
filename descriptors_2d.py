import torch
import numpy as np
from scipy.spatial import Voronoi
from shapely.geometry import Polygon, box
import matplotlib.pyplot as plt
import os
import utils
from rve import RVE

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

def ripleys_k(points: torch.Tensor, hs: torch.Tensor, domain_size: torch.Tensor, offsets: torch.Tensor):
    """
    Compute Ripley's K function for a set of 2D points using periodic offsets,
    counting all valid offset distances (not just the minimum).

    Args:
        points (torch.Tensor): (N, 2) point coordinates.
        hs (torch.Tensor): (M,) distance thresholds.
        domain_size (torch.Tensor): (3,) domain size.
        offsets (torch.Tensor): (K, 2) periodic offsets (e.g., 9x2 for periodic domain).

    Returns:
        torch.Tensor: (M,) K values for each distance in hs.
    """
    #TODO implement weight for edge correction when not apply_pbc
    device = points.device
    n = points.shape[0]
    hs = hs.to(device)
    offsets = offsets.to(device)

    # Pairwise difference matrix (N, N, 2)
    diff = points[:, None, :] - points[None, :, :]  # (N, N, 2)

    # Apply all offsets: (K, N, N, 2)
    shifted = diff[None, :, :, :] + offsets[:, None, None, :]

    # Compute distances for all offset copies
    dist_sq = torch.sum(shifted ** 2, dim=-1)  # (K, N, N)
    dist = torch.sqrt(dist_sq)                 # (K, N, N)

    # Mask out self-distances (only when offset == [0,0])
    mask_self = torch.eye(n, dtype=torch.bool, device=device)
    # Find index of the [0,0] offset if it exists
    zero_offset_idx = (offsets == 0).all(dim=1).nonzero(as_tuple=True)[0]
    if len(zero_offset_idx) > 0:
        dist[zero_offset_idx[0], mask_self] = float('inf')

    # Compare distances with thresholds
    # dist: (K, N, N), hs: (M,)
    # -> (K, N, N, M)
    within_h = (dist[..., None] < hs)  # broadcasted comparison
    counts = within_h.sum(dim=(0, 1, 2))  # sum over K, N, N

    # Normalize
    area = domain_size[0] * domain_size[1]
    k_values = (area / (n * (n - 1))) * counts
    return k_values

def load_fibres(file_name):
    """
    Load fibre data from a file and normalize it to fit within a unit cube.
    
    Args:
        file_name (str): The path to the file containing fibre data.
    Returns:
        np array: A tensor of shape (N, M, 3) representing N fibres with M points each in 3D space.
    """
    model = torch.load(file_name)
    fibre_coords = model["fibre_coords"].cpu().detach().numpy()
    fibre_r = model["fibre_r"].cpu().numpy()
    domain_size = model["domain_size"].cpu().numpy()
    n_fibres, n_points, _ = fibre_coords.shape
    print(f"Loaded {n_fibres} fibres with {n_points} points each.")
    # Normalize to unit cube
    fibre_coords /= domain_size
    return fibre_coords, domain_size, fibre_r

def eval(name, step, device, save_figs):
    if step is None: step = utils.latest_rve(f"results/{name}/rve")
    rve = RVE.eval(name, step, device)
    fibre_r_mean = rve.fibre_r.mean()
    root = f"results/{name}/figs/{step}"
    if save_figs:
        os.makedirs(root, exist_ok=True)
    
    # Get a horizontal slice at z = 0.5
    z = 0.5
    slice_points = utils.get_slice(rve.fibre_coords, z)
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
    print(f"Voronoi polygon areas (finite): mean={np.mean(finite_areas):.4f}, std={np.std(finite_areas):.4f}")
    
    # Plot histogram of areas
    plt.figure()
    plt.hist(finite_areas, bins=30, edgecolor='black')
    plt.title("Histogram of Voronoi Polygon Areas")
    plt.xlabel("Area")
    plt.ylabel("Frequency")
    if save_figs:
        plt.savefig(f"{root}/voronoi_areas_hist.png")
    else:
        plt.show()
    
    # Compute Ripley's K function
    hs = torch.linspace(0.1, 30, 50).to(rve.device)*fibre_r_mean
    # Divide hs by fibre radius for plotting
    hs_norm = (hs / fibre_r_mean).cpu().numpy()
    k_values = ripleys_k(slice_points, hs, rve.domain_size, offsets)
    
    # Plot Ripley's K function
    csr = (torch.pi * hs*hs).cpu().numpy()
    plt.figure()
    plt.plot(hs_norm, k_values.cpu().numpy(), label="Ripley's K")
    plt.plot(hs_norm, csr, 'r--', label="Theoretical K (CSR)")
    plt.title("Ripley's K Function")
    plt.xlabel("Distance divided by fibre radius")
    plt.ylabel("K(h)")
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
    plt.title("Ripley's K - Poisson")
    plt.xlabel("Distance devided by fibre radius")
    plt.ylabel("L(h)")
    plt.legend()
    if save_figs:
        plt.savefig(f"{root}/ripleys_l.png")
    else:
        plt.show()

    # Plot Radial Distribution Function (RDF)
    rdf_values = torch.gradient(k_values, spacing=(hs,))[0] / (2 * torch.pi * hs)
    plt.figure()
    plt.plot(hs_norm, rdf_values.cpu().numpy(), label="RDF")
    plt.axhline(1, color='r', linestyle='--', label="CSR Line") # Complete Spatial Randomness line
    plt.title("Radial Distribution Function (RDF)")
    plt.xlabel("Distance divided by fibre radius")
    plt.ylabel("g(h)")
    plt.legend()
    if save_figs:
        plt.savefig(f"{root}/rdf.png")
    else:
        plt.show()

def main():
    name = 'big_angles_small_curv_k'
    step = 4501
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)
    eval(name, step, device, save_figs=True)

if __name__ == "__main__":
    main()