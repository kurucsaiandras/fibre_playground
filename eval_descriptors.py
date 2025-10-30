import torch
import numpy as np
from scipy.spatial import Voronoi
import matplotlib.pyplot as plt
import os
import utils

def get_slice(fibres, z):
    """
    Get the intersection points of fibres with a horizontal plane at height z.
    
    Args:
        fibres (np array): A tensor of shape (N, M, 3) representing N fibres with M points each in 3D space.
        z (float): The height of the horizontal plane.
        
    Returns:
        np array: A tensor of shape (K, 2) containing the projected intersection points.
    """
    intersection_points = []
    for fibre in fibres:
        for i in range(len(fibre) - 1):
            p1 = fibre[i]
            p2 = fibre[i + 1]
            if (p1[2] - z) * (p2[2] - z) < 0:  # Check if the segment crosses the plane
                t = (z - p1[2]) / (p2[2] - p1[2])
                intersection_point = p1 + t * (p2 - p1)
                intersection_points.append(intersection_point)
            elif p1[2] == z:  # If p1 is exactly on the plane
                intersection_points.append(p1)
            elif p2[2] == z:  # If p2 is exactly on the plane
                intersection_points.append(p2)
    
    if intersection_points:
        return np.array(intersection_points)[:, :2]  # Return only x, y coordinates
    else:
        return np.empty((0, 2))

def voronoi_polygon_areas(points):
    """
    Compute the areas of Voronoi polygons given a set of 2D points, treating the points as if they are in a periodic domain.
    
    Args:
        points (np array): A tensor of shape (N, 2) representing N 2D points.
        
    Returns:
        np array: A tensor of shape (N,) containing the area of each Voronoi polygon.
    """
    
    # Create periodic copies of the points
    offsets = np.array([[0, 0], [1, 0], [-1, 0], [0, 1], [0, -1], [1, 1], [1, -1], [-1, 1], [-1, -1]])
    all_points = np.vstack([points + offset for offset in offsets])
    
    vor = Voronoi(all_points)
    
    areas = np.zeros(len(points))
    
    for i in range(len(points)):
        region_index = vor.point_region[i]
        vertices = vor.regions[region_index]
        
        if -1 in vertices:  # Infinite region
            areas[i] = np.inf
            continue
        
        polygon = vor.vertices[vertices]
        
        # Compute area using the shoelace formula
        x = polygon[:, 0]
        y = polygon[:, 1]
        area = 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))
        areas[i] = area
    
    return np.array(areas)

def ripleys_k(points, hs):
    """
    Compute Ripley's K function for a set of 2D points, treating the points as if they are in a periodic domain.
    
    Args:
        points (np array): A tensor of shape (N, 2) representing N 2D points.
        hs (np array): A tensor of shape (M,) representing distances at which to compute K.
    Returns:
        np array: A tensor of shape (M,) containing the K values for each distance in hs.
    """
    n = len(points)
    counts = np.zeros(len(hs))
    for i in range(n):
        for j in range(n):
            if i != j:
                dist = np.linalg.norm(points[i] - points[j])
                # Consider periodic boundary conditions
                dist = min(dist, np.linalg.norm(points[i] - points[j] + np.array([1, 0])))
                dist = min(dist, np.linalg.norm(points[i] - points[j] + np.array([-1, 0])))
                dist = min(dist, np.linalg.norm(points[i] - points[j] + np.array([0, 1])))
                dist = min(dist, np.linalg.norm(points[i] - points[j] + np.array([0, -1])))
                dist = min(dist, np.linalg.norm(points[i] - points[j] + np.array([1, 1])))
                dist = min(dist, np.linalg.norm(points[i] - points[j] + np.array([1, -1])))
                dist = min(dist, np.linalg.norm(points[i] - points[j] + np.array([-1, 1])))
                dist = min(dist, np.linalg.norm(points[i] - points[j] + np.array([-1, -1])))
                
                for k, h in enumerate(hs):
                    if dist < h:
                        counts[k] += 1
    area = 1.0  # Assuming unit square domain
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

def main():
    name = "fibre_sim_random"
    step = None
    if step is None:
        path = utils.latest_rve(f"results/{name}/rve")
    else:
        path = f"results/{name}/rve/{step}.pt"
    fibres, domain_size, fibre_r = load_fibres(path)
    save_figs = True
    fibre_r_mean = np.mean(fibre_r)

    if save_figs:
        if not os.path.exists("figs"):
            os.makedirs("figs")
        if not os.path.exists(f"figs/{name}"):
            os.makedirs(f"figs/{name}")
    
    # Get a horizontal slice at z = 0.5
    z = 0.5
    slice_points = get_slice(fibres, z)
    print(f"Number of intersection points at z={z}: {len(slice_points)}")
    
    if len(slice_points) == 0:
        print("No intersection points found.")
        return
    
    # Compute Voronoi polygon areas
    areas = voronoi_polygon_areas(slice_points)
    finite_areas = areas[np.isfinite(areas)]
    print(f"Voronoi polygon areas (finite): mean={np.mean(finite_areas):.4f}, std={np.std(finite_areas):.4f}")
    
    # Plot histogram of areas
    plt.figure()
    plt.hist(finite_areas, bins=30, edgecolor='black')
    plt.title("Histogram of Voronoi Polygon Areas")
    plt.xlabel("Area")
    plt.ylabel("Frequency")
    if save_figs:
        plt.savefig(f"figs/{name}/voronoi_areas_hist.png")
    else:
        plt.show()
    
    # Compute Ripley's K function
    hs = np.linspace(0.01, 0.5, 50)
    # Divide hs by fibre radius for plotting
    hs_norm = hs * (domain_size[0]) / fibre_r_mean
    k_values = ripleys_k(slice_points, hs)
    
    # Plot Ripley's K function
    plt.figure()
    plt.plot(hs_norm, k_values, label="Ripley's K")
    plt.plot(hs_norm, np.pi * hs**2, 'r--', label="Theoretical K (CSR)")
    plt.title("Ripley's K Function")
    plt.xlabel("Distance divided by fibre radius")
    plt.ylabel("K(h)")
    plt.legend()
    if save_figs:
        plt.savefig(f"figs/{name}/ripleys_k.png")
    else:
        plt.show()

    # Plot L(h)
    L_values = np.sqrt(k_values / np.pi) - hs
    plt.figure()
    plt.plot(hs_norm, L_values, label="L(h)")
    plt.axhline(0, color='r', linestyle='--', label="CSR Line") # Complete Spatial Randomness line
    plt.title("Ripley's K - Poisson")
    plt.xlabel("Distance devided by fibre radius")
    plt.ylabel("L(h)")
    plt.legend()
    if save_figs:
        plt.savefig(f"figs/{name}/ripleys_l.png")
    else:
        plt.show()

    # Plot Radial Distribution Function (RDF)
    rdf_values = np.gradient(k_values, hs) / (2 * np.pi * hs)
    plt.figure()
    plt.plot(hs_norm, rdf_values, label="RDF")
    plt.axhline(1, color='r', linestyle='--', label="CSR Line") # Complete Spatial Randomness line
    plt.title("Radial Distribution Function (RDF)")
    plt.xlabel("Distance divided by fibre radius")
    plt.ylabel("g(h)")
    plt.legend()
    if save_figs:
        plt.savefig(f"figs/{name}/rdf.png")
    else:
        plt.show()

if __name__ == "__main__":
    main()