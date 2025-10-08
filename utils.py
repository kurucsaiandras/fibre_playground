import torch
import poisson_disc as pd
import os

def get_bounding_boxes(points, radius):
    """
    points: (n_fibres, resolution, 3)
    radius: scalar
    returns: (n_fibres, 2, 3) tensor of bounding boxes: p_min, p_max
    """
    min_coords = points - radius
    max_coords = points + radius
    boxes = torch.stack([min_coords.min(dim=1).values, max_coords.max(dim=1).values], dim=1)
    return boxes

def get_bbox_intersections(boxes, domain_size):
    """
    boxes: (n_fibres, 2, 3) tensor of bounding boxes: p_min, p_max
    domain_size: (3,) tensor/list specifying [x_max, y_max, z_max]
    returns: (n_fibres, n_fibres) boolean tensor indicating which boxes intersect
    """
    p_min = boxes[:, 0, :]  # (n, 3)
    p_max = boxes[:, 1, :]  # (n, 3)

    # Expand dimensions for broadcasting
    p_min_exp = p_min[:, None, :]  # (n, 1, 3)
    p_max_exp = p_max[:, None, :]  # (n, 1, 3)

    # Check for overlap in each dimension
    overlap_x = (p_min_exp[:, :, 0] <= p_max_exp[:, :, 0].T) & (p_max_exp[:, :, 0] >= p_min_exp[:, :, 0].T)
    overlap_y = (p_min_exp[:, :, 1] <= p_max_exp[:, :, 1].T) & (p_max_exp[:, :, 1] >= p_min_exp[:, :, 1].T)
    overlap_z = (p_min_exp[:, :, 2] <= p_max_exp[:, :, 2].T) & (p_max_exp[:, :, 2] >= p_min_exp[:, :, 2].T)

    # PBCs
    overlap_x_pbc = (p_min_exp[:, :, 0] - domain_size[0] <= p_max_exp[:, :, 0].T) & (p_max_exp[:, :, 0] - domain_size[0] >= p_min_exp[:, :, 0].T)
    overlap_y_pbc = (p_min_exp[:, :, 1] - domain_size[1] <= p_max_exp[:, :, 1].T) & (p_max_exp[:, :, 1] - domain_size[1] >= p_min_exp[:, :, 1].T)

    # Boxes intersect if they overlap in all three dimensions
    intersections = overlap_x & overlap_y & overlap_z

    # PBC intersections
    intersections_x_pbc = overlap_x_pbc & overlap_y & overlap_z
    intersections_y_pbc = overlap_x & overlap_y_pbc & overlap_z
    intersections_xy_pbc = overlap_x_pbc & overlap_y_pbc & overlap_z
    intersections_yx_pbc = overlap_x_pbc & overlap_y_pbc.T & overlap_z

    # remove double counting (i,j) and (j,i) and self-intersections
    intersections = torch.triu(intersections, diagonal=1)
    # only remove self-intersections for pbc (keep the symmetric ones)
    intersections_x_pbc = intersections_x_pbc & (~torch.eye(intersections_x_pbc.shape[0], dtype=bool, device=intersections_x_pbc.device))
    intersections_y_pbc = intersections_y_pbc & (~torch.eye(intersections_y_pbc.shape[0], dtype=bool, device=intersections_y_pbc.device))
    intersections_xy_pbc = intersections_xy_pbc & (~torch.eye(intersections_xy_pbc.shape[0], dtype=bool, device=intersections_xy_pbc.device))
    intersections_yx_pbc = intersections_yx_pbc & (~torch.eye(intersections_yx_pbc.shape[0], dtype=bool, device=intersections_yx_pbc.device))

    return intersections, intersections_x_pbc, intersections_y_pbc, intersections_xy_pbc, intersections_yx_pbc

def generate_fibres_poisson(domain_size, num_points, radius, std_angle, device):
    """
    Generate fibres in a cubic domain of given size.
    domain_size: (3,) tensor/list specifying [x_max, y_max, z_max]
    num_points: number of points per fibre
    std_angle: standard deviation of angle perturbation from vertical (z) direction
    device: torch device
    returns:
    - fibres: (n_fibres, num_points, 3) tensor of coordinates
    - step_lengths: (n_fibres, 1) tensor of step lengths between points. used as rest lengths for springs.
    """
    # Random starting coordinates at z=0
    # Generate Poisson disk samples (2D)
    samples = pd.Bridson_sampling(dims=domain_size[:2].cpu().numpy(), radius=radius, k=30)
    samples = torch.tensor(samples, dtype=torch.float32, device=device)
    n_fibres = samples.shape[0]
    z0 = torch.zeros(samples.shape[0], 1, device=device)
    coords0 = torch.cat([samples, z0], dim=1)  # (n_fibres, 3)

    # Base direction (0, 0, 1) with Gaussian perturbations in x/y
    dx = torch.randn(n_fibres, 1, device=device) * std_angle
    dy = torch.randn(n_fibres, 1, device=device) * std_angle
    dz = torch.ones(n_fibres, 1, device=device)  # mostly upward
    dirs = torch.cat([dx, dy, dz], dim=1)  # (n_fibres, 3)
    dirs = dirs / torch.norm(dirs, dim=1, keepdim=True)  # normalize

    # Calculate lengths until they hit the top (z = domain_size)
    lengths = (domain_size[2] - z0) / dirs[:, 2:3]  # (n_fibres, 1)
    step_lengths = lengths / (num_points - 1)  # (n_fibres, 1)

    # Calculate step vectors
    steps = dirs * step_lengths  # (n_fibres, 3)

    # Indices for steps
    idx = torch.arange(num_points, device=device).view(1, -1, 1)  # (1, num_points, 1)

    # Compute all points
    fibres = coords0[:, None, :] + steps[:, None, :] * idx  # (n_fibres, num_points, 3)
    return fibres, step_lengths

def generate_fibres_random(domain_size, num_points, n_fibres, std_angle, device):
    """
    Generate n_fibres fibres in a cubic domain of given size.
    domain_size: (3,) tensor/list specifying [x_max, y_max, z_max]
    num_points: number of points per fibre
    std_angle: standard deviation of angle perturbation from vertical (z) direction
    returns:
    - fibres: (n_fibres, num_points, 3) tensor of coordinates
    - step_lengths: (n_fibres, 1) tensor of step lengths between points. used as rest lengths for springs.
    """
    # Random starting coordinates at z=0
    x0 = torch.rand(n_fibres, 1, device=device) * domain_size[0]
    y0 = torch.rand(n_fibres, 1, device=device) * domain_size[1]
    z0 = torch.zeros(n_fibres, 1, device=device)
    coords0 = torch.cat([x0, y0, z0], dim=1)  # (n_fibres, 3)

    # Base direction (0, 0, 1) with Gaussian perturbations in x/y
    dx = torch.randn(n_fibres, 1, device=device) * std_angle
    dy = torch.randn(n_fibres, 1, device=device) * std_angle
    dz = torch.ones(n_fibres, 1, device=device)  # mostly upward
    dirs = torch.cat([dx, dy, dz], dim=1)  # (n_fibres, 3)
    dirs = dirs / torch.norm(dirs, dim=1, keepdim=True)  # normalize

    # Calculate lengths until they hit the top (z = domain_size)
    lengths = (domain_size[2] - z0) / dirs[:, 2:3]  # (n_fibres, 1)
    step_lengths = lengths / (num_points - 1)  # (n_fibres, 1)

    # Calculate step vectors
    steps = dirs * step_lengths  # (n_fibres, 3)

    # Indices for steps
    idx = torch.arange(num_points, device=device).view(1, -1, 1)  # (1, num_points, 1)

    # Compute all points
    fibres = coords0[:, None, :] + steps[:, None, :] * idx  # (n_fibres, num_points, 3)
    return fibres, step_lengths

# Stable 3D angle computation
def angle_between(p1, p2, p3, eps=1e-8):
    # vectors v1 = p1 - p2, v2 = p3 - p2
    v1 = p1 - p2
    v2 = p3 - p2
    # 3D cross product scalar
    cross = torch.cross(v1, v2, dim=-1)  # (..., 3)
    cross_norm = torch.norm(cross, dim=-1)  # (...)
    dot = (v1 * v2).sum(dim=-1)  # (...)
    # angle in [0, pi]
    angle = torch.atan2(cross_norm + eps, dot)
    # atan2(y, x) returns in (-pi, pi); because we use abs(cross) angle is in (0, pi]
    return angle

def get_offsets(domain_size_current, apply_pbo, device):
    """
    domain_size_current: (3,) tensor/list specifying [x_max, y_max, z_max]
    apply_pbo: bool, whether to apply periodic boundary offsets in x and y directions
    device: torch device
    returns: list of (3,) tensors specifying offsets to apply to points for PBC
    """
    offsets = [torch.zeros(3, device=device)]
    if apply_pbo:
        offsets += [
            torch.tensor([domain_size_current[0], 0, 0], device=device),
            torch.tensor([0, domain_size_current[1], 0], device=device),
            torch.tensor([domain_size_current[0], domain_size_current[1], 0], device=device),
        ]
    return offsets

def save_model(jobname, fibres, steps, time, domain_size, diameter):
    if not os.path.exists(f"results/{jobname}/models"):
        os.makedirs(f"results/{jobname}/models")
    save_dict = {
        "fibres": fibres,
        "steps": steps,
        "time": time,
        "domain_size": domain_size,
        "diameter": diameter
    }

    torch.save(save_dict, f"results/{jobname}/models/{jobname}_{steps}.pt")
