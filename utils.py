import torch
import poisson_disc as pd
import os
import curl_noise_perlin
import curl_noise_structured
import math

def get_bounding_boxes(points, radii):
    """
    points: (n_fibres, resolution, 3)
    radii: scalar or (n_fibres,) tensor of fibre radii
    returns: (n_fibres, 2, 3) tensor of bounding boxes: p_min, p_max
    """
    min_coords = points - radii.view(-1, 1, 1)
    max_coords = points + radii.view(-1, 1, 1)
    boxes = torch.stack([min_coords.min(dim=1).values, max_coords.max(dim=1).values], dim=1)
    return boxes

def get_pbc_offsets(domain_size, device):
    """
    Calculates the offsets by which bboxes have to be shifted to get the correct pbc distances
    Naming has to align with `get_bbox_intersections()`
    """
    offsets = {
        "normal": {
            "i": torch.tensor([0.0, 0.0, 0.0], device=device),
            "j": torch.tensor([0.0, 0.0, 0.0], device=device),
        },
        "x_pbc": {
            "i": torch.tensor([domain_size[0], 0.0, 0.0], device=device),
            "j": torch.tensor([0.0, 0.0, 0.0], device=device),
        },
        "y_pbc": {
            "i": torch.tensor([0.0, domain_size[1], 0.0], device=device),
            "j": torch.tensor([0.0, 0.0, 0.0], device=device),
        },
        "xy_pbc": {
            "i": torch.tensor([domain_size[0], domain_size[1], 0.0], device=device),
            "j": torch.tensor([0.0, 0.0, 0.0], device=device),
        },
        "yx_pbc": {
            "i": torch.tensor([domain_size[0], 0.0, 0.0], device=device),
            "j": torch.tensor([0.0, domain_size[1], 0.0], device=device),
        },
    }

    return offsets

def get_bbox_intersections(boxes, domain_size, apply_pbc):
    """
    boxes: (n_fibres, 2, 3) tensor of bounding boxes: p_min, p_max
    domain_size: (3,) tensor/list specifying [x_max, y_max, z_max]
    returns: (n_fibres, n_fibres) boolean tensor indicating which boxes intersect
    """
    intersections = {}
    p_min = boxes[:, 0, :]  # (n, 3)
    p_max = boxes[:, 1, :]  # (n, 3)

    # Expand dimensions for broadcasting
    p_min_exp = p_min[:, None, :]  # (n, 1, 3)
    p_max_exp = p_max[:, None, :]  # (n, 1, 3)

    # Check for overlap in each dimension
    overlap_x = (p_min_exp[:, :, 0] <= p_max_exp[:, :, 0].T) & (p_max_exp[:, :, 0] >= p_min_exp[:, :, 0].T)
    overlap_y = (p_min_exp[:, :, 1] <= p_max_exp[:, :, 1].T) & (p_max_exp[:, :, 1] >= p_min_exp[:, :, 1].T)
    overlap_z = (p_min_exp[:, :, 2] <= p_max_exp[:, :, 2].T) & (p_max_exp[:, :, 2] >= p_min_exp[:, :, 2].T)

    # Boxes intersect if they overlap in all three dimensions
    # remove double counting (i,j) and (j,i) and self-intersections
    intersections["normal"] = torch.triu(overlap_x & overlap_y & overlap_z, diagonal=1)

    if apply_pbc:
        # PBCs
        overlap_x_pbc = (p_min_exp[:, :, 0] - domain_size[0] <= p_max_exp[:, :, 0].T) & (p_max_exp[:, :, 0] - domain_size[0] >= p_min_exp[:, :, 0].T)
        overlap_y_pbc = (p_min_exp[:, :, 1] - domain_size[1] <= p_max_exp[:, :, 1].T) & (p_max_exp[:, :, 1] - domain_size[1] >= p_min_exp[:, :, 1].T)

        # PBC intersections
        intersections["x_pbc"] = overlap_x_pbc & overlap_y & overlap_z
        intersections["y_pbc"] = overlap_x & overlap_y_pbc & overlap_z
        intersections["xy_pbc"] = overlap_x_pbc & overlap_y_pbc & overlap_z
        intersections["yx_pbc"] = overlap_x_pbc & overlap_y_pbc.T & overlap_z

        # only remove self-intersections for pbc (keep the symmetric ones)
        intersections["x_pbc"] = intersections["x_pbc"] & (~torch.eye(intersections["x_pbc"].shape[0], dtype=bool, device=intersections["x_pbc"].device))
        intersections["y_pbc"] = intersections["y_pbc"] & (~torch.eye(intersections["y_pbc"].shape[0], dtype=bool, device=intersections["y_pbc"].device))
        intersections["xy_pbc"] = intersections["xy_pbc"] & (~torch.eye(intersections["xy_pbc"].shape[0], dtype=bool, device=intersections["xy_pbc"].device))
        intersections["yx_pbc"] = intersections["yx_pbc"] & (~torch.eye(intersections["yx_pbc"].shape[0], dtype=bool, device=intersections["yx_pbc"].device))

    return intersections

def generate_radii(n_fibres, config, device):
    """
    Generate fibre radii with normal distribution, clipped to positive values.
    returns: (n_fibres,) tensor of initial radii, (n_fibres,) tensor of target radii
    """
    initial_r = config.initialization.generate.fibre_r_initial
    mean_r = config.evolution.fibre_r_target
    std_r = config.evolution.fibre_r_std
    r_initial = torch.full((n_fibres,), initial_r, device=device)
    r_target = torch.randn(n_fibres, device=device) * std_r + mean_r
    r_target = torch.clamp(r_target, min=0.005)  # avoid non-positive radii
    return r_initial, r_target

def generate_fibres_poisson(config, device):
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
    # extract params from config
    radius = config.initialization.generate.fibre_r_initial
    num_points = config.initialization.generate.resolution
    std_angle = config.initialization.generate.angle_std_dev
    domain_size = config.initialization.generate.domain_size_initial
    # Random starting coordinates at z=0
    # Generate Poisson disk samples (2D)
    samples = pd.Bridson_sampling(dims=domain_size[:2], radius=radius, k=30)
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
    return fibres, step_lengths, math.pi

def importance_sample(n_samples, domain_size, device):
    shape = (5, 5, 0)
    grid = curl_noise_perlin.generate_perlin_grid(shape, device=device)
    res = 100
    eval_points = torch.meshgrid(
        torch.linspace(0, shape[0], steps=res, device=device),
        torch.linspace(0, shape[1], steps=res, device=device),
        torch.tensor([0.0], device=device)
    )
    eval_points = torch.stack(eval_points, dim=-1) # shape (res, res, 1, 3)
    perlin_values = curl_noise_perlin.perlin_noised(grid, eval_points)
    perlin_values_2d = perlin_values[:, :, 0, 0]
    pdf = (perlin_values_2d - perlin_values_2d.min()).flatten()
    pdf = pdf / pdf.max()
    # saturate high and low values with smooth function
    sat_strength = 5.0
    pdf = torch.sigmoid((pdf - 0.5) * sat_strength)
    pdf = pdf / pdf.sum()
    indices = torch.multinomial(pdf, n_samples, replacement=True)
    xs = indices % res
    ys = indices // res
    xs = xs.float() + torch.rand(n_samples, device=device)
    ys = ys.float() + torch.rand(n_samples, device=device)
    xs = xs / res * domain_size[0]
    ys = ys / res * domain_size[1]
    # plot for debugging
    pdf = pdf.reshape(res, res)
    import matplotlib.pyplot as plt
    plt.figure()
    plt.imshow(pdf.cpu().numpy(), extent=(0, domain_size[0], 0, domain_size[1]), origin='lower', cmap='gray')
    plt.scatter(xs.cpu().numpy(), ys.cpu().numpy(), color='red', s=1, alpha=0.5)
    plt.title("Importance Sampling Distribution")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.savefig("importance_sampling_debug.png", dpi=400)
    plt.clf()
    return xs.view(-1, 1), ys.view(-1, 1)

def generate_fibres_random(config, device):
    """
    Generate n_fibres fibres in a cubic domain of given size.
    domain_size: (3,) tensor/list specifying [x_max, y_max, z_max]
    num_points: number of points per fibre
    std_angle: standard deviation of angle perturbation from vertical (z) direction
    returns:
    - fibres: (n_fibres, num_points, 3) tensor of coordinates
    - step_lengths: (n_fibres, 1) tensor of step lengths between points. used as rest lengths for springs.
    """
    # extract params from config
    n_fibres = config.initialization.generate.num_of_fibres
    num_points = config.initialization.generate.resolution
    std_angle = config.initialization.generate.angle_std_dev
    domain_size = config.initialization.generate.domain_size_initial
    # Random starting coordinates at z=0
    x0, y0 = importance_sample(n_fibres, domain_size, device)
    #x0 = torch.rand(n_fibres, 1, device=device) * domain_size[0]
    #y0 = torch.rand(n_fibres, 1, device=device) * domain_size[1]
    z0 = torch.zeros(n_fibres, 1, device=device)

    # Base direction (0, 0, 1) with Gaussian perturbations in x/y
    # NOTE: This is not an actual normal distribution of angles in 3D.
    # This gives a slightly flatter distribution for the initial projected
    # angles in the xz and yz planes than using spherical coordinates.
    # Using spherical coordinates gives a more 'spiky' distribution that
    # resembles the distrib. in converged stages. (although using this
    # method also converges to the same results)
    
    dx = torch.randn(n_fibres, 1, device=device) * std_angle
    dy = torch.randn(n_fibres, 1, device=device) * std_angle
    dz = torch.ones(n_fibres, 1, device=device)  # mostly upward
    #x_ = domain_size[0]
    #tilted_bundles = (0.1*x_ < x0) & (x0 < 0.15*x_) | (0.6*x_ < x0) & (x0 < 0.62*x_)
    #dy = torch.where(tilted_bundles, dy+0.5, dy)
    #y0 = torch.where(tilted_bundles, y0-0.25*domain_size[1], y0)
    dirs = torch.cat([dx, dy, dz], dim=1)  # (n_fibres, 3)
    dirs = dirs / torch.norm(dirs, dim=1, keepdim=True)  # normalize
    coords0 = torch.cat([x0, y0, z0], dim=1)  # (n_fibres, 3)

    # Draw inclination angle from normal distribution (angle with z axis)
    #thetas = torch.randn(n_fibres, device=device) * std_angle
    # Draw azimuthal angle from uniform distribution (rotate around z axis)
    #phis = torch.rand(n_fibres, device=device) * torch.pi # only until pi as theta can be negative
    # Calculate coordinates of direction vectors
    #dx = torch.sin(thetas) * torch.cos(phis)
    #dy = torch.sin(thetas) * torch.sin(phis)
    #dz = torch.cos(thetas)
    #dirs = torch.stack([dx, dy, dz], dim=1)  # (n_fibres, 3) already normalized

    # Calculate lengths until they hit the top (z = domain_size)
    lengths = (domain_size[2] - z0) / dirs[:, 2:3]  # (n_fibres, 1)
    step_lengths = lengths / (num_points - 1)  # (n_fibres, 1)

    # Calculate step vectors
    steps = dirs * step_lengths  # (n_fibres, 3)

    # Indices for steps
    idx = torch.arange(num_points, device=device).view(1, -1, 1)  # (1, num_points, 1)

    # Compute all points
    fibres = coords0[:, None, :] + steps[:, None, :] * idx  # (n_fibres, num_points, 3)
    return fibres, step_lengths, math.pi

def generate_fibres_curl(config, device):
    n_fibres = config.initialization.generate.num_of_fibres
    num_points = config.initialization.generate.resolution
    domain_size = torch.tensor(
        config.initialization.generate.domain_size_initial, device=device)
    noise_waves = curl_noise_structured.generate_noise_waves(100, device)
    # remaps the scale of the noise function in each dimension
    remap_scale = torch.tensor([1.0, 1.5, 5.0], device=device)*0.25 * domain_size
    c = 1.0 # warp factor for the wave distortion
    x0 = torch.rand(n_fibres, 1, device=device) * domain_size[0]
    y0 = torch.rand(n_fibres, 1, device=device) * domain_size[1]
    z0 = torch.zeros(n_fibres, 1, device=device)
    coords0 = torch.cat([x0, y0, z0], dim=1)  # (n_fibres, 3)
    # scaling factor for gradient TODO: make config param, and independent of step size
    step_size_in_plane = 0.00025
    # scaling factor for gradient
    s = torch.randn(n_fibres, device=device) * step_size_in_plane * 0.5 + step_size_in_plane  # (n_fibres,)
    step_size_vertical = domain_size[2] / num_points
    fibre_coords = torch.zeros(n_fibres, num_points, 3, device=device)
    fibre_coords[:, 0, :] = coords0
    for step in range(1, num_points):
        prev_coords = fibre_coords[:, step-1, :]
        noise = curl_noise_structured.eval_with_derivatives(noise_waves, prev_coords, remap_scale, c)
        gradients = noise[:, 1:]  # (n_fibres, 3)
        #norm = torch.norm(gradients[:, :2], dim=1) + 1e-8
        gradx = gradients[:, 0] * -s #/ norm
        grady = gradients[:, 1] * -s #/ norm

        steps = torch.stack([grady, -gradx, torch.full_like(gradx, step_size_vertical)], dim=1)  # (n_fibres, 3)
        new_coords = prev_coords + steps
        fibre_coords[:, step, :] = new_coords

    # calculate segment lengths
    diffs = fibre_coords[:, 1:, :] - fibre_coords[:, :-1, :]
    segment_lengths = torch.norm(diffs, dim=2)  # (n_fibres, num_points-1)
    p1 = fibre_coords[:,:-2]       # (n_fibres, resolution-2, 3)
    p2 = fibre_coords[:,1:-1]
    p3 = fibre_coords[:,2:]
    phi0_curvature = angle_between(p1, p2, p3)  # (n_fibres, resolution-2)
    return fibre_coords, segment_lengths, phi0_curvature

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

def get_offsets(domain_size_current, apply_pbc, device):
    """
    domain_size_current: (3,) tensor/list specifying [x_max, y_max, z_max]
    apply_pbc: bool, whether to apply periodic boundary offsets in x and y directions
    device: torch device
    returns: list of (3,) tensors specifying offsets to apply to points for PBC
    """
    offsets = [torch.zeros(3, device=device)]
    if apply_pbc:
        offsets += [
            torch.tensor([domain_size_current[0], 0, 0], device=device),
            torch.tensor([0, domain_size_current[1], 0], device=device),
            torch.tensor([domain_size_current[0], domain_size_current[1], 0], device=device),
        ]
    return offsets

def latest_rve(root: str) -> int:
    """Return the largest numeric step number among .pt files in a directory."""
    files = [
        f for f in os.listdir(root)
        if f.endswith(".pt") and f[:-3].isdigit()
    ]
    if not files:
        raise FileNotFoundError(f"No numeric .pt files found in {root}")

    # Sort numerically by filename (excluding ".pt")
    latest_step = max(int(f[:-3]) for f in files)
    return latest_step

def get_slice(fibres: torch.Tensor, z: float) -> torch.Tensor:
    """
    Get the intersection points of fibres with a horizontal plane at height z using PyTorch.
    
    Args:
        fibres (torch.Tensor): Tensor of shape (N, M, 3) representing N fibres with M points each in 3D space.
        z (float): Height of the horizontal plane.
        
    Returns:
        torch.Tensor: Tensor of shape (K, 2) containing the projected intersection points (x, y).
    """
    # Compute start and end points of each segment
    p1 = fibres[:, :-1, :]   # (N, M-1, 3)
    p2 = fibres[:, 1:, :]    # (N, M-1, 3)

    # Compute z differences
    z1 = p1[..., 2]
    z2 = p2[..., 2]

    # Check if segment crosses the plane (one above, one below)
    crosses = (z1 - z) * (z2 - z) < 0

    # Parameter t for intersection points (only valid where crosses=True)
    # Avoid division by zero
    denom = (z2 - z1)
    denom[denom == 0] = float('nan')  # prevent division by zero
    t = (z - z1) / denom

    # Compute intersection points
    intersection_points = p1 + t.unsqueeze(-1) * (p2 - p1)
    # Mask out non-crossing segments
    intersection_points = intersection_points[crosses]
    # Handle cases where points lie exactly on the plane
    on_plane_points = torch.cat([p1[z1 == z], p2[z2 == z]], dim=0)
    # Combine crossing and on-plane points
    all_points = torch.cat([intersection_points, on_plane_points], dim=0)

    if all_points.numel() == 0:
        return torch.empty((0, 2), dtype=fibres.dtype, device=fibres.device)

    return all_points[:, :2]  # return x, y only