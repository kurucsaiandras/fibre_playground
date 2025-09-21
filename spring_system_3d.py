import torch
import math
import torch.nn.functional as F
import pyvista as pv
import threading
import time
import poisson_disc as pd

# Use GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#device = torch.device("cpu")
print(f"Using device: {device}")

def collision_loss(points, k, D):
    n_fibres, resolution, _ = points.shape

    diff = points[:, :, None, None, :] - points[None, None, :, :, :]
    dists = torch.linalg.norm(diff, dim=-1)

    # mask: only between fibres
    mask_diff_fibre = ~torch.eye(n_fibres, dtype=torch.bool, device=points.device)
    mask_diff_fibre = mask_diff_fibre[:, None, :, None]

    # penalty = k * relu(D - dist)
    penalties = 0.5 * k * F.relu(D - dists)**2 * mask_diff_fibre

    # keep only i < j (upper triangle in fibre–fibre matrix)
    fibre_ids = torch.arange(n_fibres, device=points.device)
    mask_upper = (fibre_ids[:, None] < fibre_ids[None, :])[:, None, :, None]
    penalties = penalties * mask_upper

    num_pairs = (penalties > 0).sum()
    if num_pairs > 0:
        loss = penalties.sum() / num_pairs
    else:
        loss = torch.tensor(0.0, device=points.device)

    return loss

def linearity_loss(points, k, L):
    """
    points: (n_fibres, resolution, 3)
    k: scalar or (n_fibres, ) tensor
    L: scalar or (n_fibres, ) tensor
    """
    diffs = points[:,:-1] - points[:,1:]      # (n_fibres, resolution-1, 3)
    dists = torch.norm(diffs, dim=2)    # (n_fibres, resolution-1)
    loss = 0.5 * k * (dists - L) ** 2
    loss = loss.sum()
    return loss

def torsional_loss(points, k, L):
    """
    points: (n_fibres, resolution, 3)
    k: scalar or (n_fibres, ) tensor
    L: scalar or (n_fibres, ) tensor
    """
    p1 = points[:,:-2]       # (n_fibres, resolution-2, 3)
    p2 = points[:,1:-1]
    p3 = points[:,2:]
    angles = angle_between(p1, p2, p3)  # (n_fibres, resolution-2)
    loss = 0.5 * k * (angles - L) ** 2
    loss = loss.sum()
    return loss

def boundary_loss(points, domain_size, k):
    """
    points: (n_fibres, resolution, 3)
    domain_size: (3,) tensor/list specifying [x_max, y_max, z_max]
    k: scalar
    """
    # lower violations: values < 0
    lower_violation = torch.clamp(-points, min=0.0)
    # upper violations: values > domain_size
    upper_violation = torch.clamp(points - domain_size, min=0.0)
    # total violation per coordinate
    violations = lower_violation + upper_violation  # shape (n_fibres, resolution, 3)
    loss = 0.5 * k * (violations ** 2).sum(dim=2)  # sum over x,y,z -> shape (n_fibres, resolution)
    loss = loss.sum()  # sum over all points
    return loss

def generate_fibres(domain_size, num_points, std_angle):
    """
    Generate fibres in a cubic domain of given size.
    domain_size: (3,) tensor/list specifying [x_max, y_max, z_max]
    num_points: number of points per fibre
    std_angle: standard deviation of angle perturbation from vertical (z) direction
    returns:
    - fibres: (n_fibres, num_points, 3) tensor of coordinates
    - step_lengths: (n_fibres, 1) tensor of step lengths between points. used as rest lengths for springs.
    """
    # Random starting coordinates at z=0
    # Generate Poisson disk samples (2D)
    samples = pd.Bridson_sampling(dims=domain_size[:2].cpu().numpy(), radius=0.55, k=30)
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


# ------------------------
# Setup
# ------------------------
resolution = 30 # TODO we need bigger number here to actually avoid collisions
#n_fibres = 200
domain_size = torch.tensor([10.0, 10.0, 10.0], device=device)
domain_size_final = torch.tensor([7.0, 7.0, 10.0], device=device)
angle_std_dev = 0.05
fibre_diameter = 0.2
fibre_diameter_final = 0.5
n_iter = 500000

fibres, spring_L_linear = generate_fibres(domain_size, resolution, angle_std_dev)
n_fibres = fibres.shape[0]
print(f"Fibre to volume ratio: {n_fibres * math.pi * (fibre_diameter/2)**2 / (domain_size[0]*domain_size[1]):.3f}")
spring_k_linear = 0.1 # TODO tune based on the step length
spring_k_torsional = 0.001 # TODO tune based on the step length
spring_L_torsional = math.pi
spring_k_boundary = 1.0
spring_k_collision = 100.0

to_plot = True

if to_plot:
    plotter = pv.Plotter()

    # Build initial tube meshes and add to plotter, storing the PolyData objects
    meshes = []
    for i in range(n_fibres):
        arr = fibres[i].cpu().numpy()
        line = pv.Spline(arr, n_points=resolution)
        tube = line.tube(radius=fibre_diameter / 2.0)
        plotter.add_mesh(tube, color="lightsteelblue", smooth_shading=True)
        meshes.append(tube)  # keep reference to update points

    plotter.show(auto_close=False, interactive_update=True)

# Optimize for all fibres
fibres_params = torch.nn.Parameter(fibres.clone())
optimizer = torch.optim.Adam([fibres_params], lr=1e-2)

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

# ------------------------
# Optimization loop
# ------------------------

max_grad_norm = 1.0

def optimize():
    global fibre_diameter, domain_size
    for step in range(n_iter+1):
        optimizer.zero_grad()

        # Linear loss (distances)
        loss_linear = linearity_loss(fibres_params, spring_k_linear, spring_L_linear)
        # Torsional loss (angles)
        loss_torsion = torsional_loss(fibres_params, spring_k_torsional, spring_L_torsional)
        # Boundary loss
        loss_boundary = boundary_loss(fibres_params, domain_size, spring_k_boundary)
        # Collision loss
        loss_collision = collision_loss(fibres_params, spring_k_collision, fibre_diameter)

        loss_sum = loss_linear + loss_torsion + loss_boundary + loss_collision

        # detect bad loss before backward
        if not torch.isfinite(loss_sum):
            print("Non-finite loss at step", step, "-> aborting")
            break

        loss_sum.backward()

        # gradient clipping to avoid explosion
        torch.nn.utils.clip_grad_norm_([fibres_params], max_grad_norm)

        optimizer.step()

        # sanity check after step
        if not torch.all(torch.isfinite(fibres_params)):
            print("Parameters became non-finite at step", step, "-> aborting")
            break

        if step % 100 == 0:
            print(f"Step {step}: Overall Energy = {loss_sum.item():.6f}, "
                f"Linear = {loss_linear.item():.6f}, "
                f"Torsional = {loss_torsion.item():.6f}, "
                f"Boundary = {loss_boundary.item():.6f}, "
                f"Collision = {loss_collision.item():.6f}")
            if to_plot:
                # Update PyVista meshes
                for i in range(n_fibres):
                    arr = fibres_params[i].detach().cpu().numpy()
                    new_line = pv.Spline(arr, n_points=resolution)
                    new_tube = new_line.tube(radius=fibre_diameter / 2.0)
                    meshes[i].points[:] = new_tube.points
            
        # adjust configuration if no collisions
        if loss_collision == 0:
            # save params
            torch.save(fibres_params.data.cpu(), "fibre_coords.pt")
            # first, increase fibre diameter until target
            if fibre_diameter < fibre_diameter_final:
                fibre_diameter += 0.01
                print(f"Increase fibre diameter to {fibre_diameter:.3f}")
                print(f"Fibre to volume ratio: {n_fibres * math.pi * (fibre_diameter/2)**2 / (domain_size[0]*domain_size[1]):.3f}")
            # then, decrease domain size until target
            elif domain_size[0] > domain_size_final[0]:
                domain_size[0] -= 0.1
                domain_size[1] -= 0.1
                # subtract 0.05 from x and y of points
                fibres_params.data[:, :, 0] -= 0.05
                fibres_params.data[:, :, 1] -= 0.05
                print(f"Decrease domain size to {domain_size[0]:.1f} x {domain_size[1]:.1f}")
                print(f"Fibre to volume ratio: {n_fibres * math.pi * (fibre_diameter/2)**2 / (domain_size[0]*domain_size[1]):.3f}")
            else:
                print("Reached target configuration -> stopping")
                break

if to_plot:
    threading.Thread(target=optimize, daemon=True).start()
    # call update in main thread until optimization is done
    while True:
        plotter.update()
        time.sleep(0.05)
        if not threading.main_thread().is_alive():
            break
    plotter.show(interactive_update=False)  
else:
    optimize()
