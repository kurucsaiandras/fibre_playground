import numpy as np
import matplotlib.pyplot as plt
import torch

def rot_from_z_to_d(z, d):
    # Rodrigues rotation formula
    device = z.device

    z = z / z.norm()
    d = d / d.norm()

    v = torch.cross(z, d)               # rotation axis
    s = torch.linalg.norm(v)            # sin(theta)
    c = torch.dot(z, d)                 # cos(theta)

    # If parallel or anti-parallel:
    if s < 1e-6:
        if c > 0:
            return torch.eye(3, device=device)  # no rotation
        else:
            # 180° rotation around x-axis is fine
            return torch.tensor([
                [1,0,0],
                [0,-1,0],
                [0,0,-1]
            ], device=device)

    # Skew-symmetric cross-product matrix of v
    vx = torch.tensor([
        [0,     -v[2],  v[1]],
        [v[2],   0,    -v[0]],
        [-v[1],  v[0],  0   ]
    ], device=device)

    R = torch.eye(3, device=device) + vx + (vx @ vx) * ((1 - c) / (s*s))
    return R

def sample_sphere_with_dir(N, d, std_angle, device):
    thetas = torch.randn(N, device=device) * std_angle
    phis = torch.rand(N, device=device) * (2.0 * torch.pi)

    dx = torch.sin(thetas) * torch.cos(phis)
    dy = torch.sin(thetas) * torch.sin(phis)
    dz = torch.cos(thetas)
    dirs_local = torch.stack([dx, dy, dz], dim=-1)

    # build rotation matrix
    z = torch.tensor([0.0, 0.0, 1.0], device=device)
    R = rot_from_z_to_d(z, d)

    # rotate to global space
    dirs = dirs_local @ R.T
    return dirs

def sample_sphere_uniform(N, device):
    # Uniform sampling on a sphere
    theta = torch.rand(N, device=device) * 2.0 * torch.pi        # azimuth
    u = torch.rand(N, device=device) * 2.0 - 1.0                 # cos(phi)
    phi_sphere = torch.acos(u)                                   # polar angle

    dx = torch.sin(phi_sphere) * torch.cos(theta)
    dy = torch.sin(phi_sphere) * torch.sin(theta)
    dz = torch.cos(phi_sphere)

    return torch.stack([dx, dy, dz], dim=-1)

def generate_noise_waves(N, device, seed=0):
    torch.manual_seed(seed)
    sigma = 1.0
    chi = torch.randn(N, device=device)
    scale = 3.0 * sigma + chi

    d = sample_sphere_uniform(N, device)
    #dominant_dir = torch.tensor([1.0, 1.0, 0.0], device=device)
    #std_angle = 0.75
    #d = sample_sphere_with_dir(N, dominant_dir, std_angle, device)

    omega = scale[:, None] * d    # (N,3)

    # phase offset
    phi = torch.rand(N, device=device) * 2.0 * torch.pi
    weight = torch.exp(-(chi * chi) / (sigma * sigma))

    return torch.stack([omega[:, 0], omega[:, 1], omega[:, 2], phi, weight], dim=0)

def eval_waves(waves, x, remap_scale=torch.tensor([1.0, 1.0, 1.0])):
    remap_scale = remap_scale.to(x.device)
    # Remap coordinates
    x = x / remap_scale

    # waves: shape (5, N) = [omega_x, omega_y, omega_z, phi, weight]
    omega_x = waves[0]          # (N,)
    omega_y = waves[1]
    omega_z = waves[2]
    phi     = waves[3]
    weight  = waves[4]

    # x: (..., 3)
    # We broadcast x against waves: result (..., N)
    phase = (
        x[..., 0][..., None] * omega_x +
        x[..., 1][..., None] * omega_y +
        x[..., 2][..., None] * omega_z +
        phi
    )

    # Noise value
    noise = (weight * torch.sin(phase)).mean(dim=-1)

    # Gradient: weight * cos(phase) * omega_dim
    common = weight * torch.cos(phase)    # (..., N)

    grad_x = (common * omega_x).mean(dim=-1)
    grad_y = (common * omega_y).mean(dim=-1)
    grad_z = (common * omega_z).mean(dim=-1)

    # Output shape: (..., 4)
    return torch.stack([noise, grad_x, grad_y, grad_z], dim=-1)

def main():
    from rve import RVE  # local import to avoid circular dependency
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Using device:", device)
    waves = generate_noise_waves(100, device)
    # generate 3D meshgrid
    sh = 1
    shape = (sh, sh, sh)
    res=200
    eval_points = torch.meshgrid(
        torch.linspace(0, shape[0], steps=res, device=device)[:-1],
        torch.linspace(0, shape[1], steps=res, device=device)[:-1],
        torch.linspace(0, shape[2], steps=res, device=device)[:-1]
    )
    eval_points = torch.stack(eval_points, dim=-1) # shape (res, res, res, 3)
    remap_scale = torch.tensor([1.0, 1.0, 2.0])
    wave_noise_lf = eval_waves(waves, eval_points, remap_scale)
    wave_noise_hf = eval_waves(waves, eval_points, remap_scale*0.05)
    wave_noise = wave_noise_lf + 0.0 * wave_noise_hf
    slices = wave_noise[:, :, ::res//5, :].cpu().numpy() # take slices along z-axis
    for i, z in enumerate(np.linspace(0, shape[0], num=5, endpoint=False)):
        image = slices[:, :, i, 0]  # noise values
        plt.title(f"Wave noise slice at z={z:.2f}")
        plt.xlabel("x")
        plt.ylabel("y")
        plt.imshow(image.T, extent=(0, shape[0], 0, shape[1]), origin='lower', cmap='gray')
        plt.quiver(
            np.linspace(0, shape[0], num=res)[::5],
            np.linspace(0, shape[1], num=res)[::5],
            slices[::5, ::5, i, 2].T,  # dy
            -slices[::5, ::5, i, 1].T,  # -dx
            color='red', alpha=0.5
        )
        plt.grid(False)
        plt.savefig(f"wave_slice_z_{z:.2f}.png", dpi=400)
        plt.clf()
    
    n_fibres = 2500
    x0 = torch.rand(n_fibres, 1, device=device) * shape[0]
    y0 = torch.rand(n_fibres, 1, device=device) * shape[1]
    z0 = torch.zeros(n_fibres, 1, device=device)
    coords0 = torch.cat([x0, y0, z0], dim=1)  # (n_fibres, 3)
    n_steps = 40
    # scaling factor for gradient based on bundle position
    step_size_in_plane = 0.025
    #s = torch.where(
    #    ((0.1*shape[0] < x0) & (x0 < 0.15*shape[0])) |
    #    ((0.6*shape[0] < x0) & (x0 < 0.62*shape[0])),
    #    1.0, -1.0).squeeze() * step_size_in_plane
    step_size_vertical = shape[2] / n_steps
    fibre_coords = torch.zeros(n_fibres, n_steps, 3, device=device)
    fibre_coords[:, 0, :] = coords0
    for step in range(1, n_steps):
        prev_coords = fibre_coords[:, step-1, :]
        wave_noise_lf = eval_waves(waves, prev_coords, remap_scale)
        if step == 1:
            tol = 0.01
            s = torch.where((wave_noise_lf[:, 0].abs() < tol), 1.0, -1.0).squeeze() * step_size_in_plane
        wave_noise_hf = eval_waves(waves, prev_coords, remap_scale*0.2)
        wave_noise = wave_noise_lf + 0.3 * wave_noise_hf
        gradients = wave_noise[:, 1:]  # (n_fibres, 3)
        gradx = gradients[:, 0] * s
        grady = gradients[:, 1] * s
        steps = torch.stack([grady, -gradx, torch.full_like(gradx, step_size_vertical)], dim=1)  # (n_fibres, 3)
        new_coords = prev_coords + steps
        fibre_coords[:, step, :] = new_coords

    rve = RVE.external(fibre_coords, radius=sh*0.001, downsample=False)
    rve.save("curl_noise_waves", 0, 0)

if __name__ == "__main__":
    main()