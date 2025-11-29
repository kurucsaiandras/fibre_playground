import numpy as np
import matplotlib.pyplot as plt
import torch

def sample_sphere_uniform(N, device):
    # Uniform sampling on a sphere
    theta = torch.rand(N, device=device) * 2.0 * torch.pi        # azimuth
    u = torch.rand(N, device=device) * 2.0 - 1.0                 # cos(phi)
    phi_sphere = torch.acos(u)                                   # polar angle

    dx = torch.sin(phi_sphere) * torch.cos(theta)
    dy = torch.sin(phi_sphere) * torch.sin(theta)
    dz = torch.cos(phi_sphere)

    return torch.stack([dx, dy, dz], dim=-1)

def generate_noise_waves(N, device):
    sigma = 1.0
    chi = torch.randn(N, device=device)
    scale = 3.0 * sigma + chi

    d = sample_sphere_uniform(N, device)
    #dominant_dir = torch.tensor([1.0, 1.0, 0.0], device=device)
    #std_angle = 0.25
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

    grad_x = (common * omega_x).mean(dim=-1) / remap_scale[0]
    grad_y = (common * omega_y).mean(dim=-1) / remap_scale[1]
    grad_z = (common * omega_z).mean(dim=-1) / remap_scale[2]

    # Output shape: (..., 4)
    return torch.stack([noise, grad_x, grad_y, grad_z], dim=-1)

def eval_triangle_wave(x, l, m, n):
    '''
    x: (...) torch tensor containing 1d input points
    l: period length
    m: assymmetric const
    n: number of frequencies to sum
    '''
    k = torch.arange(1, n, dtype=x.dtype, device=x.device)  # shape (k,)
    # Fourier coefficiekts
    bk = -(2.0 * ((-1.0)**k) * (m**2)) / (k*k * (m-1) * (torch.pi**2)) * torch.sin( k * (m-1) * torch.pi / m )
    # argument: k * π * x / L
    arg = (torch.pi / l) * x[..., None] * k   # shape (..., k)
    # sum series
    fx = torch.sum(bk * torch.sin(arg), dim=-1)
    dfdx = torch.sum(bk * torch.cos(arg) * ((torch.pi / l) * k), dim=-1)
    return torch.stack([fx, dfdx], dim=-1)

def eval_with_derivatives(noise_waves, x, remap_scale, c):
    wave_noise = eval_waves(noise_waves, x, remap_scale)
    dg = wave_noise[..., 1:]
    modified_wave = eval_triangle_wave(x[..., 0] + c*wave_noise[..., 0], 0.15, 5.0, 10)
    df = modified_wave[..., 1]
    dfg = torch.stack([
        df * (1.0 + c * dg[..., 0]),
        df * (c * dg[..., 1]),
        df * (c * dg[..., 2]),
    ], dim=-1)
    # Add higher freq noise
    base_scale = torch.tensor([1.0, 1.0, 2.0], device=remap_scale.device)
    # mid frequency
    w_mf = 5.0
    remap_scale_mf = base_scale * 0.1
    wave_noise_mf = eval_waves(noise_waves, x, remap_scale_mf)
    # high frequency
    w_hf = 1.0
    remap_scale_hf = base_scale* 0.05
    wave_noise_hf = eval_waves(noise_waves, x, remap_scale_hf)

    val = modified_wave[..., 0] + w_mf * wave_noise_mf[..., 0] + w_hf * wave_noise_hf[..., 0]
    grad = dfg + w_mf * wave_noise_mf[..., 1:] + w_hf * wave_noise_hf[..., 1:]
    return torch.cat([val.unsqueeze(-1), grad], dim=-1)

def main():
    torch.manual_seed(0)
    from rve import RVE  # local import to avoid circular dependency
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Using device:", device)
    noise_waves = generate_noise_waves(100, device)
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
    remap_scale = torch.tensor([1.0, 1.5, 5.0])*0.25
    c = 1.0
    if False:
        modified_wave = eval_with_derivatives(noise_waves, eval_points, remap_scale, c)
        slices = modified_wave[:, :, ::res//5, :].cpu().numpy() # take slices along z-axis
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
            #plt.show()
            plt.savefig(f"wave_slice_z_{z:.2f}.png", dpi=400)
            plt.clf()

    n_fibres = 2500
    x0 = torch.rand(n_fibres, 1, device=device) * shape[0]
    y0 = torch.rand(n_fibres, 1, device=device) * shape[1]
    z0 = torch.zeros(n_fibres, 1, device=device)
    coords0 = torch.cat([x0, y0, z0], dim=1)  # (n_fibres, 3)
    n_steps = 40
    # scaling factor for gradient based on bundle position
    step_size_in_plane = 0.00025
    # scaling factor for gradient
    s = torch.randn(n_fibres, device=device) * step_size_in_plane * 0.5 + step_size_in_plane  # (n_fibres,)
    step_size_vertical = shape[2] / n_steps
    fibre_coords = torch.zeros(n_fibres, n_steps, 3, device=device)
    fibre_coords[:, 0, :] = coords0
    for step in range(1, n_steps):
        prev_coords = fibre_coords[:, step-1, :]
        noise = eval_with_derivatives(noise_waves, prev_coords, remap_scale, c)
        gradients = noise[:, 1:]  # (n_fibres, 3)
        #norm = torch.norm(gradients[:, :2], dim=1) + 1e-8
        gradx = gradients[:, 0] * -s #/ norm
        grady = gradients[:, 1] * -s #/ norm

        steps = torch.stack([grady, -gradx, torch.full_like(gradx, step_size_vertical)], dim=1)  # (n_fibres, 3)
        new_coords = prev_coords + steps
        fibre_coords[:, step, :] = new_coords

    rve = RVE.external(fibre_coords, radius=sh*0.001, downsample=False)
    rve.save("curl_noise_structured", 0, 0)

if __name__ == "__main__":
    main()