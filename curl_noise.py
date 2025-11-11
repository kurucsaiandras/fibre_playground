import numpy as np
import matplotlib.pyplot as plt
import torch
from rve import RVE

def generate_perlin_grid(shape, device='cpu', seed=0):
    """
    Generate a 3D Perlin-style grid of random unit vectors.

    Args:
        shape: tuple of ints (nx, ny, nz) for the number of cells in each dimension
        device: 'cpu' or 'cuda'
        seed: random seed for reproducibility
    Returns:
        unit_vectors: torch.Tensor of shape (nx+1, ny+1, nz+1, 3)
    """
    torch.manual_seed(seed)
    grid_shape = (shape[0]+1, shape[1]+1, shape[2]+1, 3)
    random_vectors = torch.randn(grid_shape, device=device)  # normal distribution
    norms = random_vectors.norm(dim=-1, keepdim=True)  # compute vector norms
    unit_vectors = random_vectors / norms  # normalize to unit length
    return unit_vectors

def quintic_interp(f):
    """Quintic interpolation and derivative."""
    u = f**3 * (f * (f*6 - 15) + 10)
    du = 30 * f**2 * (f * (f - 2) + 1)
    return u, du

def perlin_noised(grid, x, remap_scale=torch.tensor([1.0, 1.0, 1.0])):
    """
    Compute 3D Perlin noise and derivatives using PyTorch.
    
    Parameters:
        x: Tensor of shape (..., 3) - points in 3D space
        grid: Tensor of shape (nx+1, ny+1, nz+1, 3) - precomputed unit vectors
        remap_scale: Tensor of shape (3,) - scales the noise in each dimension

    Returns:
        Tensor of shape (..., 4) - [noise, dx, dy, dz]
    """
    # Remap coordinates
    x = x / remap_scale
    # Separate integer/fractional part
    i = torch.floor(x).long()
    f = x - i.float()
    
    u, du = quintic_interp(f)
    
    # Helper to fetch gradient from grid
    def g(ix, iy, iz):
        return grid[ix%grid.shape[0], iy%grid.shape[1], iz%grid.shape[2]]
    
    ga = g(i[...,0], i[...,1], i[...,2])
    gb = g(i[...,0]+1, i[...,1], i[...,2])
    gc = g(i[...,0], i[...,1]+1, i[...,2])
    gd = g(i[...,0]+1, i[...,1]+1, i[...,2])
    ge = g(i[...,0], i[...,1], i[...,2]+1)
    gf = g(i[...,0]+1, i[...,1], i[...,2]+1)
    gg = g(i[...,0], i[...,1]+1, i[...,2]+1)
    gh = g(i[...,0]+1, i[...,1]+1, i[...,2]+1)
    
    # Projections
    va = torch.sum(ga * (f - torch.tensor([0.,0.,0.], device=x.device)), dim=-1)
    vb = torch.sum(gb * (f - torch.tensor([1.,0.,0.], device=x.device)), dim=-1)
    vc = torch.sum(gc * (f - torch.tensor([0.,1.,0.], device=x.device)), dim=-1)
    vd = torch.sum(gd * (f - torch.tensor([1.,1.,0.], device=x.device)), dim=-1)
    ve = torch.sum(ge * (f - torch.tensor([0.,0.,1.], device=x.device)), dim=-1)
    vf = torch.sum(gf * (f - torch.tensor([1.,0.,1.], device=x.device)), dim=-1)
    vg = torch.sum(gg * (f - torch.tensor([0.,1.,1.], device=x.device)), dim=-1)
    vh = torch.sum(gh * (f - torch.tensor([1.,1.,1.], device=x.device)), dim=-1)
    
    # Interpolated noise
    v = (va +
         u[...,0]*(vb-va) +
         u[...,1]*(vc-va) +
         u[...,2]*(ve-va) +
         u[...,0]*u[...,1]*(va-vb-vc+vd) +
         u[...,1]*u[...,2]*(va-vc-ve+vg) +
         u[...,2]*u[...,0]*(va-vb-ve+vf) +
         u[...,0]*u[...,1]*u[...,2]*(-va+vb+vc-vd+ve-vf-vg+vh))
    
    # Derivative of noise
    d = (ga +
         u[...,0].unsqueeze(-1)*(gb-ga) +
         u[...,1].unsqueeze(-1)*(gc-ga) +
         u[...,2].unsqueeze(-1)*(ge-ga) +
         u[...,0].unsqueeze(-1)*u[...,1].unsqueeze(-1)*(ga-gb-gc+gd) +
         u[...,1].unsqueeze(-1)*u[...,2].unsqueeze(-1)*(ga-gc-ge+gg) +
         u[...,2].unsqueeze(-1)*u[...,0].unsqueeze(-1)*(ga-gb-ge+gf) +
         u[...,0].unsqueeze(-1)*u[...,1].unsqueeze(-1)*u[...,2].unsqueeze(-1)*(-ga+gb+gc-gd+ge-gf-gg+gh) +
         du * torch.stack([
             (vb-va) + u[...,1]*(va-vb-vc+vd) + u[...,2]*(va-vb-ve+vf) + u[...,1]*u[...,2]*(-va+vb+vc-vd+ve-vf-vg+vh),
             (vc-va) + u[...,2]*(va-vc-ve+vg) + u[...,0]*(va-vb-vc+vd) + u[...,2]*u[...,0]*(-va+vb+vc-vd+ve-vf-vg+vh),
             (ve-va) + u[...,0]*(va-vb-ve+vf) + u[...,1]*(va-vc-ve+vg) + u[...,0]*u[...,1]*(-va+vb+vc-vd+ve-vf-vg+vh)
         ], dim=-1))
    
    return torch.cat([v.unsqueeze(-1), d], dim=-1)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Using device:", device)
sh = 5
shape = (sh, sh, sh)
grid = generate_perlin_grid(shape, device=device)
'''
res = 200
# generate 3D meshgrid
eval_points = torch.meshgrid(
    torch.linspace(0, shape[0], steps=res, device=device)[:-1],
    torch.linspace(0, shape[1], steps=res, device=device)[:-1],
    torch.linspace(0, shape[2], steps=res, device=device)[:-1]
)
eval_points = torch.stack(eval_points, dim=-1) # shape (res, res, res, 3)
perlin_values = perlin_noised(grid, eval_points)
perlin_slices = perlin_values[:, :, ::200//5, :].cpu().numpy() # take slices along z-axis
for i, x in enumerate(np.linspace(0, shape[0], num=5, endpoint=False)):
    image = perlin_slices[:, :, i, 0]  # noise values
    plt.title(f"Perlin noise slice at x={x:.2f}")
    plt.xlabel("y")
    plt.ylabel("z")
    plt.imshow(image, extent=(0, shape[0], 0, shape[1]), origin='lower', cmap='gray')
    plt.quiver(
        np.linspace(0, shape[0], num=res)[::5],
        np.linspace(0, shape[1], num=res)[::5],
        -perlin_slices[::5, ::5, i, 1],  # -dx
        perlin_slices[::5, ::5, i, 2],  # dy
        color='red', alpha=0.5
    )
    plt.grid(False)
    plt.savefig(f"perlin_slice_x_{x:.2f}.png", dpi=400)
    plt.clf()
'''
# remaps the scale of the noise function in each dimension
remap_scale = torch.tensor([1.0, 1.0, 1.0], device=device)
n_fibres = 500
x0 = torch.rand(n_fibres, 1, device=device) * shape[0]
y0 = torch.rand(n_fibres, 1, device=device) * shape[1]
z0 = torch.zeros(n_fibres, 1, device=device)
coords0 = torch.cat([x0, y0, z0], dim=1)  # (n_fibres, 3)
n_steps = 200
s = 0.01 # scaling factor for gradient
step_size = shape[2] / n_steps
fibre_coords = torch.zeros(n_fibres, n_steps, 3, device=device)
fibre_coords[:, 0, :] = coords0
for step in range(1, n_steps):
    prev_coords = fibre_coords[:, step-1, :]
    # zero out z coordinate for 2D perlin noise in xy plane
    prev_coords_flat = torch.cat([prev_coords[:, :2], torch.zeros(n_fibres, 1, device=device)], dim=1)
    perlin_out = perlin_noised(grid, prev_coords, remap_scale)
    gradients = perlin_out[:, 1:]  # (n_fibres, 3)
    gradx = gradients[:, 0] * s
    grady = gradients[:, 1] * s
    steps = torch.stack([grady, -gradx, torch.full_like(gradx, step_size)], dim=1)  # (n_fibres, 3)
    new_coords = prev_coords + steps
    fibre_coords[:, step, :] = new_coords

rve = RVE.external(fibre_coords, radius=sh*0.005, downsample=False)
rve.save("curl_init_2", 0, 0)