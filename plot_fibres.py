import torch
import pyvista as pv
import utils

plotter = pv.Plotter()
model = torch.load("results/test_run_radii/rve/20494.pt")
fibre_coords = model["fibre_coords"].cpu().detach()  # (n_fibres, n_points, 3)
fibre_r = model["fibre_r"].cpu()  # (n_fibres,)
domain_size = model["domain_size"].cpu().numpy()

offsets = utils.get_offsets(domain_size, model["apply_pbc"], device="cpu")
colors = ["lightsteelblue", "blue", "green", "purple"]

n_fibres, n_points, _ = fibre_coords.shape

for offset, color in zip(offsets, colors):
        for i in range(n_fibres):
            arr = (fibre_coords[i] + offset).cpu().numpy()
            line = pv.Spline(arr, n_points=n_points)
            tube = line.tube(radius=fibre_r[i])
            plotter.add_mesh(tube, color=color, smooth_shading=True)
# plot domain box
box = pv.Box(bounds=(0, domain_size[0], 0, domain_size[1], 0, domain_size[2]))
plotter.add_mesh(box, style="wireframe", color="black")

plotter.show(auto_close=False)