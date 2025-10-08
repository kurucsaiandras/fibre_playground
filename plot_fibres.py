import torch
import pyvista as pv
import utils

plot_pbo = True

plotter = pv.Plotter()
model = torch.load("results/pbc/models/pbc_28761.pt")
fibres = model["fibres"]  # (n_fibres, n_points, 3)
fibre_diameter = model["diameter"]
domain_size = model["domain_size"].cpu().numpy()

offsets = utils.get_offsets(domain_size, plot_pbo, device="cpu")
colors = ["lightsteelblue", "blue", "green", "purple"]

n_fibres, n_points, _ = fibres.shape

for offset, color in zip(offsets, colors):
        for i in range(n_fibres):
            arr = (fibres[i] + offset).cpu().numpy()
            line = pv.Spline(arr, n_points=n_points)
            tube = line.tube(radius=fibre_diameter / 2.0)
            plotter.add_mesh(tube, color=color, smooth_shading=True)
# plot domain box
box = pv.Box(bounds=(0, domain_size[0], 0, domain_size[1], 0, domain_size[2]))
plotter.add_mesh(box, style="wireframe", color="black")

plotter.show(auto_close=False)