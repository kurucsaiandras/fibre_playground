import torch
import pyvista as pv

plotter = pv.Plotter()
fibres = torch.load("fibre_coords.pt").to("cpu")  # (n_fibres, n_points, 3)
n_fibres, n_points, _ = fibres.shape
fibre_diameter = 0.5

for i in range(n_fibres):
    arr = fibres[i].cpu().numpy()
    line = pv.Spline(arr, n_points=n_points)
    tube = line.tube(radius=fibre_diameter / 2.0)
    plotter.add_mesh(tube, color="lightsteelblue", smooth_shading=True)

plotter.show(auto_close=False)