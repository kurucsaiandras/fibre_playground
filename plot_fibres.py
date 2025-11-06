from rve import RVE
import pyvista as pv
import utils

plotter = pv.Plotter()
name = 'big_angles_small_curv_k'
step = 4501
if step is None: step = utils.latest_rve(f"results/{name}/rve")
rve = RVE.eval(name, step, "cpu")

offsets = utils.get_offsets(rve.domain_size, rve.apply_pbc, device="cpu")
colors = ["lightsteelblue", "blue", "green", "purple"]

n_fibres, n_points, _ = rve.fibre_coords.shape

for offset, color in zip(offsets, colors):
        for i in range(n_fibres):
            arr = (rve.fibre_coords[i] + offset).cpu().numpy()
            line = pv.Spline(arr, n_points=n_points)
            tube = line.tube(radius=rve.fibre_r[i])
            plotter.add_mesh(tube, color=color, smooth_shading=True)
# plot domain box
box = pv.Box(bounds=(0, rve.domain_size[0], 0, rve.domain_size[1], 0, rve.domain_size[2]))
plotter.add_mesh(box, style="wireframe", color="black")

plotter.show(auto_close=False)