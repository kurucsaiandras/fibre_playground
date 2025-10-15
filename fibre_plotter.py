import pyvista as pv
import utils

class FibrePlotter:
    def __init__(self, rve):
        self.plotter = pv.Plotter()
        offsets = utils.get_offsets(rve.domain_size, rve.apply_pbc, rve.device)
        colors = ["lightsteelblue", "blue", "green", "purple"]

        # Build initial tube meshes and add to plotter, storing the PolyData objects
        self.meshes = []
        for offset, color in zip(offsets, colors):
            for i in range(rve.fibre_coords.shape[0]):
                arr = (rve.fibre_coords[i] + offset).detach().cpu().numpy()
                line = pv.Spline(arr, n_points=rve.fibre_coords.shape[1])
                tube = line.tube(radius=rve.fibre_r[i])
                actor = self.plotter.add_mesh(tube, color=color, smooth_shading=True)
                self.meshes.append((actor, tube)) # keep reference to update points
        
        # plot domain box
        domain_size = rve.domain_size.cpu().numpy()
        self.box = pv.Box(bounds=(0, domain_size[0], 0, domain_size[1], 0, domain_size[2]))
        self.plotter.add_mesh(self.box, style="wireframe", color="black")

        self.plotter.show(auto_close=False, interactive_update=True)

    def update(self, rve):
        offsets = utils.get_offsets(rve.domain_size, rve.apply_pbc, rve.device)
        n_fibres = rve.fibre_coords.shape[0]
        for i, offset in enumerate(offsets):
            # Update PyVista meshes
            for fibre_idx in range(n_fibres):
                arr = (rve.fibre_coords[fibre_idx] + offset).detach().cpu().numpy()
                new_line = pv.Spline(arr, n_points=rve.fibre_coords.shape[1])
                new_tube = new_line.tube(radius=rve.fibre_r[fibre_idx])
                actor, tube = self.meshes[fibre_idx + i*n_fibres]
                tube.points[:] = new_tube.points
                #if (i in i_idx_x) or (i in j_idx_x):
                #    actor.prop.color = "red"
                #else:
                #    actor.prop.color = colors[j]
        # update box
        domain_size = rve.domain_size.cpu().numpy()
        new_box = pv.Box(bounds=(0, domain_size[0], 0, domain_size[1], 0, domain_size[2]))
        self.box.points[:] = new_box.points
        #self.plotter.update()