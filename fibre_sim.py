import torch
import math
import threading
import time
import argparse
import utils
import losses
import logger

# Use GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#device = torch.device("cpu")
print(f"Using device: {device}")

# ------------------------
# Setup
# ------------------------
resolution = 50 # TODO we need bigger number here to actually avoid collisions
#n_fibres = 200
domain_size = torch.tensor([10.0, 10.0, 10.0], device=device)
domain_size_final = torch.tensor([6.5, 6.5, 10.0], device=device)
angle_std_dev = 0.05
fibre_diameter = 0.05
fibre_diameter_final = 0.3
n_iter = 500000

fibres, spring_L_linear = utils.generate_fibres(domain_size, resolution, angle_std_dev, device)
n_fibres = fibres.shape[0]
print(f"Fibre to volume ratio: {n_fibres * math.pi * (fibre_diameter/2)**2 / ((domain_size[0]+fibre_diameter)*(domain_size[1]+fibre_diameter)):.3f}")
print(f"Number of fibres: {n_fibres}")
spring_k_linear = 0.1 # TODO tune based on the step length
spring_k_torsional = 0.1 # TODO tune based on the step length
spring_L_torsional = math.pi
spring_k_boundary = 1.0
spring_k_collision = 100.0

to_plot = False

# parse arg that specifies jobname
jobname = "fibre_sim"
parser = argparse.ArgumentParser(description="Fibre simulation")
parser.add_argument("--jobname", type=str, nargs='?', default="fibre_sim", help="Job name for output files")
args = parser.parse_args()
jobname = args.jobname

if to_plot:
    import pyvista as pv
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

# ------------------------
# Optimization loop
# ------------------------

max_grad_norm = 1.0

def optimize():
    global fibre_diameter, domain_size
    # start timer
    start_time = time.time()
    global_start_time = start_time
    for step in range(n_iter+1):
        optimizer.zero_grad()

        # Length loss (distances)
        loss_length = losses.length_loss(fibres_params, spring_k_linear, spring_L_linear)
        # Curvature loss (angles)
        loss_curvature = losses.curvature_loss(fibres_params, spring_k_torsional, spring_L_torsional)
        # Boundary loss
        loss_boundary = losses.boundary_loss(fibres_params, domain_size, spring_k_boundary)
        # Collision loss
        loss_collision = losses.collision_loss(fibres_params, spring_k_collision, fibre_diameter)

        loss_sum = loss_length + loss_curvature + loss_boundary + loss_collision

        # detect bad loss before backward
        if not torch.isfinite(loss_sum):
            print("Non-finite loss at step", step, "-> aborting")
            break

        loss_sum.backward()

        # gradient clipping to avoid explosion
        torch.nn.utils.clip_grad_norm_([fibres_params], max_grad_norm)

        # save params and log if no collisions or every 100 steps
        if loss_collision == 0 or step % 100 == 0:
            if loss_collision == 0:
                torch.save(fibres_params.data.cpu(), f"models/{jobname}.pt")
                log_file_name = "model_saves"
                elapsed = (time.time() - global_start_time)
            elif step % 100 == 0:
                log_file_name = "progress"
                elapsed = (time.time() - start_time) / 100
                if to_plot:
                    # Update PyVista meshes
                    for i in range(n_fibres):
                        arr = fibres_params[i].detach().cpu().numpy()
                        new_line = pv.Spline(arr, n_points=resolution)
                        new_tube = new_line.tube(radius=fibre_diameter * 0.5)
                        meshes[i].points[:] = new_tube.points
                start_time = time.time()
            logger.log(jobname, log_file_name, step, elapsed,
                loss_length.item(), loss_curvature.item(),
                loss_boundary.item(), loss_collision.item(),
                loss_sum.item(), n_fibres, fibre_diameter, domain_size.cpu().numpy())

        # fibres_params gets updated here!
        optimizer.step()

        # sanity check after step
        if not torch.all(torch.isfinite(fibres_params)):
            print("Parameters became non-finite at step", step, "-> aborting")
            break
            
        # adjust configuration if no collisions
        if loss_collision == 0:
            # first, increase fibre diameter until target
            if fibre_diameter < fibre_diameter_final:
                fibre_diameter += 0.01
                print(f"Increase fibre diameter to {fibre_diameter:.3f}")
                with open(f"volume_ratio_{jobname}.txt", "a") as f:
                    f.write(f"Increase fibre diameter to {fibre_diameter:.3f}\n")
            # then, decrease domain size until target
            elif domain_size[0] > domain_size_final[0]:
                domain_size[0] -= 0.1
                domain_size[1] -= 0.1
                # subtract 0.05 from x and y of points
                fibres_params.data[:, :, 0] -= 0.05
                fibres_params.data[:, :, 1] -= 0.05
                print(f"Decrease domain size to {domain_size[0]:.3f} x {domain_size[1]:.3f}")
                with open(f"volume_ratio_{jobname}.txt", "a") as f:
                    f.write(f"Decrease domain size to {domain_size[0]:.3f} x {domain_size[1]:.3f}\n")
            else:
                print("Reached target configuration -> stopping")
                break

def main():
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

if __name__ == "__main__":
    main()