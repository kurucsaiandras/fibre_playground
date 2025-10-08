import torch
import math
import threading
import time
import argparse
import utils
import losses
import logger
import config_parser
import os
import shutil
import sys
#sys.path.append("E:/DTU/thesis/Pytorch-PCGrad")
#from pcgrad import PCGrad

# parse arg that specifies jobname
parser = argparse.ArgumentParser(description="Fibre simulation")
parser.add_argument("--jobname", type=str, nargs='?', default="fibre_sim", help="Job name for output files")
parser.add_argument("--config_name", type=str, nargs='?', default="default_small", help="Config file name used for all parameters")
args = parser.parse_args()
jobname = args.jobname
config_name = args.config_name
if not os.path.exists("results"):
    os.makedirs("results")
if not os.path.exists(f"results/{jobname}"):
    os.makedirs(f"results/{jobname}")

# Use GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#device = torch.device("cpu")
print(f"Using device: {device}")

# ------------------------
# Setup
# ------------------------
shutil.copy(f"config/{config_name}.yaml", f"results/{jobname}/config.yaml")
config = config_parser.load_config(f"config/{config_name}.yaml")

if config.initialization.method == 'generate':
    if config.initialization.generate.method == 'poisson':
        fibres, l0_length = utils.generate_fibres_poisson(
            torch.tensor(config.initialization.generate.domain_size_initial, device=device),
            config.initialization.generate.resolution,
            config.initialization.generate.poisson_radius,
            config.initialization.generate.angle_std_dev,
            device)
    elif config.initialization.generate.method == 'random':
        fibres, l0_length = utils.generate_fibres_random(
            torch.tensor(config.initialization.generate.domain_size_initial, device=device),
            config.initialization.generate.resolution,
            config.initialization.generate.num_of_fibres,
            config.initialization.generate.angle_std_dev,
            device)
    # save initial fibres and Ls
    if not os.path.exists("init"):
        os.makedirs("init")
    if not os.path.exists(f"init/{jobname}"):
        os.makedirs(f"init/{jobname}")
    torch.save(fibres.cpu(), f"init/{jobname}/fibre_coords_initial.pt")
    torch.save(l0_length.cpu(), f"init/{jobname}/fibre_l0_initial.pt")
    shutil.copy(f"config/{config_name}.yaml", f"init/{jobname}/config.yaml")
    # set current domain size
    domain_size_current = torch.tensor(config.initialization.generate.domain_size_initial, device=device)
elif config.initialization.method == 'load':
    fibres = torch.load( f"init/{config.initialization.load.name}/fibre_coords_initial.pt").to(device)
    l0_length = torch.load(f"init/{config.initialization.load.name}/fibre_l0_initial.pt").to(device)
    config_init = config_parser.load_config(f"init/{config.initialization.load.name}/config.yaml").initialization
    # set domain size from saved config
    domain_size_current = torch.tensor(config_init.generate.domain_size_initial, device=device)
phi0_curvature = math.pi
n_fibres = fibres.shape[0]
fibre_diameter_current = config.initialization.generate.fibre_diameter_initial
print(f"Fibre to volume ratio: {n_fibres * math.pi * (fibre_diameter_current/2)**2 /((domain_size_current[0]+fibre_diameter_current)*(domain_size_current[1]+fibre_diameter_current)):.3f}")
print(f"Number of fibres: {n_fibres}")

if config.stats.to_plot:
    import pyvista as pv
    plotter = pv.Plotter()
    offsets = utils.get_offsets(domain_size_current, config.stats.plot_pbc, device)
    colors = ["lightsteelblue", "blue", "green", "purple"]

    # Build initial tube meshes and add to plotter, storing the PolyData objects
    meshes = []
    for offset, color in zip(offsets, colors):
        for i in range(n_fibres):
            arr = (fibres[i] + offset).cpu().numpy()
            line = pv.Spline(arr, n_points=config.initialization.generate.resolution)
            tube = line.tube(radius=fibre_diameter_current / 2.0)
            actor = plotter.add_mesh(tube, color=color, smooth_shading=True)
            meshes.append((actor, tube)) # keep reference to update points
    
    # plot domain box
    box = pv.Box(bounds=(0, domain_size_current[0].cpu().numpy(), 0, domain_size_current[1].cpu().numpy(), 0, domain_size_current[2].cpu().numpy()))
    plotter.add_mesh(box, style="wireframe", color="black")

    plotter.show(auto_close=False, interactive_update=True)

# Optimize for all fibres
fibres_params = torch.nn.Parameter(fibres.clone())
if config.optimization.optimizer == 'adam':
    optimizer = torch.optim.Adam([fibres_params], lr=config.optimization.learning_rate)
elif config.optimization.optimizer == 'lbfgs':
    optimizer = torch.optim.LBFGS([fibres_params], lr=config.optimization.learning_rate, max_iter=10, history_size=10)

# globals for logging
last_losses = {}
global_start_time = time.time()
current_step = 0
def closure():
    optimizer.zero_grad()

    loss_length = losses.length_loss(fibres_params, config.spring_system.k_length, l0_length)
    loss_curvature = losses.curvature_loss(fibres_params, config.spring_system.k_curvature, phi0_curvature)
    loss_boundary = losses.boundary_loss(fibres_params, domain_size_current, config.spring_system.k_boundary)
    loss_collision = losses.collision_loss(fibres_params, config.spring_system.k_collision, fibre_diameter_current, domain_size_current)

    loss_sum = loss_length + loss_curvature + loss_boundary + loss_collision

    if not torch.isfinite(loss_sum):
        print("Non-finite loss in closure -> aborting")
        return loss_sum

    loss_sum.backward()
    #optimizer.pc_backward([loss_length, loss_curvature, loss_boundary, loss_collision])

    # store individual losses for later
    last_losses["length"] = loss_length.detach()
    last_losses["curvature"] = loss_curvature.detach()
    last_losses["boundary"] = loss_boundary.detach()
    last_losses["collision"] = loss_collision.detach()

    # save params if no collisions before updating the points
    if last_losses["collision"] <= config.evolution.collision_threshold:
        utils.save_model(jobname, fibres_params.data.cpu(),
                         current_step, time.time()-global_start_time,
                         domain_size_current, fibre_diameter_current)

    return loss_sum

# ------------------------
# Optimization loop
# ------------------------

def optimize():
    global fibre_diameter_current, domain_size_current, current_step, global_start_time
    # start timer
    start_time = time.time()
    global_start_time = start_time
    for step in range(config.evolution.max_iterations+1):
        current_step = step
        loss_sum = optimizer.step(closure)
        loss_length = last_losses["length"]
        loss_curvature = last_losses["curvature"]
        loss_boundary = last_losses["boundary"]
        loss_collision = last_losses["collision"]
        # gradient clipping to avoid explosion
        if config.optimization.grad_clipping:
            torch.nn.utils.clip_grad_norm_([fibres_params], config.optimization.max_grad_norm)

        # save params and log if no collisions or every specified steps
        if loss_collision <= config.evolution.collision_threshold or step % config.stats.logging_freq == 0:
            if loss_collision <= config.evolution.collision_threshold:
                log_file_name = "model_saves"
                elapsed = (time.time() - global_start_time)
            elif step % config.stats.logging_freq == 0:
                log_file_name = "progress"
                elapsed = (time.time() - start_time) / config.stats.logging_freq
                if config.stats.to_plot:
                    offsets = utils.get_offsets(domain_size_current, config.stats.plot_pbc, device)
                    for j, offset in enumerate(offsets):
                        # Update PyVista meshes
                        for i in range(n_fibres):
                            arr = (fibres_params[i] + offset).detach().cpu().numpy()
                            new_line = pv.Spline(arr, n_points=config.initialization.generate.resolution)
                            new_tube = new_line.tube(radius=fibre_diameter_current * 0.5)
                            actor, tube = meshes[i + j*n_fibres]
                            tube.points[:] = new_tube.points
                            #if (i in i_idx_x) or (i in j_idx_x):
                            #    actor.prop.color = "red"
                            #else:
                            #    actor.prop.color = colors[j]
                    # update box
                    new_box = pv.Box(bounds=(0, domain_size_current[0].cpu().numpy(), 0, domain_size_current[1].cpu().numpy(), 0, domain_size_current[2].cpu().numpy()))
                    box.points[:] = new_box.points
                start_time = time.time()
            logger.log(jobname, log_file_name, step, elapsed,
                loss_length.item(), loss_curvature.item(),
                loss_boundary.item(), loss_collision.item(),
                loss_sum.item(), n_fibres, fibre_diameter_current, domain_size_current.cpu().numpy())

        # sanity check after step
        if not torch.all(torch.isfinite(fibres_params)):
            print("Parameters became non-finite at step", step, "-> aborting")
            break
            
        # adjust configuration if no collisions
        if loss_collision <= config.evolution.collision_threshold:
            # first, increase fibre diameter until target
            if fibre_diameter_current < config.evolution.fibre_diameter_final:
                fibre_diameter_current += 0.01
            # then, decrease domain size until target
            elif domain_size_current[0] > config.evolution.domain_size_final[0]:
                domain_size_current[0] -= 0.1
                domain_size_current[1] -= 0.1
                # subtract 0.05 from x and y of points
                fibres_params.data[:, :, 0] -= 0.05
                fibres_params.data[:, :, 1] -= 0.05
            else:
                if config.evolution.collision_threshold > 0:
                    config.evolution.collision_threshold = 0
                else:
                    print("Reached target configuration -> stopping")
                    break

def main():
    if config.stats.to_plot:
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