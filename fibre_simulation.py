import torch
import torch.nn.functional as F
import logger
import os
import time
import shutil
import config_parser
from rve import RVE
import threading

class FibreSimulation:
    def __init__(self, config_name, job_name, device):
        self.config = config_parser.load_config(f"config/{config_name}.yaml")
        self.job_name = job_name
        self.device = device
        self.global_start_time = time.time()
        self.step_start_time = time.time()
        self.current_step = 0
        self.losses = {}
        self.rve = RVE(self.config, self.device)
        self._setup_optimizer()
        self.progress_logger = logger.Logger(self.job_name, "progress")
        self.rve_saves_logger = logger.Logger(self.job_name, "rve_saves")

        os.makedirs(f"results/{self.job_name}/rve", exist_ok=True)
        shutil.copy(f"config/{config_name}.yaml", f"results/{self.job_name}/used_config.yaml")
        
        self.rve.save(self.job_name, self.current_step, time=0.0)

        if self.config.stats.to_plot:
            from fibre_plotter import FibrePlotter
            self.plotter = FibrePlotter(self.rve)

    def _setup_optimizer(self):
        if self.config.optimization.optimizer == 'adam':
            self.optimizer = torch.optim.Adam([self.rve.fibre_coords], lr=self.config.optimization.learning_rate)
        elif self.config.optimization.optimizer == 'lbfgs':
            self.optimizer = torch.optim.LBFGS([self.rve.fibre_coords], lr=self.config.optimization.learning_rate,
                                               max_iter=10, history_size=10)

    def closure(self):
        self.optimizer.zero_grad()

        self.losses["length"] = self.rve.length_loss()
        self.losses["curvature"] = self.rve.curvature_loss()
        self.losses["boundary"] = self.rve.boundary_loss()
        self.losses["collision"] = self.rve.collision_loss()

        loss_sum = (sum(self.losses.values()))

        if not torch.isfinite(loss_sum):
            print("Non-finite loss in closure -> aborting")
            return loss_sum

        loss_sum.backward()
        #optimizer.pc_backward([loss_length, loss_curvature, loss_boundary, loss_collision])

        # Compute individual gradients
        '''grads = []
        for loss in [loss_length, loss_curvature, loss_boundary, loss_collision]:
            optimizer.zero_grad()
            loss.backward(retain_graph=True)
            grads.append(fibres_params.grad.clone())

        # Apply projection: ensure collision gradient is not increased
        g_collision = grads[3]
        g_total = sum(grads)
        # element-wise dot product and norm
        dot = (g_total * g_collision).sum(dim=-1, keepdim=True)
        g_proj = dot / (g_collision * g_collision).sum(dim=-1, keepdim=True) * g_collision
        g_total = torch.where(dot < 0, g_total - g_proj, g_total)

        # Set projected gradients and step
        with torch.no_grad():
            fibres_params.grad = g_total'''

        # save params if no collisions before updating the points
        if self.losses["collision"] <= self.config.evolution.collision_threshold:
            self.rve.save(self.job_name, self.current_step, time.time()-self.global_start_time)
            self.rve_saves_logger.log(self, time.time() - self.global_start_time)
        return loss_sum

    def step(self):
        self.current_step += 1
        self.optimizer.step(self.closure)
        
        # gradient clipping to avoid explosion
        if self.config.optimization.grad_clipping:
            torch.nn.utils.clip_grad_norm_([self.rve.fibre_coords],
                                           self.config.optimization.max_grad_norm)
        
        # logging and plotting
        if self.current_step % self.config.stats.logging_freq == 0:
            elapsed = (time.time() - self.step_start_time) / self.config.stats.logging_freq
            self.progress_logger.log(self, elapsed)
            if self.config.stats.to_plot:
                self.plotter.update(self.rve)
            self.step_start_time = time.time()

        # sanity check after step
        if not torch.all(torch.isfinite(self.rve.fibre_coords)):
            print("Parameters became non-finite at step", self.current_step, "-> aborting")
            return False
            
        # adjust configuration if no collisions
        if self.losses["collision"] <= self.config.evolution.collision_threshold:
            # first, increase fibre diameter until target
            if self.rve.fibre_diameter < self.config.evolution.fibre_diameter_final:
                self.rve.fibre_diameter += self.config.evolution.fibre_diameter_step
            # then, decrease domain size until target
            elif self.rve.domain_size[0] > self.config.evolution.domain_size_final[0]:
                self.rve.domain_size[0] -= 0.1
                self.rve.domain_size[1] -= 0.1
                # subtract 0.05 from x and y of points
                self.rve.fibre_coords.data[:, :, 0] -= 0.05
                self.rve.fibre_coords.data[:, :, 1] -= 0.05
            # finally, decrease collision threshold until zero for final polishing
            else:
                if self.config.evolution.collision_threshold > 0:
                    self.config.evolution.collision_threshold = 0
                else:
                    print("Reached target configuration -> stopping")
                    return False
                
        return True

    def run(self):
        for _ in range(self.config.evolution.max_iterations + 1):
            continue_sim = self.step()
            if not continue_sim:
                break

    def launch(self):
        if self.config.stats.to_plot:
            threading.Thread(target=self.run, daemon=True).start()
            # call update in main thread until optimization is done
            while True:
                self.plotter.plotter.update()
                time.sleep(0.05)
                if not threading.main_thread().is_alive():
                    break
            self.plotter.plotter.show(interactive_update=False)  
        else:
            self.run()

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="default_small", help="Configuration file name (without .yaml)")
    parser.add_argument("--job_name", type=str, default="test_run", help="Job name for saving results")
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    sim = FibreSimulation(args.config, args.job_name, device)
    sim.launch()

if __name__ == "__main__":
    main()