import torch
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
        self.milestone = False
        if self.config.optimization.alternate_phases:
            self.optim_phase = 'collision'  # or 'all'
            self.current_iter_per_phase = 0
            self.avg_window = self.config.optimization.moving_average_window
            self.moving_avg = torch.zeros(self.avg_window*2, device=device)
            self.min_loss = torch.inf
            self.fibre_coords_checkpoint = self.rve.fibre_coords.clone()

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

        if self.config.optimization.line_loss:
            self.losses["collision"] = self.rve.collision_line_loss()
        else:
            self.losses["collision"] = self.rve.collision_loss()
        self.losses["boundary"] = self.rve.boundary_loss()
        self.losses["length"] = self.rve.length_loss()
        self.losses["curvature"] = self.rve.curvature_loss()

        if self.config.optimization.alternate_phases:

            if self.optim_phase == 'collision':

                self.current_iter_per_phase += 1
                loss_sum = self.losses["collision"] #+ self.losses["boundary"]

                if self.losses["collision"] <= self.config.evolution.collision_threshold:
                    self.optim_phase = 'all'
                    print(f"Switching to 'all' optimization phase at step {self.current_step}")
                    print(f"Took {self.current_iter_per_phase} iterations in 'collision' phase")
                    self.current_iter_per_phase = 0
                    self.milestone = True
                    self.moving_avg = torch.zeros(self.avg_window*2, device=self.device)
                    self.min_loss = torch.inf
                    #for param_group in self.optimizer.param_groups:
                    #    param_group['lr'] = 0.01

            elif self.optim_phase == 'all':

                loss_sum = (sum(self.losses.values()))

                if loss_sum < self.min_loss:
                    self.min_loss = loss_sum
                    self.fibre_coords_checkpoint = self.rve.fibre_coords.clone()

                if self.current_iter_per_phase < self.avg_window*2:
                    self.moving_avg[self.current_iter_per_phase] = loss_sum
                else:
                    self.moving_avg = torch.roll(self.moving_avg, -1)
                    self.moving_avg[-1] = loss_sum

                self.current_iter_per_phase += 1

                if (loss_sum < self.config.optimization.cumulative_loss_threshold
                    or self.current_iter_per_phase == self.config.optimization.max_iter_per_phase
                    or self.moving_avg[:self.avg_window].sum() < self.moving_avg[self.avg_window:].sum()):
                    with torch.no_grad():
                        self.rve.fibre_coords.data.copy_(self.fibre_coords_checkpoint)
                        self.rve.fibre_coords.grad = None
                    self.optim_phase = 'collision'
                    print(f"Switching to 'collision' optimization phase at step {self.current_step}")
                    print(f"Took {self.current_iter_per_phase} iterations in 'all' phase")
                    self.current_iter_per_phase = 0
                    #for param_group in self.optimizer.param_groups:
                    #    param_group['lr'] = 0.001
        else:
            loss_sum = (sum(self.losses.values()))
            if self.losses["collision"] <= self.config.evolution.collision_threshold:
                self.milestone = True

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

        # save params if milestone before updating the points
        if self.milestone:
            self.rve.save(self.job_name, self.current_step, time.time()-self.global_start_time)
            self.rve_saves_logger.log(self, time.time() - self.global_start_time)
        return loss_sum

    def step(self):
        self.current_step += 1
        self.milestone = False
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
            
        # adjust configuration if milestone
        if self.milestone:
            at_target = self.rve.evolve()
            if at_target:
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