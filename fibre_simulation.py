import torch
import logger
import os
import time
import shutil
import config_parser
from rve import RVE
import threading
from optimizer import Optimizer
import descriptors_2d
import descriptors_3d

class FibreSimulation:
    def __init__(self, config_name, job_name, device):
        self.config = config_parser.load_config(f"config/{config_name}.yaml")
        self.job_name = job_name
        self.device = device
        self.global_start_time = time.time()
        self.step_start_time = time.time()
        self.current_step = 0
        self.rve = RVE(self.config, self.device)
        self.optimizer = Optimizer(self.config.optimization, self.rve)
        self.progress_logger = logger.Logger(self.job_name, "progress")
        self.rve_saves_logger = logger.Logger(self.job_name, "rve_saves")
        self.refine_phase = False

        os.makedirs(f"results/{self.job_name}/rve", exist_ok=True)
        shutil.copy(f"config/{config_name}.yaml", f"results/{self.job_name}/used_config.yaml")
        
        self.rve.save(self.job_name, self.current_step, time=0.0)

        if self.config.stats.to_plot:
            from fibre_plotter import FibrePlotter
            self.plotter = FibrePlotter(self.rve)

    def save_and_evolve(self):
        self.rve.save(self.job_name, self.current_step, time.time()-self.global_start_time)
        self.rve_saves_logger.log(self, time.time() - self.global_start_time)
        at_target = self.rve.evolve()
        if at_target:
            do_continue = False
            if self.config.evolution.overlap_threshold > 0:
                self.config.evolution.overlap_threshold = 0
                do_continue = True
            if self.config.optimization.refine_phase == True and self.refine_phase == False:
                self.refine_phase = True
                do_continue = True
            if not do_continue:
                print("Reached target configuration -> stopping")
                return False
        return True

    def step(self):
        self.optimizer.loss()

        self.current_step += 1
        if self.current_step % self.config.stats.logging_freq == 0:
            elapsed = (time.time() - self.step_start_time) / self.config.stats.logging_freq
            self.progress_logger.log(self, elapsed)
            if self.config.stats.to_plot:
                self.plotter.update(self.rve)
            self.step_start_time = time.time()

        if self.config.optimization.alternate_phases:
            self.optimizer.phase_iter += 1

            if (self.optimizer.phase == 'overlap' and
                self.optimizer.losses["overlap"] <= self.config.evolution.overlap_threshold):
                    self.optimizer.switch_phase()
                    return self.save_and_evolve()

            elif self.optimizer.phase == 'joint':
                self.optimizer.save_checkpoint()
                self.optimizer.update_moving_avg()
                if self.optimizer.eval_joint_criteria():
                    with torch.no_grad():
                        self.rve.fibre_coords.data.copy_(self.optimizer.params_checkpoint)
                        self.rve.fibre_coords.grad = None
                    self.optimizer.switch_phase()
                    return True
        else:
            if self.optimizer.losses["overlap"] <= self.config.evolution.overlap_threshold:
                return self.save_and_evolve() # NOTE: We are not doing step here

        self.optimizer.step()

        # sanity check after step
        if not torch.all(torch.isfinite(self.rve.fibre_coords)):
            print("Parameters became non-finite at step", self.current_step, "-> aborting")
            return False
        
        return True

    def run(self):
        for _ in range(self.config.evolution.max_iterations + 1):
            continue_sim = self.step()
            if not continue_sim:
                if self.config.stats.eval_statistics:
                    descriptors_2d.eval(self.job_name, None, self.device, save_figs=True)
                    descriptors_3d.eval(self.job_name, None, self.device, save_figs=True)
                    self.progress_logger.plot(save_figs=True)
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
    parser.add_argument("--config", type=str, default="default", help="Configuration file name (without .yaml)")
    parser.add_argument("--job_name", type=str, default="test_run", help="Job name for saving results")
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    sim = FibreSimulation(args.config, args.job_name, device)
    sim.launch()

if __name__ == "__main__":
    main()