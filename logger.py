import os

class Logger:
    def __init__(self, job_name, file_name):
        self.job_name = job_name
        self.file_name = file_name
        os.makedirs(f"results/{self.job_name}/logs", exist_ok=True)
        with open(f"results/{self.job_name}/logs/{self.file_name}.csv", "a") as f:
            f.write(
                "step,time,length,curvature,boundary,overlap,total,"
                "n_fibres,fibre_r_mean,domain_size_x,domain_size_y,"
                "domain_size_z,volume_ratio,phase\n"
            )

    def log(self, simulation, time):
        loss = simulation.optimizer.losses
        n_fibres = simulation.rve.fibre_coords.shape[0]
        fibre_r_mean = simulation.rve.fibre_r.mean().item()
        domain_size = simulation.rve.domain_size.cpu().numpy()
        if simulation.optimizer.phase == 'collision': phase = 0
        elif simulation.optimizer.phase == 'joint': phase = 1
        with open(f"results/{self.job_name}/logs/{self.file_name}.csv", "a") as f:
            fibre_to_volume_ratio = simulation.rve.get_fibre_to_volume_ratio()
            f.write(
                f"{simulation.current_step},{time:.6f},{loss['length']:.6f},{loss['curvature']:.6f},"
                f"{loss['boundary']:.6f},{loss['overlap']:.12f},{sum(loss.values()):.6f},{n_fibres},"
                f"{fibre_r_mean:.3f},{domain_size[0]:.3f},{domain_size[1]:.3f},"
                f"{domain_size[2]:.3f},{fibre_to_volume_ratio:.6f},{phase}\n"
            )