import numpy as np
import os

def log(jobname, filename, step, time, length, curvature, boundary, collision,
        total, n_fibres, fibre_diameter, domain_size):
    # check if folder jobname/logs exists, if not create it
    if not os.path.exists(f"results/{jobname}/logs"):
        os.makedirs(f"results/{jobname}/logs")
    with open(f"results/{jobname}/logs/{filename}.csv", "a") as f:
        if step == 0:
            f.write(
                "step,time,length,curvature,boundary,collision,total,"
                "n_fibres,fibre_diameter,domain_size_x,domain_size_y,"
                "domain_size_z,volume_ratio\n"
            )
        fibre_to_volume_ratio = (n_fibres * np.pi * (fibre_diameter*0.5)**2
            / ((domain_size[0] + fibre_diameter) * (domain_size[1] + fibre_diameter)))
        f.write(
            f"{step},{time:.6f},{length:.6f},{curvature:.6f},"
            f"{boundary:.6f},{collision:.6f},{total:.6f},{n_fibres},"
            f"{fibre_diameter:.3f},{domain_size[0]:.3f},"
            f"{domain_size[1]:.3f},{domain_size[2]:.3f},{fibre_to_volume_ratio:.6f}\n"
        )