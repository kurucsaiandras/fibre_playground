from line_profiler import LineProfiler
import spring_system_3d as fibre_sim

lp = LineProfiler()
# add the functions you want to inspect
lp.add_function(fibre_sim.get_bounding_boxes)
lp.add_function(fibre_sim.get_bbox_intersections)
lp.add_function(fibre_sim.collision_loss)
lp.add_function(fibre_sim.linearity_loss)
lp.add_function(fibre_sim.angle_between)
lp.add_function(fibre_sim.torsional_loss)
lp.add_function(fibre_sim.boundary_loss)
lp.add_function(fibre_sim.optimize)

def run():
    fibre_sim.main()

lp_wrapper = lp(run)
lp_wrapper()

# make sure GPU ops are accounted for
import torch
if torch.cuda.is_available():
    torch.cuda.synchronize()

lp.print_stats()
