import cProfile
import pstats
import spring_system_3d as fibre_sim

profiler = cProfile.Profile()
profiler.enable()

fibre_sim.main()   # run your simulation

profiler.disable()
stats = pstats.Stats(profiler).sort_stats("cumulative")
stats.print_stats(20)  # top 20 functions
