# ################################################################################
# # Conventional ACO for IPPS is based on edge selection, i.e., from an op_jomt to another op_jomt.
# # , which increases the size of problem exponentially, and consequently, computational time.
# # Thus, We propose a two stage ACO for IPPS, which assign operation, machine and tool sequentially
# ################################################################################
# ## including packages
# ################################################################################
#
# from ipps.Params import *
# import time
#
#
# ################################################################################
# ## run ACO iteratively
# ################################################################################
# def run_ACO(aco_type):
#
#     global LOCAL_UPDATE
#
#     # results
#     data_makespan = [[0] * NUM_TEST for _ in range(NUM_PROBLEM)]
#     data_comptime = [[0] * NUM_TEST for _ in range(NUM_PROBLEM)]
#     data_fitness = [[0] * NUM_TEST for _ in range(NUM_PROBLEM)]
#     data_penalty = [[0] * NUM_TEST for _ in range(NUM_PROBLEM)]
#     data_numgen = [[0] * NUM_TEST for _ in range(NUM_PROBLEM)]
#     data_bestfit_gen = [[[0] * MAX_GENERATION for _ in range(NUM_TEST)] for _ in range(NUM_PROBLEM)]
#     data_individual = [[0] * NUM_TEST for _ in range(NUM_PROBLEM)]
#     opt_individual = [0] * NUM_PROBLEM
#     opt_makespan = [0] * NUM_PROBLEM
#     opt_penalty = [0] * NUM_PROBLEM
#     opt_comptime = [0] * NUM_PROBLEM
#
#     copt_makespan = [9999999.0] * NUM_PROBLEM
#
#     LOCAL_UPDATE = True
#
#     for p in range(SET_PROBLEM):
#         for n in range(NUM_TEST):
#             print(p)
#             ipps = PROBLEM[p]
#
#             start_time = time.time()
#             data_individual[p][n], data_fitness[p][n], data_penalty[p][n], data_bestfit_gen[p],[n], data_numgen[p][n] = ACO_standard(ipps)
#             end_time = time.time()
#             data_comptime[p][n] = float(end_time - start_time) / 1000 # sec
#             data_makespan[p][n] = data_individual[p][n].objective
#
#             # save optimal individual
#             if data_makespan[p][n] <= min(data_makespan[p][:n]):
#                 opt_individual[p] = data_individual[p][n]
#                 opt_makespan[p] = data_makespan[p][n]
#                 opt_penalty[p] = data_penalty[p][n]
#                 opt_comptime[p] = data_comptime[p][n]
#
#
# # standard ACO algorithm: Ant Colony System
# def ACO_standard(problem):
#
#   # generate pheromone table
#   global TAU, J2T
#
#   generate_TAU_table(problem)
#
#   num_nonimproved = 0
#   num_generation = 1
#   colony = Array(Trail,COLONY_SIZE)
#   colony_fitness = zeros(Float,COLONY_SIZE)
#   colony_penalty = zeros(Float,COLONY_SIZE)
#   best_fitness_gen = zeros(Float,MAX_GENERATION)
#   #ave_fitness_gen = zeros(Float,MAX_GENERATION)
#
#   # initial cbest
#   cbest_trail = construct_trail_empty(problem)
#   cbest_fitness = 0.0
#
#   # perform evolution of colony
#   while (num_generation<=MAX_GENERATION)&(num_nonimproved<=STOP_GENERATION)
#
#     # construct trail for each Ant
#     for k in 1:COLONY_SIZE
#       colony[k] = construct_trail(problem,construct_trail_empty(problem))
#       colony_fitness[k], colony_penalty[k] = calculate_fitness(colony[k])
#     end
#
#     # update generation best
#     #ave_fitness_gen[num_generation] = mean(pop_fitness)
#     best_fitness_gen[num_generation], best_ix = findmax(colony_fitness)
#
#     if cbest_fitness < best_fitness_gen[num_generation]
#       cbest_trail = dcopy_ind(colony[best_ix])
#       cbest_fitness = colony_fitness[best_ix]
#       num_nonimproved = 0
#     else
#       num_nonimproved += 1
#     end
#
#     # update global TAU
#     update_TAU_global(problem,cbest_trail,cbest_fitness)
#
#     num_generation += 1
#   end
#
#   cbest_fitness, cbest_penalty = calculate_fitness(cbest_trail)
#
#   return cbest_trail, cbest_fitness, cbest_penalty, best_fitness_gen, num_generation
#
# end
#
#
# ################################################################################
# ## functions for problem definitions
# ################################################################################
# def generate_TAU_table(problem):
#     global TAU, J2T
#     num_job = len(problem) # number of jobs in problem
#     max_oper = max(NUM_OPER[problem]) # maximum number of operations in jobs
#     J2T = [0] * MAX_JOB # index vector from job id into TAU_TABLE index
#     J2T[problem] = collect(1:num_job)
#     # initial TAU as zero matrix
#     TAU = [[[[0] * num_job for _ in range(max_oper)] for _ in range(num_job)]for _ in range(max_oper)]
#     # assign ACO_TAU_0 for problem
#     for j1 in problem, o1 = 1:NUM_OPER[j1], j2 in problem, o2 = 1:NUM_OPER[j2]
#         TAU[J2T[j1],o1,J2T[j2],o2] = ACO_TAU_0
#
#     # make 0 for precedence relations
#     for j in problem, o = 1:NUM_OPER[j]
#         for out in OUT_OP[j,o]
#           TAU[J2T[j],out,J2T[j],o] = 0.0
