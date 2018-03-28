from ipps.Params import *
from ipps.IPPS import *
import time
import copy
import csv
import matplotlib.pyplot as plt


def run_GA():
    global FIT_PARAM, HYBRID_THETA, GO

    data_makespan = [[999999999] * NUM_PROBLEM for _ in range(NUM_TEST)]
    data_comptime = [[0] * NUM_PROBLEM for _ in range(NUM_TEST)]
    data_fitness = [[0] * NUM_PROBLEM for _ in range(NUM_TEST)]
    data_penalty = [[0] * NUM_PROBLEM for _ in range(NUM_TEST)]
    data_numgen = [[0] * NUM_PROBLEM for _ in range(NUM_TEST)]
    data_bestfit_gen = [[0] * NUM_PROBLEM for _ in range(NUM_TEST)]
    data_individual = [[0] * NUM_PROBLEM for _ in range(NUM_TEST)]

    opt_individual = [0] * NUM_PROBLEM
    opt_makespan = [0] * NUM_PROBLEM
    opt_penalty = [0] * NUM_PROBLEM


    # for p in SET_PROBLEM:
    p = 0
    # for n in range(NUM_TEST):
    n = 1
    print(p)
    ipps = PROBLEM[p]

    start_time = time.time()
    data_individual[p][n], data_fitness[p][n], data_penalty[p][n], data_bestfit_gen[p][n], data_numgen[p][n] = ga_standard(ipps)
    data_comptime[p][n] = (time.time() - start_time) / 1000 # sec
    data_makespan[p][n] = data_individual[p][n].makespan

    # save optimal individual
    # if data_makespan[p][n] == min(data_makespan[p]):
    #     opt_individual[p] = data_individual[p][n]
    #     opt_makespan[p] = data_makespan[p][n]
    #     opt_penalty[p] = data_penalty[p][n]


    # save results
    # results = [stat_minmax(data_makespan), stat_minmax(data_comptime), stat_minmax(data_numgen), stat_minmax(data_fitness)]
    # with open('./result.csv', 'w') as f:
    #     writer = csv.writer(f)
    #     writer.writerow(results)
    #     f.close()


    # save figures
    # for p in len(NUM_PROBLEM):
    # plt.plot(data_individual[p][n])
    # plt.plot([1,2,3,4])
    # savefig("gantt_P$p _$add_text.png", dpi=400)
    plt.show()

    # plt.trend("Best Fitness Trend", PLOT_TREND, data_bestfit_gen)
    # # savefig("trend_$add_text.png", dpi=400)


def ga_standard(problem):
    # scheduling = scheduling_active
    # selection = TOURNAMENT
    # crossover = ECX
    # mutation = LMTCMNP

    num_nonimproved = 0
    num_generation = 0
    next_population = [0] * POP_SIZE
    next_pop_fitness = [0] * POP_SIZE
    best_fitness_gen = [0] * MAX_GENERATION

    # generate initial population
    population, pop_fitness = generate_initial_population(problem, POP_SIZE)

    # initialize current best individual
    cbest_fitness = max(pop_fitness)
    cbest_idx = pop_fitness.index(cbest_fitness)
    cbest_individual = copy.deepcopy(population[cbest_idx])

    # calculate number of elitist. num_elitist must be even number
    num_elite = int(ELITISM_PROB * POP_SIZE)


    # perform evolution of population
    while num_generation < MAX_GENERATION and num_nonimproved < STOP_GENERATION:

        # if elitism is used, insert the elitists into end of next_population
        elite_idxs = indmaxn(pop_fitness, num_elite)

        # assign idxs of eilitists into next generation idxs
        for i in range(num_elite):
            next_population[POP_SIZE - 1 - i] = copy.deepcopy(population[elite_idxs[i]])
            next_pop_fitness[POP_SIZE - 1 - i] = pop_fitness[elite_idxs[i]]

        num_offspring = POP_SIZE - num_elite
        # perform selection procedure and generate new generation
        parents_idxs = TOURNAMENT(pop_fitness, num_offspring)

        # genetic operations procedure
        for i in [i * 2 for i in range((int)(num_offspring/2))]:
            # crossover procedure
            if len(population[i].seq) != len(population[i + 1].seq):
                print(len(population[i].seq), len(population[i + 1].seq))
                print(i)

            mother = copy.deepcopy(population[parents_idxs[i]])
            father = copy.deepcopy(population[parents_idxs[i + 1]])
            if random.random() < CROSSOVER_PROB:
                offspring1, offspring2 = crossover(mother, father)
            else:
                offspring1, offspring2 = mother, father

            # mutation procedure
            if random.random() < MUTATION_PROB:
                mutation(offspring1)
            if random.random() < MUTATION_PROB:
                mutation(offspring2)

            # scheduling and calculate fitness of offsprings
            scheduling_active(offspring1)
            scheduling_active(offspring2)
            fitness1, penalty1 = calculate_fitness(offspring1)
            fitness2, penalty2 = calculate_fitness(offspring1)

            # replace offsprings into next population
            next_population[i] = offspring1
            next_population[i+1] = offspring2
            next_pop_fitness[i] = fitness1
            next_pop_fitness[i+1] = fitness2

        # update population and fitness array
        population = next_population
        pop_fitness = next_pop_fitness

        # update generation best
        # ave_fitness_gen[num_generation] = mean(pop_fitness)
        best_fitness_gen[num_generation] = max(pop_fitness)
        best_ix = pop_fitness.index(best_fitness_gen[num_generation])

        if cbest_fitness < best_fitness_gen[num_generation]:
            cbest_individual = copy.deepcopy(population[best_ix])
            cbest_fitness = best_fitness_gen[num_generation]
            num_nonimproved = 0
        else:
            num_nonimproved += 1

        num_generation += 1

    cbest_fitness, cbest_penalty = calculate_fitness(cbest_individual)
    return cbest_individual, cbest_fitness, cbest_penalty, best_fitness_gen, num_generation


def generate_initial_population(problem, size):
    population = [0]*size
    pop_fitness = [0]*size
    pop_penalty = 0.0

    for i in range(size):
        population[i] = construct_schedule(problem)
        pop_fitness[i], pop_penalty = calculate_fitness(population[i])

    return population, pop_fitness


# calculate fitness values of a schedule
def calculate_fitness(Sch):
    FIT_PARAM = (10.0, 10.0, 0.5, 0.5)  # Kim et al. 2007
    F_C1, F_C2, F_ALPHA, F_BETA = FIT_PARAM

    tool_over = [Sch.tool_used[i] - TOOL_NUM[i] for i in range(len(TOOL_NUM))]
    tool_penalty = 0
    for t in tool_over:
        if t > 0:
            tool_penalty += 2 * pow(t, F_ALPHA)
    slot_over = [Sch.slot_used[i] - TOOL_MAG[i] for i in range(len(TOOL_MAG))]
    slot_penalty = 0
    for s in slot_over:
        if s > 0:
            slot_penalty += 2 * pow(s, F_BETA)

    penalty = F_C1 * tool_penalty + F_C2 * slot_penalty
    fitness = 100000.0 / (Sch.makespan + penalty)

    return fitness, penalty

################################################################################
## basic function definitions
################################################################################

def make_seq_order(Mother, Father):
    # define sequence order Array
    len_seq = len(Mother.seq)
    order = [0] * len_seq
    # make jo_matrix of Father
    jo_mat = make_jo_matrix(Father)

    for i in range(len_seq):
        order[i] = jo_mat[Mother.seq[i].j][Mother.seq[i].o]

    return order


################################################################################
## selection procedures
################################################################################
# selection_tournament
def TOURNAMENT(pop_fitness, num_offspring):
    len_pop = len(pop_fitness)
    # empty array
    idx_off = [0]* num_offspring
    rand_vec = [random.random() for _ in range(num_offspring)]

    # random individuals with replacement
    idx1 = [random.randrange(len_pop) for _ in range(num_offspring)]
    idx2 = [random.randrange(len_pop) for _ in range(num_offspring)]

    for i in range(num_offspring):
        if rand_vec[i] < TOURNAMENT_PROB:
            idx_off[i] = idx1[i] if pop_fitness[idx1[i]] > pop_fitness[idx2[i]] else idx2[i]
        else:
            idx_off[i] = idx2[i] if pop_fitness[idx1[i]] > pop_fitness[idx2[i]] else idx1[i]

    return idx_off


################################################################################
## crossover operator function definitions
################################################################################

# exchange crossover
# precedence, or_relation, mag_const, tool_const = Break
# ECX(Mother, Father):
# def crossover(Mother, Father):
#     len_off = len(Mother.seq)
#
#     # define offspring sequences
#     fa_order = make_seq_order(Father, Mother) # Mother 1:end
#
#     # replace operations
#     random_index = [random.randint(0, 1) for _ in range(len_off)]
#     mo_idxs = [i for i in range(len_off) if random_index[i] == 1]
#     fa_idxs = [fa_order[i] for i in mo_idxs]            #find location which are in cut1:cut2 of Mother
#     for i in range(len(mo_idxs)):
#         Mother.seq[mo_idxs[i]], Father.seq[fa_idxs[i]] = Father.seq[fa_idxs[i]], Mother.seq[mo_idxs[i]]
#
#     return Mother, Father

# precedence preserve crossover
# precedence = OK, or_relation, mag_const, tool_const = Break
def crossover(mother, father):
    len_off = len(mother.seq)
    # copy parents
    p11 = copy.deepcopy(mother.seq)
    p12 = copy.deepcopy(father.seq)
    p21 = copy.deepcopy(father.seq)
    p22 = copy.deepcopy(mother.seq)

    # random vector with 1 or 2
    rand_num = [random.randint(0, 1) for _ in range(len_off)]
    #println(rand_ix)
    off_seq1 = []
    off_seq2 = []

    for i in range(len_off):
        if rand_num[i]:
            off_seq1.append(p11.pop(0))
            remove_op(p12, off_seq1[i])
            off_seq2.append(p21.pop(0))
            remove_op(p22, off_seq2[i])
        else:
            off_seq1.append(p12.pop(0))
            remove_op(p11, off_seq1[i])
            off_seq2.append(p22.pop(0))
            remove_op(p21, off_seq2[i])

    # generate new Offsprings with sequence off_seqs
    offspring1 = Schedule(off_seq1)
    offspring2 = Schedule(off_seq2)

    return offspring1, offspring2


################################################################################
## mutation operator function definitions
################################################################################

# LMTCM(location change mutation) keeps precedence and or relations
def mutation(ind):
    # define sequence order Array
    len_seq = len(ind.seq)

    # make jo_matrix of Ind for fast search
    jo_mat = make_jo_matrix(ind)

    # select an operation randomly
    selected_op_idx = random.randrange(len_seq)
    selected_op = ind.seq.pop(selected_op_idx)
    j, o = selected_op.j, selected_op.o # assign field values to variables

    # initial_value
    range_start = -1 # range start
    range_end = len_seq # range end

    # find range start that keeps precedence relationship
    for inop in IN_OP[j][o]:
        idx = jo_mat[j][inop]
        range_start = idx + 1

    # find range end that keeps precedence relationship
    for outop in OUT_OP[j][o]:
        idx = jo_mat[j][outop]
        range_end = idx

    if range_end - range_start > 0:
        index = random.randrange(range_start, range_end)
        ind.seq.insert(index, selected_op)
    else: # if there is a problem return to original
        ind.seq.insert(selected_op_idx, selected_op)

    # machine and tool change
    num_change = round(MTCM_PROB * len_seq)
    rand_idx = [random.randrange(len_seq) for _ in range(num_change)]
    for i in rand_idx:
        j, o = ind.seq[i].j, ind.seq[i].o
        ind.seq[i].m, ind.seq[i].t, ind.seq[i].pt = op_assign_mt_rand(j, o)
