from ipps.Data import *

# Parameters
MAX_GENERATION = 2
NUM_TEST = 20

NUM_PROBLEM = len(PROBLEM)
SET_PROBLEM = [x for x in range(NUM_PROBLEM)]

POP_SIZE = 300
STOP_GENERATION = 100

CROSSOVER_PROB = 0.3
MUTATION_PROB = 0.1
ELITISM_PROB = 0.1
TOURNAMENT_PROB = 0.7
MTCM_PROB = 0.05

# F_C1,F_C2,F_ALPHA,F_BETA
FIT_NON = (10.0,10.0,0.5,0.5) # Kim et al. 2007
FIT_FEA = (20.0,20.0,2.0,2.0)

# (scheduling, selection, crossover, mutation)
# GO_SGA = (PPX, LMTCMPP!)        #preserve crossover, location change mutation
