import numpy as np
from ipps.Data import *
import random
import copy


class Operation:
    def __init__(self, j=0, o=0, m=0, t=0, pt=0, active=False):
        self.j = j
        self.o = o
        self.m = m
        self.t = t
        self.pt = pt
        self.active = active


class Schedule:
    def __init__(self, seq=[], st=[], ct=[], slot_used=[], tool_used=[], makespan=0.0):
        self.seq = seq
        self.st = st
        self.ct = ct
        self.slot_used = slot_used
        self.tool_used = tool_used
        self.makespan = makespan


# return fiels of an operation
def op_fields(op):
    return op.j, op.o, op.m, op.t, op.pt, op.active


# scheduling algorithm
def construct_schedule(problem):
    # calculate length of schedule
    # len_sch = sum(NUM_OPER[problem])
    seq = []  # sequence of assigned operations

    # setup initial operations: push all operations in the Unvisit
    for j in problem:
        for o in range(1, NUM_OPER[j]+1):
            m, t, pt = op_assign_mt_rand(j, o)  # select a machine and a tool
            new_op = Operation(j, o, m, t, pt, True)  # construct a new operation
            seq.append(new_op)

    new_schedule = Schedule(seq)
    scheduling_active(new_schedule)
    return new_schedule


# select a set of machine and tool from pt_table x = (job,id, oper_id) randomly
def op_assign_mt_rand(j, o):
    idx = random.randrange(1, len(PT_TABLE[j][o][1]))
    return PT_TABLE[j][o][0][idx], PT_TABLE[j][o][1][idx], PT_TABLE[j][o][2][idx]


# active scheduling with non precedence preserved operation sequence
def scheduling_active(Sch):
    # calculate length of schedule
    len_sch = len(Sch.seq)

    # assign st and ct for new schedule
    Sch.st = [] #start time
    Sch.ct = [] #completion time

    Unvisit = Sch.seq # a set of unassigned operations
    Available = [] # a set of current avaliable operations
    Assigned = [] # sequence of assigned operations

    tool_mag = [[0] * MAX_TOOL for _ in range(MAX_MACH)]
    ast_op = [[0] * MAX_OPER for _ in range(MAX_JOB)]  # allowable start time for operations
    ast_mach = [0] * MAX_MACH

    # if start node, then also push the operation to Available
    for i in range(len_sch):
        j, o, m, t, pt, active = op_fields(Unvisit[i])
        Unvisit[i].active = True
        if o in START_OP[j]:
            Available.append(Unvisit[i])

    while Available:
        selected_op = select_operation_hybrid(Unvisit, Available, ast_op, ast_mach)
        j, o, m, t, pt, active = op_fields(selected_op)     # assign field values to variables

        # update Sets
        test = len(Unvisit)
        remove_op(Unvisit, selected_op)
        remove_op(Available, selected_op)
        Assigned.append(selected_op)

        # calculate start time
        st_op = max(ast_op[j-1][o-1], ast_mach[m-1])

        if active:
            # if the selected_op is OR node, make the alternative process inactive
            for d in DUMMY_OP[j][o]:
                if d in Unvisit:
                    op_inactive(Unvisit, j, d)
                    op_inactive(Available, j, d)

            tool_mag[m - 1][t - 1] = 1    # machine m use tool t
            # calculate compl etion time
            ct_op = st_op + pt
            # update available start times
            for out_ops in OUT_OP[j][o]:
                ast_op[j][out_ops] = ct_op
            ast_mach[m-1] = ct_op
        # active == false
        else:
            ct_op = st_op

        # insert st and ct
        Sch.st.append(st_op)
        Sch.ct.append(ct_op)

        # add available operations in Available
        for out_op in OUT_OP[j][o]:
            if check_all_prev_ops_in_visited(Assigned, j, out_op):
                if op_find(Unvisit, j, out_op) != -1:
                    Available.append(op_find(Unvisit, j, out_op))
                else:
                    arr = []
                    for i in range(len(Unvisit)):
                        if Unvisit[i].j == j:
                            arr.append(Unvisit[i])
                    print("can't find!")

        # while

    Sch.seq = Assigned
    # calculate status of tool magazine
    Sch.tool_used = [sum(tool_mag[i][j] for i in range(MAX_MACH) ) for j in range(MAX_TOOL)]
    # Sch.slot_used = np.array(tool_mag) * np.array(TOOL_SLOT)
    Sch.slot_used = [sum(tool_mag[i][j]*TOOL_SLOT[j] for j in range(MAX_TOOL)) for i in range(MAX_MACH)]
    Sch.makespan = max(Sch.ct)


# operation selection for hybrid scheduling algorithm
def select_operation_hybrid(Unvisit, Available, ast_op, ast_mach):
    HYBRID_THETA = random.choice(Available)

    # construct arrays for calculation
    len_A = len(Available)
    # row:: j, o, m, order in Available, order in Unvisit
    A = [[0] * 4 for _ in range(len_A)]
    # est: earlest start time, ect: earlest completion time
    est = [0] * len_A
    ect = [0] * len_A

    # calculate early start time of operations
    for i in range(len_A):
        if type(Available[i]) == int:
            print("!")
        j, o, m, t, pt, active = op_fields(Available[i]) # assign field values to variables
        est[i] = max(ast_op[j-1][o-1], ast_mach[m-1])
        ect[i] = est[i] + pt
        A[i] = [j, o, m, i]

    # find machine with earlest ct
    tau_opt = min(ect)
    tau_opt_index = ect.index(tau_opt)
    m_opt = A[tau_opt_index]

    ##################
    # 임시 추가 코드
    ##################

    return Available[tau_opt_index]

    ##################
    # 여기까지
    ##################

    # # generate a set B
    # B_idxs = find(A[:,3] .== m_opt)
    # B = A[B_idxs,:]
    # est = est[B_idxs]
    #
    # # find operation with earlest st
    # sigma_opt, sigma_opt_index = min(est)
    #
    # # generate conflict set C
    # C_idxs = find(est .< theta*tau_opt + (1-theta)*sigma_opt)
    # C = B[C_idxs,:]
    #
    # if HYBRID_THETA > 1
    # # hybrid GA find first operation among C
    # order = Int[]
    # for i in 1:length(C_idxs)
    #   push!(order, (Unvisit,C[i,1],C[i,2]))[1]
    # end
    # sid = C[indmin(order),4]
    # else
    # # select operation randomly from set C
    # sid = rand(C[:,4])
    # end
    #
    # return Available[sid]


def op_index(S, j, o):
    for i in range(len(S)):
        if S[i].j == j and S[i].o == o:
            return i
    return -1


def op_find(sch, j, o):
    for i in range(len(sch)):
        if sch[i].j == j and sch[i].o == o:
            return sch[i]
    return -1


# inactive operation x from S
def op_inactive(S, j, o):
    index = op_index(S, j, o)
    if index != -1:
        S[index].active = False


def check_all_prev_ops_in_visited(s, j, o):
    for pre in IN_OP[j][o]:
        if op_index(s, j, pre) == -1:
            return False  # visit = false

    return True  # all previous operations in Visited


def make_jo_matrix(Sch):
    jo_mat = [[0] * MAX_OPER for _ in range(MAX_JOB)]
    for i in range(len(Sch.seq)):
        jo_mat[Sch.seq[i].j][Sch.seq[i].o] = i

    return jo_mat


def remove_op(sch, op):
    for i in range(len(sch)):
        if sch[i].j == op.j and sch[i].o == op.o:
            sch.pop(i)
            break


def indmaxn(fitness, n):
    temp = copy.deepcopy(fitness)
    arr = []
    for i in range(n):
        ind = temp.index(max(temp))
        arr.append(ind)
    return arr


def stat_minmax(data):
    return [min(data), max(data), sum(data)/len(data)]

