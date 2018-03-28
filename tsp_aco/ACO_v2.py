import random as rand
import matplotlib.pyplot as plt
import math
import time

graph = []
c = 1.0             # original amount of trail
alpha = 1           # trail preference
beta = 2            # greedy preference
global_evap = 0.05  # 전체 증발 정도
local_evap = 0.02   # local 증발 정도
Q = 500             # 추가 정도

pr = 0.01           # 랜덤 선택 probability

n = 0               # 노드의 개수
m = 0               # 개미의 수
numAntFactor = 0.8  # 개미의 수 = 노드의 개수 * numAntFactor
trails = []         # 페로몬
ants = []
currentIndex = 0

bestTour = 0
bestTourLength = 0
d_rate = 0.9        # discount rate


class Ant:
    def __init__(self):
        self.tour = []        # 방문한 곳들
        self.visited = [False] * n     # 방문했었는지 체크

    def visit_town(self, town):
        self.tour.append(town)
        self.visited[town] = True

    def tour_length(self):
        length = 0
        for i in range(-1, n - 1):
            length += get_distance(self.tour[i], self.tour[i + 1])

        return length


def data_init():
    global n, m, trails, ants, local_evap
    with open("_data1.txt", "r") as f:
        lines = f.readlines()
        for line in lines:
            arr = line.split(",")
            f_arr = [float(val) for val in arr]
            graph.append(f_arr)
        f.close()

    n = len(graph)
    m = int(n * numAntFactor)
    trails = [[c] * n for _ in range(n)]
    ants = [Ant() for _ in range(m)]
    local_evap = math.pow(global_evap, 1.0 / n)


def get_distance(node1, node2):
    global graph
    x1 = graph[node1][0]
    y1 = graph[node1][1]
    x2 = graph[node2][0]
    y2 = graph[node2][1]
    return math.sqrt(pow(x1 - x2, 2) + pow(y1 - y2, 2))


def nearest_neighbor(node):
    town = -1
    min_value = get_distance(node, 0)
    for i in range(1, n):
        value = get_distance(node, i)
        if i != node and value < min_value:
            town = i
    return town


# m ants with random start city
def solve():
    global n, trails, c, bestTourLength, bestTour
    result_data = []
    start_time = time.time()
    for i in range(200):
        setup_ants()
        move_ants()
        update_global_trails()
        update_best()
        result_data.append(bestTourLength)

    end_time = time.time()
    print("Time : " + str(end_time - start_time))
    print("Best tour length : " + str(bestTourLength))
    print(bestTour)
    plt.plot(result_data)
    plt.show()


############################
### main_method
############################
def setup_ants():
    global currentIndex, m, ants
    for i in range(m):
        for j in range(len(ants[i].visited)):
            ants[i].visited[j] = False
        ants[i].tour = []
        ants[i].visit_town(rand.randrange(n))


# m ants with random start city
def move_ants():
    global n, ants
    for _ in range(n - 1):
        for ant in ants:
            selected_town = select_next_town(ant)                   # 이동할 도시 탐색
            update_local_trails(ant.tour[-1], selected_town)        # 이동했을 경우 경로의 페로몬(지역) 업데이트
            ant.visit_town(selected_town)                           # 방문


def update_best():
    global bestTour, bestTourLength
    for ant in ants:
        if bestTourLength == 0 or ant.tour_length() < bestTourLength:
            bestTour = ant.tour
            bestTourLength = ant.tour_length()


#######################
### sub_method
#######################

# 다음 도시 탐색
def select_next_town(ant):
    if rand.random() < pr:
        # exploit
        while True:
            t = rand.randrange(n)  # random town
            if not ant.visited[t]:
                return t

    # 가능한 길들의 확률 파악
    probs = prob_to(ant)
    r = rand.random()
    tot = 0
    for i in range(n):
        tot += probs[i]
        if tot >= r:
            return i


# 가능한 길들의 확률 계산
def prob_to(ant):
    town = ant.tour[-1]
    total = 0.0
    probs = []

    # 방문할 수 있는 곳들의 페로몬 총량
    for i in range(n):
        if not ant.visited[i]:
            total += pow(trails[town][i], alpha) * pow(1.0 / get_distance(town, i), beta)

    # 각각의 길의 경향성 파악
    for i in range(n):
        if ant.visited[i]:
            probs.append(0.0)
        else:
            prob = pow(trails[town][i], alpha) * pow(1.0 / get_distance(town, i), beta)
            try:
                probs.append(prob / total)
            except():
                print("방문할 수 있는 곳이 없음")

    return probs


# 지역강화 (강화학습)
def update_local_trails(town1, town2):
    # 강화학습 방법
    # best = trails[town2][0]
    # for i in range(1, n):
    #     if best < trails[town2][i]:
    #         best = trails[town2][i]
    #
    # trails[town1][town2] += d_rate * best

    # Antcolony 지역강화 방법
    nearest_town = nearest_neighbor(town1)
    trails[town1][town2] = (1 - local_evap) * trails[town1][town2] + local_evap * (1 / (n * get_distance(town1, nearest_town)))


# 전역강화 (페로몬)
def update_global_trails():
    global n, global_evap, ants, trails

    # 증발
    for i in range(n):
        for j in range(n):
            trails[i][j] *= global_evap

    # 전역강화
    best_length = 0
    best_ant = -1
    for ant in ants:
        if best_length == 0 or ant.tour_length() < best_length:  # best 개미가 누구인지 체크
            best_length = ant.tour_length()
            best_ant = ant

    for i in range(-1, n - 1):
        trails[best_ant.tour[i]][best_ant.tour[i + 1]] += 1 / best_length


if __name__ == "__main__":
    data_init()
    solve()