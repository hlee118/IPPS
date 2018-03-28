import random as rand
import matplotlib.pyplot as plt
import math
import time

# main variables
graph = []          # 노드들
n = 0               # 노드의 개수
q_value = []        # q value들
init_q = 10000.0
visited = []        # 방문했던 곳 배열 TRUE / FALSE
path = []           # 현재까지 방문했던 곳 리스트
c_city = 0          # 현재 위치


# parameters
pr = 0.1           # exploit rate
d_rate = 0.9        # discount rate

# results
bestTour = 0
bestTourLength = 0


def data_init():
    global n, q_value, visited, c_city
    with open("_data3.txt", "r") as f:
        lines = f.readlines()
        for line in lines:
            arr = line.split(",")
            f_arr = [float(val) for val in arr]
            graph.append(f_arr)
        f.close()

    n = len(graph)
    q_value = [[init_q] * n for _ in range(n)]
    visited = [0] * n
    c_city = rand.randrange(n)          # random 시작지점


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


def solve():
    global n, bestTourLength, bestTour
    result_data = []
    start_time = time.time()
    for i in range(2000):
        init()
        move()
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
def init():
    global visited, c_city, path
    visited = [0] * n
    path = []
    c_city = rand.randrange(n)
    visited[c_city] = True
    path.append(c_city)


def move():
    for i in range(n - 1):                      # 모든곳을 다 탐색할 때까지
        selected_town = select_next_town()  # 이동할 도시 탐색
        visit_town(selected_town)           # 이동 및 q-value 업데이트


def update_best():
    global bestTourLength, bestTour
    total_distance = 0
    for i in range(n):
        total_distance += get_distance(path[i - 1], path[i])

    if bestTourLength == 0 or total_distance < bestTourLength:
        bestTourLength = total_distance
        bestTour = path


#######################
### sub_method
#######################
def select_next_town():
    # 모혐형 랜덤탐색
    if rand.random() < pr:
        while True:
            t = rand.randrange(n)  # random town
            if not visited[t]:
                return t

    # 최소값 선택방식
    q_min = 0
    q_min_city = -1
    for next_city in range(n):
        if not visited[next_city] and (q_min == 0 or q_value[c_city][next_city] < q_min):
            q_min = q_value[c_city][next_city]
            q_min_city = next_city

    # if q_min_city == -1:
    #     print("break")

    return q_min_city


def visit_town(next_city):
    global c_city, path
    # visit
    visited[next_city] = True
    path.append(c_city)

    # q-value update
    q_value[c_city][next_city] = get_distance(c_city, next_city) + 0.9 * min(q_value[next_city])
    c_city = next_city


if __name__ == "__main__":
    data_init()
    solve()