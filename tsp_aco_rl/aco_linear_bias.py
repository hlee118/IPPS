import random as rand
import matplotlib.pyplot as plt
import math
import time
import tensorflow as tf

c = 1.           # original amount of trail
alpha = 1          # trail preference
beta = 1           # greedy preference

pr = 0.01           # probability of pure random selection of the next town

n = 0               # 노드의 개수
m = 0               # 개미의 수
numAntFactor = 0.8  # 개미의 수 = 노드의 개수 * numAntFactor
graph = []
trails_w = []         # 페로몬
trails_b = []
ants = []
currentIndex = 0

bestTour = 0
bestTourLength = 0

X = 0
W = 0
B = 0
sess = 0
train = 0
cost = 0


class Ant:
    def __init__(self):
        self.tour = []        #방문한 곳들
        self.visited = [False] * n     #방문했었는지 체크

    def visit_town(self, town):
        self.tour.append(town)
        self.visited[town] = True

    def tour_length(self):
        length = 0
        for i in range(-1, n - 1):
            length += get_distance(self.tour[i], self.tour[i + 1])

        return length


def data_init():
    global n, m, trails_w, trails_b, ants, c, X, W, B, sess, train, cost
    with open("_data1.txt", "r") as f:
        lines = f.readlines()
        for line in lines:
            arr = line.split(",")
            f_arr = [float(val) for val in arr]
            graph.append(f_arr)
        f.close()

    n = len(graph)
    m = int(n * numAntFactor)
    c = get_distance(0, 1)
    trails_w = [[c] * n for _ in range(n)]
    trails_b = [[c] * n for _ in range(n)]
    ants = [Ant() for _ in range(m)]


    ###############
    # tensorflow
    ###############
    # placeholders for a tensor that will be always fed.
    X = tf.placeholder(tf.float32, shape=[1, n])
    W = tf.Variable(tf.random_normal([n, 1]), name='weight')
    B = tf.Variable(tf.random_normal([1]), name='bias')

    # Hypothesis
    hypothesis = tf.matmul(X, W) + B

    # Simplified cost/loss function
    cost = tf.reduce_mean(tf.square(hypothesis - 1400))

    optimizer = tf.train.GradientDescentOptimizer(learning_rate=1e-5)
    train = optimizer.minimize(cost)

    # Launch the graph in a session.
    sess = tf.Session()
    # Initializes global variables in the graph.
    sess.run(tf.global_variables_initializer())


def get_distance(node1, node2):
    global graph
    x1 = graph[node1][0]
    y1 = graph[node1][1]
    x2 = graph[node2][0]
    y2 = graph[node2][1]
    return math.sqrt(pow(x1 - x2, 2) + pow(y1 - y2, 2))


def solve():
    global n, trails_w, trails_b, c, bestTourLength, bestTour

    result_data = []
    start_time = time.time()
    for i in range(50):
        setup_ants()
        move_ants()
        update_trails()
        update_best()
        result_data.append(bestTourLength)

    sess.close()
    end_time = time.time()
    print("Time : " + str(end_time - start_time))
    print("Best tour length : " + str(bestTourLength))
    plt.plot(result_data)
    plt.show()


# m ants with random start city
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
            selected_town = select_next_town(ant)
            try:
                ant.visit_town(selected_town)
            except:
                print("selected_town has none type value")


def select_next_town(ant):
    global n, pr
    if rand.random() < pr:
        # 가끔은 그냥 바로 선택 되기도 함
        while True:
            t = rand.randrange(n)  # random town
            if not ant.visited[t]:
                return t

    else:
        # 가능한 길들의 확률 파악
        probs = prob_to(ant)

        r = rand.random()
        tot = 0
        for i in range(n):
            tot += probs[i]
            if tot >= r:
                return i


def prob_to(ant):
    global n, alpha, beta
    town = ant.tour[-1]
    total = 0.0
    probs = []
    # 방문할 수 있는 곳들의 페로몬 총량
    for i in range(n):
        if not ant.visited[i]:
            total += pow(trails_w[town][i], alpha) * pow(1.0 / get_distance(town, i), beta)

    for i in range(n):
        if ant.visited[i]:
            probs.append(0.0)
        else:
            # 각각의 길의 경향성 파악
            prob = pow(trails_w[town][i], alpha) * pow(1.0 / get_distance(town, i), beta)
            try:
                probs.append(prob / total)
            except:
                print("방문할 수 있는 곳이 없음")

    return probs


def update_trails():
    global trails_w

    for ant in ants:
        # x_data = [[get_distance(ant.tour[i - 1], ant.tour[i]) for i in range(n)]]     # X = 거리데이터
        x_data = [ant.tour]                                                             # X = 순서데이터

        w_data = [[trails_w[ant.tour[i - 1]][ant.tour[i]]] for i in range(n)]
        W.assign(w_data)

        b_data = [trails_b[ant.tour[i - 1]][ant.tour[i]] for i in range(n)]
        b_data = [sum(b_data) / n]
        B.assign(b_data)

        cost_val, w_val, b_val, _ = sess.run([cost, W, B, train], feed_dict={X: x_data})
        b_diff = b_data[0] - b_val[0]

        # 페로몬 업데이트
        for i in range(n):
            if not math.isnan(float(w_val[i][0])):
                trails_w[ant.tour[i - 1]][ant.tour[i]] = float(w_val[i][0])
                trails_b[ant.tour[i - 1]][ant.tour[i]] -= b_diff


def update_best():
    global bestTour, bestTourLength
    for ant in ants:
        if bestTourLength == 0 or ant.tour_length() < bestTourLength:
            bestTourLength = ant.tour_length()

if __name__ == "__main__":
    data_init()
    solve()