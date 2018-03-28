import random
import matplotlib.pyplot as plt
import time
from keras.layers import Dense
from keras.optimizers import Adam
from keras.models import Sequential
from keras import backend as K
import numpy as np
import math
from data_model import Data

# parameters
data = Data()
pr = 0.01            # exploit rate
d_rate = 0.90        # discount rate

# results
bestTour = 0
bestTourLength = 0
result_data = []

# keras_params
learning_rate = 0.001


class ReinforceAgent:
    def __init__(self):
        self.node_size = data.node_size
        self.visited = []
        self.path = []
        self.distance = 0
        self.c_city = -1
        self.n_city = -1
        self.model = self.build_model()
        self.optimizer = self.build_optimizer()

    def build_model(self):
        model = Sequential()
        model.add(Dense(24, input_dim=1, activation='relu'))
        model.add(Dense(24, activation='relu'))
        model.add(Dense(self.node_size, activation='softmax'))
        model.summary()
        return model

    # 정책신경망을 업데이트 하기 위한 오류함수와 훈련함수의 생성
    def build_optimizer(self):
        action = K.placeholder(shape=[None, self.node_size])
        discounted_rewards = K.placeholder(shape=[None, ])

        # 크로스 엔트로피 오류함수 계산
        action_prob = K.sum(action * self.model.output, axis=1)
        cross_entropy = K.log(action_prob) * discounted_rewards
        loss = -K.sum(cross_entropy)

        # 정책신경망을 업데이트하는 훈련함수 생성
        optimizer = Adam(lr=learning_rate)
        updates = optimizer.get_updates(self.model.trainable_weights, [],
                                        loss)
        train = K.function([self.model.input, action, discounted_rewards], [],
                           updates=updates)
        return train

    # 정책신경망으로 행동 선택
    def get_action(self):
        # 모델로부터 행동 산출
        policy = self.model.predict(np.array([[self.c_city]]))[0]

        # 방문한 곳은 제외
        for i in range(self.node_size):
            if self.visited[i]:
                policy[i] = 0

        # 확률 재계산
        p_sum = np.sum(policy)
        policy /= p_sum
        self.n_city = np.random.choice(range(self.node_size), 1, p=policy)[0]

        # 방문표시
        self.visited[self.n_city] = True
        self.path.append(self.n_city)
        self.distance += data.get_distance(self.c_city, self.n_city)

        # 다음 도시를 현재도시로 변경
        self.c_city = self.n_city

    # 정책신경망 업데이트
    def train_model(self):
        reward = 1 / self.distance
        discounted_rewards = [0] * self.node_size
        for i in range(self.node_size):
            discounted_rewards[self.node_size - 1 - i] = (reward * math.pow(d_rate, i))
            discounted_rewards -= np.mean(discounted_rewards)
            discounted_rewards /= np.std(discounted_rewards)

        next_states = [np.zeros(self.node_size) for _ in range(self.node_size)]
        for index, node in enumerate(self.path):
            next_states[index-1][node] = 1

        states = [np.array([node]) for node in self.path]
        self.optimizer([states, next_states, discounted_rewards])

    def cal_total_distance(self):
        total_distance = 0
        for i in range(self.node_size):
            city1 = self.path[i - 1]
            city2 = self.path[i]
            total_distance += data.get_distance(city1, city2)
        return total_distance


####################
# Main function
####################
# init
def init():
    agent.c_city = random.randrange(agent.node_size)
    agent.n_city = -1
    agent.visited = [False] * agent.node_size
    agent.visited[agent.c_city] = True
    agent.path = [agent.c_city]
    agent.distance = 0


# move
def move():
    # 첫번째 곳은 이미 방문한 상태로 시작
    for i in range(data.node_size - 1):
        agent.get_action()


# train
def train_model():
    agent.train_model()


# update
def update_best():
    global bestTourLength, bestTour
    distance = agent.cal_total_distance()
    result_data.append(distance)
    if bestTourLength == 0 or distance < bestTourLength:
        bestTourLength = distance
        bestTour = agent.path


###################
# start!
###################
if __name__ == "__main__":
    agent = ReinforceAgent()

    start_time = time.time()

    for e in range(500):
        init()            # 초기화
        move()            # 한바퀴
        train_model()     # 학습
        update_best()

    end_time = time.time()

    print("Time : " + str(end_time - start_time))
    print("Best tour length : " + str(bestTourLength))
    print(bestTour)
    plt.plot(result_data)
    plt.show()
