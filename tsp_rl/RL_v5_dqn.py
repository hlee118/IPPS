import random
import matplotlib.pyplot as plt
import time
from keras.layers import Dense
from keras.optimizers import Adam
from keras.models import Sequential
import numpy as np
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
epsilon = 1.0
epsilon_decay = 0.999
epsilon_min = 0.01
batch_size = 64
discount_factor = 0.99


class DQNAgent:
    def __init__(self):
        self.node_size = data.node_size
        self.visited = []
        self.path = []
        self.distance = 0
        self.c_city = -1
        self.n_city = -1

        # 모델과 타깃 모델 생성
        self.model = self.build_model()

    # 상태가 입력, 큐함수가 출력인 인공신경망 생성
    def build_model(self):
        model = Sequential()
        model.add(Dense(24, input_dim=1, activation='relu', kernel_initializer='he_uniform'))
        model.add(Dense(24, activation='relu', kernel_initializer='he_uniform'))
        model.add(Dense(self.node_size, activation='linear', kernel_initializer='he_uniform'))
        model.summary()
        model.compile(loss='mse', optimizer=Adam(lr=learning_rate))
        return model

    # 예측된 q함수의 최대값으로 행동 선택
    def get_action(self):
        # 남은 도시들
        cities = np.array(range(self.node_size))
        not_visited= np.array([not flag for flag in self.visited])
        left_cities = cities[not_visited]

        # 모델로부터 행동 산출
        if np.random.rand() <= epsilon:
            self.n_city = np.random.choice(left_cities)
        else:
            state = np.array([[self.c_city]])
            q_value = self.model.predict(state)
            q_value = q_value[0]

            # 방문한 곳은 제외
            for i in range(self.node_size):
                if self.visited[i]:
                    q_value[i] = -999999

                self.n_city = np.argmax(q_value)

        # 방문표시
        self.visited[self.n_city] = True
        self.path.append(self.n_city)
        self.distance += data.get_distance(self.c_city, self.n_city)

        # 다음도시 return
        return self.n_city

    # 정책신경망 update
    def train_model(self):
        global epsilon
        if epsilon > epsilon_min:
            epsilon *= epsilon_decay

        # 현재 상태에 대한 모델의 큐함수
        # 다음 상태에 대한 타깃 모델의 큐함수
        for idx in range(self.node_size):

            state = np.array([[self.path[idx-1]]])
            next_state = np.array([[self.path[idx]]])
            reward = 10000 / data.get_distance(state[0][0], next_state[0][0])

            target = self.model.predict(state)

            # 벨만 최적 방정식을 이용한 업데이트 타깃
            target[0][next_state] = reward + discount_factor * (np.amax(target[0]))
            self.model.fit(state, target, batch_size=self.node_size, epochs=1, verbose=0)

    def cal_total_distance(self):
        total_distance = 0
        for i in range(self.node_size):
            city1 = self.path[i - 1]
            city2 = self.path[i]
            total_distance += data.get_distance(city1, city2)

        return total_distance


# init
def init():
    agent.c_city = random.randrange(agent.node_size)
    agent.n_city = -1
    agent.visited = [False] * agent.node_size
    agent.visited[agent.c_city] = True
    agent.path = [agent.c_city]
    agent.distance = 0


# update
def update_best():
    global bestTourLength, bestTour
    distance = agent.cal_total_distance()
    result_data.append(distance)
    if bestTourLength == 0 or distance < bestTourLength:
        bestTourLength = distance
        bestTour = agent.path


###################
# Main Function
###################
if __name__ == "__main__":
    agent = DQNAgent()
    start_time = time.time()

    for e in range(5000):
        init()            # 초기화
        # 첫번째 곳은 이미 방문한 상태로 시작
        done = 0
        for i in range(data.node_size - 1):
            agent.n_city = agent.get_action()
            # 다음 도시를 현재도시로 변경
            agent.c_city = agent.n_city

        agent.train_model()
        if e and e % 100 == 0:
            print(e, " done")
            print(bestTourLength)
        update_best()

    end_time = time.time()

    print("Time : " + str(end_time - start_time))
    print("Best tour length : " + str(bestTourLength))
    print(bestTour)
    plt.plot(result_data)
    plt.show()
