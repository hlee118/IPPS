# Designed by Hyunsik Lee
# 2017.09.09
# Referred to github. robbs92's TSP_Neural-Net Project
# Boltzmann Machine
# annealing process

import random
import math
import copy
import time
import matplotlib.pyplot as plt

graph = []      # point of cities
n = 0           # number of cities
m = 0           # epoch
distances = []
min_distance = 0


def main():
    # start
    start_time = time.time()

    data_init()
    neuron_matrix = create_neuron_matrix()
    annealed_neuron_matrix = anneal(neuron_matrix)
    optimal_distance = calculate_distance(annealed_neuron_matrix)

    # end
    end_time = time.time()

    print("Time : " + str(end_time - start_time))
    print("Best tour length : " + str(optimal_distance))
    plt.plot(distances)
    plt.show()


def data_init():
    global n, m
    with open("_data1.txt", "r") as f:
        lines = f.readlines()
        for line in lines:
            arr = line.split(",")
            f_arr = [float(val) for val in arr]
            graph.append(f_arr)
        f.close()

    n = len(graph)
    m = n + 1


def get_distance(node1, node2):
    global graph
    x1 = graph[node1][0]
    y1 = graph[node1][1]
    x2 = graph[node2][0]
    y2 = graph[node2][1]
    return math.sqrt(pow(x1 - x2, 2) + pow(y1 - y2, 2))


def calculate_distance(neuron_matrix):
    prev_city = 0
    optimal_distance = 0
    for epoch in range(m):
        for curr_city in range(n):
            if neuron_matrix[epoch][curr_city] == 1:
                optimal_distance = optimal_distance + get_distance(prev_city, curr_city)
                prev_city = curr_city

    return optimal_distance


# row : city
# sequence of rows : tsp sequence
def create_neuron_matrix():
    global n
    neuron_matrix = [[0] * n] * m

    # Randomly choose start city
    first_and_last_epoch = [0] * n
    first_and_last_epoch[random.randrange(n)] = 1
    neuron_matrix[0] = first_and_last_epoch
    neuron_matrix[m - 1] = first_and_last_epoch

    # Create middle ephochs with random binary states
    for i in range(1, m - 1):
        neuron_matrix[i] = [random.randint(0, 1) for _ in range(n)]

    # Convert randomly assigned lists into list of lists
    # neuron_matrix = list(zip(*neuron_matrix))
    # neuron_matrix = [list(elem) for elem in neuron_matrix]
    return neuron_matrix


# The annealing process takes place in this function.
def anneal(neuron_matrix):
    global distances, min_distance
    # initial parameter
    curr_temp = 100000.0
    # cooling_rate = 0.99995
    cooling_rate = 0.9995
    absolute_temp = 1000

    # Annealing
    while curr_temp > absolute_temp:
        # Establish a candidate neuron matrix
        candidate_neuron_matrix = copy.deepcopy(neuron_matrix)

        # Pick a random candidate neuron to update
        x, y = pick_random_neuron()

        # Update the randomly chose neuron in the candidate matrix
        if candidate_neuron_matrix[x][y] == 1:
            candidate_neuron_matrix[x][y] = 0
        else:
            candidate_neuron_matrix[x][y] = 1

        # Calculate the initial consensus value and the candidate consensus value
        consensus = consensus_function(neuron_matrix, x, y)
        candidate_consensus = consensus_function(candidate_neuron_matrix, x, y)
        delta_consensus = candidate_consensus - consensus

        # To promote valid city visits, convert the sign of the consensus value
        if -35 <= delta_consensus <= 35:
            delta_consensus = delta_consensus * -1

        # Randomly generate an acceptance criteria between 0 and 1
        acceptance_criteria = random.uniform(0, 1)

        # Generate an acceptance probability from the consensus delta and the current annealing temperature
        acceptance_probability = sigmoid_function(delta_consensus, curr_temp)

        # If the acceptance criteria is less that than the acceptance probability,
        # update the neuron to matrix with the candidate neuron
        if acceptance_criteria < acceptance_probability:
            neuron_matrix[x][y] = candidate_neuron_matrix[x][y]
            distance = calculate_distance(neuron_matrix)
            distances.append(distance)
            if min_distance == 0 or distance < min_distance:
                min_distance = distance

        # Update the annealing process temperature
        curr_temp = curr_temp * cooling_rate

    return neuron_matrix


# The consensus function determines the overall consensus value of a neuron matrix. In order for the program to
# properly determine the result for this problem, the consensus function has to be calculated with weight values
# that inhibit invalid updates to the neuron matrix. TO do this, the consensus function looks for invalid update
# scenarios and will calculate the consensus value with very high weights to inhibit the potential for an invalid
# matrix. Such scenarios include the salesman visiting previously visited cities or multiple cities in the same
# epoch.
def consensus_function(neuron_matrix, x, y):
    consensus_value = 0
    weight = 0
    # Consensus value does not need to be calculated if the neuron is off, it will be 0 in that scenario
    if neuron_matrix[x][y] == 1:
        # Check to see if any of the other neurons representing the city the current neuron is related to are on
        # Column(all epoch) check
        visited = False
        for row in range(m):
            if neuron_matrix[row][y] == 1 and row != x:
                visited = True

        # Check to see if any of the other neurons in the same epoch
        simultaneous = False
        for city in range(n):
            if neuron_matrix[x][city] == 1 and city != y:
                # If there are other neurons in the same epoch that are on, compare the distances of the respective
                # cities from the previously visited city to determine which neuron being on would give the salesman
                # a shorter journey. If there is another neuron in the epoch that is on and represents a shorter
                # journey for the salesman, then inhibit this neuron with a weight of 500 otherwise use the distance
                # from the last city as the weight
                # Row check
                for previous in range(n):
                    if neuron_matrix[x - 1][previous] == 1:
                        if get_distance(previous, city) < get_distance(previous, y):
                            simultaneous = True

        if simultaneous or visited:
            weight = 1000000
        else:
            # If none of the prohibited scenarios exist,
            # use the city distances from all of the other 'on' neurons as te weight
            for prev_city in range(n):
                if neuron_matrix[x - 1][prev_city] == 1:
                    weight = weight + get_distance(prev_city, y)

        current_neuron_state = neuron_matrix[x][y]
        # Calculate the consensus value with the current neuron state and the weights determined above
        consensus_value = 0.5 * weight * current_neuron_state
    return consensus_value


# The sigmoid function sim ply takes the Consensus delta and the current temperature from the annealing process
# and returns the curret activation function
def sigmoid_function(delta_consensus, curr_temp):
    activation_value = 1 / (1 + math.exp(delta_consensus / curr_temp))
    return activation_value


# Randomly picks a neuron in the neuron matrix to update, note the first and last epochs are ignored
def pick_random_neuron():
    x = random.randrange(1, m - 1)        # 처음과 끝 제외
    y = random.randrange(0, n)
    return x, y


if __name__ == '__main__':
    main()
