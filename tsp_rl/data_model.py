import math


class Data:
    def __init__(self):
        # main variables
        self.graph = []          # 노드들
        self.node_size = 0       # 노드의 개수
        self.data_init()

    def data_init(self):
        with open("_data1.txt", "r") as f:
            lines = f.readlines()
            for line in lines:
                arr = line.split(",")
                f_arr = [float(val) for val in arr]
                self.graph.append(f_arr)
            f.close()

        self.node_size = len(self.graph)

    def get_distance(self, node1, node2):
        x1 = self.graph[node1][0]
        y1 = self.graph[node1][1]
        x2 = self.graph[node2][0]
        y2 = self.graph[node2][1]
        return math.sqrt(pow(x1 - x2, 2) + pow(y1 - y2, 2))