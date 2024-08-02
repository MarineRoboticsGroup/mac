import numpy as np

class NaiveGreedy:
    def __init__(self, edges):
        self.weights = np.array([e.weight for e in edges])

    def subset(self, k):
        print("Weights: ", len(self.weights))
        print("k: ", k)
        idx = np.argpartition(self.weights, -k)[-k:]
        solution = np.zeros(len(self.weights))
        if k > 0:
            solution[idx] = 1.0
        return solution
