import numpy as np

class NaiveGreedy:
    def __init__(self, G_lc):
        self.G_lc = G_lc
        self.kappas = []
        print("num edges:")
        print(len(G_lc.edges()))
        for u,v,a in G_lc.edges(data=True):
            self.kappas.append(a['weight'])
        self.kappas = np.array(self.kappas)

    def subset(self, k):
        print("Kappas: ", len(self.kappas))
        print("k: ", k)
        idx = np.argpartition(self.kappas, -k)[-k:]
        solution = np.zeros(len(self.kappas))
        if k > 0:
            solution[idx] = 1.0
        return solution
