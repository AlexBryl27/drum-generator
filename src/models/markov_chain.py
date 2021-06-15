import matplotlib.pyplot as plt

from collections import Counter
from typing import List

class MarkovChain():
    def __init__(self):
        self._aprior = Counter()
        self._N = 0
        self._state_transitions_counter = Counter()
    
    def fit(self, X: List[List[int]]):
        self.__init__()
        [self.fit_partial(x) for x in X]
        
    def fit_partial(self, X: List[int]):
        self._aprior.update(X)
        N = len(X)
        self._N += N
        state_transitions = [
            (X[i], X[i+1])
            for i in range(N-1)
        ]
        self._state_transitions_counter.update(state_transitions)
    
    def predict_proba_state_transition(self, current_state, future_state):
        return self._state_transitions_counter[(current_state, future_state)] / self._aprior[current_state]
    
    def predict_proba_state(self, state):
        return self._aprior[state] / self._N
    
    def heatmap_of_transitions(self):
        # Матрица переходов
        max_uri_label = max(self._aprior.keys())
        uri_map = [[0 for j in range(max_uri_label)] for i in range(max_uri_label)]
        for i, row in enumerate(uri_map):
            for j, _ in enumerate(row):
                if (self._aprior[i] == 0) or (self._aprior[j] == 0):
                    continue
                else:
                    uri_map[i][j] = self.predict_proba_state_transition(i, j)

        plt.figure(figsize=(7, 7))
        ax = plt.imshow(
            X=uri_map,
            cmap='hot',
            interpolation='nearest'
        )
        return uri_map
