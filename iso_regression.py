import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.collections import LineCollection
from sklearn.isotonic import IsotonicRegression

class IsoRegressor:

    dataset = []
    ir = None

    def __init__(self, dataset):
        self.dataset = dataset
        self.ir = IsotonicRegression()

    def select_columns(self, x, y):
        self.dataset = self.dataset.sort_values(x)
        x = np.array(self.dataset[x])
        y = np.array(self.dataset[y])

        return x, y

    def get_graph(self, col1, col2):
        x, y = self.select_columns(col1, col2)
        transformed = self.ir.fit_transform(x, y)

        n = len(x)
        segments = [[[i, y[i]], [i, transformed[i]]] for i in range(n)]
        lc = LineCollection(segments, zorder=0)
        lc.set_array(np.ones(len(y)))
        lc.set_linewidth(np.full(n, 0.5))

        fig = plt.figure(figsize=(15,8))
        plt.plot(x, y, 'r.', markersize=12)
        plt.plot(x, transformed, 'b.-', markersize=12)
        plt.gca().add_collection(lc)
        plt.legend(('Data', 'Isotonic Fit'), loc='lower right')
        plt.title('Isotonic regression', fontsize=24)
        plt.xlabel(col1, fontsize=18)
        plt.ylabel(col1, fontsize=18)
        
        return fig

    def make_prediction(self, input):
        feature = [input]
        pred = self.ir.predict(feature)

        return pred
