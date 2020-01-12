import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

class PolyRegressor:

    dataset = []
    poly_reg = None
    poly_features = None

    def __init__(self, dataset):
        self.dataset = dataset
        self.ir = LinearRegression()

    def select_columns(self, x, y):
        self.dataset = self.dataset.sort_values(x)
        x = np.array(self.dataset[[x]])
        y = np.array(self.dataset[[y]])

        return x, y

    def get_graph(self, col1, col2):
        x, y = self.select_columns(col1, col2)

        self.poly_features = PolynomialFeatures(degree=15)
        x_poly = self.poly_features.fit_transform(x)

        self.poly_reg = LinearRegression()
        self.poly_reg.fit(x_poly, y)
        y_poly_pred = self.poly_reg.predict(x_poly)

        fig = plt.figure(figsize=(15,8))
        raw = plt.scatter(x, y, color='red')
        fit, = plt.plot(x, y_poly_pred, linewidth=3, color='blue')
        plt.legend((raw, fit), ('Data', 'Polynomial Fit'), loc='lower right')

        plt.title('Polynomial regression', fontsize=24)
        plt.xlabel(col1, fontsize=18)
        plt.ylabel(col1, fontsize=18)
        
        return fig

    def make_prediction(self, input):
        feature = [[input]]
        pred = self.poly_reg.predict(self.poly_features.fit_transform(feature))

        return pred
