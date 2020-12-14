import numpy as np
import pandas as pd
from sklearn.datasets import load_boston
from sklearn.datasets import load_iris
from sklearn.datasets import load_wine
from sklearn.linear_model import LinearRegression
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt


#%%
# Question 1
def boston_reg(Results = True):
    # get variable names and training data to fit the regression model
    boston = load_boston()['feature_names']
    X, y = load_boston(return_X_y = True)
    lineareg = LinearRegression()
    lineareg.fit(X, y)
    coef = pd.DataFrame([(name, coeff) for name, coeff in zip(boston, lineareg.coef_)], columns=["Parameter Name", "Coefficient"])
    coef.sort_values("Coefficient", ascending=False, inplace=True)

    # print ranking
    if Results:
        print("Boston Dataset Results:")
        print(coef)

    return coef


#%%
# Question 2
def elbow_method():
    # load data
    data_wine, _ = load_wine(return_X_y = True)
    # iris dataset
    data_iris, _ = load_iris(return_X_y = True)
    # SSD
    iris_SSD = []
    wine_SSD = []
    # make loop
    for i in range(1, 11):
        iris = KMeans(n_clusters = i).fit(data_iris)
        wine = KMeans(n_clusters = i).fit(data_wine)
        iris_SSD.append(iris.inertia_)
        wine_SSD.append(wine.inertia_)

    # plot results
    # iris
    plt.plot(range(1, 11), iris_SSD)
    plt.title("Iris dataset")
    plt.xlabel("Num of clusters")
    plt.ylabel("SSD")
    plt.show()

    # wine
    plt.plot(range(1, 11), wine_SSD)
    plt.title("Wine Dataset")
    plt.xlabel("Num of clusters")
    plt.ylabel("SSD")
    plt.show()

#%%
if __name__ == "__main__":
        print(boston_reg())
        print(elbow_method())

