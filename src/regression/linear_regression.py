from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_predict


def linear_regression(matrix, classes):
    lr = LinearRegression()
    y_pred = cross_val_predict(lr, matrix, classes, cv=10)
    return y_pred
