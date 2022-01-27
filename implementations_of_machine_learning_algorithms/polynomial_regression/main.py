import matplotlib.pyplot as mtp
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    # importing datasets
    data_set = pd.read_csv('Position_Salaries.csv')

    # Extracting Independent and dependent Variable
    x = data_set.iloc[:, 1:2].values
    y = data_set.iloc[:, 2].values

    # Fitting the Linear Regression to the dataset

    lin_regs = LinearRegression()
    lin_regs.fit(x, y)

    # Fitting the Polynomial regression to the dataset

    poly_regs = PolynomialFeatures(degree=2)
    x_poly = poly_regs.fit_transform(x)
    lin_reg_2 = LinearRegression()
    lin_reg_2.fit(x_poly, y)

    # Visulaizing the result for Linear Regression model
    plot1 = mtp.figure(1)
    mtp.scatter(x, y, color="blue")
    mtp.plot(x, lin_regs.predict(x), color="red")
    mtp.title("Bluff detection model(Linear Regression)")
    mtp.xlabel("Position Levels")
    mtp.ylabel("Salary")
    #mtp.show()

    # Visulaizing the result for Polynomial Regression
    plot2 = mtp.figure(2)
    mtp.scatter(x, y, color="blue")
    mtp.plot(x, lin_reg_2.predict(poly_regs.fit_transform(x)), color="red")
    mtp.title("Bluff detection model(Polynomial Regression)")
    mtp.xlabel("Position Levels")
    mtp.ylabel("Salary")
    mtp.show()

    lin_pred = lin_regs.predict([[6.5]])
    print("result with the Linear Regression model =", lin_pred)

    poly_pred = lin_reg_2.predict(poly_regs.fit_transform([[6.5]]))
    print("result with the Polynomial Regression model =", poly_pred)
