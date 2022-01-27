import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    # importing datasets
    dataset = pd.read_csv('Data.csv')

    # Extracting Independent Variable
    x = dataset.iloc[:, :-1].values
    # Extracting Dependent variable
    y = dataset.iloc[:, -1].values

    # handling missing data(Replacing missing data with the mean value)
    imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
    # Fitting imputer object to the independent variables x.
    imputer_fit = imputer.fit(x[:, 1:3])
    # Replacing missing data with the calculated mean value
    x[:, 1:3] = imputer.transform(x[:, 1:3])

    label_encoder_x = LabelEncoder()
    x[:, 0] = label_encoder_x.fit_transform(x[:, 0])

    # Encoding for dummy variables
    # Encoding for dummy variables
    columnTransformer = ColumnTransformer([('encoder', OneHotEncoder(), [0])],     remainder='passthrough')
    x = columnTransformer.fit_transform(x)

    # encoding for purchased variable
    labelencoder_y = LabelEncoder()
    y = labelencoder_y.fit_transform(y)

    # Splitting the dataset into training and test set.
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)
    # print("x_train= ", x_train)
    # print("x_test= ", x_test)
    # print("y_train= ", y_train)
    # print("y_test= ", y_test)

    # Feature Scaling of datasets

    st_x = StandardScaler()
    x_train = st_x.fit_transform(x_train)
    x_test = st_x.transform(x_test)

    print("x_train= ", x_train)
    print("x_test= ", x_test)

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
