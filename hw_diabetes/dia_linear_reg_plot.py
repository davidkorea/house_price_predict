import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

DATA_FILE = '../data_ai/diabetes.csv'
FEAT_COLS = ['AGE','SEX','BMI','BP','S1','S2','S3','S4','S5','S6']

def plot(linear_reg_model,X_train,y_train,feat):
    w = linear_reg_model.coef_
    b = linear_reg_model.intercept_

    plt.figure()
    plt.scatter(X_train,y_train,alpha=0.5)
    plt.plot(X_train,w * X_train + b,c='red')
    plt.title(feat)
    plt.show()


def main():
    diabetes_data = pd.read_csv(DATA_FILE)

    for feat in FEAT_COLS:
        X = diabetes_data[feat].values.reshape(-1,1)
        y = diabetes_data['Y'].values
        X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=1/5,random_state=20)
        linear_reg_model = LinearRegression()
        linear_reg_model.fit(X_train,y_train)
        r2_score = linear_reg_model.score(X_test,y_test)
        print('Feature:{}, R2:{}'.format(feat,r2_score))

        plot(linear_reg_model,X_train,y_train,feat)


main()