import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

DATA_FILE = './data_ai/house_data.csv'
FEAT_COLS = ['bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot', 'sqft_above', 'sqft_basement']

def train_plot(linear_reg_model,X,y,feat):
    w = linear_reg_model.coef_
    b = linear_reg_model.intercept_

    plt.figure()
    # sample dots
    plt.scatter(X,y,alpha=0.5)

    #linear_regression line
    plt.plot(X, w * X + b, c='red')
    plt.title('train_'+feat)
    plt.savefig('./plot/train_{}.png'.format(feat))
    plt.show()

def test_plot(linear_reg_model,X,y,feat):
    w = linear_reg_model.coef_
    b = linear_reg_model.intercept_

    plt.figure()
    # sample dots
    plt.scatter(X,y,alpha=0.5)

    #linear_regression line
    plt.plot(X, w * X + b, c='red')
    plt.title('test_'+feat)
    plt.savefig('./plot/test_{}.png'.format(feat))
    plt.show()

def main():
    house_data = pd.read_csv(DATA_FILE)

    for feat in FEAT_COLS:
        X = house_data[feat].values.reshape(-1,1)
        y = house_data['price']

        X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=1/3,random_state=10)
        linear_reg_model = LinearRegression()
        linear_reg_model.fit(X_train,y_train)
        r2_score = linear_reg_model.score(X_test,y_test)
        print('Feature:{}, R2:{}'.format(feat,r2_score))

        train_plot(linear_reg_model,X_train,y_train,feat)
        test_plot(linear_reg_model,X_test,y_test,feat)

main()

# Results:
# Feature:bedrooms, R2:0.09258779614933521
# Feature:bathrooms, R2:0.27043501608845366
# Feature:sqft_living, R2:0.4961737855768537
# Feature:sqft_lot, R2:0.006598795645454514
# Feature:sqft_above, R2:0.35340999022703223
# Feature:sqft_basement, R2:0.1180671186537966