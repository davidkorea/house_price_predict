import os
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import pandas as pd
import ai_utils

DATA_FILE = './data_ai/house_data.csv'
# print(os.path.exists(DATA_FILE))
FEATURE_COL = ['bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot', 'sqft_above', 'sqft_basement']


def main():
    house_data = pd.read_csv(DATA_FILE,usecols=FEATURE_COL + ['price'])
    # ai_utils.plot_feat_and_price(house_data)

    X = house_data[FEATURE_COL].values
    y = house_data['price'].values

    X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=1/3,random_state=10)

    linear_reg_model = LinearRegression()
    linear_reg_model.fit(X_train,y_train)
    r2_score = linear_reg_model.score(X_test,y_test)
    print('R2_score: {}'.format(r2_score))

    idx = 201
    single_test_feat = X_test[idx,:]
    true_price = y_test[idx]
    pred_price = linear_reg_model.predict([single_test_feat])
    print('Sample:',single_test_feat)
    print('True :{}, Pred: {}'.format(true_price,pred_price))

main()

