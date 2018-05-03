import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression


DATA_FILE = '../data_ai/diabetes.csv'
# print(os.path.exists(DATA_FILE))
FEATURE_COLS = ['AGE','SEX','BMI','BP','S1','S2','S3','S4','S5','S6']
diabetes_data = pd.read_csv(DATA_FILE)
X = diabetes_data[FEATURE_COLS].values
y = diabetes_data['Y'].values

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=1/5,random_state=20)

linear_reg_model = LinearRegression()
linear_reg_model.fit(X_train,y_train)
r2_score = linear_reg_model.score(X_test,y_test)
print('R2 score: ',r2_score)

idx = 22
single_test_feat = X_test[idx,:]
true = y_test[idx]
pred = linear_reg_model.predict([single_test_feat])
print('Sample {} - true:{}, pred:{}'.format(idx,true,pred))

# Results:
# R2 score:  0.4179775463198647
# Sample 22 - true:104, pred:[31.5716432]

# CSV data description
# * AGE：年龄 
# * SEX: 性别 
# * BMI: 体质指数（Body Mass Index） 
# * BP: 平均血压（Average Blood Pressure） 
# * S1~S6: 一年后的6项疾病级数指标 
# * Y: 一年后患疾病的定量指标，为需要预测的标签
