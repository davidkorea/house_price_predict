import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

DATA_FILE = './data_ai/house_data.csv'

print(os.path.exists(DATA_FILE))
COLS = ['price','bedrooms','bathrooms','sqft_living','sqft_lot',
        'floors','waterfront','view','condition','grade','sqft_above',
        'sqft_basement','yr_built','yr_renovated','zipcode','lat',
        'long','sqft_living15','sqft_lot15']

data_df = pd.read_csv(DATA_FILE,usecols=COLS)
corr_df = data_df.corr()
sns.heatmap(corr_df,annot=True,cmap='rainbow',fmt='.2f')
plt.title('Features heatmap')
plt.tight_layout()
plt.savefig('./plot/heatmap.png')
plt.show()

