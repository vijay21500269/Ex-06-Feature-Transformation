# Ex-06-Feature-Transformation
## AIM
To read the given data and perform Feature Transformation process and save the data to a file.
## EXPLANATION
Feature Transformation is a technique by which we can boost our model performance. Feature transformation is a mathematical transformation in which we apply a mathematical formula to a particular column(feature) and transform the values which are useful for our further analysis.
## ALGORITHM
### STEP 1:
Read the given Data
### STEP 2:
Clean the Data Set using Data Cleaning Processv
### STEP 3:
Apply Feature Transformation techniques to all the features of the data set
### STEP 4:
Print the transformed features
## PPROGRAM:
~~~
Developed by : R.Vijay
Registration Number : 212221230121
~~~
~~~
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
import scipy.stats as stats
from sklearn.preprocessing import QuantileTransformer

df=pd.read_csv("data_trans.csv")
df

sm.qqplot(df.HighlyPositiveSkew,fit=True,line='45')
plt.show()

sm.qqplot(df.HighlyNegativeSkew,fit=True,line='45')
plt.show()

sm.qqplot(df.ModeratePositiveSkew,fit=True,line='45')
plt.show()

sm.qqplot(df.ModerateNegativeSkew,fit=True,line='45')
plt.show()

df['HighlyPositiveSkew']=np.log(df.HighlyPositiveSkew)
sm.qqplot(df.HighlyPositiveSkew,fit=True,line='45')
plt.show()

df['HighlyNegativeSkew']=np.log(df.HighlyNegativeSkew)
sm.qqplot(df.HighlyPositiveSkew,fit=True,line='45')
plt.show()

df['ModeratePositiveSkew_1'], parameters=stats.yeojohnson(df.ModeratePositiveSkew)
sm.qqplot(df.ModeratePositiveSkew_1,fit=True,line='45')
plt.show()

df['ModerateNegativeSkew_1'], parameters=stats.yeojohnson(df.ModerateNegativeSkew)
sm.qqplot(df.ModerateNegativeSkew_1,fit=True,line='45')
plt.show()

from sklearn.preprocessing import PowerTransformer
transformer=PowerTransformer("yeo-johnson")
df['ModerateNegativeSkew_2']=pd.DataFrame(transformer.fit_transform(df[['ModerateNegativeSkew']]))
sm.qqplot(df.ModerateNegativeSkew_2,fit=True,line='45')
plt.show()

from sklearn.preprocessing import QuantileTransformer
qt= QuantileTransformer(output_distribution = 'normal')
df['ModerateNegativeSkew_2']=pd.DataFrame(qt.fit_transform(df[['ModerateNegativeSkew']]))

sm.qqplot(df.ModerateNegativeSkew_2,fit=True,line='45')
plt.show()

df2=df.copy()

df2['HighlyPositiveSkew']= 1/df2.HighlyPositiveSkew
sm.qqplot(df2.HighlyPositiveSkew,fit=True,line='45')

plt.show()
~~~
## OUTPUT:
![img](https://github.com/vijay21500269/Ex-06-Feature-Transformation/blob/main/ds1.png)
![img](https://github.com/vijay21500269/Ex-06-Feature-Transformation/blob/main/ds2.png)
![img](https://github.com/vijay21500269/Ex-06-Feature-Transformation/blob/main/ds3.png)
![img](https://github.com/vijay21500269/Ex-06-Feature-Transformation/blob/main/ds4.png)
![img](https://github.com/vijay21500269/Ex-06-Feature-Transformation/blob/main/ds5.png)
![img](https://github.com/vijay21500269/Ex-06-Feature-Transformation/blob/main/ds6.png)
![img](https://github.com/vijay21500269/Ex-06-Feature-Transformation/blob/main/ds7.png)
![img](https://github.com/vijay21500269/Ex-06-Feature-Transformation/blob/main/ds8.png)
![img](https://github.com/vijay21500269/Ex-06-Feature-Transformation/blob/main/ds9.png)
![img](https://github.com/vijay21500269/Ex-06-Feature-Transformation/blob/main/ds10.png)
## RESULT:
Thus feature transformation is done for the given dataset.
