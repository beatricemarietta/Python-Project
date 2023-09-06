import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
%matplotlib inline

from google.colab import drive
drive.mount('/content/drive')

data = '/content/drive/MyDrive/time_series.xlsx'

df = pd.read_excel(data)
df.head()

df.columns=["Tanggal","Kasus Harian"]
df.head()
df.describe()
df.set_index('Tanggal',inplace=True)

from pylab import rcParams
rcParams['figure.figsize'] = 16, 9
df.plot()

df["Kasus Harian"]=df["Kasus Harian"]**(1/2)
df

from pylab import rcParams
rcParams['figure.figsize'] = 16, 9
df.plot()

from statsmodels.tsa.stattools import adfuller
test_result=adfuller(df["Kasus Harian"])

def adf_test(dataset):
     dftest = adfuller(dataset, autolag = 'AIC')
     print("1. ADF : ",dftest[0])
     print("2. P-Value : ", dftest[1])
     print("3. Num Of Lags : ", dftest[2])
     print("4. Num Of Observations Used For ADF Regression:",      dftest[3])
     print("5. Critical Values :")
     for key, val in dftest[4].items():
         print("\t",key, ": ", val)
adf_test(df['Kasus Harian'])

def test_p_value(data):
        fuller_test = adfuller(data)
        print('P-value: ',fuller_test[1])
        if fuller_test[1] <= 0.05:
            print('Reject null hypothesis, data is stationary')
        else:
            print('Do not reject null hypothesis, data is not stationary')
test_p_value(df['Kasus Harian'])

from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
plt_ACF = plot_acf(df['Kasus Harian'].dropna())
plt_PACF = plot_pacf(df['Kasus Harian'].dropna())

from pmdarima import auto_arima
import pmdarima as pm

pm.auto_arima
model = pm.auto_arima(df["Kasus Harian"], start_p=0, start_q=0,
                         test='adf',
                         max_p=2, max_q=2,
                         d=0, trace=True,
                         error_action='ignore',  
                         suppress_warnings=True, 
                         stepwise=True)
print(model.summary())

model = pm.auto_arima(df["Kasus Harian"], start_p=0, start_q=0,
                      test='adf',       # use adftest to find optimal 'd'
                      max_p=2, max_q=2, # maximum p and q
                      m=1,              # frequency of series
                      d=0,              # let model determine 'd'
                      seasonal=False,   # No Seasonality
                      start_P=0, 
                      D=0, 
                      trace=True,
                      error_action='ignore',  
                      suppress_warnings=True, 
                      stepwise=True)

print(model.summary())

model.plot_diagnostics(figsize=(10,8))
plt.show()

# Forecast
fitted, confint = model.predict(n_periods=12, return_conf_int=True)
df_forecast = fitted**(1/(1/2))
df_forecast
