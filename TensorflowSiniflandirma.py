import pandas as pd
import numpy as np

dataFrame = pd.read_excel("maliciousornot.xlsx")

dataFrame.head()
"""
   Type  URL_LENGTH  ...    SOURCE_R    SOURCE_S
0     1   23.303047  ...    0.595983    0.154015
1     1   26.645007  ...  356.216667    0.115311
2     1   25.505113  ...    0.468004    0.113445
3     1   14.792707  ...    0.859842  224.092667
4     1   26.282313  ...    0.306217    0.099456
"""

dataFrame.info()
"""
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 548 entries, 0 to 547
Data columns (total 31 columns):
 #   Column                     Non-Null Count  Dtype  
---  ------                     --------------  -----  
 0   Type                       548 non-null    int64  
 1   URL_LENGTH                 548 non-null    float64
 2   NUMBER_SPECIAL_CHARACTERS  548 non-null    float64
 3   TCP_CONVERSATION_EXCHANGE  548 non-null    float64
 4   DIST_REMOTE_TCP_PORT       548 non-null    float64
 5   REMOTE_IPS                 548 non-null    float64
 6   APP_BYTES                  548 non-null    float64
 7   SOURCE_APP_PACKETS         548 non-null    float64
 8   REMOTE_APP_PACKETS         548 non-null    float64
 9   SOURCE_APP_BYTES           548 non-null    float64
 10  REMOTE_APP_BYTES           548 non-null    float64
 11  APP_PACKETS                548 non-null    float64
 12  DNS_QUERY_TIMES            548 non-null    float64
 13  SOURCE_A                   548 non-null    float64
 14  SOURCE_B                   548 non-null    float64
 15  SOURCE_C                   548 non-null    float64
 16  SOURCE_D                   548 non-null    float64
 17  SOURCE_F                   548 non-null    float64
 18  SOURCE_E                   548 non-null    float64
 19  SOURCE_G                   548 non-null    float64
 20  SOURCE_H                   548 non-null    float64
 21  SOURCE_I                   548 non-null    float64
 22  SOURCE_J                   548 non-null    float64
 23  SOURCE_K                   548 non-null    float64
 24  SOURCE_M                   548 non-null    float64
 25  SOURCE_L                   548 non-null    float64
 26  SOURCE_N                   548 non-null    float64
 27  SOURCE_O                   548 non-null    float64
 28  SOURCE_P                   548 non-null    float64
 29  SOURCE_R                   548 non-null    float64
 30  SOURCE_S                   548 non-null    float64
dtypes: float64(30), int64(1)
"""

dataFrame.describe()
"""
             Type    URL_LENGTH  ...    SOURCE_R    SOURCE_S
count  548.000000    548.000000  ...  548.000000  548.000000
mean     0.383212    949.973475  ...   40.829159    2.637820
std      0.486613   3202.802599  ...  119.531119   19.086225
min      0.000000     10.051787  ...    0.202720    0.071295
25%      0.000000     15.838688  ...    0.331022    0.093099
50%      0.000000     18.069900  ...    0.374869    0.103743
75%      1.000000     23.264187  ...    0.430342    0.119375
max      1.000000  12828.981333  ...  704.661333  224.092667
"""

dataFrame.corr()["Type"].sort_values()
"""
URL_LENGTH                  -0.228422
SOURCE_I                    -0.138708
SOURCE_B                    -0.128587
SOURCE_APP_BYTES            -0.086080
SOURCE_C                    -0.075369
REMOTE_APP_BYTES            -0.048806
SOURCE_G                    -0.017433
DNS_QUERY_TIMES             -0.011055
SOURCE_F                    -0.007551
SOURCE_E                     0.001985
SOURCE_L                     0.022932
SOURCE_D                     0.029479
SOURCE_H                     0.055045
SOURCE_O                     0.063622
SOURCE_R                     0.069140
SOURCE_N                     0.088076
APP_BYTES                    0.096659
REMOTE_IPS                   0.126232
SOURCE_APP_PACKETS           0.129433
REMOTE_APP_PACKETS           0.139874
SOURCE_S                     0.141134
SOURCE_P                     0.205141
APP_PACKETS                  0.240818
NUMBER_SPECIAL_CHARACTERS    0.412095
SOURCE_J                     0.453197
SOURCE_A                     0.536539
DIST_REMOTE_TCP_PORT         0.710294
SOURCE_M                     0.734002
TCP_CONVERSATION_EXCHANGE    0.744570
SOURCE_K                     0.784173
Type                         1.000000
"""

import matplotlib.pyplot as plt
import seaborn as sbn

#type
sbn.countplot(x="Type", data = dataFrame)
#typebar
dataFrame.corr()["Type"].sort_values().plot(kind="bar")


y = dataFrame["Type"].values
x = dataFrame.drop("Type",axis = 1).values

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.3,random_state=15)

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaler.fit(x_train)

x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Dropout
from tensorflow.keras.callbacks import EarlyStopping

x_train.shape
"""
(383, 30)
"""

model = Sequential()

model.add(Dense(units=30,activation = "relu"))
model.add(Dense(units=15,activation = "relu"))
model.add(Dense(units=15,activation = "relu"))
model.add(Dense(units=1,activation = "sigmoid"))

model.compile(loss="binary_crossentropy",optimizer = "adam")

#model.fit
model.fit(x=x_train, y=y_train, epochs=700,validation_data=(x_test,y_test),verbose=1)
#model.history
model.history.history


#modelkaybi
modelKaybi = pd.DataFrame(model.history.history)
modelKaybi.plot()


model = Sequential()

model.add(Dense(units=30,activation = "relu"))
model.add(Dense(units=15,activation = "relu"))
model.add(Dense(units=15,activation = "relu"))
model.add(Dense(units=1,activation = "sigmoid"))

model.compile(loss="binary_crossentropy",optimizer = "adam")
earlyStopping = EarlyStopping(monitor="val_loss",mode="min",verbose=1,patience=25)
model.fit(x=x_train, y=y_train, epochs = 700, validation_data = (x_test,y_test), verbose = 1, callbacks=[earlyStopping])

#modelkaybi2
modelKaybi = pd.DataFrame(model.history.history)
modelKaybi.plot()


model = Sequential()

model.add(Dense(units=30,activation = "relu"))
model.add(Dropout(0.6))

model.add(Dense(units=15,activation = "relu"))
model.add(Dropout(0.6))

model.add(Dense(units=15,activation = "relu"))
model.add(Dropout(0.6))

model.add(Dense(units=1,activation = "sigmoid"))

model.compile(loss="binary_crossentropy",optimizer = "adam")

model.fit(x=x_train, y=y_train, epochs = 700, validation_data = (x_test,y_test), verbose = 1, callbacks=[earlyStopping])

kayipDf = pd.DataFrame(model.history.history)
kayipDf.plot()

tahminlerimiz = model.predict_classes(x_test)
tahminlerimiz

from sklearn.metrics import classification_report, confusion_matrix

classification_report(y_test,tahminlerimiz)

confusion_matrix(y_test,tahminlerimiz)