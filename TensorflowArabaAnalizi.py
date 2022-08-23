import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sbn

dataFrame = pd.read_excel("merc.xlsx")

dataFrame.head()
"""
   year  price transmission  mileage  tax   mpg  engineSize
0  2005   5200    Automatic    63000  325  32.1         1.8
1  2017  34948    Automatic    27000   20  61.4         2.1
2  2016  49948    Automatic     6200  555  28.0         5.5
3  2016  61948    Automatic    16000  325  30.4         4.0
4  2016  73948    Automatic     4000  325  30.1         4.0
"""

dataFrame.describe()
"""
               year          price  ...           mpg    engineSize
count  13119.000000   13119.000000  ...  13119.000000  13119.000000
mean    2017.296288   24698.596920  ...     55.155843      2.071530
std        2.224709   11842.675542  ...     15.220082      0.572426
min     1970.000000     650.000000  ...      1.100000      0.000000
25%     2016.000000   17450.000000  ...     45.600000      1.800000
50%     2018.000000   22480.000000  ...     56.500000      2.000000
75%     2019.000000   28980.000000  ...     64.200000      2.100000
max     2020.000000  159999.000000  ...    217.300000      6.200000
"""

dataFrame.isnull().sum()
"""
year            0
price           0
transmission    0
mileage         0
tax             0
mpg             0
engineSize      0
"""

#car1
plt.figure(figsize=(7,5))
sbn.distplot(dataFrame["price"])

#car2
sbn.countplot(dataFrame["year"])

#corr
dataFrame.corr()
"""
                year     price   mileage       tax       mpg  engineSize
year        1.000000  0.520712 -0.738027  0.012480 -0.094626   -0.142147
price       0.520712  1.000000 -0.537214  0.268717 -0.438445    0.516126
mileage    -0.738027 -0.537214  1.000000 -0.160223  0.202850    0.063652
tax         0.012480  0.268717 -0.160223  1.000000 -0.513742    0.338341
mpg        -0.094626 -0.438445  0.202850 -0.513742  1.000000   -0.339862
engineSize -0.142147  0.516126  0.063652  0.338341 -0.339862    1.000000
"""

dataFrame.corr()["price"].sort_values()
"""
mileage      -0.537214
mpg          -0.438445
tax           0.268717
engineSize    0.516126
year          0.520712
price         1.000000
"""
#car3
sbn.scatterplot(x="mileage",y="price",data=dataFrame)

dataFrame.sort_values("price",ascending = False).head(20)
"""
       year   price transmission  mileage  tax   mpg  engineSize
6199   2020  159999    Semi-Auto     1350  145  21.4         4.0
10044  2020  154998    Automatic     3000  150  21.4         4.0
5      2011  149948    Automatic     3000  570  21.4         6.2
8737   2019  140319    Semi-Auto      785  150  22.1         4.0
6386   2018  139995    Semi-Auto    13046  145  21.4         4.0
8      2019  139948    Automatic    12000  145  21.4         4.0
9133   2019  139559    Semi-Auto     1000  145  22.1         4.0
8821   2020  138439    Semi-Auto     1000  145  22.1         4.0
5902   2018  135771    Semi-Auto    19000  145  21.4         4.0
7864   2018  135124    Semi-Auto    18234  150  21.4         4.0
8673   2019  134219    Semi-Auto     1000  145  24.8         4.0
6210   2019  129990    Automatic     1000  145  24.8         4.0
4759   2019  126000    Automatic      250  145  24.6         4.0
2647   2019  125796    Automatic      637  145  24.8         4.0
6223   2019  124999    Automatic     1500  145  31.7         4.0
4094   2019  124366    Semi-Auto      880  145  24.8         4.0
2629   2019  123846    Semi-Auto     2951  145  22.1         4.0
7134   2019  115359    Semi-Auto     1000  145  30.1         4.0
9159   2019  114199    Semi-Auto      891  145  22.6         4.0
1980   2019  109995    Semi-Auto     4688  150  31.7         4.0
"""

dataFrame.sort_values("price",ascending = True).head(20)
"""
       year  price transmission  mileage  tax   mpg  engineSize
11816  2003    650       Manual   109090  235  40.0         1.4
12008  2010   1350       Manual   116126  145  54.3         2.0
11765  2000   1490    Automatic    87000  265  27.2         3.2
11549  2002   1495    Automatic    13800  305  39.8         2.7
12594  2004   1495       Manual   119000  300  34.5         1.8
11174  2001   1695    Automatic   108800  325  31.7         3.2
12710  2006   1695    Automatic   153000  300  33.6         1.8
12766  2004   1780    Automatic   118000  265  41.5         2.2
12009  2007   1800    Automatic    84000  200  42.8         1.5
11764  1998   1990    Automatic    99300  265  32.1         2.3
11808  1998   1990    Automatic   113557  265  32.1         2.3
11383  2005   1995    Automatic   105000  260  43.5         2.1
11378  2004   1995    Semi-Auto   165000  330  20.0         3.7
11857  2002   2140    Automatic    52700  325  31.4         2.0
11906  2007   2478    Automatic    81000  160  49.6         2.0
11795  2005   2490    Automatic   101980  200  47.9         2.0
12765  2004   2495    Automatic   104000  325  31.7         1.8
11943  2005   2690    Automatic   109000  325  32.1         1.8
11263  2007   2795       Manual    79485  200  45.6         1.5
49     2006   2880    Automatic    66000  160  52.3         2.0
"""

len(dataFrame) 
# 13119
len(dataFrame) * 0.01
# 131.19

yuzdeDoksanDokuzDf = dataFrame.sort_values("price",ascending = False).iloc[131:]
yuzdeDoksanDokuzDf.describe()
"""
               year         price  ...           mpg    engineSize
count  12988.000000  12988.000000  ...  12988.000000  12988.000000
mean    2017.281876  24074.926933  ...     55.437142      2.050901
std        2.228515   9866.224575  ...     15.025999      0.532596
min     1970.000000    650.000000  ...      1.100000      0.000000
25%     2016.000000  17357.500000  ...     45.600000      1.675000
50%     2018.000000  22299.000000  ...     56.500000      2.000000
75%     2019.000000  28706.000000  ...     64.200000      2.100000
max     2020.000000  65990.000000  ...    217.300000      6.200000
"""

#car4
plt.figure(figsize=(7,5))
sbn.distplot(yuzdeDoksanDokuzDf["price"])

dataFrame.describe()
"""
               year          price  ...           mpg    engineSize
count  13119.000000   13119.000000  ...  13119.000000  13119.000000
mean    2017.296288   24698.596920  ...     55.155843      2.071530
std        2.224709   11842.675542  ...     15.220082      0.572426
min     1970.000000     650.000000  ...      1.100000      0.000000
25%     2016.000000   17450.000000  ...     45.600000      1.800000
50%     2018.000000   22480.000000  ...     56.500000      2.000000
75%     2019.000000   28980.000000  ...     64.200000      2.100000
max     2020.000000  159999.000000  ...    217.300000      6.200000
"""

dataFrame.groupby("year").mean()["price"]
"""
year
1970    24999.000000
1997     9995.000000
1998     8605.000000
1999     5995.000000
2000     5743.333333
2001     4957.900000
2002     5820.444444
2003     4878.000000
2004     4727.615385
2005     4426.111111
2006     4036.875000
2007     5136.045455
2008     6967.437500
2009     6166.764706
2010     8308.473684
2011    12624.894737
2012    10845.140351
2013    11939.842466
2014    14042.936864
2015    16731.780020
2016    19307.892948
2017    21514.307854
2018    25720.162918
2019    31290.020865
2020    35433.282337
"""

yuzdeDoksanDokuzDf.groupby("year").mean()["price"]
"""
year
1970    24999.000000
1997     9995.000000
1998     8605.000000
1999     5995.000000
2000     5743.333333
2001     4957.900000
2002     5820.444444
2003     4878.000000
2004     4727.615385
2005     4426.111111
2006     4036.875000
2007     5136.045455
2008     6967.437500
2009     6166.764706
2010     8308.473684
2011     8913.459459
2012    10845.140351
2013    11939.842466
2014    14042.936864
2015    16647.822222
2016    19223.558943
2017    21356.280421
2018    24800.844506
2019    30289.524832
2020    34234.794872
"""

dataFrame[dataFrame.year != 1970].groupby("year").mean()["price"]
"""
year
1997     9995.000000
1998     8605.000000
1999     5995.000000
2000     5743.333333
2001     4957.900000
2002     5820.444444
2003     4878.000000
2004     4727.615385
2005     4426.111111
2006     4036.875000
2007     5136.045455
2008     6967.437500
2009     6166.764706
2010     8308.473684
2011    12624.894737
2012    10845.140351
2013    11939.842466
2014    14042.936864
2015    16731.780020
2016    19307.892948
2017    21514.307854
2018    25720.162918
2019    31290.020865
2020    35433.282337
"""

dataFrame = yuzdeDoksanDokuzDf
dataFrame.describe()
"""
               year         price  ...           mpg    engineSize
count  12988.000000  12988.000000  ...  12988.000000  12988.000000
mean    2017.281876  24074.926933  ...     55.437142      2.050901
std        2.228515   9866.224575  ...     15.025999      0.532596
min     1970.000000    650.000000  ...      1.100000      0.000000
25%     2016.000000  17357.500000  ...     45.600000      1.675000
50%     2018.000000  22299.000000  ...     56.500000      2.000000
75%     2019.000000  28706.000000  ...     64.200000      2.100000
max     2020.000000  65990.000000  ...    217.300000      6.200000
"""

dataFrame = dataFrame[dataFrame.year != 1970]
dataFrame.groupby("year").mean()["price"]
"""
year
1997     9995.000000
1998     8605.000000
1999     5995.000000
2000     5743.333333
2001     4957.900000
2002     5820.444444
2003     4878.000000
2004     4727.615385
2005     4426.111111
2006     4036.875000
2007     5136.045455
2008     6967.437500
2009     6166.764706
2010     8308.473684
2011     8913.459459
2012    10845.140351
2013    11939.842466
2014    14042.936864
2015    16647.822222
2016    19223.558943
2017    21356.280421
2018    24800.844506
2019    30289.524832
2020    34234.794872
"""

dataFrame.head()
"""
      year  price transmission  mileage  tax   mpg  engineSize
6177  2019  65990    Semi-Auto     5076  150  30.4         3.0
5779  2020  65990    Semi-Auto      999  145  28.0         4.0
3191  2020  65980    Semi-Auto     3999  145  28.0         4.0
4727  2019  65000    Semi-Auto     3398  145  27.2         4.0
8814  2019  64999    Semi-Auto      119  145  40.9         3.0
"""

dataFrame = dataFrame.drop("transmission",axis=1)
y = dataFrame["price"].values
x = dataFrame.drop("price",axis=1).values

y
"""
array([65990, 65990, 65980, ...,  1490,  1350,   650], dtype=int64)
"""
x
"""
array([[2.01900e+03, 5.07600e+03, 1.50000e+02, 3.04000e+01, 3.00000e+00],
       [2.02000e+03, 9.99000e+02, 1.45000e+02, 2.80000e+01, 4.00000e+00],
       [2.02000e+03, 3.99900e+03, 1.45000e+02, 2.80000e+01, 4.00000e+00],
       ...,
       [2.00000e+03, 8.70000e+04, 2.65000e+02, 2.72000e+01, 3.20000e+00],
       [2.01000e+03, 1.16126e+05, 1.45000e+02, 5.43000e+01, 2.00000e+00],
       [2.00300e+03, 1.09090e+05, 2.35000e+02, 4.00000e+01, 1.40000e+00]])
"""


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.3,random_state=10)

len(x_train)
"""
9090
"""
len(x_test)
"""
3897
"""

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

x_train.shape
"""
(9090, 5)
"""

model = Sequential()

model.add(Dense(12,activation="relu"))
model.add(Dense(12,activation="relu"))
model.add(Dense(12,activation="relu"))
model.add(Dense(12,activation="relu"))

model.add(Dense(1))

model.compile(optimizer="adam",loss="mse")

model.fit(x=x_train, y = y_train,validation_data=(x_test,y_test),batch_size=250,epochs=300)


kayipVerisi = pd.DataFrame(model.history.history)
kayipVerisi.head()
"""
          loss     val_loss
0  672134912.0  688082240.0
1  672084352.0  687996672.0
2  671937536.0  687751168.0
3  671521024.0  687057280.0
4  670380032.0  685230016.0
"""

#car5
kayipVerisi.plot()


from sklearn.metrics import mean_squared_error, mean_absolute_error
tahminDizisi = model.predict(x_test)
tahminDizisi
"""
array([[21509.025],
       [23239.734],
       [24777.615],
       ...,
       [25948.443],
       [14277.055],
       [24497.87 ]], dtype=float32)
"""

mean_absolute_error(y_test,tahminDizisi)
# 3182.481310900462

dataFrame.describe()
"""
               year         price  ...           mpg    engineSize
count  12987.000000  12987.000000  ...  12987.000000  12987.000000
mean    2017.285516  24074.855779  ...     55.438392      2.051059
std        2.189633   9866.601115  ...     15.025902      0.532313
min     1997.000000    650.000000  ...      1.100000      0.000000
25%     2016.000000  17355.000000  ...     45.600000      1.700000
50%     2018.000000  22299.000000  ...     56.500000      2.000000
75%     2019.000000  28706.000000  ...     64.200000      2.100000
max     2020.000000  65990.000000  ...    217.300000      6.200000
"""

#car6
plt.scatter(y_test,tahminDizisi)
plt.plot(y_test,y_test,"g-*")

dataFrame.iloc[2]
"""
year           2020.0
price         65980.0
mileage        3999.0
tax             145.0
mpg              28.0
engineSize        4.0
Name: 3191, dtype: float64
"""

yeniArabaSeries = dataFrame.drop("price",axis=1).iloc[2]
type(yeniArabaSeries)
#pandas.core.series.Series
yeniArabaSeries = scaler.transform(yeniArabaSeries.values.reshape(-1,5))
model.predict(yeniArabaSeries)