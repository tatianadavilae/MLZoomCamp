```python
import pandas as pd
import numpy as np
import os
```


```python
os.getcwd()
```




    'C:\\Users\\Tatiana\\Downloads'




```python
data=pd.read_csv('data.csv')
```


```python
np.__version__
```




    '1.19.2'




```python
data.shape
```




    (11914, 16)




```python
data.groupby(['Make']).count().sort_values('Model', ascending=False).head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Model</th>
      <th>Year</th>
      <th>Engine Fuel Type</th>
      <th>Engine HP</th>
      <th>Engine Cylinders</th>
      <th>Transmission Type</th>
      <th>Driven_Wheels</th>
      <th>Number of Doors</th>
      <th>Market Category</th>
      <th>Vehicle Size</th>
      <th>Vehicle Style</th>
      <th>highway MPG</th>
      <th>city mpg</th>
      <th>Popularity</th>
      <th>MSRP</th>
    </tr>
    <tr>
      <th>Make</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Chevrolet</th>
      <td>1123</td>
      <td>1123</td>
      <td>1123</td>
      <td>1117</td>
      <td>1121</td>
      <td>1123</td>
      <td>1123</td>
      <td>1123</td>
      <td>622</td>
      <td>1123</td>
      <td>1123</td>
      <td>1123</td>
      <td>1123</td>
      <td>1123</td>
      <td>1123</td>
    </tr>
    <tr>
      <th>Ford</th>
      <td>881</td>
      <td>881</td>
      <td>881</td>
      <td>868</td>
      <td>881</td>
      <td>881</td>
      <td>881</td>
      <td>881</td>
      <td>499</td>
      <td>881</td>
      <td>881</td>
      <td>881</td>
      <td>881</td>
      <td>881</td>
      <td>881</td>
    </tr>
    <tr>
      <th>Volkswagen</th>
      <td>809</td>
      <td>809</td>
      <td>809</td>
      <td>809</td>
      <td>805</td>
      <td>809</td>
      <td>809</td>
      <td>809</td>
      <td>585</td>
      <td>809</td>
      <td>809</td>
      <td>809</td>
      <td>809</td>
      <td>809</td>
      <td>809</td>
    </tr>
    <tr>
      <th>Toyota</th>
      <td>746</td>
      <td>746</td>
      <td>746</td>
      <td>744</td>
      <td>745</td>
      <td>746</td>
      <td>746</td>
      <td>746</td>
      <td>303</td>
      <td>746</td>
      <td>746</td>
      <td>746</td>
      <td>746</td>
      <td>746</td>
      <td>746</td>
    </tr>
    <tr>
      <th>Dodge</th>
      <td>626</td>
      <td>626</td>
      <td>626</td>
      <td>626</td>
      <td>626</td>
      <td>626</td>
      <td>626</td>
      <td>626</td>
      <td>320</td>
      <td>626</td>
      <td>626</td>
      <td>626</td>
      <td>626</td>
      <td>626</td>
      <td>626</td>
    </tr>
  </tbody>
</table>
</div>




```python
data.loc[data.Make=='Audi',['Model']].drop_duplicates().count()
```




    Model    34
    dtype: int64




```python
data.isna().sum()
```




    Make                    0
    Model                   0
    Year                    0
    Engine Fuel Type        3
    Engine HP              69
    Engine Cylinders       30
    Transmission Type       0
    Driven_Wheels           0
    Number of Doors         6
    Market Category      3742
    Vehicle Size            0
    Vehicle Style           0
    highway MPG             0
    city mpg                0
    Popularity              0
    MSRP                    0
    dtype: int64




```python
data["Engine Cylinders"].median()
```




    6.0




```python
data["Engine Cylinders"].mode()
```




    0    4.0
    dtype: float64




```python
data.loc[:,"Engine Cylinders"] = data.loc[:,"Engine Cylinders"].fillna(4)
```


```python
data["Engine Cylinders"].median()
```




    6.0




```python
X=data.loc[data.Make == 'Lotus',["Engine HP", "Engine Cylinders"]].drop_duplicates().values
```


```python
X
```




    array([[189.,   4.],
           [218.,   4.],
           [217.,   4.],
           [350.,   8.],
           [400.,   6.],
           [276.,   6.],
           [345.,   6.],
           [257.,   4.],
           [240.,   4.]])




```python
XTX=X.T.dot(X)
```


```python
XTX
```




    array([[7.31684e+05, 1.34100e+04],
           [1.34100e+04, 2.52000e+02]])




```python
np.linalg.inv(XTX)
```




    array([[ 5.53084235e-05, -2.94319825e-03],
           [-2.94319825e-03,  1.60588447e-01]])




```python
y=[1100, 800, 750, 850, 1300, 1000, 1000, 1300, 800]
```


```python
w=np.linalg.inv(XTX).dot(X.T)
w.dot(y)
```




    array([  4.59494481, -63.56432501])


