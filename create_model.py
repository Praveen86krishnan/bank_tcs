# Models are created in this file, need to be run only on refresh of the training data.
# Or if the image changes

import  pandas as pd
import numpy
from math import radians, cos, sin, asin, sqrt, pow
import random
from datetime import timedelta, date
import pickle as pk
from PIL import Image
import itertools
import matplotlib.pyplot as plt


dataset = pd.read_csv('AUSWEA.csv')

# F -> C
dataset['a_temp'] = (dataset['a_temp'] - 32)/(1.8)
dataset['a_dewp'] = (dataset['a_dewp'] - 32)/(1.8)


#using image to calibrate elevation in meters
im = Image.open('gebco_08_rev_elev_D2_grey_geo.tif')
# im.show()
imarray = numpy.array(im)
# imarray.shape
# IMage D2 is 90 Degree east to west and 90 north to south
# Upper left	Lower right
# 0N 90E	90S 180E
# each pixel is 90/float(imarray.shape[0]) degrees
deg_per_pixcel = 90/float(imarray.shape[0])

print 'reading the elev details from image & from dataset...'
learn_data = []
for i in range(len(dataset)):
    lat,lon,elev = dataset.iloc[i,20:].values
    # print imarray[int(abs(lat/deg_per_pixcel)),int(abs((lon-90)/deg_per_pixcel))]
    learn_data.append([imarray[int(abs(lat/deg_per_pixcel)),int(abs((lon-90)/deg_per_pixcel))] , float(elev)])

# removing duplicates
learn_data.sort()
t = list(t for t,_ in itertools.groupby(learn_data))

t = numpy.array(t)
X = numpy.array(t[:,0]).reshape(t.shape[0],1)
y = numpy.array(t[:,1]).reshape(t.shape[0],1)

print 'Creating Model....Using Linear Regression (Elevation & Pulse Reading)'
# using simple regression
from sklearn.linear_model import LinearRegression
reg = LinearRegression()
reg.fit(X,y)
model_name = 'calibrate_elev'
pk.dump(reg,open(model_name,'wb'))

print 'Saved Model as %s' % model_name



# regression for temp & dewp
# model 1 - temp

X = dataset.iloc[:,[3,4,20,21,22]].values
y = dataset.iloc[:,5].values

X = numpy.append(numpy.array([1 if x in (1,2,12) else -1 if x in (6,7) else 0 for x in list(dataset['a_mo'])]).reshape(X.shape[0],1),values=X,axis=1)
print 'Creating Model....For Predicting temp, Using Linear Regression'

from sklearn.linear_model import LinearRegression
reg = LinearRegression()
reg.fit(X,y)

model_name = 'predict_temp'
pk.dump(reg,open(model_name,'wb'))

print 'Saved Model as %s' % model_name



## Classification
# predicting weather condition

X = dataset.iloc[:,[3,5,20,21,22]].values

weat = []
sn_i = 0
ra_i = 0
ss_i = 0
for i in range(dataset.shape[0]):
    if dataset['a_snow_ice_pellets'][i] == 1:
        weat.append('Sn')
        sn_i = sn_i + 1
    elif dataset['a_rain_drizzle'][i] == 1:
        weat.append('Ra')
        ra_i = ra_i + 1
    else:
        weat.append('Ss')
        ss_i = ss_i + 1
n = min(ss_i,sn_i,ra_i)
y = numpy.array(weat).reshape(dataset.shape[0],1)

X_temp = numpy.append(y,values=X,axis=1)
X_fin = random.sample(X_temp[X_temp[:,0] == 'Sn'],n)
X_fin = numpy.append(random.sample(X_temp[X_temp[:,0] == 'Ra'],n),values=X_fin,axis=0)
X_fin = numpy.append(random.sample(X_temp[X_temp[:,0] == 'Ss'],n),values=X_fin,axis=0)

y = X_fin[:,0]
X = X_fin[:,1:].astype(float)


from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators=100,
                                    criterion='entropy')

classifier.fit(X,y)

model_name = 'predict_weather_cond'
pk.dump(classifier,open(model_name,'wb'))

print 'Saved Model as %s' % model_name


# model 2 - dewpoint

t = dataset[dataset.a_dewp < 110]['a_temp'].reshape(dataset[dataset.a_dewp < 110].shape[0],1)
d = dataset[dataset.a_dewp < 110]['a_dewp'].reshape(dataset[dataset.a_dewp < 110].shape[0],1)

from sklearn.linear_model import LinearRegression
reg = LinearRegression()
reg.fit(t,d)

model_name = 'predict_dewpoint'
pk.dump(reg,open(model_name,'wb'))

print 'Saved Model as %s' % model_name





# creating city lists with lat & long

city = dataset[['b_name','b_lat','b_lon']].drop_duplicates()
file_name = 'cities'
pk.dump(city,open(file_name,'wb'))

print 'Saved file as %s' % file_name

