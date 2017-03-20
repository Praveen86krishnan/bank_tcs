# -*- coding: utf-8 -*-

import numpy
from math import radians, cos, sin, asin, sqrt, pow, exp
import pickle as pk
from PIL import Image

def  get_pressure_from_elev_m(elev): # elev in meters
    '''
    #calculate pressure at give elevation (https://en.wikipedia.org/wiki/Density_of_air#Altitude)
    # dataset['b_elev'] gives elevation in meters
    p = []
    for h in range(1,10000,10):
        p.append(p0*(pow((1-(B*h)),A)))
    h = range(1,10000,10)
    plt.plot(p,h)
    plt.show()
    Looks similar to the graph
    '''
    p0 = 101325.00 #Pa
    L = 0.0065 #K/m
    T0 = 288.15 #K
    g = 9.80665 #m/s2
    M = 0.0289644 #kg/mol
    R = 8.31447 #J/(mol*K)
    A = (g*M)/(R*L)
    B = L/T0
    return (p0*(pow((1-(B*elev)),A))/100.00)


def cal_RH(temp):
    model_name = 'predict_dewpoint'
    reg = pk.load(open(model_name, 'r'))
    dewpt = reg.predict(temp)
    return 100*((exp((17.625*dewpt)/(243.04+dewpt)))/(exp((17.625*temp)/(243.04+temp))))


def dist_cal(lon1, lat1, lon2, lat2):
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * asin(sqrt(a))
    km = 6367 * c
    return km


def get_elev(lat,lon):
    #get the pixcel values

    im = Image.open('gebco_08_rev_elev_D2_grey_geo.tif')
    imarray = numpy.array(im)
    deg_per_pixcel = 90 / float(imarray.shape[0])
    pulse_v = imarray[int(abs(lat/deg_per_pixcel)), int(abs((lon-90)/deg_per_pixcel))]

    model_name = 'calibrate_elev'
    reg = pk.load(open(model_name, 'r'))
    return reg.predict(pulse_v)




def find_nearest_city(lat,lon):
    file_name = 'cities'
    cities = pk.load(open(file_name, 'r'))
    distance = 4e10
    out = ''
    for index, row in cities.iterrows():
        temp = dist_cal(lon,lat,row['b_lon'],row['b_lat'])
        if temp < distance:
            distance = temp
            out = row['b_name']

    return out



def get_temp(lat,lon,month,day,elev):
    model_name = 'predict_temp'
    reg = pk.load(open(model_name, 'r'))
    ses = 1 if month in (1, 2, 12) else -1 if month in (6, 7) else 0
    X = numpy.array([ses,month, day, lat, lon, elev])
    return reg.predict(X)


def get_cond(X):
    temp_d = {'Sn' : 'Snow',
              'Ra' : 'Rain',
              'Ss' : 'Sunny'}
    model_name = 'predict_weather_cond'
    classfier = pk.load(open(model_name, 'r'))
    # X = numpy.array([month,temp,lat,lon,ele])
    return temp_d[classfier.predict(X)[0]]

