import random
from datetime import timedelta, date, datetime, time
import fns


# Set the initial values where the character starts from
# t - 1 point & time # using sydney cordinates
lat = -33.86
lon = 151.21
dt = date.today()

# Assuming the char speed of travel ( kilometers in days)
Speed_of_char_kms_per_day = 50


# Using Random fun to see at which lat & long he reachs next
while 1:

    to_print = ''

    dec_lat = random.random() *  random.choice([-1,1])      # 10 *
    dec_lon = random.random() * random.choice([-1,1])       # 10 *
    distnace = fns.dist_cal(lon,lat,lon+dec_lat,lat+dec_lat)

    Time = distnace / float(Speed_of_char_kms_per_day)  # days

    t = dt + timedelta(days=Time)

    lat = lat + dec_lat
    lon = lon + dec_lon
    dt = t
    if lat <= -90:
        lat = -70
    if lon >= 180:
        lon = 170

    #finding closest point from training data
    to_print = fns.find_nearest_city(lat,lon)+'|' # city Name
    temp = '%.2f,%.2f,%.0f|' % (lat,lon,fns.get_elev(lat,lon))
    to_print = to_print + temp
    temp_dt = dt + timedelta(minutes=random.randrange(1,(60*24)))
    temp = datetime.combine(dt, time(random.randrange(9,18),random.randrange(0,60),random.randrange(0,60)))
    temp = '%s|' % temp.strftime('%Y-%m-%dT%H:%M:%SZ')
    to_print = to_print + temp
    ele = fns.get_elev(lat,lon)[0][0]
    t = fns.get_temp(lat,lon,dt.month,dt.day,ele)[0]
    pre = fns.get_pressure_from_elev_m(ele)
    to_print  = to_print+ fns.get_cond((dt.month,t,lat,lon,ele)) + '|%.1f|%.1f|%.0f' % (t, pre,fns.cal_RH(t))
    print to_print

