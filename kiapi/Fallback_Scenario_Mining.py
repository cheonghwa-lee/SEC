
"""
https://github.com/minsuk-heo/kaggle-titanic/blob/master/titanic-solution.ipynb
https://www.youtube.com/watch?v=aqp_9HV58Ls
"""
import pandas as pd
# import seaborn as sns
# sns.set()

# realmap
import folium
# ...
import numpy as np
import time

from preprocessing_fmrc import *


# 3. feature engineering
def feature_engineering(obj1, col1, obj2, col2):
    """
    1. shared_data()
    2. realmap()
    
    1. mapping (digitization)
    2. filtering
    3. sorting
    4. filling
    5. binning
    6. ranging
    7. digit placing
    """
    realmap_obu(obj1, col1)
    obj1, obj2, sd = shared_data(obj1, col1, obj2, col2)
    # realmap_obu(obj1, col1)
    realmap(obj1, col1, obj2, col2, sd)

def shared_data(obj1, col1, obj2, col2):
    """
    1. bitmap_f()
    2. extract()
    """
    bitmap1 = bitmap_f(obj1, col1)
    bitmap2 = bitmap_f(obj2, col2)
    bitmap = np.logical_and(bitmap1, bitmap2)
    sd = np.where(bitmap == True)
    sd = np.array(sd)
    sd = sd.T
    obj1 = extract(obj1, col1, sd)
    obj2 = extract(obj2, col2, sd)
    return obj1, obj2, sd

def bitmap_f(obj, col):
    bitmap = [[[[[False] * 60] * 60] * 12] * 2]
    bitmap = np.array(bitmap[0])
    apmlist = []
    hlist = []
    mlist = []
    slist = []
    apmhmslist = []
    for row in obj.index:
        apm = obj[col][row].split(' ')
        hms = apm[0].split(':')

        apm = apm[1]
        if apm == "AM":
            apm = 0
        elif apm == "PM":
            apm = 1
        else:
            raise
        h = int(hms[0])
        if h == 12:
            h = 0
        else:
            pass
        m = int(hms[1])
        s = int(hms[2])
        bitmap[apm][h][m][s] = True
        apmlist.append(apm)
        hlist.append(h)
        mlist.append(m)
        slist.append(s)
        apmhms = "{}{}{}{}".format(apm, h, m, s)
        apmhmslist.append(apmhms)
    obj['apm'] = apmlist
    obj['h'] = hlist
    obj['m'] = mlist
    obj['s'] = slist
    obj['apmhms'] = apmhmslist
    print("@@@@@@@@@@@@@@@@@@@@@@", obj.head)
    return bitmap

def extract(obj, col, sd):
    del_idx = []
    for row in obj.index:
        apm = obj[col][row].split(' ')
        hms = apm[0].split(':')

        apm = apm[1]
        if apm == "AM":
            apm = 0
        elif apm == "PM":
            apm = 1
        else:
            raise
        h = int(hms[0])
        if h == 12:
            h = 0
        else:
            pass
        m = int(hms[1])
        s = int(hms[2])

        cd = [apm, h, m, s]
        
        if cd in sd.tolist():
            pass
        else:
            del_idx.append(row)
    obj = obj.drop(index=del_idx)
    return obj

def realmap(obj1, col1, obj2, col2, sd):
    DEN = 10000000
    lat_init = obj1["latitude"][160120]/DEN
    lon_init = obj1["longitude"][160120]/DEN
    for sd_i in sd:
        print("******************************", sd_i[0], sd_i[1], sd_i[2], sd_i[3])
        apmhms = "{}{}{}{}".format(sd_i[0], sd_i[1], sd_i[2], sd_i[3])
        is_apmhms_1 = obj1.loc[obj1["apmhms"] == apmhms] 
        is_apmhms_2 = obj2.loc[obj2["apmhms"] == apmhms] 
        globals()['map_{}_{}_{}_{}'.format(sd_i[0], sd_i[1], sd_i[2], sd_i[3])] = folium.Map(location=[lat_init, lon_init], zoom_start=15)
        for row in is_apmhms_1.index:
            lat = obj1["latitude"][row]/DEN
            lon = obj1["longitude"][row]/DEN
            folium.Marker(location=[lat, lon], icon=folium.Icon(color="yellow", icon="star")).add_to(globals()['map_{}_{}_{}_{}'.format(sd_i[0], sd_i[1], sd_i[2], sd_i[3])])
            rsu_lat = obj1["rx_latitude"][row]/DEN
            rsu_lon = obj1["rx_longitude"][row]/DEN
            folium.Marker(location=[rsu_lat, rsu_lon], icon=folium.Icon(color="blue", icon="star")).add_to(globals()['map_{}_{}_{}_{}'.format(sd_i[0], sd_i[1], sd_i[2], sd_i[3])])
        for row in is_apmhms_2.index:    
            lat = obj2["detect_latitude"][row]/DEN
            lon = obj2["detect_longitude"][row]/DEN
            folium.Marker(location=[lat, lon], icon=folium.Icon(color="green", icon="star")).add_to(globals()['map_{}_{}_{}_{}'.format(sd_i[0], sd_i[1], sd_i[2], sd_i[3])])
        globals()['map_{}_{}_{}_{}'.format(sd_i[0], sd_i[1], sd_i[2], sd_i[3])].save('map_{}_{}_{}_{}.html'.format(sd_i[0], sd_i[1], sd_i[2], sd_i[3]))
        print("done")

def realmap_obu(obj1, col1):
    DEN = 10000000
    lat_init = obj1["latitude"][160120]/DEN
    lon_init = obj1["longitude"][160120]/DEN
    for row in obj1:
        # print("******************************", idx[0], idx[1], idx[2], idx[3])
        # apmhms = "{}{}{}{}".format(idx[0], idx[1], idx[2], idx[3])
        # is_apmhms_1 = obj1.loc[obj1["apmhms"] == apmhms] 
        # globals()['map_{}_{}_{}_{}'.format(idx[0], idx[1], idx[2], idx[3])] = folium.Map(location=[lat_init, lon_init], zoom_start=15)
        m = folium.Map(location=[lat_init, lon_init], zoom_start=15)
        for row in obj1.index:
            # print(row)
            lat = obj1["latitude"][row]/DEN
            lon = obj1["longitude"][row]/DEN
            # folium.Marker(location=[lat, lon], icon=folium.Icon(color="yellow", icon="star")).add_to(globals()['map_{}_{}_{}_{}'.format(idx[0], idx[1], idx[2], idx[3])])
            folium.Marker(location=[lat, lon], icon=folium.Icon(color="pink", icon="star")).add_to(m)
            rsu_lat = obj1["rx_latitude"][row]/DEN
            rsu_lon = obj1["rx_longitude"][row]/DEN
            # folium.Marker(location=[rsu_lat, rsu_lon], icon=folium.Icon(color="blue", icon="star")).add_to(globals()['map_{}_{}_{}_{}'.format(idx[0], idx[1], idx[2], idx[3])])
            folium.Marker(location=[rsu_lat, rsu_lon], icon=folium.Icon(color="blue", icon="star")).add_to(m)
            # folium.Marker(location=[lat, lon], icon=folium.Icon(color="green", icon="star")).add_to(globals()['map_{}_{}_{}_{}'.format(idx[0], idx[1], idx[2], idx[3])])
        # globals()['obu_map_{}_{}_{}_{}'.format(idx[0], idx[1], idx[2], idx[3])].save('map_{}_{}_{}_{}.html'.format(idx[0], idx[1], idx[2], idx[3]))
        m.save("obu_map.html")
        print("done")

"""
Types of variable
1. binary
2. category
3. integer
4. floats
"""

# 4. clustering
from haversine import haversine
# haversine(point1, point2, unit=m)
# https://stricky.tistory.com/284


def main():
    # 0. data load
    obu = pd.read_csv("PVD_200623_obu_state.csv")
    rsu_g1 = pd.read_csv("RSA_200623_rsu_accidentstate.csv")
    rsu_g2 = pd.read_csv("SPaT_200623_rsu_signalstate.csv")

    # 1. pre-processing
    obu = pre_processing_obu(obu)
    rsu_g1 = pre_processing_rsu1(rsu_g1)
    # rsu_g2 = pre_processing(rsu_g2)

    # 3. feature engineering
    feature_engineering(obu, "msg_received_time", rsu_g1, "msg_received_time")

    # 4. clustering

    # 5. post-processing

if __name__ == "__main__":
    main()