
from haversine import haversine, inverse_haversine, Direction, Unit
from math import pi, radians, cos, sin, asin, sqrt
import numpy as np
def haversine_distance(lat1, lat2, lon1, lon2, in_meters = True):
     
    lon1 = radians(lon1)
    lon2 = radians(lon2)
    lat1 = radians(lat1)
    lat2 = radians(lat2)
        
    # Haversine formula
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = sin(dlat / 2)**2 + cos(lat1) * cos(lat2) * sin(dlon / 2)**2
   
    c = 2 * asin(sqrt(a))
        
    # Radius of earth in kilometers
    r = 6371
        
    # calculate the result
    res = c*r
    if in_meters :
        return 1000*res
    else:
        return res

def latlon_cart_2(p1,p2):
    x = 6371 * np.cos(np.radians(p1[0])) * np.cos(np.radians(p1[1]))
    y = 6371 * np.cos(np.radians(p1[0])) * np.sin(np.radians(p1[1]))
                
    x1 = 6371 * np.cos(np.radians(p2[0])) * np.cos(np.radians(p2[1]))
    y1 = 6371 * np.cos(np.radians(p2[0])) * np.sin(np.radians(p2[1]))

    temp = np.array((x,y))-np.array((x1,y1))
    dist1 = np.sqrt(np.dot(temp.T,temp))*1000
    return dist1
    
p1 = [55.6528,12.488184]
    
m = 10

psouth = inverse_haversine(p1,m,Direction.SOUTH,unit=Unit.METERS)
pwest = inverse_haversine(p1,m,Direction.WEST,unit=Unit.METERS)
psouthwest = inverse_haversine(p1,m,pi * 1.25,unit=Unit.METERS)
    
dist1south = latlon_cart_2(p1,psouth)
dist2south = haversine_distance(p1[0],psouth[0],p1[1],psouth[1])
dist1west = latlon_cart_2(p1,pwest)
dist2west = haversine_distance(p1[0],pwest[0],p1[1],pwest[1])
dist1southwest = latlon_cart_2(p1,psouthwest)
dist2southwest = haversine_distance(p1[0],psouthwest[0],p1[1],psouthwest[1])
    
print('South diff:',abs(dist1south-dist2south))
print('West diff:',abs(dist1west-dist2west))
print('Southwest diff:',abs(dist1southwest-dist2southwest))

# 4th decimal change lonitude
haversine([55.6606,12.48355],[55.6606,12.48365],unit=Unit.METERS)
latlon_cart_2([55.6606,12.48355],[55.6606,12.48365])

# 4th decimal change latitude
haversine([55.6606,12.48355],[55.6607,12.48355],unit=Unit.METERS)
latlon_cart_2([55.6606,12.48355],[55.6607,12.48355])














