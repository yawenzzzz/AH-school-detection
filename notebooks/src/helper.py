import math
from math import radians, cos, sin, asin, sqrt
import random

def point_distance(lon1, lat1, lon2, lat2):
    """
    Calculate the great circle distance between two points
    on the earth (specified in decimal degrees)
    """
    # convert decimal degrees to radians
    if (lon1 != None) and (lat1 != None) and (lon2 != None) and (lat2 != None): # can be sub_id without lat/lon
        lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
        # haversine formula
        dlon = lon2 - lon1
        dlat = lat2 - lat1
        a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
        c = 2 * asin(sqrt(a))
        r = 6371 # Radius of earth in kilometers. Use 3956 for miles
        return c * r

def divide_train_test(filename_list):
    """
    separate into train & test files
    """
    test_filenames = ['D20190927-T072325', 'D20191016-T184753', 'D20191016-T213424', 'D20191018-T081659', 'D20191018-T110329', 'D20191020-T145420', 'D20191024-T103607', 'D20191024-T172924', 'D20191102-T144417', 'D20191102-T160647'] # manually selected (10)
    random.seed(0)

    other_test_filenames = random.sample(list(set(filename_list) - set(test_filenames)), k=40) # 50 in total
    # get train and test
    test_filenames = test_filenames + other_test_filenames
    train_filenames = [i for i in filename_list if i not in test_examples]

    return train_filenames, test_filenames
