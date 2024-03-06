import os
import numpy as np
import pandas as pd
import cv2
from scipy import signal
import math
from shapely.geometry import Polygon
from .helper import point_distance

class FeatureExtractor(object):
    """
    Extract features of ROIs, validate each ROI (polygon or not)
    Input: ROIs (contours), npy, depth, time, positions
    Output: valid ROIs, ROI features, save ROI figures
    """
    def __init__(self, filename=None, contours=None, Sv_npy=None, bottom_idx=None, time=None, depth=None, positions=None):
        self.filename = filename
        self.contours = contours
        self.Sv_npy = Sv_npy
        self.bottom_idx = bottom_idx
        self.time = time
        self.depth = depth
        self.positions = positions

    def __call__(self):
        # select ROIs
        contours_sel = []
        contours_features = []

        for contour in self.contours:
            try:
                x, y, w, h = cv2.boundingRect(contour)
                # geographic features
                center_x, center_y = int(x + w/2.0), int(y + h/2.0)
                total_water_column = self.depth[self.bottom_idx[center_x]]
                median_depth = self.depth[center_y]
                relative_altitude = float(median_depth) / total_water_column # 1: sea floor, 0: sea surface
                latitude = self.positions['latitude'][center_x]
                longitude = self.positions['longitude'][center_x]
                # time features
                median_time = self.time[center_x]
            except:
                continue
#             # filter, for train_new (w <= 2 or h <= 2)
#             if w <= 2 or h <= 2:
#                     continue
            # if success, get contextual features!
            contour_dict = {'total_water_column': total_water_column, 'depth': median_depth, 'relative_altitude': relative_altitude, 'latitude': latitude, 'longitude': longitude, 'time': median_time}            
            contours_sel.append(contour)
            contours_features.append(contour_dict)

        return contours_sel, contours_features
