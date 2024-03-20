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
        # apply median filter (4 * x * y)
        Sv_npy = np.array([signal.medfilt2d(i, 3) for i in self.Sv_npy])

        # convert xy index into Unit meters!
        x_dim = Sv_npy.shape[1]
        y_dim = Sv_npy.shape[2]
        x_latlon = [[self.positions['longitude'][i], self.positions['latitude'][i]] for i in range(x_dim)]
        x_latlon_meters = []
        for i in range(x_dim):
            if i == 0:
                x_latlon_meters.append(0)
            else:
                if np.isnan(point_distance(x_latlon[i][0], x_latlon[i][1], x_latlon[i-1][0], x_latlon[i-1][1])):
                    x_latlon_meters.append(x_latlon_meters[i-1]) # use last
                else:
                    x_latlon_meters.append(x_latlon_meters[i-1] + point_distance(x_latlon[i][0], x_latlon[i][1], x_latlon[i-1][0], x_latlon[i-1][1]) * 1000) # accumulate meters
        # convert y
        y_depth_meters = [self.depth[j] for j in range(y_dim)]

        # select ROIs
        contours_sel = []
        contours_features = []

        for contour in self.contours:
            try:
                # acoustic features
                img_contours = np.zeros([y_dim, x_dim], dtype=np.int32)
                cv2.drawContours(img_contours, [contour] , -1, (1), thickness=-1)
                xy_indices = np.nonzero(img_contours) # y, x
                xy_indices = [(i, j) for i, j in zip(xy_indices[1], xy_indices[0])] # switch x/y
                Sv_values = np.array([Sv_npy[:, i, j] for i, j in xy_indices])
                perc_li = [0, 5, 25, 50, 75, 95, 100]
                Sv_features = []
                for i in range(4):
                    Sv_values_freq = Sv_values[:, i]
                    Sv_features_freq = []
                    for perc in perc_li:
                        Sv_features_freq.append(np.percentile(Sv_values_freq, perc))
                    # add std
                    Sv_features_freq.append(np.std(Sv_values_freq))
                    Sv_features.append(Sv_features_freq)
                # add relative dif (to 38kHz)
                for i in [0, 2, 3]:
                    Sv_features.append(Sv_features[i][3] / float(Sv_features[1][3])) # perc 50

                # geometric features
                x, y, w, h = cv2.boundingRect(contour)
                x_min, x_max = x, x+w
                y_min, y_max = y, y+h
                length = x_latlon_meters[x_max] - x_latlon_meters[x_min]
                thickness = self.depth[y_max] - self.depth[y_min]
#                 # remove tiny objects! thickness - width, key parameter
#                 if thickness < 2 or length < 1:
#                     continue
                
                x_li = contour[:, 0, 0]
                y_li = contour[:, 0, 1]
                xy_meters = [(x_latlon_meters[i], y_depth_meters[j]) for i, j in zip(x_li, y_li)]
                # form polygon
                polygon = Polygon(xy_meters)
                area = polygon.area
                perimeter = polygon.length
                compact = float(perimeter) / area
                rectangularity = float(length * thickness) / area
                circularity = float(4 * math.pi * area) / (perimeter * perimeter)
                elongation = float(length) / thickness

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

            # if success, get features + crop echogram
            contour_dict = {'filename': self.filename, 'Sv_18kHz_min': Sv_features[0][0], 'Sv_18kHz_p5': Sv_features[0][1], 'Sv_18kHz_p25': Sv_features[0][2], 'Sv_18kHz_p50': Sv_features[0][3], 'Sv_18kHz_p75': Sv_features[0][4], 'Sv_18kHz_p95': Sv_features[0][5], 'Sv_18kHz_max': Sv_features[0][6], 'Sv_18kHz_std': Sv_features[0][7], 'Sv_38kHz_min': Sv_features[1][0], 'Sv_38kHz_p5': Sv_features[1][1], 'Sv_38kHz_p25': Sv_features[1][2], 'Sv_38kHz_p50': Sv_features[1][3], 'Sv_38kHz_p75': Sv_features[1][4], 'Sv_38kHz_p95': Sv_features[1][5], 'Sv_38kHz_max': Sv_features[1][6], 'Sv_38kHz_std': Sv_features[1][7], 'Sv_120kHz_min': Sv_features[2][0], 'Sv_120kHz_p5': Sv_features[2][1], 'Sv_120kHz_p25': Sv_features[2][2], 'Sv_120kHz_p50': Sv_features[2][3], 'Sv_120kHz_p75': Sv_features[2][4], 'Sv_120kHz_p95': Sv_features[2][5], 'Sv_120kHz_max': Sv_features[2][6], 'Sv_120kHz_std': Sv_features[2][7], 'Sv_200kHz_min': Sv_features[3][0], 'Sv_200kHz_p5': Sv_features[3][1], 'Sv_200kHz_p25': Sv_features[3][2], 'Sv_200kHz_p50': Sv_features[3][3], 'Sv_200kHz_p75': Sv_features[3][4], 'Sv_200kHz_p95': Sv_features[3][5], 'Sv_200kHz_max': Sv_features[3][6], 'Sv_200kHz_std': Sv_features[3][7], 'Sv_ref_18kHz': Sv_features[4], 'Sv_ref_120kHz': Sv_features[5], 'Sv_ref_200kHz': Sv_features[6], 'length': length, 'thickness': thickness, 'area': area, 'perimeter': perimeter, 'rectangularity': rectangularity, 'compact': compact, 'circularity': circularity, 'elongation': elongation, 'total_water_column': total_water_column, 'depth': median_depth, 'relative_altitude': relative_altitude, 'latitude': latitude, 'longitude': longitude, 'time': median_time}
            
            contours_sel.append(contour)
            contours_features.append(contour_dict)

        return contours_sel, contours_features
