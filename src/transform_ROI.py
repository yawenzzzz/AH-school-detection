import os
import numpy as np
from datetime import datetime, timedelta
import cv2

class ROITransformer(object):
    """
    TRransform each ROI to depth/time ranges, depth/time coords
    Input: contour, depth, time
    Output: point count, left x (date, time), top y (upper depth), right x, bottom y, [Point 1, ..., Point N]
    """
    def __init__(self, contour=None, depth=None, time=None):
        self.contour = contour # (#points, 1, xy indices)
        self.depth = depth
        self.time = time
        self.offset = self.depth[0] - 6 # minus 6 meters

    def __call__(self):
        # get bbox
        x, y, w, h = cv2.boundingRect(self.contour) # minAreaRect (rotated rec)
        x_min = x
        x_max = x + w
        y_min = y
        y_max = y + h # can outside boundary

        left_datetime = self.time[x_min].astype(datetime)
        right_datetime = self.time[x_max].astype(datetime)
        top_depth = self.depth[y_min] - self.offset
        bottom_depth = self.depth[y_max] - self.offset
        
        # formating: 20190925 1749451605 22.0286015595
        left_date = left_datetime.strftime('%Y%m%d') # to string
        left_time = left_datetime.strftime('%H%M%S%f')[:10] # %f: Microsecond as a decimal number, zero-padded on the left.
        right_date = right_datetime.strftime('%Y%m%d')
        right_time = right_datetime.strftime('%H%M%S%f')[:10]

        bbox_points = [left_date, left_time, top_depth, right_date, right_time, bottom_depth]

        # get polygon
        x_indices = self.contour[:, 0, 0] # 1D
        y_indices = self.contour[:, 0, 1]
        mask_points = []
        for i, j in zip(x_indices, y_indices):
            i_date = self.time[i].astype(datetime).strftime('%Y%m%d')
            i_time = self.time[i].astype(datetime).strftime('%H%M%S%f')[:10]
            j_depth = self.depth[j] - self.offset
            mask_points.append([i_date, i_time, j_depth])

        point_count = len(mask_points)

        return point_count, bbox_points, mask_points
