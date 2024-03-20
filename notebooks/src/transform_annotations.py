import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

class AnnotationTransformer(object):
    """
    Transform annotation mask (one echogram) into x/y coords
    Input: annotations (with mask), filename, depth, time
    Output: mask x/y indices
    """
    def __init__(self, annotations=None, filename=None, depth=None, time=None, label_map=None):
        self.annotations = annotations
        self.filename = filename
        self.depth = depth
        self.time = time
        self.label_map = label_map # {'Unclassified regions': 1, 'krill_schools': 2, 'fish_school': 3, 'AH_School': 4}
        self.offset = self.depth[0] - 6 # original offset for 6m

    def __call__(self):
        # select annotations
        annotations_sel = self.annotations[self.annotations['file_dir'].str.contains(self.filename)]        
            
        all_annotations = []
        all_labels = []
        for row_idx, row in annotations_sel.iterrows():
            label = row['label']
        
            # only schools
            if label not in self.label_map:
                continue

            mask = row['mask'].split(' ')
            num_points = len(mask) // 3 # date, time, depth
            points = []
            for point_idx in range(num_points):
                Date = mask[point_idx * 3 + 0]
                Time = mask[point_idx * 3 + 1]
                Depth = float(mask[point_idx * 3 + 2]) + self.offset # replace!
                Date_Time = str(Date) + str(Time).zfill(10)
                Date_Time_dt = datetime.strptime(Date_Time, '%Y%m%d%H%M%S%f')
                Date_Time_ns = np.datetime64(Date_Time_dt)
                # find nearest point
                x_idx = (np.where(self.time == min(self.time, key = lambda x: abs(x - Date_Time_ns)))[0]).tolist()[0]
                # correction: 6 meters buffer
                y_idx = (np.where(self.depth == min(self.depth, key = lambda y: abs(y - Depth)))[0]).tolist()[0]
                points.append([[x_idx, y_idx]]) # CHECK

            all_annotations.append(np.array(points))
            all_labels.append(self.label_map[label])

        return np.array(all_annotations, dtype=object), all_labels # #annotations, #points, 1, xy indices
