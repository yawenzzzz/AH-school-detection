import os
import numpy as np
import pandas as pd
import cv2

class OverlapAnnotation(object):
    """
    Overlap contours (one echogram) and annotations (transformed)
    Input: all annotations, filename, contours
    Output: overlap ratio, contour labels
    """
    def __init__(self, filename=None, img_shape=None, annotations=None, labels=None, contours=None, fig_dir=None):
        self.filename = filename
        self.img_shape = img_shape
        self.annotations = annotations # mask points
        self.labels = labels # annotation labels (1, 2, 3, 4)
        self.contours = contours
        self.fig_dir = fig_dir
    
    def final_overlap_metrics(self):
        """
        final metrics
        """
        # for contours
        img_contours = np.zeros(self.img_shape, dtype=np.int32)
        cv2.drawContours(img_contours, self.contours, -1, (1), thickness=-1)
        
        # for annotations, only AH schools
        img_annotations = np.zeros(self.img_shape, dtype=np.int32)
        annotations_sel = self.annotations[[idx for idx, i in enumerate(self.labels) if i == 4]]
        try:
            cv2.drawContours(img_annotations, annotations_sel, -1, (1), thickness=-1) # thickness = -1, fill
        except:
            pass
        
        # positive overlap
        intersection = np.logical_and(img_annotations, img_contours)
        if np.sum(img_annotations) == 0:
            pos_overlap = float("nan") # remove later
        else:
            pos_overlap = np.sum(intersection) / np.sum(img_annotations)
        
        # for annotations, other classes
        img_annotations = np.zeros(self.img_shape, dtype=np.int32)
        annotations_sel = self.annotations[[idx for idx, i in enumerate(self.labels) if i != 4]]
        try:
            cv2.drawContours(img_annotations, annotations_sel, -1, (1), thickness=-1) # thickness = -1, fill
        except:
            pass
        
        # negative overlap
        intersection = np.logical_and(img_annotations, img_contours)
        if np.sum(img_annotations) == 0:
            neg_overlap = float("nan")
        else:
            neg_overlap = np.sum(intersection) / np.sum(img_annotations)
        
        return pos_overlap, neg_overlap
    
    def pixel_overlap_metrics(self):
        """
        pixel-wise, recall & precision, iou score
        """
        if 4 not in self.labels:
            return float("nan"), float("nan"), float("nan")
        
        # for annotations
        img_annotations = np.zeros(self.img_shape, dtype=np.int32)
        # only AH schools
        annotations_sel = self.annotations[[idx for idx, i in enumerate(self.labels) if i == 4]]
        cv2.drawContours(img_annotations, annotations_sel, -1, (1), thickness=-1) # thickness = -1, fill
        
        # for contours
        img_contours = np.zeros(self.img_shape, dtype=np.int32)
        cv2.drawContours(img_contours, self.contours, -1, (1), thickness=-1)

        # pixel intersections
        intersection = np.logical_and(img_annotations, img_contours)
        union = np.logical_or(img_annotations, img_contours)
        if np.sum(img_annotations) == 0:
            recall = float("nan")
        else:
            recall = np.sum(intersection) / float(np.sum(img_annotations))
        if np.sum(img_contours) == 0:
            precision = float("nan")
        else:
            precision = np.sum(intersection) / float(np.sum(img_contours))
        if np.sum(union) == 0:
            iou_score = float("nan")
        else:
            iou_score = np.sum(intersection) / float(np.sum(union))
        
        if self.fig_dir != None:
            cv2.drawContours(img_annotations, annotations_sel, -1, (255), thickness=1) # thickness = -1, fill
            cv2.imwrite(self.fig_dir + f'{self.filename}_annotations.png', img_annotations)

        return recall, precision, iou_score
        
    def object_overlap_metrics(self, threshold):
        """
        object-wise, recall & precision, when overlap ratio >= threshold
        """ 
        # draw all contours
        if self.fig_dir != None:
            img_contours = np.zeros(self.img_shape, dtype=np.int32)
            cv2.drawContours(img_contours, self.contours, -1, (255), thickness=1) 
            cv2.imwrite(self.fig_dir + f'{self.filename}_contours.png', img_contours)
              
        # draw all AH schools
        annotations_sel = self.annotations[[idx for idx, i in enumerate(self.labels) if i == 4]]
        if self.fig_dir != None:
            img_annotations = np.zeros(self.img_shape, dtype=np.int32)
            cv2.drawContours(img_annotations, annotations_sel, -1, (255), thickness=1) # thickness = -1, fill
            cv2.imwrite(self.fig_dir + f'{self.filename}_annotations.png', img_annotations)
        
        # draw all contours
        img_contours = np.zeros(self.img_shape, dtype=np.int32)
        cv2.drawContours(img_contours, self.contours, -1, (1), thickness=-1)
        
        # check annotations one by one
        count = 0
        for annotation in annotations_sel:
            img_annotations = np.zeros(self.img_shape, dtype=np.int32)
            cv2.drawContours(img_annotations, [annotation], -1, (1), thickness=-1) # thickness = -1, fill
            # overlap
            intersection = np.logical_and(img_annotations, img_contours)
            if np.sum(intersection) / np.sum(img_annotations) > threshold:
                count += 1
        recall = float(count) / len(annotations_sel)
        
        # draw all annotations
        img_annotations = np.zeros(self.img_shape, dtype=np.int32)
        cv2.drawContours(img_annotations, annotations_sel, -1, (1), thickness=-1) # thickness = -1, fill
        
        # check contours one by one
        count = 0
        for contour in self.contours:
            img_contours = np.zeros(self.img_shape, dtype=np.int32)
            cv2.drawContours(img_contours, [contour], -1, (1), thickness=-1)
            # overlap
            intersection = np.logical_and(img_annotations, img_contours)
            if np.sum(intersection) / np.sum(img_contours) > threshold:
                count += 1
        precision = float(count) / (len(self.contours) + 0.00000001)
        
        return recall, precision
    
    def object_overlap_count(self, threshold):
        """
        object-wise, get count only, when overlap ratio >= threshold
        """ 
#         # draw all contours
#         if self.fig_dir != None:
#             img_contours = np.zeros(self.img_shape, dtype=np.int32)
#             cv2.drawContours(img_contours, self.contours, -1, (255), thickness=1) 
#             cv2.imwrite(self.fig_dir + f'{self.filename}_contours.png', img_contours)
              
        # draw all AH schools
        annotations_sel = self.annotations[[idx for idx, i in enumerate(self.labels) if i == 4]]
#         if self.fig_dir != None:
#             img_annotations = np.zeros(self.img_shape, dtype=np.int32)
#             cv2.drawContours(img_annotations, annotations_sel, -1, (255), thickness=1) # thickness = -1, fill
#             cv2.imwrite(self.fig_dir + f'{self.filename}_annotations.png', img_annotations)
        
        # draw all contours
        img_contours = np.zeros(self.img_shape, dtype=np.int32)
        cv2.drawContours(img_contours, self.contours, -1, (1), thickness=-1)
        
        # check annotations one by one
        count = 0
        for annotation in annotations_sel:
            img_annotations = np.zeros(self.img_shape, dtype=np.int32)
            cv2.drawContours(img_annotations, [annotation], -1, (1), thickness=-1) # thickness = -1, fill
            # overlap
            intersection = np.logical_and(img_annotations, img_contours)
            if np.sum(intersection) / float(np.sum(img_annotations)) >= threshold:
                count += 1
        annotations_valid = count
        annotations_all = len(annotations_sel)
        
#         # draw all annotations
#         img_annotations = np.zeros(self.img_shape, dtype=np.int32)
#         cv2.drawContours(img_annotations, annotations_sel, -1, (1), thickness=-1) # thickness = -1, fill
        
#         # check contours one by one
#         count = 0
#         for contour in self.contours:
#             img_contours = np.zeros(self.img_shape, dtype=np.int32)
#             cv2.drawContours(img_contours, [contour], -1, (1), thickness=-1)
#             # overlap
#             intersection = np.logical_and(img_annotations, img_contours)
#             if np.sum(intersection) / np.sum(img_contours) >= threshold:
#                 count += 1
#         roi_valid = count
#         roi_all = len(self.contours)
        
        # return annotations_valid, annotations_all, roi_valid, roi_all
        return annotations_valid, annotations_all

    def assign_label(self, threshold):
        """
        pixel-wise, assign label to contours
        threshold: % of certain pixels to assign a label
        """
        # plot annotations by label first
        img_annotations = np.zeros(self.img_shape, dtype=np.int32)
        for val in range(1, 5):
            annotations_sel = self.annotations[[idx for idx, i in enumerate(self.labels) if i == val]]
            try:
                cv2.drawContours(img_annotations, annotations_sel, -1, (val), thickness=-1)
            except:
                continue # not all value included
        # get annotations with 1, 2, 3, 4, correct!

        contours_labels = [0] * len(self.contours)
        for contour_idx, contour in enumerate(self.contours):
            img_contours = np.zeros(self.img_shape, dtype=np.int32)
            cv2.drawContours(img_contours, [contour], -1, (1), thickness=-1)
            total_pixel = np.sum(img_contours)
            # overlap pixel-wise
            res = img_contours * img_annotations
            values, counts = np.unique(res, return_counts=True)
            for idx, val in enumerate(values):
                if float(counts[idx]) / total_pixel > threshold:
                    contours_labels[contour_idx] = val # assign label
        return contours_labels
    
    def compute_iou(self, threshold):
        """
        object-wise (find corresponding annotations), assign labels to contours
        threshold: minimum iou
        """
        contours_labels = [0] * len(self.contours)
        for contour_idx, contour in enumerate(self.contours):
            # draw contours
            img_contours = np.zeros(self.img_shape, dtype=np.int32)
            cv2.drawContours(img_contours, [contour], -1, (1), thickness=-1)
            # get boundary
            try:
                x1, y1, w1, h1 = cv2.boundingRect(contour)
            except:
                continue
            x1_min, x1_max = x1, x1+w1
            y1_min, y1_max = y1, y1+h1
            # draw annotations
            for idx, annotation in enumerate(self.annotations):
                # early stopping (if no overlap!)
                try:
                    x2, y2, w2, h2 = cv2.boundingRect(annotation)
                except:
                    continue
                x2_min, x2_max = x2, x2+w2
                y2_min, y2_max = y2, y2+h2
                if x1_max <= x2_min or x1_min >= x2_max or y1_max <= y2_min or y1_min >= y2_max:
                    continue               
                img_annotations = np.zeros(self.img_shape, dtype=np.int32)
                # get label
                val = self.labels[idx]
                cv2.drawContours(img_annotations, [annotation], -1, (1), thickness=-1) # thickness = -1, fill
                # check overlap 
                intersection = np.logical_and(img_annotations, img_contours)
                if np.sum(intersection) == 0:
                    continue
                union = np.logical_or(img_annotations, img_contours)
                iou = np.sum(intersection) / float(np.sum(union))
                if iou >= threshold:
                    contours_labels[contour_idx] = val # stop
                    break
        return contours_labels
                  
