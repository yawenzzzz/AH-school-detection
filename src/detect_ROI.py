import os
import numpy as np
import cv2
from scipy import signal

class ROIDetector(object):
    """
    Read Sv npy and conduct contour detection
    Input: npy array, surface/bottom index
    Output: contours
    """
    def __init__(self, filename=None, Sv_npy=None, surface_idx=None, bottom_idx=None, fig_dir=None, threshold=None, kernel_size=None):
        self.filename = filename
        self.Sv_npy = Sv_npy
        self.surface_idx = surface_idx
        self.bottom_idx = bottom_idx
        self.fig_dir = fig_dir
        # add parameters
        self.threshold = threshold
        self.kernel_size = kernel_size

    def __call__(self):
        # preprocess npy (4 * x * y), Remember: x: time, y: depth
        # median filter
        Sv_npy = np.array([signal.medfilt2d(i, 3) for i in self.Sv_npy])
        Sv_npy = np.transpose(Sv_npy, (1, 2, 0)) # (x * y * 4)
        # thresholding (all frequencies)
        process_npy = Sv_npy.reshape(-1, 4)
        process_npy = np.where(process_npy >= self.threshold, 1, 0)
        process_npy = process_npy.sum(1)
        process_npy = process_npy.reshape(Sv_npy.shape[0], Sv_npy.shape[1]) # shape: (x * y)

        # TODO: what about using masked 38kHz for detection?

        # add surface and bottom mask
        for idx, bottom in enumerate(self.bottom_idx):
            process_npy[idx, :self.surface_idx] = 0
            process_npy[idx, bottom:] = 0
        # get result (binary: 0/255)
        process_npy = np.where(process_npy == 4, 255, 0)
        
        # ADD: dilate image (helps a lot), [0] kernel size, [1] 1.5 * kernel size
        kernel = np.ones((int(self.kernel_size), int(self.kernel_size)), np.uint8)
        process_npy = process_npy.astype(np.uint8)
        process_npy = cv2.dilate(process_npy, kernel, iterations=1)
        
#         # ADD: remove openings (if not, precision would be very low)
#         kernel = np.ones((5, 5), np.uint8)
#         process_npy = cv2.morphologyEx(process_npy, cv2.MORPH_OPEN, kernel)
        
        # for contour detection, shape: (y * x)
        process_npy = np.transpose(process_npy, (1, 0))
        process_npy = process_npy.astype(np.uint8)

        # ROI detection
        contours, hierarchy = cv2.findContours(process_npy, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        return process_npy.shape, contours
