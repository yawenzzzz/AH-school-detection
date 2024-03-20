import numpy as np
import cv2
from skimage.transform import resize
from scipy import signal

class ROICropper(object):
    """
    for each ROI (contour), crop & resize, save as npy
    Input: ROIs (contours), Sv npy, npy dir
    Output: save npy
    """
    def __init__(self, filename=None, contours=None, features=None, labels=None, Sv_npy=None, npy_dir=None):
        self.filename = filename
        self.contours = contours
        self.features = features
        self.labels = labels
        self.Sv_npy = Sv_npy
        self.npy_dir = npy_dir

    def __call__(self):
        # apply median filter (4 * x * y)
        Sv_npy = np.array([signal.medfilt2d(i, 3) for i in self.Sv_npy])

        # for cropping (0-255)
        Sv_img = np.transpose(Sv_npy, (1, 2, 0)) # (x * y * 4)
        Sv_img = np.where(Sv_img >= -85, Sv_img, -85)
        Sv_img = np.where(Sv_img < 0, Sv_img, 0)
        Sv_img = (Sv_img + 85.0) * (255.0 / 85.0)
        Sv_img = Sv_img.astype(np.uint8)

        for idx, contour in enumerate(self.contours):
            label = self.labels[idx]
            feature = self.features[idx] # get a dict
            # expand
            total_water_column = feature['total_water_column']
            depth = feature['depth']
            relative_altitude = feature['relative_altitude']
            latitude = feature['latitude']
            longitude = feature['longitude']
            time = feature['time']
            # get bbox
            x, y, w, h = cv2.boundingRect(contour)
            x_min, x_max = x, x+w
            y_min, y_max = y, y+h
            # crop image (x * y * 4)
            Sv_img_crop = Sv_img[x_min:x_max, y_min:y_max, :]
            res = resize(Sv_img_crop, (100, 100, 4), preserve_range=True)
            res = res.astype(np.uint8)
            np.save(self.npy_dir + f'{self.filename}_{idx}_{total_water_column}_{depth}_{relative_altitude}_{latitude}_{longitude}_{time}_{label}.npy', res)
