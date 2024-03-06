import os
import numpy as np
from echolab2.instruments import EK60
from echolab2.processing import align_pings
from echolab2.processing.mask import Mask
from echolab2.processing.line import Line
from echolab2.plotting.matplotlib import echogram
from echolab2.processing import mask, line

class EchogramReader(object):
    """
    Read one echogram
    Input: raw and bottom path
    Output: npy array, surface/bottom index, time, depth, latitude/longitude
    """
    def __init__(self, raw_path=None, bot_path=None, freq_li=None):
        self.raw_path = raw_path
        self.bot_path = bot_path
        self.freq_li = freq_li

    def __call__(self):
        # get filename
        filename = (self.raw_path.split('/')[-1]).split('.')[0]
        # print(filename)
        ek60 = EK60.EK60()
        ek60.read_raw(self.raw_path)
        ek60.read_bot(self.bot_path)
        # create a dictionary of RawData objects from the channels in the data
        raw_data = {}
        for channel in ek60.channel_id_map:
            raw = ek60.get_raw_data(channel_number=channel)
            frequency = int(raw.frequency[0]/1000)
            if frequency in self.freq_li:
                raw_data[frequency] = raw
        # get Sv data
        Sv_data = []
        for frequency in self.freq_li:
            Sv_data.append(raw_data[frequency].get_Sv(heave_correct=True))
        # align pings across channels
        align_pings.AlignPings(channels=Sv_data, mode='pad')

        # impute bottom data (38kHz)
        raw_bottom = raw_data[38].get_bottom(heave_correct=True)
        non_zero = np.nonzero(raw_bottom.data)
        idx = np.arange(len(raw_bottom.data))
        # apply interpolation (interp: depth at each ping)
        interp = np.interp(idx, idx[non_zero], raw_bottom.data[non_zero])

        # get time & depth & position (38kHz)
        time = Sv_data[1].ping_time
        depth = Sv_data[1].depth
        positions = ek60.nmea_data.interpolate(Sv_data[1], 'position')

        # get bottom & surface depth_index, for masking purpose
        bottom_idx = []
        bottom_buffer = 1 # 1 meter above bottom
        for i in interp:
            i -= bottom_buffer
            x_idx = (np.where(depth == min(depth, key=lambda x: abs(x - i)))[0]).tolist()[0]
            bottom_idx.append(x_idx)

        # surface idx (constant for pings)
        surface_depth = 10 # 10 meters below surface
        surface_idx = (np.where(depth == min(depth, key=lambda x: abs(x - surface_depth)))[0]).tolist()[0]

        # get npy
        Sv_npy = np.array([i.data for i in Sv_data]) # 4 * x * y

        return filename, Sv_npy, surface_idx, bottom_idx, time, depth, positions
