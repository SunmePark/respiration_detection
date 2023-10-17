import sys
from optparse import OptionParser
from time import sleep

import pymoduleconnector
from pymoduleconnector import DataType

import numpy as np


def reset(device_name):
    mc = pymoduleconnector.ModuleConnector(device_name)
    xep = mc.get_xep()
    xep.module_reset()
    mc.close()
    sleep(3)


def on_file_available(data_type, filename):
    print("new file available for data type: {}".format(data_type))
    print("  |- file: {}".format(filename))
    if data_type == DataType.FloatDataType:
        print("processing Float data from file")


def on_meta_file_available(session_id, meta_filename):
    print("new meta file available for recording with id: {}".format(session_id))
    print("  |- file: {}".format(meta_filename))


def clear_buffer(xep):
    """Clears the frame buffer"""
    while xep.peek_message_data_float():
        xep.read_message_data_float()


class setmode():
    def __init__(self, device_name, fps, record=False, baseband=False):
        self.device_name = device_name
        self.FPS = fps
        self.record = False
        self.baseband = False
        self.directory = '.'
        reset(device_name)
        self.mc = pymoduleconnector.ModuleConnector(self.device_name)

    def stopAPPmode(self):
        # Assume an X4M300/X4M200 module and try to enter XEP mode
        app = self.mc.get_x4m200()
        # Stop running application and set module in manual mode.
        try:
            app.set_sensor_mode(0x13, 0)  # Make sure no profile is running.
        except RuntimeError:
            # Profile not running, OK
            pass

        try:
            app.set_sensor_mode(0x12, 0)  # Manual mode.
        except RuntimeError:
            # Maybe running XEP firmware only?
            pass

        if self.record:
            recorder = self.mc.get_data_recorder()
            recorder.subscribe_to_file_available(pymoduleconnector.AllDataTypes, on_file_available)
            recorder.subscribe_to_meta_file_available(on_meta_file_available)

    def setXEPmode(self):
        xep = self.mc.get_xep()
        # Set DAC range
        xep.x4driver_set_dac_min(900)
        xep.x4driver_set_dac_max(1150)

        # Set integration
        xep.x4driver_set_iterations(16)
        xep.x4driver_set_pulses_per_step(26)

        xep.x4driver_set_downconversion(int(self.baseband))
        # Start streaming of data
        xep.x4driver_set_fps(self.FPS)

        return xep

    def read_frame(self, xep):
        """Gets frame data from module"""
        d = xep.read_message_data_float()
        frame = np.array(d.data)

        if self.baseband:
            n = len(frame)
            frame = frame[:n // 2] + 1j * frame[n // 2:]

        return frame


def setParser():
    parser = OptionParser()
    parser.add_option(
        "-d",
        "--device",
        dest="device_name",
        default = "COM7",               ##### please check before running
        help="device file to use",
        metavar="FILE")
    parser.add_option(
        "-b",
        "--baseband",
        action="store_true",
        default=False,
        dest="baseband",
        help="Enable baseband, rf data is default")
    parser.add_option(
        "-r",
        "--record",
        action="store_true",
        default=False,
        dest="record",
        help="Enable recording")
    parser.add_option(
        "-f",
        "--file",
        dest="meta_filename",
        metavar="FILE",
        help="meta file from recording")

    (options, args) = parser.parse_args()


