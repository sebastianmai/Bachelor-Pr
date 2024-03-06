#!/usr/bin/env python3
import zmq
import numpy as np

class ZMQ_Publisher():

    def __init__(self, address='localhost'):
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.PUB)
        self.socket.connect(f"tcp://{address}:5556")


    # function for sending a custom multipart message
    # refer to readme for more information
    def publish(self, header, additionalSensors, payload, flags=0):
        header_dict = dict(name=header[0], msg_type=header[1], add_sensor=header[2])
        self.socket.send_json(header_dict, flags|zmq.SNDMORE)

        if header[1] == 0:
            return self.socket.send_string(payload, flags)
        if header[2]:
            self.socket.send_json(additionalSensors, flags|zmq.SNDMORE)
        return self.send_array(payload, flags)


    # function for serializing and sending np arrays
    def send_array(self, array, flags=0, copy=True, track=False):
        md = dict(
            dtype = str(array.dtype),
            shape = array.shape,
        )
        self.socket.send_json(md, flags|zmq.SNDMORE)
        return self.socket.send(array, flags, copy=copy, track=track)
