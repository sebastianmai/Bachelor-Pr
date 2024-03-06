# Sensor
This directory contains the files only needed for the Sensor Node, which will collect the data from the CYBRES Measurement Unit and sends them to the Edge Device. The Sensor Node also requires the directory *Utilities*.
### main.py
The entry point and only file that has to be executed to run the scripts.
### sensor_node.py
Contains the main class *Sensor_Node*, which handles the program logic of the sensor. It communicates with the Cybres MU, collects data and sends them with the *zmq_publisher* to the Edge Device. The measurement data will also be locally saved.
### cybres_mu.py
Contains the class *Cybres_MU*, an interface for the CYBRES Measurement Unit.
### zmq_publisher.py
Contains the class *ZMQ_Publisher* for wireless communication between sensor and edge device using [ZeroMQ](https://zeromq.org). It provides methods for sending numpy arrays and the multipart data message.
