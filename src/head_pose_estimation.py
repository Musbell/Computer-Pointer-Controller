import os
import sys
import logging as log
import cv2
import numpy as np
import math
from openvino.inference_engine import IENetwork, IECore


class Head_Pose_Estimation:
    def __init__(self):
        self.plugin = None
        self.network = None
        self.input_blob = None
        self.output_blob = None
        self.exec_network = None
        self.infer_request = None

    def load_model(self, model, device="CPU", cpu_extension=None, num_request=1):
        """
       Load the model given IR files.
       Defaults to CPU as device for use in the workspace.
       Synchronous requests made within.
       """
        model_xml = model
        model_bin = os.path.splitext(model_xml)[0] + ".bin"

        # Initialize the plugin
        self.plugin = IECore()

        # Add a CPU extension, if applicable
        if cpu_extension and "CPU" in device:
            self.plugin.add_extension(cpu_extension, device)
            log.info("CPU extension loaded: {}".format(cpu_extension))

        # Read the IR as a IENetwork
        try:
            self.network = IENetwork(model=model_xml, weights=model_bin)
        except Exception as e:
            raise ValueError("Could not Initialise the network. Have you enterred the correct model path?")

        # Check Network layer support
        if "CPU" in device:
            supported_layers = self.plugin.query_network(self.network, "CPU")
            not_supported_layers = [l for l in self.network.layers.keys() if l not in supported_layers]
            if len(not_supported_layers) != 0:
                log.error("Following layers are not supported by the plugin for specified device {}:\n {}".
                          format(device, ', '.join(not_supported_layers)))
                log.error(
                    "Please try to specify cpu extensions library path in sample's command line parameters using -l "
                    "or --cpu_extension command line argument")
                sys.exit(1)

        # Load the IENetwork into the plugin
        self.exec_network = self.plugin.load_network(self.network, device, num_requests=1)

        # Get the input layer
        self.input_blob = next(iter(self.network.inputs))
        self.output_blob = next(iter(self.network.outputs))
        return

    def get_input_shape(self):
        return self.network.inputs[self.input_blob].shape

    def preprocess_input(self, image):

        image_resized = cv2.resize(image, (self.get_input_shape()[3], self.get_input_shape()[2]))

        img_processed = np.transpose(np.expand_dims(image_resized, axis=0), (0, 3, 1, 2))

        return img_processed

    def preprocess_output(self, outputs):
        outs = []

        outs.append(outputs['angle_y_fc'].tolist()[0][0])
        outs.append(outputs['angle_p_fc'].tolist()[0][0])
        outs.append(outputs['angle_r_fc'].tolist()[0][0])

        return outs

    def predict(self, image):
        img_processed = self.preprocess_input(image.copy())

        outputs = self.exec_network.infer({self.input_blob: img_processed})

        result = self.preprocess_output(outputs)

        return result

    def clean(self):
        """
        Deletes all the instances
        :return: None
        """
        del self.exec_network
        del self.plugin
        del self.network
