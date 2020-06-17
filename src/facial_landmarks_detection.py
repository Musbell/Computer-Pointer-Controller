import os
import sys
import logging as log
import cv2
import numpy as np
from openvino.inference_engine import IENetwork, IECore


class Facial_Landmarks_Detection:
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
        img_cvt = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        img_resized = cv2.resize(img_cvt, (self.get_input_shape()[3], self.get_input_shape()[2]))
        img_processed = np.transpose(np.expand_dims(img_resized, axis=0), (0, 3, 1, 2))

        return img_processed

    def preprocess_output(self, outputs):
        result = outputs[self.output_blob][0]
        lefteye_x = result[0].tolist()[0][0]
        lefteye_y = result[1].tolist()[0][0]
        righteye_x = result[2].tolist()[0][0]
        righteye_y = result[3].tolist()[0][0]

        return lefteye_x, lefteye_y, righteye_x, righteye_y

    def predict(self, image):
        img_processed = self.preprocess_input(image.copy())
        outputs = self.exec_network.infer(inputs={self.input_blob: img_processed})
        coords = self.preprocess_output(outputs)

        h = image.shape[0]
        w = image.shape[1]

        coords = coords * np.array([w, h, w, h])
        coords = coords.astype(np.int32)

        le_xmin = coords[0] - 10
        le_ymin = coords[1] - 10
        le_xmax = coords[0] + 10
        le_ymax = coords[1] + 10

        re_xmin = coords[2] - 10
        re_ymin = coords[3] - 10
        re_xmax = coords[2] + 10
        re_ymax = coords[3] + 10

        le = image[le_ymin:le_ymax, le_xmin:le_xmax]
        re = image[re_ymin:re_ymax, re_xmin:re_xmax]

        eye_coords = [[le_xmin, le_ymin, le_xmax, le_ymax], [re_xmin, re_ymin, re_xmax, re_ymax]]

        return le, re, eye_coords

    def clean(self):
        """
        Deletes all the instances
        :return: None
        """
        del self.exec_network
        del self.plugin
        del self.network
