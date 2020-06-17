import os
import sys
import logging as log
import cv2
import numpy as np
import math
from openvino.inference_engine import IENetwork, IECore


class Gaze_Estimation:
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
        self.input_blob = [i for i in self.network.inputs.keys()]
        self.output_blob = [i for i in self.network.outputs.keys()]
        return

    def get_input_shape(self):
        return self.network.inputs[self.input_blob[1]].shape

    def preprocess_input(self, left_eye_image, right_eye_image):
        le_image_resized = cv2.resize(left_eye_image, (self.get_input_shape()[3], self.get_input_shape()[2]))
        le_img_processed = np.transpose(np.expand_dims(le_image_resized, axis=0), (0, 3, 1, 2))

        re_image_resized = cv2.resize(right_eye_image, (self.get_input_shape()[3], self.get_input_shape()[2]))
        re_img_processed = np.transpose(np.expand_dims(re_image_resized, axis=0), (0, 3, 1, 2))

        return le_img_processed, re_img_processed

    def preprocess_output(self, outputs, head_pose_angle):
        gaze_vec = outputs[self.output_blob[0]].tolist()[0]
        angle_r_fc = head_pose_angle[2]
        cosine = math.cos(angle_r_fc * math.pi / 180.0)
        sine = math.sin(angle_r_fc * math.pi / 180.0)

        x_val = gaze_vec[0] * cosine + gaze_vec[1] * sine
        y_val = -gaze_vec[0] * sine + gaze_vec[1] * cosine

        return (x_val, y_val), gaze_vec

    def predict(self, left_eye_image, right_eye_image, head_pose_angle):
        le_img_processed, re_img_processed = self.preprocess_input(left_eye_image.copy(), right_eye_image.copy())

        outputs = self.exec_network.infer({'head_pose_angles': head_pose_angle, 'left_eye_image': le_img_processed,
                                           'right_eye_image': re_img_processed})

        mouse_coords, gaze_vec = self.preprocess_output(outputs, head_pose_angle)

        return mouse_coords, gaze_vec

    def clean(self):
        """
        Deletes all the instances
        :return: None
        """
        del self.exec_network
        del self.plugin
        del self.network
