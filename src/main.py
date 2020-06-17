import os
import numpy as np
import cv2
import logging
import time
from argparse import ArgumentParser

from input_feeder import InputFeeder
from face_detection import FaceDetection
from facial_landmarks_detection import Facial_Landmarks_Detection
from gaze_estimation import Gaze_Estimation
from head_pose_estimation import Head_Pose_Estimation
from mouse_controller import MouseController


def build_argparser():
    parser = ArgumentParser()

    parser.add_argument("-fd", "--face_detection_model", required=True, type=str,
                        help=" Path to .xml file of Face Detection model.")
    parser.add_argument("-fld", "--facial_landmark_model", required=True, type=str,
                        help=" Path to .xml file of Facial Landmark Detection model.")
    parser.add_argument("-ge", "--gaze_estimation_model", required=True, type=str,
                        help=" Path to .xml file of Gaze Estimation model.")
    parser.add_argument("-hp", "--head_pose_model", required=True, type=str,
                        help=" Path to .xml file of Head Pose Estimation model.")
    parser.add_argument("-i", "--input", required=True, type=str,
                        help=" Path to video file or enter cam for webcam")
    parser.add_argument("-l", "--cpu_extension", required=False, type=str,
                        default=None,
                        help="path of extensions if any layers is incompatible with  hardware")
    parser.add_argument("-prob", "--prob_threshold", required=False, type=float,
                        default=0.6,
                        help="Probability threshold for model to identify the face .")
    parser.add_argument("-d", "--device", type=str, default="CPU",
                        help="Specify the target device to run on: "
                             "CPU, GPU, FPGA or MYRIAD is acceptable. Sample "
                             "will look for a suitable plugin for device "
                             "(CPU by default)")

    return parser


def main():
    args = build_argparser().parse_args()

    inputFile = args.input
    logger = logging.getLogger()

    start_loading = time.time()

    fd = FaceDetection()
    fld = Facial_Landmarks_Detection()
    ge = Gaze_Estimation()
    hp = Head_Pose_Estimation()

    model_loading_time = time.time() - start_loading

    fd.load_model(args.face_detection_model, args.device, args.cpu_extension)
    fld.load_model(args.facial_landmark_model, args.device, args.cpu_extension)
    ge.load_model(args.gaze_estimation_model, args.device, args.cpu_extension)
    hp.load_model(args. head_pose_model, args.device, args.cpu_extension)

    mc = MouseController('medium', 'fast')

    if inputFile.lower() == "cam":
        feed = InputFeeder("cam")
    else:
        if not os.path.isfile(inputFile):
            logger.error("Unable to find input file")
            exit(1)

        feed = InputFeeder("video", inputFile)
    feed.load_data()

    counter = 0
    frame_count = 0
    inference_time = 0
    start_inf_time = time.time()
    for frame in feed.next_batch():
        if frame is not None:
            frame_count += 1

            key = cv2.waitKey(60)
            start_inference = time.time()

            croppedFace, face_coords = fd.predict(frame.copy(), args.prob_threshold)
            if type(croppedFace) == int:
                logger.error("No face detected.")
                if key == 27:
                    break

                continue

            hp_output = hp.predict(croppedFace.copy())

            left_eye, right_eye, eye_coords = fld.predict(croppedFace.copy())
            new_mouse_coord, gaze_vector = ge.predict(left_eye, right_eye, hp_output)

            stop_inference = time.time()
            inference_time = inference_time + stop_inference - start_inference
            counter = counter + 1

            img_hor = cv2.resize(frame, (500, 500))
            cv2.imshow('Visualization', img_hor)

            if frame_count % 5 == 0:
                mc.move(new_mouse_coord[0], new_mouse_coord[1])

            if key == 27:
                break

    fps = frame_count / inference_time

    logger.error("video ended...")
    logger.error("Total loading time of the models: " + str(model_loading_time) + " s")
    logger.error("total inference time {} seconds".format(inference_time))
    logger.error("Average inference time: " + str(inference_time / frame_count) + " s")
    logger.error("fps {} frame/second".format(fps / 5))

    cv2.destroyAllWindows()
    feed.close()


if __name__ == '__main__':
    main()
