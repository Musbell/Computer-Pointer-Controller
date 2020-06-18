import os
import cv2
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

    frame_num = 0
    inference_time = 0
    counter = 0

    # Initialize the Inference Engine
    fd = FaceDetection()
    fld = Facial_Landmarks_Detection()
    ge = Gaze_Estimation()
    hp = Head_Pose_Estimation()

    # Load Models
    fd.load_model(args.face_detection_model, args.device, args.cpu_extension)
    fld.load_model(args.facial_landmark_model, args.device, args.cpu_extension)
    ge.load_model(args.gaze_estimation_model, args.device, args.cpu_extension)
    hp.load_model(args.head_pose_model, args.device, args.cpu_extension)

    # Mouse Controller precision and speed
    mc = MouseController('medium', 'fast')

    # feed input from an image, webcam, or video to model
    if args.input == "CAM":
        feed = InputFeeder("cam")
    else:
        assert os.path.isfile(args.input), "Specified input file doesn't exist"
        feed = InputFeeder("video", args.input)
    feed.load_data()

    frame_count = 0
    for frame in feed.next_batch():
        frame_count += 1
        inf_start = time.time()
        if frame is not None:
            img_frame = cv2.resize(frame, (500, 500))
            cv2.imshow('Computer Pointer Controller', img_frame)
            key = cv2.waitKey(60)

            det_time = time.time() - inf_start

            # make predictions
            detected_face, face_coords = fd.predict(frame.copy(), args.prob_threshold)
            hp_output = hp.predict(detected_face.copy())
            left_eye, right_eye, eye_coords = fld.predict(detected_face.copy())

            stop_inference = time.time()
            inference_time = inference_time + stop_inference - inf_start
            counter = counter + 1

            # set speed
            if frame_count % 5 == 0:
                mouse_x, mouse_y = ge.predict(left_eye, right_eye, hp_output)
                mc.move(mouse_x[0], mouse_y[1])

            # INFO
            print(f'[INFO] approx. NUMBER OF FRAMES: {frame_num}')
            print(f'[INFO] approx. INFERENCE TIME: {det_time * 1000}ms')

            frame_num += 1

            if key == 27:
                break
    feed.close()


if __name__ == '__main__':
    main()
