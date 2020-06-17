# Computer Pointer Controller

The application uses gaze detection points with the use of deep learning model to estimate the gaze of user's eyes to move the mouse pointer position. It supports camera video stream and video file as input.

## Project Set Up and Installation
OpenVINO™ toolkit and its dependencies must be installed to run the application. OpenVINO 2020.2.130 is used on this project.  
 Installation instructions may be found at: https://software.intel.com/en-us/articles/OpenVINO-Install-Linux or https://github.com/udacity/nd131-openvino-fundamentals-project-starter/blob/master/linux-setup.md  

There are certain pretrained models you need to download after initializing the openVINO environment:

`source /opt/intel/openvino/bin/setupvars.sh`
-   [Face Detection Model](https://docs.openvinotoolkit.org/latest/_models_intel_face_detection_adas_binary_0001_description_face_detection_adas_binary_0001.html)
-   [Facial Landmarks Detection Model](https://docs.openvinotoolkit.org/latest/_models_intel_landmarks_regression_retail_0009_description_landmarks_regression_retail_0009.html)
-   [Head Pose Estimation Model](https://docs.openvinotoolkit.org/latest/_models_intel_head_pose_estimation_adas_0001_description_head_pose_estimation_adas_0001.html)
-   [Gaze Estimation Model](https://docs.openvinotoolkit.org/latest/_models_intel_gaze_estimation_adas_0002_description_gaze_estimation_adas_0002.html)

To download the above pretrained model, run the following commands after creating `model folder` in the project directory and `cd` into it:

*Face Detection*

```
$ python3 /opt/intel/openvino/deployment_tools/tools/model_downloader/downloader.py --name "face-detection-adas-binary-0001"
```

*Head Pose Estimation*

```
$ python3 /opt/intel/openvino/deployment_tools/tools/model_downloader/downloader.py --name "head-pose-estimation-adas-0001"
```

*Gaze Estimation Model*

```
$ python3 /opt/intel/openvino/deployment_tools/tools/model_downloader/downloader.py --name "gaze-estimation-adas-0002"
```

*Facial Landmarks Detection*

```
$ python3 /opt/intel/openvino/deployment_tools/tools/model_downloader/downloader.py --name "landmarks-regression-retail-0009"
```

*Install the requirements*
```
 Install the requirements
```

*Project structure*

```
|
|--bin
    |--demo.mp4
|--model
    |--face-detection-adas-binary-0001
    |--gaze-estimation-adas-0002
    |--head-pose-estimation-adas-0001
    |--landmarks-regression-retail-0009
|--src
    |--face_detection.py
    |--facial_landmarks_detection.py
    |--gaze_estimation.py
    |--head_pose_estimation
    |--input_feeder.py
    |--main.py
    |--mouse_controller.py
|--README.md
|--requirements.txt
```
## Demo
Use the following command to run the application

```
$ python3 main.py -f model/intel/face-detection-adas-binary-0001/FP32-INT1/face-detection-adas-binary-0001.xml -fl model/intel/landmarks-regression-retail-0009/FP32/landmarks-regression-retail-0009.xml -hp model/intel/head-pose-estimation-adas-0001/FP32/head-pose-estimation-adas-0001.xml -g model/intel/gaze-estimation-adas-0002/FP32/gaze-estimation-adas-0002.xml -i bin/demo.mp4
```

## Documentation

command line parameters
```
$ python3 main.py --help
usage: main.py [-h] -fd FACE_DETECTION_MODEL -fld FACIAL_LANDMARK_MODEL -ge
               GAZE_ESTIMATION_MODEL -hp HEAD_POSE_MODEL -i INPUT
               [-l CPU_EXTENSION] [-prob PROB_THRESHOLD] [-d DEVICE]

optional arguments:
  -h, --help            show this help message and exit
  -fd FACE_DETECTION_MODEL, --face_detection_model FACE_DETECTION_MODEL
                        Path to .xml file of Face Detection model.
  -fld FACIAL_LANDMARK_MODEL, --facial_landmark_model FACIAL_LANDMARK_MODEL
                        Path to .xml file of Facial Landmark Detection model.
  -ge GAZE_ESTIMATION_MODEL, --gaze_estimation_model GAZE_ESTIMATION_MODEL
                        Path to .xml file of Gaze Estimation model.
  -hp HEAD_POSE_MODEL, --head_pose_model HEAD_POSE_MODEL
                        Path to .xml file of Head Pose Estimation model.
  -i INPUT, --input INPUT
                        Path to video file or enter cam for webcam
  -l CPU_EXTENSION, --cpu_extension CPU_EXTENSION
                        path of extensions if any layers is incompatible with
                        hardware
  -prob PROB_THRESHOLD, --prob_threshold PROB_THRESHOLD
                        Probability threshold for model to identify the face .
  -d DEVICE, --device DEVICE
                        Specify the target device to run on: CPU, GPU, FPGA or
                        MYRIAD is acceptable. Sample will look for a suitable
                        plugin for device (CPU by default)
```

## Benchmarks
Measuring performance (Start inference asyncronously, 4 inference requests using 4 streams for CPU, limits: 60000 ms duration)

Hardware Configuration: Intel® Core™ i5-6200U CPU @ 2.30GHz × 4

_______________________________________
*face-detection-adas-binary-0001*

*FP32*

* Count: 5320 iterations
* Duration: 60037.26 ms
* Latency: 42.58 ms
* Throughput: 88.61 FPS

____________________________________
*gaze-estimation-adas-0002*

*FP16-INT8*
* Count: 78160 iterations
* Duration: 60002.77 ms
* Latency: 2.89 ms
* Throughput: 1302.61 FPS

*FP16*
* Count:  51312 iterations
* Duration: 60007.48 ms
* Latency: 4.43 ms
* Throughput: 855.09 FPS

*FP32*
* Count: 48736 iterations
* Duration: 60007.04 ms
* Latency: 4.48 ms
* Throughput: 812.17 FPS

________________________________________
*head-pose-estimation-adas-0001*

*FP16*
* Count:  62812 iterations
* Duration: 60004.12 ms
* Latency: 3.55 ms
* Throughput: 1046.79 FPS

*FP16-INT8*
* Count:  81888 iterations
* Duration: 60004.36 ms
* Latency: 2.79 ms
* Throughput: 1364.70 FPS

*FP32*
* Count: 55084 iterations
* Duration: 60005.58 ms
* Latency: 3.63 ms
* Throughput: 917.98 FPS

___________________________________________
*landmarks-regression-retail-0009*

*FP16*
* Count:  335108 iterations
* Duration: 60001.13 ms
* Latency: 0.68 ms
* Throughput: 5585.03 FPS

*FP16-INT8*
* Count:  295848 iterations
* Duration: 60000.93 ms
* Latency: 0.72 ms
* Throughput: 4930.72 FPS

*FP32*
* Count: 322800 iterations
* Duration: 60000.69 ms
* Latency:  0.67 ms
* Throughput: 5379.94 FPS
________________________________________________

## Results
From the above results, the best model precision combination is that of Face detection 32 bits preand loose accuracycision and other models in 16 bits. 
This reduce the model size and load time, although models with lower precision gives low accuracy but better inference time.

## Stand Out Suggestions
This is where you can provide information about the stand out suggestions that you have attempted.

### Async Inference
If you have used Async Inference in your code, benchmark the results and explain its effects on power and performance of your project.

### Edge Cases
There will be certain situations that will break your inference flow. For instance, lighting changes or multiple people in the frame. Explain some of the edge cases you encountered in your project and how you solved them to make your project more robust.