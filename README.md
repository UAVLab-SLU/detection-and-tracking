# Detection-and-Tracking

This repository has code to connect with drone, YOLO object detector and Siamese Tracker. 

While streaming, we are transmitting both frames and their associated metadata. Each frame undergoes processing by the Yolo detector, which then generates bounding boxes as metadata. When a user decides to track a Point of Interest (POI), we input the bounding box coordinates into the Siamese tracker. This tracker continuously monitors the POI until the average confidence across ten consecutive frames falls below a set threshold.

### Setup:


1. Clone the repository
2. Install Parrot Olympe ```pip3 install parrot-olympe```
3. Setup requirements for GeoLocation.
   
    ```pip install requirements.txt```
   
    This package(droneresponse_mathtools) requires an external package(geoids).[link](https://sourceforge.net/projects/geographiclib/files/geoids-distrib/egm96-5.tar.bz2/download?use_mirror=cfhcable)

    Add path in `__init__.py`

4. Install YOLO for object detection. [link](https://docs.ultralytics.com/)

5. Install requirements for Siamese Tracker. ```pip install requirements_tracker.txt```
6. We are using Siamese Network to track objects/people.[reference](https://github.com/vision4robotics/SiamAPN)
7. 
    We used SiamAPN++. Pretrained model can be downloaded from [link](https://github.com/vision4robotics/SiamAPN#siamapn-1)
   
    Download the Pretrained model and rename it as SiamAPNPlusModel.pth.


### Connectivity

To stream data to a different laptop/ or when connected to a real drone.[refer this link](https://github.com/UAVLab-SLU/Aerial-video-annotator/tree/repo-split#connectivity)
