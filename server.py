
import os
import sys
sys.path.append('../')
sys.path.append('../AerialReId')
sys.path.append('../GeoRegistration')

import argparse
import cv2
import torch
from glob import glob

from pysot.core.config_adapn import cfg
from pysot.models.model_builder_adapn import ModelBuilderADAPN
from pysot.tracker.adsiamapn_tracker import ADSiamAPNTracker
from pysot.utils.model_load import load_pretrain
from collections import deque
from quatToEuler import quaternion_to_euler
from bbox_calculations import bbox_to_string
from extract_features import extractFeatures
from ReId import check_match
from Drone import Drone
from camera import Camera
from GeoLocation import GeoLocation
from SimpleTargetCalculator import SimpleTargetCalculator
from Coordinates import Coordinates
from PIL import Image

torch.set_num_threads(1)

import time
import UdpComms as U
import time
import cv2
import base64
import imutils
import json
import pandas as pd
import GeoCoordinationHandler as GC
from object_detector import ObjectDetector
import requests

from shapely.geometry import Point
import geopandas

# Create UDP socket to use for sending (and receiving)
sock = U.UdpComms(udpIP="127.0.0.1", portTX=8080, portRX=8081, enableRX=True, suppressWarnings=True)
sock2 = U.UdpComms(udpIP="127.0.0.1", portTX=8000, portRX=8002, enableRX=True, suppressWarnings=True)
sock3 = U.UdpComms(udpIP="127.0.0.1", portTX=8005, portRX=8003, enableRX=True, suppressWarnings=True)
cam = cv2.VideoCapture('Samples/east_1.mp4')
i = 1
lat = 50
lon = 50
alt = 40

df = pd.read_csv('Samples/east_1.csv')


cfg.merge_from_file('config.yaml')
cfg.CUDA = torch.cuda.is_available()
device = torch.device('cuda' if cfg.CUDA else 'cpu')

# create model
model = ModelBuilderADAPN()

# load model
model = load_pretrain(model, 'SiamAPNPlusModel.pth').eval().to(device)

# build tracker

trackers = {}
tracker_active = {}
# tracker = ADSiamAPNTracker(model)

detector = ObjectDetector()

track_di = {}
track_avg = {}

import random
import string

def generate_uid(length=6):
    characters = string.ascii_letters + string.digits
    return ''.join(random.choice(characters) for _ in range(length))


# start_time = time.time()
last_time_called = time.time()

def save_frame(uid,frame,st):
    print(st)
    x1,y1,x2,y2 = st.split('.')
    print(x1,y1,x2,y2,">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
    cropped_img = frame[int(y1):int(y2), int(x1):int(x2)]
    width = 64  
    height = 128
    if cropped_img is not None and cropped_img.size > 0:  
        resized_img = cv2.resize(cropped_img, (width, height))
        id = int(time.time())
        cv2.imwrite('../AerialReId/PRAI/pytorch/gallery/00000/{}_{}.jpg'.format(uid,id), resized_img)

def save_location(id, lat,lon):
    print(id,lat,lon)
    json_file_path = 'last_locations.json'
    if os.path.isfile(json_file_path):
        with open(json_file_path, 'r') as json_file:
            address_dict = json.load(json_file)
    else:
        address_dict = {}

    df = geopandas.tools.reverse_geocode(  
        [Point(lon,lat)]
    )
    print(df)
    addr = df['address'].iloc[0]
    address_dict[id] = addr
    
    with open(json_file_path, 'w') as new_json_file:
        new_json_file.write(json.dumps(address_dict, indent=4))
     

def track(frame):
    global last_time_called
    if len(trackers.keys()) == 0:
        return ''
    else:
        temp_li = []
        for i in list(trackers.keys()):
            outputs = trackers[i].track(frame)
            track_di[i].append(outputs['best_score'])
            # print(track_di[i],'....................')
            bbox = list(map(int, outputs['bbox']))
            if bbox:
                bbox[2]=bbox[0]+bbox[2]
                bbox[3]=bbox[1]+bbox[3]
            st = '.'.join(map(str, bbox))
            temp_li.append(st)
            print(st)
            current_time = time.time()
            if current_time - last_time_called >= 3:
                 save_frame(i,frame,st)
                 last_time_called = current_time

            if len(track_di[i]) == 1:
                di = {}
                di["point"] = "start"
                di["box"] = st
                sock3.SendData(json.dumps(di).encode('utf-8'))
                save_location(i,data['lat'],data['lon'])
                print(st,"Starting pointtttttttttttttttttttttt")
            if len(track_di[i]) == 10:
                track_di[i].popleft()
                if sum(track_di[i])/10 <0.3:
                    # print("Siamesee low confidence")
                    # trackers.remove(i)
                    del trackers[i]
                    tracker_active[i]=False
                    di = {}
                    di["point"] = "last"
                    di["box"] = st
                    sock3.SendData(json.dumps(di).encode('utf-8'))
                    print(st, "Last marked pointtttttttttttttttttttttt")
                    # print(data['lat'],data['lon'])
                    save_location(i,data['lat'],data['lon'])
                    # extractFeatures()
        return 'n'.join(temp_li)

def ReIdentification(img, bbox):
     bbox = bbox.split('n')
     for li in bbox:
            if li != "":
                lis = li.split('.')
                x1,y1,x2,y2 = [int(x) for x in lis]
                # print(x1,y1,x2,y2)
                cropped_img = img[int(y1):int(y2), int(x1):int(x2)]
                width = 64  
                height = 128  
                resized_img = cv2.resize(cropped_img, (width, height))
                img_rgb = cv2.cvtColor(resized_img, cv2.COLOR_BGR2RGB)
                pil_img = Image.fromarray(img_rgb)
                # id = int(time.time())
                # name = "../cropped/test {}.jpg".format(id)
                # pil_img.save(name)
                similarity_score = check_match(pil_img)
                print(similarity_score,":::::::::")
                json_file_path = 'last_locations.json'
                if os.path.isfile(json_file_path):
                    with open(json_file_path, 'r') as json_file:
                        address_dict = json.load(json_file)
                else:
                    address_dict = {}
                for id in similarity_score:
                    if similarity_score[id]>0.7:
                        if id in address_dict:
                            print(address_dict[id])


     print("------------------")
    #  print(bbox)


def LocationCalculator(dat):

    drone = Drone()
    drone.set_current_location(GeoLocation(float(dat["lat"]),float(dat["lon"]),float(dat["alt"])))
    camera_FOVh = 69
    camera_img_resolution = [int(float(dat["resw"])),int(float(dat["resh"]))]
    target_xy_coors_in_image = Coordinates(int(float(dat["xpos"])),int(float(dat["ypos"])))
    camera_quaternion = [data['w'],data['x'],data['y'],data['z']]
    c = Camera(camera_FOVh,camera_img_resolution,camera_quaternion)
    
    simpleTargetCalulator = SimpleTargetCalculator(c, drone, target_xy_coors_in_image)

    

    target_geo_location_simple_calculation = simpleTargetCalulator.calculate()
    
    lat,lon,alt =target_geo_location_simple_calculation.getGeoLocation()

    # c = GC.CameraRayProjection(69,[float(dat["lat"]),float(dat["lon"]),float(dat["alt"])],
    #                         [int(float(dat["resw"])),int(float(dat["resh"]))],
    #                         GC.Coordinates(int(float(dat["xpos"])),int(float(dat["ypos"]))),

    #                         [data['w'],data['x'],data['y'],data['z']])
    
    # target_direction_ENU = c.target_ENU()
    # target_direction_ECEF = c.ENU_to_ECEF(target_direction_ENU)
    # intersect_ECEF = c.target_location(target_direction_ECEF)
    
    # intersect_LLA = c.ECEFtoLLA(intersect_ECEF.x,intersect_ECEF.y,intersect_ECEF.z)
    # print(c.LLAtoXYZ(intersect_LLA[0], intersect_LLA[1], intersect_LLA[2]))
    # print("CALCULATED LOCATION IS",intersect_LLA)
    
    di2 = {
        'lat' : str(lat),
        'lon' : str(lon),
        'alt' : str(alt),
        'obj' : str(dat["obj"]),
        'ctr' : str(dat["ctr"])
    }
    return json.dumps(di2).encode('utf-8')

while True:
    ret,camImage = cam.read()
    height, width, _ = camImage.shape
    frame = imutils.resize(camImage,width=400)
    encoded,buffer = cv2.imencode('.jpg',frame,[cv2.IMWRITE_JPEG_QUALITY,80])
    byteString = base64.b64encode(buffer)

    frameBytes = buffer.tobytes()
    encoded_string= base64.b64encode(frameBytes)
    
    # if(i==501):
    #     end_time = time.time()
    #     elapsed_time = end_time - start_time
    #     print(f"The program took {elapsed_time:.2f} seconds to run.")
    if(i>1):
        data = {
            'image':encoded_string.decode(),
            # 'image':'im',
        }
        di = df.iloc[i].to_dict()
        data['lat'] = di['drone/location/latitude']
        data['lon'] =di['drone/location/longitude']
        data['alt'] = di['drone/ground_distance']

        data['w'] = di['camera/quat/w'] 
        data ['x'] =di['camera/quat/x']
        data ['y'] =di['camera/quat/y']
        data ['z'] =di['camera/quat/z']

        data['roll'],data['pitch'],data['yaw'] = quaternion_to_euler(data['w'],data['x'],data['y'],data['z'])

        #Object Detection

        res = detector.predict(camImage)
        bbox_data = res[0].boxes.xyxy.cpu().numpy().tolist()
        data['tracker'] = track(camImage)
        data['bbox_data'] = bbox_to_string(data['tracker'],bbox_data)
        ReIdentification(camImage,data['bbox_data'])
        data['tracker_status'] = '.'.join(['1' if value else '0' for value in tracker_active.values()])
        

    
        sock.SendData(json.dumps(data).encode('utf-8')) # Send this string to other application
    i += 2


    dat = sock.ReadReceivedData() # read data

    if dat != None: # if NEW data has been received since last ReadReceivedData function call
            # print(type(dat)) # print new received data
            # print(dat)
            dat = "{"+dat+"}"
            dat = json.loads(dat)
            if dat["track"] == "":
                res = LocationCalculator(dat)
                sock2.SendData(res)

            else:
                print("tracker initaliseddddddddddddddddddddd")
                li = list(map(int,dat["track"].split('.')))
                print(li)
                temp_tracker = ADSiamAPNTracker(model)
                key = generate_uid()
                trackers[key] = temp_tracker
                tracker_active[key] = True
                track_di[key] = deque(maxlen = 10)
                trackers[key].init(frame, (li[0],li[1], li[2]-li[0], li[3]-li[1]))
                

    dat2 = sock2.ReadReceivedData() # read data
    #implement timer function
    if dat2 != None: 
            # print(type(dat2),"oooooooooooooooooooooooooo") 
            # print(dat2)
            dat2 = "{"+dat2+"}"
            dat2 = json.loads(dat2)
            res = LocationCalculator(dat2)
            sock2.SendData(res)
            

    time.sleep(0.1)


