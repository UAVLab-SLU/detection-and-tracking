#!/usr/bin/env python

# NOTE: Line numbers of this example are referenced in the user guide.
# Don't forget to update the user guide after every modification of this example.

import csv
import math
import time
import os
import queue
import shlex
import subprocess
import tempfile
import threading
import UdpComms as U
import imutils
import cv2
import base64
import json
import GeoCoordinationHandler as GC
import numpy as np

import olympe
from olympe.messages.ardrone3.Piloting import TakeOff, Landing
from olympe.messages.ardrone3.Piloting import moveBy, CancelMoveTo, moveTo, NavigateHome
from olympe.messages.ardrone3.PilotingEvent import moveByEnd
from olympe.messages.ardrone3.PilotingState import FlyingStateChanged, SpeedChanged, moveByChanged, moveToChanged
from olympe.messages.ardrone3.PilotingSettings import MaxTilt
from olympe.messages.ardrone3.PilotingSettingsState import MaxTiltChanged
from olympe.enums.ardrone3.PilotingState import MoveToChanged_Status as status                                                               
import olympe.enums.move as mode 
from olympe.messages.ardrone3.GPSSettingsState import GPSFixStateChanged, HomeChanged
from olympe.video.renderer import PdrawRenderer

from olympe.messages.gimbal import (
   set_target
)

olympe.log.update_config({"loggers": {"olympe": {"level": "WARNING"}}})

counter = 0
interrupt = False
DRONE_IP = os.environ.get("DRONE_IP", "10.202.0.1")
# DRONE_IP = os.environ.get("DRONE_IP", "192.168.42.1")

DRONE_RTSP_PORT = os.environ.get("DRONE_RTSP_PORT")
# sock = U.UdpComms(udpIP="10.0.0.1", portTX=8080, portRX=8001, enableRX=True, suppressWarnings=True)
# sock2 = U.UdpComms(udpIP="10.0.0.1", portTX=8000, portRX=8002, enableRX=False, suppressWarnings=True)
sock = U.UdpComms(udpIP="127.0.0.1", portTX=8080, portRX=8001, enableRX=True, suppressWarnings=True)
sock2 = U.UdpComms(udpIP="127.0.0.1", portTX=8000, portRX=8002, enableRX=False, suppressWarnings=True)
ct = 0
class StreamingExample:
    def __init__(self):
        # Create the olympe.Drone object from its IP address
        self.drone = olympe.Drone(DRONE_IP)
        
        self.tempd = tempfile.mkdtemp(prefix="olympe_streaming_test_")

        print(f"Olympe streaming example output dir: {self.tempd}")
        self.frame_queue = queue.Queue()
        self.processing_thread = threading.Thread(target=self.yuv_frame_processing)
        self.renderer = None

    def start(self):
        # Connect to drone
        assert self.drone.connect(retry=3)

        if DRONE_RTSP_PORT is not None:
            self.drone.streaming.server_addr = f"{DRONE_IP}:{DRONE_RTSP_PORT}"

        # You can record the video stream from the drone if you plan to do some
        # post processing.
        self.drone.streaming.set_output_files(
            video=os.path.join(self.tempd, "streaming.mp4"),
            metadata=os.path.join(self.tempd, "streaming_metadata.json"),
        )

        # Setup your callback functions to do some live video processing
        self.drone.streaming.set_callbacks(
            raw_cb=self.yuv_frame_cb,
            flush_raw_cb=self.flush_cb,
        )
        # Start video streaming
        self.drone.streaming.start()
        # self.renderer = PdrawRenderer(pdraw=self.drone.streaming)
        self.running = True
        self.processing_thread.start()

    def stop(self):
        self.running = False
        self.processing_thread.join()
        if self.renderer is not None:
            self.renderer.stop()
        # Properly stop the video stream and disconnect
        assert self.drone.streaming.stop()
        assert self.drone.disconnect()
    def yuv_frame_cb(self, yuv_frame):
        """
        This function will be called by Olympe for each decoded YUV frame.

            :type yuv_frame: olympe.VideoFrame
        """
        # print("vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv")
        self.show_yuv_frame(yuv_frame)
        yuv_frame.ref()
        self.frame_queue.put_nowait(yuv_frame)

    def yuv_frame_processing(self):
        while self.running:
            try:
                yuv_frame = self.frame_queue.get(timeout=0.1)
            except queue.Empty:
                continue
            # You should process your frames here and release (unref) them when you're done.
            # Don't hold a reference on your frames for too long to avoid memory leaks and/or memory
            # pool exhaustion.
            yuv_frame.unref()

    def flush_cb(self, stream):
        if stream["vdef_format"] != olympe.VDEF_I420:
            return True
        while not self.frame_queue.empty():
            self.frame_queue.get_nowait().unref()
        return True



    def show_yuv_frame(self, yuv_frame):
        global counter
        global interrupt
        counter +=1
        # print(counter)
        # if counter == 500:
            # interrupt = True
            # self.make_hover()
        # if counter == 700:
        #     interrupt = False

        # the VideoFrame.info() dictionary contains some useful information
        # such as the video resolution
        info = yuv_frame.info()

        height, width = (  # noqa
            info["raw"]["frame"]["info"]["height"],
            info["raw"]["frame"]["info"]["width"],
        )
        # print(yuv_frame.vmeta())
        
        di = {}
        di['lat'] = yuv_frame.vmeta()[1]["drone"]["location"]["latitude"]
        di['lon'] = yuv_frame.vmeta()[1]["drone"]["location"]["longitude"]
        di['w'] = yuv_frame.vmeta()[1]["camera"]["quat"]["w"]
        di['x'] = yuv_frame.vmeta()[1]["camera"]["quat"]["x"]
        di['y'] = yuv_frame.vmeta()[1]["camera"]["quat"]["y"]
        di['z'] = yuv_frame.vmeta()[1]["camera"]["quat"]["z"]
        di['alt'] = yuv_frame.vmeta()[1]["drone"]["ground_distance"]
        # di['roll'],di['pitch'],di['yaw'] = self.quaternion_to_euler(di['y']*-1.0,di['z']*-1.0,di['w'],di['x'])
        di['roll'],di['pitch'],di['yaw'] = self.quaternion_to_euler(di['w'],di['x'],di['y'],di['z'])
        # print(di['roll'],di['pitch'],di['yaw'])
        di['pitch'] = 180.0 - di['pitch']
        di['roll'] = 180.0 - di['roll']
        di['yaw'] = di['yaw']+270
        
        # print(di['roll'],di['pitch'],di['yaw'])
        # print(di['roll'],di["pitch"],di['yaw'])
        # print(di)
        # convert pdraw YUV flag to OpenCV YUV flag
        cv2_cvt_color_flag = {
            olympe.VDEF_I420: cv2.COLOR_YUV2BGR_I420,
            olympe.VDEF_NV12: cv2.COLOR_YUV2BGR_NV12,
        }[yuv_frame.format()]
        cv2frame = cv2.cvtColor(yuv_frame.as_ndarray(), cv2_cvt_color_flag)  # noqa
        
        frme = imutils.resize(cv2frame,width=400)
        encoded,buffer = cv2.imencode('.jpg',frme,[cv2.IMWRITE_JPEG_QUALITY,80])
        frameBytes = buffer.tobytes()
        encoded_string = base64.b64encode(frameBytes)
        di['image'] = encoded_string.decode()
        sock.SendData(json.dumps(di).encode('utf-8'))
        # print("...")
        data = sock.ReadReceivedData() 
        if data != None: # if NEW data has been received since last ReadReceivedData function call
            print(type(data)) # print new received data
            print(data)
            data = "{"+data+"}"
            data = json.loads(data)
            if "hover" in data.keys():
                if data['hover'] == "True":
                    interrupt = True
                elif data['hover'] == "False":
                    interrupt = False
                    
            else:
                c = GC.CameraRayProjection(69,[float(data["lat"]),float(data["lon"]),float(data["alt"])],[int(float(data["resw"])),int(float(data["resh"]))],GC.Coordinates(int(float(data["xpos"])),int(float(data["ypos"]))),[float(data["w"]),float(data["x"]), float(data["y"]), float(data["z"])])
                target_direction_ENU = c.target_ENU()
                target_direction_ECEF = c.ENU_to_ECEF(target_direction_ENU)
                intersect_ECEF = c.target_location(target_direction_ECEF)
                #print("Intersect ECEF", intersect_ECEF.x,intersect_ECEF.y,intersect_ECEF.z)
                intersect_LLA = c.ECEFtoLLA(intersect_ECEF.x,intersect_ECEF.y,intersect_ECEF.z)
                print(intersect_LLA)
                di2 = {
                    'lat' : str(intersect_LLA[0]),
                    'lon' : str(intersect_LLA[1]),
                    'alt' : str(intersect_LLA[2])
                }
                sock2.SendData(json.dumps(di2).encode('utf-8'))
    def print_test(self):
        print("Testinggggggggggggggggggggggggggggggggggggggggggggggg")

    def make_hover(self):
        print("makeeeeeeeeeeeeeeeeeeeeeeeeeee______hover")
        self.drone(CancelMoveTo(_timeout = 100)).wait()
        print(self.drone.get_state(moveToChanged))
        self.print_test()
        # # time.sleep(100)
        # self.drone(moveByEnd(_timeout = 100)).wait()
        # # self.drone(moveByChanged(_timeout = 100)).wait()
        # print("Changed................",self.drone.get_state(moveByEnd))
            
    def check_hover(self):
        global interrupt
        print("In hoverrrrrrrrrrrrr")
        while interrupt:
            print("hoveringgggggggggg")
            time.sleep(1)

    def returnHomeandLand(self):
        print("-------------------------------------------------------------------------------")
        print(self.drone.get_state(HomeChanged))
        self.drone(                                                      
            moveTo(self.drone.get_state(HomeChanged)['latitude'], self.drone.get_state(HomeChanged)['longitude'], 65.0, mode.orientation_mode.to_target, 0.0, _timeout = 100)                                      
            >> moveToChanged(status=status.DONE, _timeout=100)
            ).wait().success()
        print(self.drone.get_state(moveToChanged))
        print("Landing...")
        self.drone(Landing() >> FlyingStateChanged(state="landed", _timeout=50)).wait()
        print("Landed\n")

    def fly(self):
        global interrupt
        # print('Streamingggggggggggg')
        # time.sleep(680)
        # Takeoff, fly, land, ...
        print("Takeoff if necessary...")
        self.drone(
            FlyingStateChanged(state="hovering", _policy="check")
            | FlyingStateChanged(state="flying", _policy="check")
            | (
                GPSFixStateChanged(fixed=1, _timeout=10, _policy="check_wait")
                >> (
                    TakeOff(_no_expect=True)
                    & FlyingStateChanged(
                        state="hovering", _timeout=10, _policy="check_wait"
                    )
                )
            )
        ).wait()
        
        self.drone( set_target( gimbal_id = 0,
                                  control_mode = "position",
                                  yaw_frame_of_reference = "relative",
                                  yaw = 0.0,
                                  pitch_frame_of_reference = "relative",
                                  pitch = -50.0,
                                  roll_frame_of_reference = "relative",
                                  roll = 0.0
                                ) ).wait()
        print("ddddddddddddddddddddddddddddddd")
        print(self.drone.get_state(HomeChanged))
        # self.drone(moveBy(0, 0, -65, 0, _timeout=100)).wait().success()
        
        # points = [
            
        #     [38.63496112075127, -90.22988829020554],
        #     [38.63448710782678, -90.23004908799908],
        #     [38.63464262029892, -90.23080522048069],
        #     [38.63511513689434, -90.23063676564934],
        # ]
        # for i in points:
        #     start_time = time.time()  # Start timing
        #     print("====================================================")
        #     self.drone(                                                      
        #         moveTo(i[0], i[1], 65.0, mode.orientation_mode.to_target, 0.0,_timeout = 30)                                      
        #         >> moveToChanged(status=status.DONE, _timeout=30)
        #         ).wait().success()
        #     print(self.drone.get_state(moveToChanged))
            

        #     end_time = time.time()  # End timing
        #     duration = end_time - start_time  # Calculate duration

        #     print(f"Time taken for iteration {i}: {duration} seconds")

        # self.drone(                                                      
        #     moveTo(38.63448831349425, -90.23005939015252, 45.0, mode.orientation_mode.to_target, 0.0)                                      
        #     >> moveToChanged(status=status.DONE, _timeout=10)
        #     ).wait().success()
        # print(self.drone.get_state(moveToChanged))
        # self.drone(                                                      
        #     moveTo(38.6340093469394, -90.23021644652499, 45.0, mode.orientation_mode.to_target, 0.0)                                      
        #     >> moveToChanged(status=status.DONE, _timeout=10)
        #     ).wait().success()
        
        # print(self.drone.get_state(moveToChanged))
        
        

        
        
        self.drone(moveBy(0, 0, -35, 0, _timeout=100)).wait().success()
        # self.drone(moveBy(20, 0, 0, 0, _timeout=100)).wait().success()
        # self.drone(moveBy(0, 25, 0, 0, _timeout=100)).wait().success()
        # self.drone(moveBy(-20, 0, 0, 0, _timeout=100)).wait().success()
        # self.drone(moveBy(0, -25, 0, 0, _timeout=100)).wait().success()
        self.drone( set_target( gimbal_id = 0,
                                  control_mode = "position",
                                  yaw_frame_of_reference = "relative",
                                  yaw = 0.0,
                                  pitch_frame_of_reference = "relative",
                                  pitch = -80.0,
                                  roll_frame_of_reference = "relative",
                                  roll = 0.0
                                ) ).wait()
        time.sleep(5)
        self.drone( set_target( gimbal_id = 0,
                                  control_mode = "position",
                                  yaw_frame_of_reference = "relative",
                                  yaw = 0.0,
                                  pitch_frame_of_reference = "relative",
                                  pitch = -40.0,
                                  roll_frame_of_reference = "relative",
                                  roll = 0.0
                                ) ).wait()
        time.sleep(5)
        self.drone( set_target( gimbal_id = 0,
                                  control_mode = "position",
                                  yaw_frame_of_reference = "relative",
                                  yaw = 0.0,
                                  pitch_frame_of_reference = "relative",
                                  pitch = -20.0,
                                  roll_frame_of_reference = "relative",
                                  roll = 0.0
                                ) ).wait()
        self.drone(moveBy(0, 0, 35, 0, _timeout=10)).wait().success()
        
        # for i in range(30):
        #     self.check_hover()
        #     self.drone(moveBy(1, 0, 0, 0, _timeout=1)).wait().success()
        #     print(i)
        #     print("-----------------")
        # for i in range(30):
        #     self.check_hover()
        #     self.drone(moveBy(-1, 0, 0, 0, _timeout=1)).wait().success()
        #     print(i)
        #     print('=================')
        

        # self.returnHomeandLand()
        # self.drone(moveBy(0, 0, 45, 0, _timeout=10)).wait().success()
        # self.drone(NavigateHome(1, _timeout = 30)).wait().success()
        

        print("Landing...")
        self.drone(Landing() >> FlyingStateChanged(state="landed", _timeout=5)).wait()
        print("Landed\n")

        import numpy as np

    def quaternion_to_euler(self,w, x, y, z):
        ysqr = y * y

        t0 = +2.0 * (w * x + y * z)
        t1 = +1.0 - 2.0 * (x * x + ysqr)
        X = np.degrees(np.arctan2(t0, t1))

        t2 = +2.0 * (w * y - z * x)
        t2 = np.where(t2>+1.0,+1.0,t2)
        #t2 = +1.0 if t2 > +1.0 else t2

        t2 = np.where(t2<-1.0, -1.0, t2)
        #t2 = -1.0 if t2 < -1.0 else t2
        Y = np.degrees(np.arcsin(t2))

        t3 = +2.0 * (w * z + x * y)
        t4 = +1.0 - 2.0 * (ysqr + z * z)
        Z = np.degrees(np.arctan2(t3, t4))

        return X, Y, Z


# print(quaternion_to_euler_angle_vectorized1(0.290620893239975, -0.6446235179901123, -0.2906208634376526,-0.6446235775947571))
        
 

def test_streaming():
    streaming_example = StreamingExample()
    # Start the video stream
    streaming_example.start()
    # Perform some live video processing while the drone is flying
    streaming_example.fly()
    # Stop the video stream
    streaming_example.stop()
    # Recorded video stream postprocessing
    # streaming_example.replay_with_vlc()


if __name__ == "__main__":
    test_streaming()