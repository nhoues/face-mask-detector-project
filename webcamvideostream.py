import os
import socket
import sys
import pickle
import struct
import numpy as np
import matplotlib.pyplot as plt
import torch
import time, datetime
import json

import imutils
from imutils.video import VideoStream
from imutils.video import FPS
from itertools import zip_longest

import argparse
import cv2
import dlib
from threading import Thread

import torch
import albumentations as A
import onnxruntime

from mylib.centroidtracker import CentroidTracker
from mylib.trackableobject import TrackableObject
from mylib.mailer import Mailer
from mylib import config, thread


def to_numpy(tensor):
    return (
        tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()
    )


def sigmod(x):
    return 1 / (1 + np.exp(-x))


class Predicator:
    def __init__(self, model_path):
        self.model = onnxruntime.InferenceSession(model_path)

    def prep_image(self, image):
        aug = A.Compose(
            [
                A.Resize(224, 224, p=1.0),
                A.Normalize(mean=[0.485], std=[0.229], max_pixel_value=255.0, p=1.0,),
            ],
            p=1.0,
        )

        image = aug(image=np.array(image))["image"]
        image = np.transpose(image, (2, 0, 1)).astype(float)
        image = torch.tensor(image, dtype=torch.float).unsqueeze(0)
        return image

    def batch_pred(self, images):
        all_images = []
        for image in images:
            all_images.append(self.prep_image(image))
        all_images = torch.cat(all_images, dim=0)
        inputs = {self.model.get_inputs()[0].name: to_numpy(all_images)}
        outs = self.model.run(None, inputs)
        outs = sigmod(outs[0])
        return outs

    def predict(self, image):
        image = self.prep_image(image)
        inputs = {self.model.get_inputs()[0].name: to_numpy(image)}
        outs = self.model.run(None, inputs)
        outs = sigmod(outs[0])
        return outs


def detect_and_predict_mask(frame, faceNet, predicator):
    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), (104.0, 177.0, 123.0))
    faceNet.setInput(blob)
    detections = faceNet.forward()
    faces = []
    locs = []
    preds = []
    for i in range(0, detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > 0.5:
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")
            (startX, startY) = (max(0, startX), max(0, startY))
            (endX, endY) = (min(w - 1, endX), min(h - 1, endY))
            face = frame[startY:endY, startX:endX]
            face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
            faces.append(face)
            locs.append((startX, startY, endX, endY))
    if len(faces) > 0:
        preds = predicator.batch_pred(faces)
    return (locs, preds)


t0 = time.time()

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument(
    "-p",
    "--prototxt",
    default="mobilenet_ssd/MobileNetSSD_deploy.prototxt",
    help="path to Caffe 'deploy' prototxt file",
)
ap.add_argument(
    "-m",
    "--model",
    default="mobilenet_ssd/MobileNetSSD_deploy.caffemodel",
    help="path to Caffe pre-trained model",
)

ap.add_argument(
    "-f",
    "--face",
    type=str,
    default="face_detector",
    help="path to face detector model directory",
)

ap.add_argument(
    "-d",
    "--model-detector",
    type=str,
    default="mask_detector.onnx",
    help="path to trained face mask detector model",
)


# confidence default 0.4
ap.add_argument(
    "-c",
    "--confidence",
    type=float,
    default=0.4,
    help="minimum probability to filter weak detections",
)
ap.add_argument(
    "-s",
    "--skip-frames",
    type=int,
    default=30,
    help="# of skip frames between detections",
)


args = vars(ap.parse_args())

CLASSES = [
    "background",
    "aeroplane",
    "bicycle",
    "bird",
    "boat",
    "bottle",
    "bus",
    "car",
    "cat",
    "chair",
    "cow",
    "diningtable",
    "dog",
    "horse",
    "motorbike",
    "person",
    "pottedplant",
    "sheep",
    "sofa",
    "train",
    "tvmonitor",
]


# load our serialized face detector model from disk
print("[INFO] loading face detector model...")
prototxtPath = os.path.sep.join([args["face"], "deploy.prototxt"])
weightsPath = os.path.sep.join(
    [args["face"], "res10_300x300_ssd_iter_140000.caffemodel"]
)
faceNet = cv2.dnn.readNet(prototxtPath, weightsPath)

# load the face mask detector model from disk
print("[INFO] loading face mask detector model...")
model_path = args["model_detector"]
predicator = Predicator(model_path)

net = cv2.dnn.readNetFromCaffe(args["prototxt"], args["model"])

HOST = "10.130.2.94"
PORT = 8026


class WebcamVideoStream:
    def __init__(self, src=0):
        self.max_people = 10
        # initialize the video stream and allow the camera sensor to warm up
        print("[INFO] starting video stream...")
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        print("Socket created")
        s.bind((HOST, PORT))
        print("Socket bind complete")
        s.listen(10)
        print("Socket now listening")
        self.conn, addr = s.accept()
        self.data = b""
        self.payload_size = struct.calcsize("L")

        # initialize the video writer (we'll instantiate later if need be)
        self.writer = None

        # initialize the frame dimensions (we'll set them as soon as we read
        # the first frame from the video)
        self.W = None
        self.H = None

        self.ct = CentroidTracker(maxDisappeared=40, maxDistance=50)
        self.trackers = []
        self.trackableObjects = {}

        self.totalFrames = 0
        self.totalDown = 0
        self.totalUp = 0
        self.x = []
        self.empty = []
        self.empty1 = []

        self.fps = FPS().start()

        # Reading frame
        while len(self.data) < self.payload_size:
            self.data += self.conn.recv(4096)
        packed_msg_size = self.data[: self.payload_size]

        self.data = self.data[self.payload_size :]
        msg_size = struct.unpack("L", packed_msg_size)[0]

        while len(self.data) < msg_size:
            self.data += self.conn.recv(4096)
        frame_data = self.data[:msg_size]
        self.data = self.data[msg_size:]
        frame = pickle.loads(frame_data)
        self.frame = imutils.resize(frame, width=400)
        self.stopped = False
        time.sleep(2.0)

    def start(self):
        print("start thread")
        t = Thread(target=self.update, args=())
        t.daemon = True
        t.start()
        return self

    def update(self):
        print("read")
        while True:
            if self.stopped:
                return
            # Reading frame
            while len(self.data) < self.payload_size:
                self.data += self.conn.recv(4096)
            packed_msg_size = self.data[: self.payload_size]

            self.data = self.data[self.payload_size :]
            msg_size = struct.unpack("L", packed_msg_size)[0]

            while len(self.data) < msg_size:
                self.data += self.conn.recv(4096)
            frame_data = self.data[:msg_size]
            self.data = self.data[msg_size:]
            frame = pickle.loads(frame_data)
            self.frame = frame

    def read(self):
        frame = self.frame
        frame = imutils.resize(frame, width=500)

        (locs, preds) = detect_and_predict_mask(frame, faceNet, predicator)

        for (box, pred) in zip(locs, preds):
            # unpack the bounding box and predictions
            (startX, startY, endX, endY) = box

            label_ = "Mask" if pred[0] > 0.7 else "No Mask"
            color = (0, 255, 0) if label_ == "Mask" else (0, 0, 255)

            # include the probability in the label
            label = "{}: {:.2f}%".format(label_, pred[0] * 100)

            # display the label and bounding box rectangle on the output
            # frame
            cv2.putText(
                frame,
                label,
                (startX, startY - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                color,
                2,
            )
            cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        if self.W is None or self.H is None:
            (self.H, self.W) = frame.shape[:2]

        status = "Waiting"
        rects = []
        if self.totalFrames % args["skip_frames"] == 0:
            status = "Detecting"
            self.trackers = []
            blob = cv2.dnn.blobFromImage(frame, 0.007843, (self.W, self.H), 127.5)
            net.setInput(blob)
            detections = net.forward()
            # loop over the detections
            for i in np.arange(0, detections.shape[2]):
                confidence = detections[0, 0, i, 2]
                idx = int(detections[0, 0, i, 1])
                if CLASSES[idx] != "person":
                    continue
                box = detections[0, 0, i, 3:7] * np.array(
                    [self.W, self.H, self.W, self.H]
                )
                (startX, startY, endX, endY) = box.astype("int")
                tracker = dlib.correlation_tracker()
                rect = dlib.rectangle(startX, startY, endX, endY)
                tracker.start_track(rgb, rect)
                self.trackers.append(tracker)
        else:
            for tracker in self.trackers:
                status = "Tracking"
                tracker.update(rgb)
                pos = tracker.get_position()
                startX = int(pos.left())
                startY = int(pos.top())
                endX = int(pos.right())
                endY = int(pos.bottom())
                rects.append((startX, startY, endX, endY))

        cv2.line(frame, (0, self.H // 2), (self.W, self.H // 2), (0, 0, 0), 3)
        objects = self.ct.update(rects)
        for (objectID, centroid) in objects.items():
            to = self.trackableObjects.get(objectID, None)
            if to is None:
                to = TrackableObject(objectID, centroid)
            else:
                y = [c[1] for c in to.centroids]
                direction = centroid[1] - np.mean(y)
                to.centroids.append(centroid)
                if not to.counted:
                    if direction < 0 and centroid[1] < self.H // 2:
                        self.totalUp += 1
                        self.empty.append(self.totalUp)
                        to.counted = True
                    elif direction > 0 and centroid[1] > self.H // 2:
                        self.totalDown += 1
                        self.empty1.append(self.totalDown)
                        self.x = []
                        self.x.append(len(self.empty1) - len(self.empty))
                        to.counted = True
            self.trackableObjects[objectID] = to
            text = "ID {}".format(objectID)
            cv2.putText(
                frame,
                text,
                (centroid[0] - 10, centroid[1] - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                2,
            )
            cv2.circle(frame, (centroid[0], centroid[1]), 4, (255, 255, 255), -1)
        status = "door locked"
        try:
            if (
                (self.totalDown - self.totalUp) >= self.max_people
            ) or label_ == "No Mask":
                status = "door locked"

            elif label_ == "Mask":
                status = "door opened"

        except:
            status = "door locked"

        info = [
            ("Total people inside", (self.totalDown - self.totalUp)),
            ("Status", status),
        ]

        for (i, (k, v)) in enumerate(info):
            text = "{}: {}".format(k, v)
            cv2.putText(
                frame,
                text,
                (10, self.H - ((i * 20) + 20)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 0, 0),
                2,
            )
        d = {}
        try:
            info.append(("Mask state", label_))
        except:
            info.append(("Mask state", "empty"))
        for (k, v) in info:
            d[k] = v
        HEADERSIZE = 10

        msg = pickle.dumps(d)
        msg = bytes(f"{len(msg):<{HEADERSIZE}}", "utf-8") + msg
        self.conn.send(msg)

        self.totalFrames += 1
        self.fps.update()
        return frame

    def stop(self):
        self.stopped = True
