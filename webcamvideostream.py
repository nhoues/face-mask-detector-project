
import cv2
from threading import Thread
import time
import numpy as np
import numpy as np 
import matplotlib.pyplot as plt
import torch 

import albumentations as A
import onnxruntime
from imutils.video import VideoStream
import argparse
import imutils
import time
import cv2
import os

def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()

def sigmod(x) : 
    return 1/(1+ np.exp(-x))

class Predicator() : 
    def __init__(self , model_path ) : 
        self.model = onnxruntime.InferenceSession(model_path)
        
    def prep_image(self , image ):
        aug = A.Compose([
                              A.Resize(224, 224, p= 1.0),
                              A.Normalize(
                                  mean=[0.485],
                                  std=[0.229],
                                  max_pixel_value=255.0,
                                  p=1.0,
                              ),
                          ],
                            p=1.0,
                        )

        image = aug(image  = np.array(image))['image']
        image = np.transpose(image , (2,0,1)).astype(float) 
        image = torch.tensor(image ,dtype = torch.float).unsqueeze(0)
        return image
    
    def batch_pred(self , images) : 
        all_images = [] 
        for image in images : 
            all_images.append(self.prep_image(image))
        all_images = torch.cat(all_images , dim = 0)
        inputs = {self.model.get_inputs()[0].name: to_numpy(all_images)}
        outs = self.model.run(None, inputs)
        outs = sigmod(outs[0])
        return outs
    
    def predict(self , image ) :
        image = self.prep_image(image)
        inputs = {self.model.get_inputs()[0].name: to_numpy(image)}
        outs = self.model.run(None, inputs)
        outs = sigmod(outs[0])
        return outs

def detect_and_predict_mask(frame, faceNet, predicator):
    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300),
        (104.0, 177.0, 123.0))
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

ap = argparse.ArgumentParser()

ap.add_argument("-f", "--face", type=str,
    default="face_detector",
    help="path to face detector model directory")

ap.add_argument("-m", "--model", type=str,
    default="mask_detector.onnx",
    help="path to trained face mask detector model")

args = vars(ap.parse_args())

# load our serialized face detector model from disk
print("[INFO] loading face detector model...")
prototxtPath = os.path.sep.join([args["face"], "deploy.prototxt"])
weightsPath = os.path.sep.join([args["face"],
	"res10_300x300_ssd_iter_140000.caffemodel"])
faceNet = cv2.dnn.readNet(prototxtPath, weightsPath)

# load the face mask detector model from disk
print("[INFO] loading face mask detector model...")
model_path = args['model']
predicator = Predicator(model_path)


class WebcamVideoStream:
    def __init__(self, src = 0):
        print("init")
        self.stream = cv2.VideoCapture(src)
        (self.grabbed, self.frame) = self.stream.read()
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
            
            (self.grabbed, self.frame) = self.stream.read()
    
    def read(self):
        frame = self.frame
        frame = imutils.resize(frame, width=400)
        (locs, preds) = detect_and_predict_mask(frame, faceNet, predicator)

        for (box, pred) in zip(locs, preds):
            # unpack the bounding box and predictions
            (startX, startY, endX, endY) = box

            label = "Mask" if pred[0] > 0.5  else "No Mask"
            color = (0, 255, 0) if label == "Mask" else (0, 0, 255)

            # include the probability in the label
            label = "{}: {:.2f}%".format(label, pred[0] * 100)

            # display the label and bounding box rectangle on the output
            # frame
            cv2.putText(frame, label, (startX, startY - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
            cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)
        return frame
    
    def stop(self):
        self.stopped = True
'''
        # initialize the video stream and allow the camera sensor to warm up
print("[INFO] starting video stream...")
vs = VideoStream(src=0).start()
time.sleep(2.0)

while True:
    frame = vs.read()
    frame = imutils.resize(frame, width=400)
    (locs, preds) = detect_and_predict_mask(frame, faceNet, predicator)
    
    for (box, pred) in zip(locs, preds):
        # unpack the bounding box and predictions
        (startX, startY, endX, endY) = box
        
        label = "Mask" if pred[0] > 0.5  else "No Mask"
        color = (0, 255, 0) if label == "Mask" else (0, 0, 255)

        # include the probability in the label
        label = "{}: {:.2f}%".format(label, pred[0] * 100)

        # display the label and bounding box rectangle on the output
        # frame
        cv2.putText(frame, label, (startX, startY - 10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
        cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)

    # show the output frame
    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF

    # if the `q` key was pressed, break from the loop
    if key == ord("q"):
        break

# do a bit of cleanup
cv2.destroyAllWindows()
vs.stop()
'''