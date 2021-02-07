import cv2
import numpy as np
import socket
import sys
import pickle
import struct
from flask import Flask, render_template

DEVELOPMENT_ENV = True

app = Flask(__name__)

app_data = {
    "title": "Covid Security System",
    "door": "Open",
    "mask": "UnMasked",
    "inst": "please wear your mask",
}
max_people = 10
HEADERSIZE = 10
cap = cv2.VideoCapture(0)
clientsocket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
clientsocket.connect(("10.130.2.94", 8026))


while True:
    ret, frame = cap.read()
    data = pickle.dumps(frame)
    clientsocket.sendall(struct.pack("L", len(data)) + data)

    full_msg = b""
    new_msg = True
    while True:
        msg = clientsocket.recv(16)
        if new_msg:
            msglen = int(msg[:HEADERSIZE])
            new_msg = False
        full_msg += msg
        if len(full_msg) - HEADERSIZE == msglen:
            data = pickle.loads(full_msg[HEADERSIZE:])
            break

    app_data["door"] = data["Status"]
    app_data["mask"] = data["Mask state"]

    if app_data["mask"] == "No Mask":
        app_data["inst"] = "Please wear your mask"
    elif data["Total people inside"] > max_people:
        app_data["inst"] = "Please wait, Limit of people inside exceeded"
    else:
        app_data["inst"] = "Welecome, Stay safe"

"""
@app.route('/')
def index():
    return render_template('index.html', app_data=app_data)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5002 , debug=DEVELOPMENT_ENV)"""
