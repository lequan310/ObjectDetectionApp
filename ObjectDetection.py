import cv2
import numpy as np

font = cv2.FONT_HERSHEY_PLAIN
colors = np.random.uniform(0, 255, size=(100, 3))
dimension = 320
confThreshold = 0.5
nmsThreshold = 0.2

classFiles = 'coco.names'
classes = []
with open(classFiles, 'r') as f:
    classes = f.read().splitlines()

modelConfig = 'yolov3.cfg'
modelWeights = 'yolov3.weights'

net = cv2.dnn.readNetFromDarknet(modelConfig, modelWeights)
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)


def find_objects(outputs, image, draw_box=True):
    # Local variables
    height, width, _ = image.shape
    bboxes = []  # bbox = list of bounding box
    class_ids = []  # list of class ID
    confidences = []  # confidences = list of confidence

    # Get bounding boxes, classes, and confidences of objects
    for output in outputs:
        for detection in output:  # det = detection
            scores = detection[5:]  # remove first 5 elements
            class_id = np.argmax(scores)  # Get class with highest accuracy
            confidence = scores[class_id]  # Get accuracy of that class

            if confidence > confThreshold:
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)

                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                bboxes.append([x, y, w, h])
                confidences.append((float(confidence)))
                class_ids.append(class_id)

    # print(len(bboxes))
    indexes = cv2.dnn.NMSBoxes(bboxes, confidences, confThreshold, nmsThreshold)
    # print(indexes.flatten())

    # Draw bounding boxes
    if draw_box:
        for i in indexes.flatten():
            x, y, w, h = bboxes[i]
            label = str(classes[class_ids[i]])
            confidence = str(round(confidences[i], 2))
            color = colors[class_ids[i]]
            cv2.rectangle(image, (x, y), (x+w, y+h), color, 2)
            cv2.putText(image, label + " " + confidence, (x, y + 20), font, 2, color, 2)


def scale_img(image, scale_percent):
    iw = int(image.shape[1] * scale_percent / 100)
    ih = int(image.shape[0] * scale_percent / 100)
    dim = (iw, ih)
    return cv2.resize(image, dim, interpolation=cv2.INTER_CUBIC)


def detect_image(img_src):
    img = cv2.imread(img_src)
    img = scale_img(img, 100)
    detect_objects(img)


def detect_video(video_src):
    cap = cv2.VideoCapture(video_src)

    while True:
        success, img = cap.read()
        detect_objects(img)


def detect_objects(img):
    blob = cv2.dnn.blobFromImage(img, 1 / 255, (dimension, dimension), (0, 0, 0), swapRB=True, crop=False)
    # for b in blob:
    #     for n, img_blob in enumerate(b):
    #         cv2.imshow(str(n), img_blob)

    net.setInput(blob)
    output_layers_names = net.getUnconnectedOutLayersNames()
    layer_outputs = net.forward(
        output_layers_names)  # (300 output layers, 85) cx, cy, w, h, confidence, 80 classes => 85
    find_objects(layer_outputs, img)

    cv2.imshow('Image', img)
    cv2.waitKey(1)


# detect_image('office.jpg')
detect_video(0)
