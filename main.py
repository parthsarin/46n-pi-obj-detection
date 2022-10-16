"""
File: main.py
----------------

This file runs the neural network on video input to identify people and 
bicycles.
"""
import cv2
import numpy as np
import sys
import time

# Configuration files
CONFIG_FILE = 'yolov3.cfg'
WEIGHTS_FILE = 'yolov3.weights'
CLASSES_FILE = 'yolov3.txt'
IMAGE_FILE = 'dog.jpeg'

# Parameters
SCALE = 0.00392
CLASSES = open(CLASSES_FILE).read().strip().split('\n')
COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))
CONF_THRESH = 0.5   # confidence threshold for making a prediction
NMS_THRESH = 0.4    # non-max suppression threshold

# Load the neural network
net = cv2.dnn.readNet(WEIGHTS_FILE, CONFIG_FILE)

def img_blob(img):
    """
    Creates a blob that can be fed into the network from
    the image input.
    """
    blob = cv2.dnn.blobFromImage(
        img, 
        SCALE, 
        (416, 416), 
        (0, 0, 0), 
        True, 
        crop=False
    )
    return blob


def get_output_names(net):
    """
    Get the names of the output layers
    """
    layer_names = net.getLayerNames()
    output_layers = [
        layer_names[i - 1] 
        for i in net.getUnconnectedOutLayers()
    ]
    return output_layers


def draw_box(img, class_id, x, y, x_plus_w, y_plus_h):
    """
    Draws a bounding box around the detected object.
    """
    label = str(CLASSES[class_id])
    color = COLORS[class_id]
    cv2.rectangle(img, (x, y), (x_plus_w, y_plus_h), color, 2)
    cv2.putText(
        img, label, (x - 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 
        0.5, color, 2
    )


def make_prediction(img):
    """
    Makes a prediction on the image.
    """
    width, height = img.shape[1], img.shape[0]
    blob = img_blob(img)
    net.setInput(blob)
    outs = net.forward(get_output_names(net))

    # keep strong predictions
    class_ids = []
    confidences = []
    boxes = []

    # for each hit, get the parameters and keep if strong
    for x in outs:
        for pred in x:
            # how confident is the model?
            scores = pred[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence < CONF_THRESH:
                continue
            
            # draw the model
            center_x = int(pred[0] * width)
            center_y = int(pred[1] * height)
            w = int(pred[2] * width)
            h = int(pred[3] * height)
            x = center_x - w / 2
            y = center_y - h / 2
            class_ids.append(class_id)
            confidences.append(float(confidence))
            boxes.append([x, y, w, h])

    return boxes, confidences, class_ids


def draw_prediction(img, boxes, confidences, class_ids):
    """
    Draws the prediction on the image.
    """
    indices = cv2.dnn.NMSBoxes(boxes, confidences, CONF_THRESH, NMS_THRESH)
    
    for i in indices:
        x, y, w, h = boxes[i]
        
        draw_box(
            img, class_ids[i], 
            round(x), round(y), 
            round(x + w), round(y + h)
        )

    filtered_boxes = [boxes[i] for i in indices]
    filtered_confidences = [confidences[i] for i in indices]
    filtered_class_ids = [class_ids[i] for i in indices]

    return filtered_boxes, filtered_confidences, filtered_class_ids


def main():
    vid = cv2.VideoCapture(0)

    while True:
        _, frame = vid.read()
        t = time.gmtime()
        date = time.strftime('%d-%m-%Y', t)
        hms = time.strftime('%H:%M:%S', t)

        boxes, confidences, class_ids = make_prediction(frame)
        boxes, confidences, class_ids = draw_prediction(frame, boxes, confidences, class_ids)

        for b, conf, ci in zip(boxes, confidences, class_ids):
            x, y, w, h = b
            c = CLASSES[ci]
            if '--verbose' in sys.argv:
                print(f'found {c} at ({x}, {y}) with confidence {conf:.2%}')
            
            l = f'{date},{hms},{c},{x},{y},{w},{h},{conf:.4}'
            print(l)

        if '--visual' in sys.argv:
            cv2.imshow('object detection', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        
    
    vid.release()
    if '--visual' in sys.argv:
        cv2.destroyAllWindows()


if __name__ == '__main__':
    main()