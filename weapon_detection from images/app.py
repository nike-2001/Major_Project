import cv2
import numpy as np
import argparse
import mimetypes

source_img = "./test_images/weapon0.jpg"
confidence_thershold = 0.1
nms_threshold = 0.1
class_names = ["Weapon"]
width_model, height_model = 416, 416
config_path = "./yolov4-custom.cfg"
weights_path = "./yolov4-custom_last.weights"

# Set Argument Parse
parser = argparse.ArgumentParser()
parser.add_argument(
    "-s",
    "--source-img",
    default=source_img,
    help="Input your image source to detect the object",
)
parser.add_argument(
    "-c",
    "--confidence",
    type=float,
    default=confidence_thershold,
    help="Input your minimal value to detect the object",
)
parser.add_argument(
    "-n",
    "--nms",
    type=float,
    default=nms_threshold,
    help="Input your minimal value nms threshold",
)
parser.add_argument(
    "-cls",
    "--class-names",
    nargs="+",
    default=class_names,
    help="Input your custom classes",
)
parser.add_argument(
    "-cfg", "--config", default=config_path, help="Input your custom config path"
)
parser.add_argument(
    "-w", "--weights", default=weights_path, help="Input your custom weights path"
)
parser.add_argument(
    "-wm", "--width-model", default=width_model, type=int, help="Input your model width requirements"
)
parser.add_argument(
    "-hm", "--height-model", default=height_model, type=int, help="Input your model height requirements"
)
value_parser = parser.parse_args()
source_img = value_parser.source_img
confidence_thershold = value_parser.confidence
nms_threshold = value_parser.nms
class_names = value_parser.class_names
width_model = value_parser.width_model
height_model = value_parser.height_model
config_path = value_parser.config
weights_path = value_parser.weights

# Check Source Img
mimestart = mimetypes.guess_type(value_parser.source_img)[0]
if mimestart != None:
    mimestart = mimestart.split("/")[0]
    if mimestart not in ["image"]:
        raise "Input image source correctly!"

def find_objects(outputs, img, confidence_thershold, nms_threshold):
    hT, wT, cT = img.shape
    bbox = []
    class_ids = []
    confidences = []

    for output in outputs:
        for det in output:
            scores = det[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > confidence_thershold:
                w, h = int(det[2] * wT), int(det[3] * hT)
                x, y = int((det[0] * wT) - w / 2), int((det[1] * hT) - h / 2)
                bbox.append([x, y, w, h])
                class_ids.append(class_id)
                confidences.append(float(confidence))

    indices = cv2.dnn.NMSBoxes(bbox, confidences, confidence_thershold, nms_threshold)
    for i in indices:
        i = i[0]
        box = bbox[i]
        x, y, w, h = box[0], box[1], box[2], box[3]
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 255), 2)
        cv2.putText(
            img,
            f"{class_names[class_ids[i]].upper()} {int(confidences[i]*100)}%",
            (x, y - 10),
            cv2.FONT_HERSHEY_PLAIN,
            1,
            (255, 0, 255),
            2,
        )


net = cv2.dnn.readNetFromDarknet(config_path, weights_path)
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

img = cv2.imread(source_img)
blob = cv2.dnn.blobFromImage(
    img, 1 / 255, (width_model, height_model), (0, 0, 0), 1, crop=False
)
net.setInput(blob)
layer_names = net.getLayerNames()
output_names = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
outputs = net.forward(output_names)
find_objects(outputs, img, confidence_thershold, nms_threshold)

cv2.imshow("Weapon", img)
cv2.waitKey(0)

cv2.destroyAllWindows()
