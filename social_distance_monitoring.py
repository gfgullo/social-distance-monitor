import cv2
import numpy as np
from scipy.spatial import distance as dist

TARGET_CLASS = "person"
MIN_CONFIDENCE = 0.5
MIN_DISTANCE = 30

SIZE = (320, 320)

GREEN = (0, 255, 0)
RED = (0, 0, 255)

LINE_THICKNESS = 1
CIRCLE_RADIUS =  3


def load_yolo(yolo_path=""):
	net = cv2.dnn.readNet(yolo_path+"yolov3.weights", yolo_path+"yolov3.cfg")
	classes = []
	with open(yolo_path+"coco.names", "r") as f:
		classes = [line.strip() for line in f.readlines()]
	layers_names = net.getLayerNames()
	output_layers = [layers_names[i[0]-1] for i in net.getUnconnectedOutLayers()]
	colors = np.random.uniform(0, 255, size=(len(classes), 3))
	return net, classes, colors, output_layers


def detect_people(img, net, outputLayers):			
    img = cv2.resize(img, SIZE)
    blob = cv2.dnn.blobFromImage(img, scalefactor=1/255, size=SIZE)
    net.setInput(blob)
    outputs = net.forward(outputLayers)
    return blob, outputs


def get_boxes(outputs, height, width):

    boxes = []

    for output in outputs:
        for detect in output:
            scores = detect[5:]
            class_id = np.argmax(scores)
            conf = scores[class_id]

            if(classes[class_id]==TARGET_CLASS and conf > MIN_CONFIDENCE):
            
                center_x = int(detect[0] * width)
                center_y = int(detect[1] * height)
                w = int(detect[2] * width)
                h = int(detect[3] * height)
                x = int(center_x - w/2)
                y = int(center_y - h / 2)

                boxes.append([x, y, w, h])
    
    return boxes


def get_centroids(rects):

    centroids = []
    for i in range(len(rects)):
        centroids.append(get_centroid(rects[i]))

    return centroids
    

def get_centroid(rect):
    return ((2*rect[0]+rect[2])//2, (2*rect[1]+rect[3])//2)


def compute_distances(centroids):
    centroids = np.array(centroids)
    dist_matrix = dist.cdist(centroids, centroids)
    dist_matrix = dist_matrix+np.eye(dist_matrix.shape[0], dist_matrix.shape[1])*1000
    return dist_matrix


def get_contact_indices(dist_matrix):
    indices = np.where((dist_matrix<=MIN_DISTANCE) & (dist_matrix>5))
    contact_indices = list(zip(indices[0],indices[1]))
    #contact_indices = np.column_stack([indices[0], indices[1]])
    return contact_indices


def draw_results(img, centroids, alert):

    centroids_drawn = set()

    for c1, c2 in alert:

        centroid1 = centroids[c1]
        centroid2 = centroids[c2]

        cv2.circle(img, (centroid1[0], centroid1[1]), CIRCLE_RADIUS, RED, cv2.FILLED)
        cv2.circle(img, (centroid2[0], centroid2[1]), CIRCLE_RADIUS, RED, cv2.FILLED)

        cv2.line(img, centroid1, centroid2, RED, thickness=LINE_THICKNESS)

        centroids_drawn.add(centroid1)
        centroids_drawn.add(centroid2)

    centroids_to_draw = set(centroids) - centroids_drawn

    for centroid in centroids_to_draw:
        cv2.circle(img, (centroid[0],centroid[1]), CIRCLE_RADIUS, (0, 255, 0), cv2.FILLED)

    return img
    


net, classes, colors, output_layers = load_yolo()

#video_path = input("Insert video path: ")
input_video_path = "video.mp4"
output_video_path = "output.avi"

cap = cv2.VideoCapture(input_video_path)
ret, frame = cap.read()

if(not ret):
    print("Errore durante il caricamento del video")
    exit(0)

codec = cv2.VideoWriter_fourcc(*'MJPG')
out = cv2.VideoWriter(output_video_path, codec, 20.0, (frame.shape[0], frame.shape[1]))


frames_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
current_frame = 1

while(cap.isOpened()):

    ret, frame = cap.read()

    if(not ret):
        break

    print("Processing frame %d of %d..." % (current_frame, frames_count), end="")

    blob, outputs = detect_people(frame, net, output_layers)
    boxes = get_boxes(outputs, frame.shape[0], frame.shape[1])
    centroids = get_centroids(boxes)

    dist_matrix = compute_distances(centroids)
    contact_indices = get_contact_indices(dist_matrix)

    frame = draw_results(frame, centroids, contact_indices)

    print("DONE")

    cv2.imshow("Social Distance Monitoring", frame)
    cv2.waitKey(1)
    out.write(frame)

    current_frame+=1

cap.release()
out.release()
cv2.destroyAllWindows()
