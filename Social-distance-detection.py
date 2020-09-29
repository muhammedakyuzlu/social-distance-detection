import cv2
import numpy as np
import time
import math
import pafy
import argparse
import sys
import os


# ----------------- Algorithm--------------------
# Loading YOLO v4 Network
# Reading stream video or just an image
# Reading frames in the loop if it is a stream
# Getting blob from the frame
# Implementing Forward Pass
# Getting Bounding Boxes
# Non-maximum Suppression
# Take the boxes and measure the space between them by Euclidean form
# Color the violates box red and the respact on green
# Drawing and Showing Bounding Boxes
#--------------------------------------------------



def choose_input():
    # initial Argument Parser to run the script from the terminal

    # -v for video
    # -c for camera
    # -y for youtube
    # -i for image
    #  then add the path/url

    ap = argparse.ArgumentParser()
    group = ap.add_mutually_exclusive_group()
    group.add_argument("-v", "--video", action="store_true",
                       help="video stream")
    group.add_argument("-c",  "--camera", action="store_true",
                       help="camera stream")
    group.add_argument("-y",  "--youtube", action="store_true",
                       help="youtube video")
    group.add_argument("-i",  "--image", action="store_true",
                       help="image file")
    ap.add_argument(
        'path', help="path to the image file / video stream", type=str,)
    args = ap.parse_args()

    # check for the optional arg
    # you need to choose at least one arg
    if (args.video == False and args.image == False and args.youtube == False and args.camera == False):
        ap.error('\n\nNo arguments provided\n -v for video \n -c for camera \n -y for youtube \n -i for image \n  then add the path/url \n ')
        sys.exit()

    # check is it a stream or just an image
    # then run detector function to get the detect objects
    if args.video:
        detector(is_video=True, path=args.path)
    elif args.image:
        # check the path if is it there
        if not os.path.isfile(args.path):
            print('The specified path does not exist')
            sys.exit()
        detector(is_video=False, path=args.path)
    elif args.camera:
        detector(is_video=True, path=int(args.path))
    else:
        # check if is it a valid youtube url
        try:
            result = pafy.new(args.path)
            best_quality_video = result.getbest()
        except:
            print(
                '\nNon valid youtube url, Make sure that you have a valid youtube url\n')
            sys.exit()
        detector(is_video=True, path=best_quality_video.url)


def get_output_layers(net):
    layer_names = net.getLayerNames()
    output_layers = [layer_names[i[0]-1]for i in net.getUnconnectedOutLayers()]
    return output_layers


def draw_prediction(img, confidence, x, y, x_plus_w, y_plus_h, color):
    cv2.rectangle(img, (x, y), (x_plus_w, y_plus_h), color, 2)
    text = "%{:.2f}".format(confidence)
    cv2.putText(img, text, (x-5, y-5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, [255, 0, 0], 2)


def paint_boxes(boxes, indices):
    colored_boxes = []
    violates = set()
    av = 0
    if len(indices) > 0:
        # get the index of the good boxes
        for i in indices:
            i = i[0]
            box = boxes[i]
            colored_boxes.append(box)
            # each box has [x, y, w, h, center_x, center_y,float(confidence), [0, 255, 0]]
            # get the width of each box
            av = av + box[2]
        # take the twice of the average width
        av = int((av/len(indices))*2)

        # get the violates
        for i in range(len(colored_boxes)):
            for j in range(i+1, len(colored_boxes)):
                # get the euclidean distances between the points
                euclidean_distance = math.sqrt(
                    pow(colored_boxes[i][4]-colored_boxes[j][4], 2) + pow(colored_boxes[i][5]-colored_boxes[j][5], 2))

                if (euclidean_distance < av):
                    # set the color on to RED
                    colored_boxes[i][7] = [0, 0, 255]
                    colored_boxes[j][7] = [0, 0, 255]
                    violates.add(i)
                    violates.add(j)

    return [colored_boxes, len(violates)]


def detector(is_video, path):
    # one function to detect ether stream or just an image
    if is_video:
        camera = cv2.VideoCapture(path)
    else:
        image = cv2.imread(path)

    while True:
        if is_video:
            ret, image = camera.read()
            if not ret:
                break
        # timer
        start = time.time()
        # set the width and the hight of an image
        Width = image.shape[1]
        Height = image.shape[0]

        # Getting blob from current frame
        # The 'cv2.dnn.blobFromImage' function returns 4-dimensional blob from current
        # frame after mean subtraction, normalizing, and RB channels swapping
        # Resulted shape has number of frames, number of channels, width and height
        # E.G.:
        # blob = cv2.dnn.blobFromImage(image, scalefactor=1.0, size, mean, swapRB=True)

        # in the end the new image has a pixel range between 0 and 1 and siz as 320*320
        blob = cv2.dnn.blobFromImage(
            image, 1 / 255.0, (320, 320), (0, 0, 0), True, crop=False)  # 416

        # setting blob as input to the network
        net.setInput(blob)

        # use get_output_layers function to get just what yolo forward need
        # ['yolo_82', 'yolo_94', 'yolo_106']
        # outs return lists each list has 84 item first four are [0,1,2,3] center x ,center y ,width,hight
        # the rest 80 items are the confidence of each detection object
        outs = net.forward(get_output_layers(net))

        confidences = []
        boxes = []

        for out in outs:
            for detection in out:
                # 80 confidence
                scores = detection[5:]
                # the max one
                class_id = np.argmax(scores)
                # the label of it
                confidence = scores[class_id]
                # just detect person that have confidence more than 50%
                # np.int64(0) person label
                # person and more than 50% confident
                if(confidence > conf_threshold and np.int64(0) == class_id):
                    center_x = int(detection[0]*Width)
                    center_y = int(detection[1]*Height)
                    w = int(detection[2]*Width)
                    h = int(detection[3]*Height)
                    # calculate the top left corner(x,y)
                    x = int(center_x-(w/2))
                    y = int(center_y-(h/2))
                    # add the confidence number to list
                    confidences.append(float(confidence))
                    # add the box to boxes list
                    boxes.append([x, y, w, h, center_x, center_y,
                                  float(confidence), [0, 255, 0]])
        # after detecting all boxes
        # apply non maximum suppression then kick out the week boxes
        indices = cv2.dnn.NMSBoxes(
            boxes, confidences, conf_threshold, nms_threshold)
        # take the index of the good boxes and the boxes them self
        # to paint boxes function to Measure the space between the objects and color it red or green
        colored_boxes = paint_boxes(boxes, indices)
        # draw all detected boxes on the image
        for box in colored_boxes[0]:
            x = box[0]
            y = box[1]
            w = box[2]
            h = box[3]
            draw_prediction(image, box[6], round(
                x), round(y), round(x+w), round(y+h), box[7])
        # set the violate number on the image
        cv2.putText(image, 'violate = {}'.format(colored_boxes[1]), (25, 25),
                    cv2.FONT_HERSHEY_TRIPLEX, 0.5, [0, 0, 255], None)
        # set the respact number on the screen
        cv2.putText(image, 'respect = {}'.format(len(colored_boxes[0])-colored_boxes[1]), (25, 50),
                    cv2.FONT_HERSHEY_TRIPLEX, 0.5, [0, 255, 0], None)
        # calculate the fps and set it on the screen
        cv2.putText(image, 'FPS = {:.0f}'.format(1/(time.time()-start)), (Width-100, 25),
                    cv2.FONT_HERSHEY_TRIPLEX, 0.5, [0, 0, 255], None)
        # show the image
        cv2.namedWindow('Object Detection', cv2.WINDOW_NORMAL)
        cv2.imshow("Object Detection", image)

        # if it's image go into empty while loop until the user press esc key
        if not is_video:
            while cv2.waitKey(1) & 0xFF != 27:
                pass
            cv2.destroyAllWindows()
            sys.exit()
            break
        # continue until the user press esc key
        if cv2.waitKey(1) & 0xFF == 27:
            print('Video has ended..\nThank you and good bay')
            cv2.destroyAllWindows()
            sys.exit()
            break

if __name__ == "__main__":

    # initial values
    classes_file_path = 'coco.names'
    weights_file_path = 'yolov4.weights'
    config_file_path = 'yolov4.cfg'
    conf_threshold = 0.5
    nms_threshold = 0.3

    # Loading trained YOLO v4 Objects Detector with the help of 'dnn'
    # dnn => Deep Neural Network
    net = cv2.dnn.readNetFromDarknet(
        config_file_path, weights_file_path)

    # get the inputs
    choose_input()
