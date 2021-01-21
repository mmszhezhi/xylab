
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys

import cv2

images_path = "test-T01"
fps = 5.0


def inference(images_path, current_directory, fps):
    video_labels_dir = current_directory + "/video_labels"
    if not os.path.exists(video_labels_dir):
        os.makedirs(video_labels_dir)

    file_name = images_path.split('/')[-1]
    video_file_name = file_name + ".avi"

    video_file_path = "%s/%s" % (video_labels_dir, video_file_name)

    img_file_names = os.listdir(images_path + "/")
    img_file_names.sort()
    print("len(img_file_names): %s" % (len(img_file_names)))

    image_file = images_path + '/' + img_file_names[0].strip()
    image = cv2.imread(image_file)
    (frame_h, frame_w, frame_c) = image.shape

    # Define the codec and create VideoWriter object
    fourcc = cv2.CV_FOURCC('M', 'J', 'P', 'G')
    videowriter = cv2.VideoWriter(video_file_path, fourcc, fps, (frame_w, frame_h))

    for img_file_name in img_file_names:
        if ".jpg" not in img_file_name:
            continue

        img_file = images_path + '/' + img_file_name.strip()
        img = cv2.imread(img_file)
        print(img_file_name)

        videowriter.write(img)

    videowriter.release()


if __name__ == '__main__':
    current_directory = os.path.dirname(os.path.abspath(__file__))
    print("current_directory:", current_directory)
    print("__file__:", __file__)
    print("os.path.abspath(__file__):", os.path.abspath(__file__))
    print("os.path.dirname(os.path.abspath(__file__)):", os.path.dirname(os.path.abspath(__file__)))
    print("os.getcwd():", os.getcwd())
    print("os.curdir:", os.curdir)
    print("os.path.abspath(os.curdir):", os.path.abspath(os.curdir))
    print("sys.path[0]:", sys.path[0])

    inference(images_path, current_directory, fps)