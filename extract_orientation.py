import csv
import json
import math
import numpy as np
import os
import matplotlib.pyplot as plt
from PIL import Image
import cv2
from pprint import pprint

from deep_orientation import beyer    # noqa # pylint: disable=unused-import
from deep_orientation import mobilenet_v2    # noqa # pylint: disable=unused-import
from deep_orientation import beyer_mod_relu     # noqa # pylint: disable=unused-import

from deep_orientation.inputs import INPUT_TYPES
from deep_orientation.inputs import INPUT_DEPTH, INPUT_RGB, INPUT_DEPTH_AND_RGB
from deep_orientation.outputs import OUTPUT_TYPES
from deep_orientation.outputs import (OUTPUT_REGRESSION, OUTPUT_CLASSIFICATION,
                                      OUTPUT_BITERNION)

import deep_orientation.preprocessing as pre
import deep_orientation.postprocessing as post

import tensorflow as tf
import tensorflow.keras.backend as K
import utils.img as img_utils
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)

tf.get_logger().setLevel('ERROR')


def load_network(model_name, weights_filepath,
                 input_type, input_height, input_width,
                 output_type,
                 sampling=False,
                 **kwargs):

    # load model --------------------------------------------------------------
    model_module = globals()[model_name]
    model_kwargs = {}
    if model_name == 'mobilenet_v2' and 'mobilenet_v2_alpha' in kwargs:
        model_kwargs['alpha'] = kwargs.get('mobilenet_v2_alpha')
    if output_type == OUTPUT_CLASSIFICATION:
        assert 'n_classes' in kwargs
        model_kwargs['n_classes'] = kwargs.get('n_classes')

    model = model_module.get_model(input_type=input_type,
                                   input_shape=(input_height, input_width),
                                   output_type=output_type,
                                   sampling=sampling,
                                   **model_kwargs)

    # load weights ------------------------------------------------------------
    model.load_weights(weights_filepath)

    return model


def calc_weighted_avg_depth(path, w, h, cx , cy):
    # print(path)
    image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    image_arr = np.asanyarray(image)
    redu_width = w * 0.9
    redu_height = h * 0.9
    x_start = int(cx - (redu_width / 2))
    x_end = int(cx + (redu_width / 2))
    if x_end+1 >= 640:
        x_end = 638
    y_start = int(cy - (redu_height / 2))
    y_end = int(cy + (redu_height / 2))
    if y_end+1 >= 480:
        y_end = 478
    values = []
    weights = []
    centroid_depth = []
    for i in range(y_start, y_end+1):
        for j in range(x_start, x_end+1):
            values.append(image_arr[i][j])
            distx = abs(cx - j)
            disty = abs(cy - i)
            dist = np.sqrt((distx*distx+disty*disty))
            if dist == 0:
                weights.append(1.1)
                centroid_depth = image_arr[i][j]
            else:
                weights.append(1 / dist)
    weighted_avg = np.average(values, weights=weights)
    # print(weighted_avg)
    return weighted_avg, centroid_depth


def calc_orientation(dep_path, x, y, ex, ey):
    image = cv2.imread(dep_path, cv2.IMREAD_GRAYSCALE)
    # get width and height of the image
    h, w = image.shape[:2]
    person = image[y: ey, x: ex]

    shape = (126, 48)
    mask = person > 0
    mask_resized = pre.resize_mask(mask.astype('uint8')*255, shape) > 0
    depth = pre.resize_depth_img(person, shape)
    depth = depth[..., None]
    depth = pre.preprocess_img(
        depth,
        mask=mask_resized,
        scale01='standardize' == 'scale01',
        standardize='standardize'== 'standardize',
        zero_mean=True,
        unit_variance=True)
    # cv2.imshow('person', depth)
    # cv2.waitKey(0)
    if K.image_data_format() == 'channels_last':
        axes = 'b01c'
    else:
        axes = 'bc01'
    depth = img_utils.dimshuffle(depth, '01c', axes)
    global model
    dep_out = model.predict(depth, batch_size=1)
    output = post.biternion2deg(dep_out)
    # print('\t\tOUTPUT:', output)
    return output


def write_data(fr_no, x, y, w, h, cx, cy, cen_dep, dep_avg, orientation, group):
    output = 'gt_db_orientation_20210412_cd_1.csv'
    # output = 'yolov4_adaria_db.csv'
    if not os.path.isfile(output):
        with open(output, 'w') as csv_f:
            csv_writer = csv.writer(csv_f, delimiter=';', lineterminator='\n')
            header = ['fr_no', 'x', 'y', 'w', 'h', 'cx', 'cy', 'cen_dep', 'dep_avg', 'orient', 'group']
            csv_writer.writerow(header)
    with open(output, 'a') as csv_f:
        csv_writer = csv.writer(csv_f, delimiter=';', lineterminator='\n')
        info = [fr_no, x, y, w, h, cx, cy, cen_dep, dep_avg, orientation, group]
        csv_writer.writerow(info)


def get_iou_score(box_a, box_b):
    # determine the (x, y)-coordinates of the intersection rectangle
    x_a = max(box_a[0], box_b[0])
    y_a = max(box_a[1], box_b[1])
    x_b = min(box_a[2], box_b[2])
    y_b = min(box_a[3], box_b[3])
    # compute the area of intersection rectangle
    interArea = max(0, x_b - x_a + 1) * max(0, y_b - y_a + 1)
    # compute the area of both the prediction and ground-truth
    # rectangles
    boxAArea = (box_a[2] - box_a[0] + 1) * (box_a[3] - box_a[1] + 1)
    boxBArea = (box_b[2] - box_b[0] + 1) * (box_b[3] - box_b[1] + 1)
    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = interArea / float(boxAArea + boxBArea - interArea)
    # return the intersection over union value
    return iou


model = load_network('beyer_mod_relu', '/home/viktor/Desktop/deep-orientation/trained_networks/beyer_mod_relu__depth__126x48__biternion__0_030000__1/weights_valid_0268',
                         'depth', 126, 48,
                         'biternion',
                         sampling=1 > 1,
                         n_classes=8,
                         mobilenet_v2_alpha=1.0)

# path = '../skeleton_tests/images/rndRICA'
path = '/home/viktor/Documents/annotated_segs/'
dep_path = '/home/viktor/Documents/exports/rgbd_dep_ts'
files = []
for (dirpath, dirnames, filenames) in os.walk(path):
    files.extend(filenames)

rgb_to_dep = {}
img_rel = '/home/viktor/Desktop/group-recognition-scripts-kclsair/data_merging/export_files/rgb_to_dep.csv'

with open('/home/viktor/Desktop/group-recognition-scripts-kclsair/data_extraction/gtmerged_frdata.json') as json_file:
    gt_frdata = json.load(json_file)

with open('/home/viktor/Desktop/group-recognition-scripts-kclsair/human_tracking/yolo_db_20210304.csv') as infile:
    reader = csv.reader(infile)
    yolo_frdata = {}
    first = True
    for row in reader:
        if first:
            first = False
        else:
            # print(row)
            bbox_data = row[0].split(';')
            frno = bbox_data[0]
            if frno not in yolo_frdata.keys():
                yolo_frdata[int(frno)] = []
            yolo_frdata[int(frno)].append(bbox_data[1:])
            # print(yolo_frdata)

with open(img_rel, 'r') as csv_f:
    csv_reader = csv.reader(csv_f, delimiter=';')
    for row in csv_reader:
        try:
            rgb_frno = int(row[1].split('frame')[-1].split('.')[0])
        except:
            continue
        rgb_to_dep[rgb_frno] = row[3]
# print(rgb_to_dep.keys())

# gt_frame_list = list(gt_frdata.keys())
# print(gt_frame_list)
# count = 0
# for file in files:
#     count += 1
#     if count % 1000 == 0:
#         print('Done with %d of %d' % (count, len(files)))
#     frame_no = int(os.path.splitext(file)[0].split('frame')[-1])
#     if str(frame_no) not in gt_frame_list:
#         continue
#     depth_image_path = os.path.join(dep_path, rgb_to_dep[file])
#     depth_image_raw = cv2.imread(depth_image_path, cv2.IMREAD_GRAYSCALE)

depth_path_prefix = '/home/viktor/Documents/exports/rgbd_dep_ts'
output_dict = {}
for fr_no, boxes in gt_frdata.items():
    print(fr_no)
    fr_no = int(fr_no)
    output_dict[fr_no] = []
    for bbox in boxes:
        try:
            x = int(bbox['x1'])
            y = int(bbox['y1'])
            w = int(bbox['x2'])-x
            h = int(bbox['y2'])-y
            label = int(bbox['label'])
        except TypeError:
            x = int(bbox[0])
            y = int(bbox[1])
            w = int(bbox[2])
            h = int(bbox[3])
            label = int(bbox[7])
        cx = int((x + (x + w)) / 2.0)
        cy = int((y + (y + h)) / 2.0)
        print('\t', x, y, w, h, cx, cy, label)
        dep_path = os.path.join(depth_path_prefix, rgb_to_dep[fr_no])
        # ===========================================================
        # YOLO OUTPUT CHANGES
        # w_avg, cdep = calc_weighted_avg_depth(dep_path, w, h, cx, cy)
        # ori = calc_orientation(dep_path, x, y, x+w, y+h)[0]
        # print('\t\t\t', w_avg, ori)
        # group = label
        # output_dict[fr_no].append(list(
        #     [str(x), str(y), str(w), str(h), str(cx), str(cy), "%.2f" % cdep, "%.2f" % w_avg, "%.2f" % ori,
        #      str(group)]))
        # ===========================================================
        try:
            yolo_frame = yolo_frdata[fr_no]
        except:
            continue
        for ybox in yolo_frame:
            yo_x = int(ybox[0])
            yo_y = int(ybox[1])
            yo_w = int(ybox[2])
            yo_h = int(ybox[3])
            yo_cx = int(ybox[4])
            yo_cy = int(ybox[5])
            w_avg = float(ybox[6])
            _, cdep = calc_weighted_avg_depth(dep_path, yo_w, yo_h, yo_cx, yo_cy)
            box_a = [x, y, x + w, y + h]
            box_b = [yo_x, yo_y, yo_x + yo_w, yo_y + yo_h]
            iou = get_iou_score(box_a, box_b)
            # print('IOU', iou)
            if iou > 0.15:
                print('FOUND ONE')
                try:
                    ori = calc_orientation(dep_path, yo_x, yo_y, yo_x + yo_w, yo_y + yo_h)[0]
                except:
                    continue
                print('\t\t\t', w_avg, ori)
                group = label
                output_dict[fr_no].append(list([str(yo_x), str(yo_y), str(yo_w), str(yo_h), str(yo_cx), str(yo_cy), "%.2f" % cdep, "%.2f" % w_avg, "%.2f" % ori, str(group)]))
        # ===========================================================
        # ===========================================================
    # break
    print('---')

# print(output_dict)
with open('yolo_db_orientation_20210810_cd_1.json', 'w') as outfile:
    json.dump(output_dict, outfile, indent=4)
