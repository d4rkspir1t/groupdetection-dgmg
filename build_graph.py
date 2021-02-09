import pandas as pd
import os
import csv
from pprint import pprint
import math


base_path_cpp = 'salsa/Annotation/salsa_cpp/'
base_path_ps = 'salsa/Annotation/salsa_cpp/'
person_log = 'geometryGT/'
fformation_log = 'fformationGT.csv'


def fetch_person_data(person_id, frame_ts, base_path):
    f_name = str(person_id).rjust(2, '0') + '.csv'
    f_path = os.path.join(base_path, person_log, f_name)
    # print(f_path)
    with open(f_path, 'r') as csvf:
        csvrdr = csv.reader(csvf, delimiter=',')
        for row in csvrdr:
            frame = float(row[0])
            data = row[1:]
            # print(person_id, frame_ts, base_path, data)
            if frame == frame_ts:
                return data


def read_frame_data(base_p, extra_t=0):
    ff_path = os.path.join(base_p, fformation_log)
    frame_data = {}
    with open(ff_path, 'r') as csvf:
        csvrdr = csv.reader(csvf, delimiter=',')
        for row in csvrdr:
            frame = str(float(row[0]) + extra_t)
            if frame not in frame_data.keys():
                frame_data[frame] = []
            group = []
            for idx in row[1:]:
                try:
                    group.append(int(idx))
                except ValueError:
                    print('BAD INPUT: ', idx)
            frame_data[frame].append(group)
    return frame_data


extra_time = 10000
frame_data_cpp = read_frame_data(base_path_cpp, 0)
frame_data_ps = read_frame_data(base_path_ps, extra_time)
frame_data = {**frame_data_cpp, **frame_data_ps}
# pprint(frame_data)
frame_node_data = {}
frame_edge_data = {}
for frame_id, frame_info in frame_data.items():
    node_data = []
    group_id_tracker = 0
    for group in frame_info:
        if len(group) == 1:
            group_id = -1
        else:
            group_id = group_id_tracker
            group_id_tracker += 1
        for person in group:
            if float(frame_id) > extra_time:
                c_frame_id = round(float(frame_id)-extra_time, 2)
                data = fetch_person_data(person, c_frame_id, base_path_ps)
            else:
                data = fetch_person_data(person, float(frame_id), base_path_cpp)
            pos_x = float(data[0])
            pos_y = float(data[1])
            body_pose = float(data[3])
            rel_head_pose = float(data[4])
            head_pose = body_pose + rel_head_pose
            # math.degrees() for degrees instead of radians
            # person_id 0, group_id 1, posx 2, posy 3, bodyp 4, rheadp 5, headp 6
            node_data.append([person, group_id, pos_x, pos_y, body_pose, rel_head_pose, head_pose])
    # pprint(node_data)
    # print(len(node_data))
    frame_node_data[frame_id] = node_data
    edge_data = []
    for person_data in node_data:
        person = person_data[0]
        group = person_data[1]
        for idx in range(len(node_data)):
            if node_data[idx][0] != person and node_data[idx][1] != -1:
                if group == node_data[idx][1]:
                    # src dst distance effort
                    distance = math.dist([person_data[2], person_data[3]], [node_data[idx][2], node_data[idx][3]])
                    angle_diff = person_data[6] - (node_data[idx][6] - math.pi)
                    if angle_diff > math.pi * 2:
                        # print('bullshit +\t', angle_diff)
                        angle_diff = angle_diff % (math.pi * 2)
                        # print('\tcorrected: ', angle_diff)
                    elif angle_diff < math.pi * -2:
                        # print('bullshit -\t', angle_diff)
                        angle_diff = angle_diff % (math.pi * 2)
                        # print('\tcorrected: ', angle_diff)
                    if angle_diff < 0:
                        effort = math.pi * 2 + angle_diff
                    else:
                        effort = angle_diff
                    edge_data.append([person, node_data[idx][0], distance, effort])
    # pprint(edge_data)
    # print(len(edge_data))
    frame_edge_data[frame_id] = edge_data
    # break
# print(df_ff.head())
# for idx in df_ff.index:
#     print(idx, df_ff.loc[idx]['time'], df_ff.loc[idx]['group'])
print(len(frame_data_cpp.keys()), len(frame_data_ps.keys()), len(frame_node_data.keys()), len(frame_edge_data.keys()))
