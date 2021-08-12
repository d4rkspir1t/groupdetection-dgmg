import pandas as pd
import os
import csv
import copy
from pprint import pprint
import math
import dgl
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import scipy.sparse as sp
from dgl.nn import SAGEConv
import itertools
from sklearn.metrics import roc_auc_score, accuracy_score
import dgl.dataloading as dgl_dl
import random
import datetime
import time
import growl_utils.utils as ut


base_path_cpp = 'salsa/Annotation/salsa_cpp/'
base_path_ps = 'salsa/Annotation/salsa_ps/'
base_path_rica = './gt_db_orientation_20210412_cd_1.csv'
person_log = 'geometryGT/'
fformation_log = 'fformationGT.csv'

extra_time = 10000

frame_data_ps = ut.read_frame_data(base_path_ps, 0)
salsa_ps_keys = list(frame_data_ps.keys())
# print(salsa_ps_keys)

random.shuffle(salsa_ps_keys)
split_idx = math.ceil(len(salsa_ps_keys)*0.6)
# print(salsa_ps_keys)
train_set_keys = salsa_ps_keys[:split_idx]
test_set_keys = salsa_ps_keys[split_idx:]
# print(len(train_set_keys), len(test_set_keys))

train_dict = dict((k, frame_data_ps[k]) for k in train_set_keys)
test_dict = dict((k, frame_data_ps[k]) for k in test_set_keys)
# print(len(train_dict.keys()), len(test_dict.keys()))

train_node_data = {}
train_edge_data = {}
max_side_dist = 0

for frame_id, frame_info in train_dict.items():
    node_data = []
    group_id_tracker = 0
    for group in frame_info:
        if len(group) == 1:
            group_id = -1
        else:
            group_id = group_id_tracker
            group_id_tracker += 1
        for person in group:
            data = ut.fetch_person_data(person, float(frame_id), base_path_ps)
            pos_x = float(data[0])
            if pos_x > max_side_dist:
                max_side_dist = pos_x
            pos_y = float(data[1])
            body_pose = float(data[3])
            rel_head_pose = float(data[4])
            head_pose = body_pose + rel_head_pose
            # math.degrees() for degrees instead of radians
            # person_id 0, group_id 1, posx 2, posy 3, bodyp 4, rheadp 5, headp 6
            # node_data.append([person, group_id, pos_x, pos_y, body_pose, rel_head_pose, round(head_pose, 4)])
            # ABLATION
            # ===============================================================
            node_data.append([person, group_id, pos_x, pos_y, round(head_pose, 4)])
            # node_data.append([person, group_id, pos_x, pos_y])
            # ===============================================================
    # pprint(node_data)
    # print(len(node_data))
    train_node_data[frame_id] = node_data
    edge_data = []
    for person_data in node_data:
        person = person_data[0]
        group = person_data[1]
        for idx in range(len(node_data)):
            if node_data[idx][0] != person and node_data[idx][1] != -1:
                if group == node_data[idx][1]:
                    # src dst distance effort
                    distance = math.dist([person_data[2], person_data[3]], [node_data[idx][2], node_data[idx][3]])
                    # ABLATION
                    # ===============================================================
                    angle_diff = person_data[-1] - (node_data[idx][-1] - math.pi)
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
                    # src dst dist eff
                    edge_data.append([person, node_data[idx][0], distance, effort])
                    # edge_data.append([person, node_data[idx][0], distance])
                    # ===============================================================
    # pprint(edge_data)
    # print(len(edge_data))
    train_edge_data[frame_id] = edge_data

train_graphs = []
skipped = 0
iters_ps = 0
for frame_id, val in train_edge_data.items():
    # print('FR ID: ', frame_id)
    if float(frame_id) >= extra_time:
        custom_node_count = 21
    else:
        # continue
        custom_node_count = 18
    srcs = []
    dsts = []
    pos = {}
    for entry in val:
        srcs.append(entry[0]-1)
        dsts.append(entry[1]-1)
    feats = []
    for person in train_node_data[frame_id]:
        pos[person[0]-1] = [person[2], person[3]]
        feat = person[2:5]
        # print(person[0])
        feats.append(feat)

    feats = torch.from_numpy(np.array(feats))
    try:
        graph = dgl.graph((srcs, dsts), num_nodes=len(train_node_data[frame_id]))
    except dgl._ffi.base.DGLError:
        skipped += 1
        continue

    # print(graph.number_of_nodes(), len(feats), len(train_node_data[frame_id]))
    draw_graph = False
    graph.ndata['feat'] = feats.float()
    # print(graph.ndata['feat'][:10])
    # print('# nodes: %d, # edges: %d' % (graph.number_of_nodes(), graph.number_of_edges()))
    if draw_graph:
        nx_g = graph.to_networkx().to_undirected()
        # pos = nx.kamada_kawai_layout(nx_g)
        print(pos)
        # should assign pos on -1:1 scale based on coordinates
        try:
            nx.draw(nx_g, pos, with_labels=True, node_color="#A0CBE2")
        except nx.exception.NetworkXError:
            node_cs = []
            for i in range(graph.number_of_nodes()):
                if i not in pos.keys():
                    pos[i] = [0, 0]
                    node_cs.append('#541E1B')
                else:
                    node_cs.append("#A0CBE2")
            nx.draw(nx_g, pos, with_labels=True, node_color=node_cs)
        base_path = 'salsa/ps_graphs'
        iters_ps += 1
        name = '%d.png' % iters_ps
        graph_path = os.path.join(base_path, name.rjust(9, '0'))
        plt.savefig(graph_path)
        plt.close()
    train_graphs.append(graph)
print('Skipped: ', skipped)

cross_val_len = len(train_graphs)*0.1

# h_feats_list = [2, 3, 4, 5, 6, 10, 15, 20]
# epochs_list = [10, 15, 20, 25, 30, 40, 50, 100, 150, 200, 250]
h_feats_list = [2]
epochs_list = [100]
total = len(h_feats_list) * len(epochs_list) * 10
count = 0
iter_count = 3
plot_tests = False

for cross_val_iter in range(10):
    validation_range_start = math.floor(cross_val_iter * cross_val_len)
    validation_range_end = math.floor((cross_val_iter+1) * cross_val_len)
    print(validation_range_start, validation_range_end)
    train_set = train_graphs[:validation_range_start] + train_graphs[validation_range_end:]
    validation_set = train_graphs[validation_range_start:validation_range_end]

    # print('idx outputs', len(train_set), len(validation_set))

    for h_feats in h_feats_list:
        for epochs in epochs_list:
            iteration = 0
            while iteration != iter_count:
                random_graph = random.sample(train_set, 1)[0]
                # print(random_graph)
                model = ut.GraphSAGE(random_graph.ndata['feat'].shape[1], h_feats)
                pred = ut.MLPPredictor(h_feats)
                optimizer = torch.optim.Adam(itertools.chain(model.parameters(), pred.parameters()), lr=0.01)
                for single_train_graph in train_set:
                    u, v = single_train_graph.edges()
                    adj = sp.coo_matrix((np.ones(len(u)), (u.numpy(), v.numpy())))
                    try:
                        adj_neg = 1 - adj.todense() - np.eye(single_train_graph.num_nodes())
                    except ValueError:
                        continue
                    neg_u, neg_v = np.where(adj_neg != 0)
                    train_pos_u, train_pos_v = u, v
                    train_neg_u, train_neg_v = neg_u, neg_v
                    train_pos_g = dgl.graph((train_pos_u, train_pos_v), num_nodes=single_train_graph.num_nodes())
                    train_neg_g = dgl.graph((train_neg_u, train_neg_v), num_nodes=single_train_graph.num_nodes())

                    #
                    # # ----------- 4. training -------------------------------- #
                    all_logits = []

                    for e in range(epochs):
                        # forward
                        # print('FEAT COUNT', len(batched_graph.ndata['feat']))
                        h = model(single_train_graph, single_train_graph.ndata['feat'])
                        pos_score = pred(train_pos_g, h)[0]['score']
                        neg_score = pred(train_neg_g, h)[0]['score']
                        loss = ut.compute_loss(pos_score, neg_score)

                        # backward
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()
                # TEST
                auc_scores = []
                precision_scores = []
                recall_scores = []
                f1_scores = []
                print('Starting tests', len(validation_set))
                test_c = 0
                for single_val_graph in validation_set:
                    test_c += 1
                    val_graph = copy.copy(single_val_graph)
                    # print('Test graph', test_graph.ndata['feat'])
                    test_eids = val_graph.edges(form='eid')
                    val_graph.remove_edges(test_eids)
                    u, v = single_val_graph.edges()
                    u_t, v_t = val_graph.edges()
                    try:
                        adj = sp.coo_matrix((np.ones(len(u)), (u.numpy(), v.numpy())))
                    except ValueError:
                        continue
                    try:
                        adj_neg = 1 - adj.todense() - np.eye(single_val_graph.num_nodes())
                        adj_t_neg = 1 - np.eye(val_graph.num_nodes())
                    except ValueError:
                        continue
                    neg_u, neg_v = np.where(adj_neg != 0)
                    neg_t_u, neg_t_v = np.where(adj_t_neg != 0)
                    test_pos_u, test_pos_v = u, v
                    test_neg_u, test_neg_v = neg_u, neg_v

                    test_full_graph = dgl.graph((neg_t_u, neg_t_v), num_nodes=val_graph.num_nodes())

                    with torch.no_grad():
                        h = model(single_val_graph, single_val_graph.ndata['feat'])
                        test_out, test_graph_out = pred(test_full_graph, h)
                        test_labels = test_out['label']

                        to_remove = []
                        for i in range(len(test_labels)):
                            if test_labels[i] == 0:
                                to_remove.append(i)

                        test_graph_out.remove_edges(to_remove)

                        original_nodec = single_val_graph.num_nodes()
                        original_u, original_v = single_val_graph.edges()
                        pred_nodec = test_graph_out.num_nodes()
                        pred_u, pred_v = test_graph_out.edges()
                        original_clusters = ut.get_clusters(original_nodec, original_u, original_v)
                        pred_clusters = ut.get_clusters(pred_nodec, pred_u, pred_v)

                        swap_original_clusters = ut.swap_clusters(original_clusters)
                        swap_pred_clusters = ut.swap_clusters(pred_clusters)

                        tp = 0
                        fp = 0
                        fn = 0
                        t = 2 / 3
                        t_ = 1 - t
                        used_pred_clusters = [-1]
                        for key, cluster in swap_original_clusters.items():
                            if key == -1:
                                continue
                            else:
                                matched_clusters = {}
                                fullsize = len(cluster)
                                for pred_key, pred_cluster in swap_pred_clusters.items():
                                    if pred_key == -1:
                                        continue
                                    match = 0
                                    miss = 0
                                    for node in cluster:
                                        if node in pred_cluster:
                                            match += 1
                                        else:
                                            miss += 1
                                    if match > 0:
                                        matched_clusters[pred_key] = [match, miss]
                                max_match = 0
                                best_match = {}
                                for match_key, match_val in matched_clusters.items():
                                    if match_val[0] > max_match:
                                        max_match = match_val[0]
                                        best_match = {match_key: match_val}
                                if len(list(best_match.keys())) == 0:
                                    continue
                                used_pred_clusters.append(list(best_match.keys())[0])
                                best_match_val = list(best_match.values())[0]
                                match = best_match_val[0]
                                miss = best_match_val[1]
                                if match / fullsize >= t and miss / fullsize <= t_:
                                    tp += 1
                                    verdict = 'tp'
                                else:
                                    fn += 1
                                    verdict = 'fn'
                                # print(key, match, miss, fullsize, verdict)
                        for key in swap_pred_clusters.keys():
                            if key not in used_pred_clusters:
                                fp += 1
                        # print('TP: %d, FN: %d, FP: %d' % (tp, fn, fp))
                        if tp + fp == 0:
                            precision = 0
                        else:
                            precision = tp / (tp + fp)
                        if tp + fn == 0:
                            recall = 0
                        else:
                            recall = tp / (tp + fn)
                        if precision + recall == 0:
                            f1 = 0.01
                            precision_scores.append(precision)
                            recall_scores.append(recall)
                            f1_scores.append(f1)
                        else:
                            f1 = 2 * (precision * recall) / (precision + recall)
                            precision_scores.append(precision)
                            recall_scores.append(recall)
                            f1_scores.append(f1)

                        if plot_tests:
                            nx_g = test_graph_out.to_networkx().to_undirected()
                            # pos = nx.kamada_kawai_layout(nx_g)
                            # print(pos)
                            # should assign pos on -1:1 scale based on coordinates
                            try:
                                nx.draw(nx_g, pos, with_labels=True, node_color="#A0CBE2")
                            except nx.exception.NetworkXError:
                                pass
                            plt.savefig(os.path.join('./quickplots', '%d.png' % test_c))
                            # plt.show()
                            plt.close()

                if len(f1_scores) > 0:
                    tracker_file_path = 'growl_param_analysis/paramopt_model_output_f1_20210520.csv'
                    iteration += 1
                    model_output_tracker = pd.DataFrame(
                        list(zip([datetime.datetime.now()], [h_feats], [epochs], [len(f1_scores)],
                                 [np.mean(precision_scores)], [np.mean(recall_scores)], [np.mean(f1_scores)])),
                        columns=['time', 'feature_count', 'epoch_count', 'test_length', 'mean_precision', 'mean_recall',
                                 'mean_f1'])
                    if os.path.exists(tracker_file_path):
                        model_output_tracker.to_csv(tracker_file_path, mode='a',
                                                    index=False, header=False)
                    else:
                        model_output_tracker.to_csv(tracker_file_path, mode='w',
                                                    index=False, header=True)
                    if np.mean(f1_scores) > 0.64:
                        break
            count += 1
            print('#%d of %d\t%d, %d' % (count, total, h_feats, epochs))
