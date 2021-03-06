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


base_path_cpp = 'salsa/Annotation/salsa_cpp/'
base_path_ps = 'salsa/Annotation/salsa_cpp/'
person_log = 'geometryGT/'
fformation_log = 'fformationGT.csv'


class GraphSAGE(nn.Module):
    def __init__(self, in_feats, h_feats):
        super(GraphSAGE, self).__init__()
        self.conv1 = SAGEConv(in_feats, h_feats, 'mean')
        self.conv2 = SAGEConv(h_feats, h_feats, 'mean')

    def forward(self, g, in_feat):
        h = self.conv1(g, in_feat)
        h = F.relu(h)
        h = self.conv2(g, h)
        return h


class MLPPredictor(nn.Module):
    def __init__(self, h_feats):
        super().__init__()
        self.W1 = nn.Linear(h_feats * 2, h_feats)
        self.W2 = nn.Linear(h_feats, 1)

    def apply_edges(self, edges):
        """
        Computes a scalar score for each edge of the given graph.

        Parameters
        ----------
        edges :
            Has three members ``src``, ``dst`` and ``data``, each of
            which is a dictionary representing the features of the
            source nodes, the destination nodes, and the edges
            themselves.

        Returns
        -------
        dict
            A dictionary of new edge features.
        """
        h = torch.cat([edges.src['h'], edges.dst['h']], 1)
        out_score = self.W2(F.relu(self.W1(h))).squeeze(1)
        out_label = torch.round(torch.sigmoid(out_score))
        # print(out_score, out_label)
        out_dict = {'score': out_score, 'label': out_label}
        return out_dict

    def forward(self, g, h):
        with g.local_scope():
            g.ndata['h'] = h
            g.apply_edges(self.apply_edges)
            # return g.edata['score']
            # print('executes', g.edata)
            out_dict = dict(g.edata)
            return out_dict, g


def compute_loss(pos_score, neg_score):
    scores = torch.cat([pos_score, neg_score])
    labels = torch.cat([torch.ones(pos_score.shape[0]), torch.zeros(neg_score.shape[0])])
    return F.binary_cross_entropy_with_logits(scores, labels)


def compute_auc(pos_score, neg_score):
    scores = torch.cat([pos_score, neg_score]).numpy()
    labels = torch.cat(
        [torch.ones(pos_score.shape[0]), torch.zeros(neg_score.shape[0])]).numpy()
    return roc_auc_score(labels, scores)


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


def get_clusters(nodec, srcn, dstn):
    clusters = {}
    cluster_idx = 0
    for node in range(nodec):
        if node not in clusters.keys():
            clusters[node] = -1
            if node in srcn:
                clusters[node] = cluster_idx
                cluster_idx += 1
                for idx, u in enumerate(list(srcn)):
                    if u == node:
                        clusters[int(dstn[idx])] = clusters[node]
    return clusters


def swap_clusters(clusters):
    swapped_clusters = {}
    for key, val in clusters.items():
        if val not in swapped_clusters.keys():
            swapped_clusters[val] = []
        swapped_clusters[val].append(key)
    return swapped_clusters


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
            node_data.append([person, group_id, pos_x, pos_y, body_pose, rel_head_pose, round(head_pose, 4)])
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
                    # src dst dist eff
                    edge_data.append([person, node_data[idx][0], distance, effort])
    # pprint(edge_data)
    # print(len(edge_data))
    frame_edge_data[frame_id] = edge_data
    # break
# print(df_ff.head())
# for idx in df_ff.index:
#     print(idx, df_ff.loc[idx]['time'], df_ff.loc[idx]['group'])
# print(len(frame_data_cpp.keys()), len(frame_data_ps.keys()), len(frame_node_data.keys()), len(frame_edge_data.keys()))
iters_cpp = 0
iters_ps = 0
all_graphs = []
for frame_id, val in frame_edge_data.items():
    # print('FR ID: ', frame_id)
    srcs = []
    dsts = []
    pos = {}
    for entry in val:
        srcs.append(entry[0]-1)
        dsts.append(entry[1]-1)
    feats = []
    for person in frame_node_data[frame_id]:
        pos[person[0]-1] = [person[2], person[3]]
        feat = person[2:7]
        # print(person[0])
        feats.append(feat)

    feats = torch.from_numpy(np.array(feats))
    graph = dgl.graph((srcs, dsts), num_nodes=18)
    missing = []
    for idx in range(0, 18):
        if idx not in pos.keys():
            missing.append(idx)
            # print(idx)
            node_list = graph.nodes().tolist()
            node_list.remove(idx)
            # print(node_list)
            graph = graph.subgraph(node_list)
    if len(missing) > 0:
        continue
    # print(graph.number_of_nodes(), len(feats), len(frame_node_data[frame_id]))
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
            for i in range(18):
                if i not in pos.keys():
                    pos[i] = [0, 0]
                    node_cs.append('#541E1B')
                else:
                    node_cs.append("#A0CBE2")
            nx.draw(nx_g, pos, with_labels=True, node_color=node_cs)
        if float(frame_id) < extra_time:
            base_path = 'salsa/cpp_graphs'
            iters_cpp += 1
            name = '%d.png' % iters_cpp
            graph_path = os.path.join(base_path, name.rjust(9, '0'))
        else:
            base_path = 'salsa/ps_graphs'
            iters_ps += 1
            name = '%d.png' % iters_ps
            graph_path = os.path.join(base_path, name.rjust(9, '0'))
        plt.savefig(graph_path)
        plt.close()
    # print('Edge count: %d, Node count: %d, Feature count: %d' % (graph.num_edges(), graph.num_nodes(), len(graph.ndata['feat'][0])))
    all_graphs.append(graph)
    # break

random.shuffle(all_graphs)
split_idx = math.ceil(len(all_graphs)*0.6)
train_graphs = all_graphs[:split_idx]
test_graphs = all_graphs[split_idx:]
train_bg = dgl.batch(train_graphs)
test_bg = dgl.batch(test_graphs)
# print(train_bg.batch_size)
# print(test_bg.batch_size)

param_opt = False

h_feats_list = [2, 3, 4, 5, 6, 10, 15, 20]
epochs_list = [10, 15, 20, 25, 30, 40, 50, 100, 150, 200, 250]
total = len(h_feats_list)*len(epochs_list)
iter_count = 10
count = 1
if param_opt:
    for h_feats in h_feats_list:
        for epochs in epochs_list:
            # h_feats = 3False
            # epochs = 100
            model = GraphSAGE(train_bg.ndata['feat'].shape[1], h_feats)
            # # You can replace DotPredictor with MLPPredictor.
            pred = MLPPredictor(h_feats)
            #
            # # ----------- 3. set up loss and optimizer -------------- #
            # # in this case, loss will in training loop
            optimizer = torch.optim.Adam(itertools.chain(model.parameters(), pred.parameters()), lr=0.01)
            auc_scores = []
            for _ in range(iter_count):
                for batched_graph in train_graphs:
                    u, v = batched_graph.edges()
                    adj = sp.coo_matrix((np.ones(len(u)), (u.numpy(), v.numpy())))
                    try:
                        adj_neg = 1 - adj.todense() - np.eye(batched_graph.number_of_nodes())
                    except ValueError:
                        continue
                    neg_u, neg_v = np.where(adj_neg != 0)
                    train_pos_u, train_pos_v = u, v
                    train_neg_u, train_neg_v = neg_u, neg_v
                    train_pos_g = dgl.graph((train_pos_u, train_pos_v), num_nodes=batched_graph.number_of_nodes())
                    train_neg_g = dgl.graph((train_neg_u, train_neg_v), num_nodes=batched_graph.number_of_nodes())

                #
                # # ----------- 4. training -------------------------------- #
                    all_logits = []
                    for e in range(epochs):
                        # forward
                        h = model(batched_graph, batched_graph.ndata['feat'])
                        pos_score = pred(train_pos_g, h)[0]['score']
                        neg_score = pred(train_neg_g, h)[0]['score']
                        loss = compute_loss(pos_score, neg_score)

                        # backward
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()

                        # if e % 5 == 0:
                            # print('In epoch {}, loss: {}'.format(e, loss))
                #
                # # ----------- 5. check results ------------------------ #
                #
                for batched_graph in test_graphs:
                    u, v = batched_graph.edges()
                    adj = sp.coo_matrix((np.ones(len(u)), (u.numpy(), v.numpy())))
                    try:
                        adj_neg = 1 - adj.todense() - np.eye(batched_graph.number_of_nodes())
                    except ValueError:
                        continue
                    neg_u, neg_v = np.where(adj_neg != 0)
                    test_pos_u, test_pos_v = u, v
                    test_neg_u, test_neg_v = neg_u, neg_v
                    test_pos_g = dgl.graph((test_pos_u, test_pos_v), num_nodes=batched_graph.number_of_nodes())
                    test_neg_g = dgl.graph((test_neg_u, test_neg_v), num_nodes=batched_graph.number_of_nodes())
                    with torch.no_grad():
                        pos_score = pred(test_pos_g, h)[0]['score']
                        neg_score = pred(test_neg_g, h)[0]['score']
                        auc = compute_auc(pos_score, neg_score)
                        # print('AUC', auc)
                        auc_scores.append(auc)

            print('#%d of %d\t%d, %d\tTested on: %d, Avg AUC: %.4f, Stdev: %.4f' % (count, total, h_feats, epochs,
                                                                                    len(auc_scores), np.mean(auc_scores),
                                                                                    np.std(auc_scores)))
            count += 1
            model_output_tracker = pd.DataFrame(
                list(zip([datetime.datetime.now()], [h_feats], [epochs], [len(auc_scores)], [np.mean(auc_scores)], [np.std(auc_scores)])),
                columns=['time', 'feature_count', 'epoch_count', 'test_length', 'mean_auc', 'std_auc'])
            if os.path.exists('model_output_tracker.csv'):
                model_output_tracker.to_csv('model_output_tracker.csv', mode='a', index=False, header=False)
            else:
                model_output_tracker.to_csv('model_output_tracker.csv', mode='w', index=False, header=True)
else:
    h_feats_list = [2, 3, 4, 5, 10, 15, 20]
    epochs_list = [15, 20, 25, 50, 100, 150, 200]
    total = len(h_feats_list) * len(epochs_list)
    count = 0
    for h_feats in h_feats_list:
        for epochs in epochs_list:
            iteration = 0
            while iteration != 5:
                # h_feats = 3
                # epochs = 20
                model = GraphSAGE(train_bg.ndata['feat'].shape[1], h_feats)
                # # You can replace DotPredictor with MLPPredictor.
                pred = MLPPredictor(h_feats)
                #
                # # ----------- 3. set up loss and optimizer -------------- #
                # # in this case, loss will in training loop
                optimizer = torch.optim.Adam(itertools.chain(model.parameters(), pred.parameters()), lr=0.01)

                start_t = datetime.datetime.now()
                for batched_graph in train_graphs:
                    u, v = batched_graph.edges()
                    adj = sp.coo_matrix((np.ones(len(u)), (u.numpy(), v.numpy())))
                    try:
                        adj_neg = 1 - adj.todense() - np.eye(batched_graph.number_of_nodes())
                    except ValueError:
                        continue
                    neg_u, neg_v = np.where(adj_neg != 0)
                    train_pos_u, train_pos_v = u, v
                    train_neg_u, train_neg_v = neg_u, neg_v
                    train_pos_g = dgl.graph((train_pos_u, train_pos_v), num_nodes=batched_graph.number_of_nodes())
                    train_neg_g = dgl.graph((train_neg_u, train_neg_v), num_nodes=batched_graph.number_of_nodes())

                    #
                    # # ----------- 4. training -------------------------------- #
                    all_logits = []

                    for e in range(epochs):
                        # forward
                        h = model(batched_graph, batched_graph.ndata['feat'])
                        pos_score = pred(train_pos_g, h)[0]['score']
                        neg_score = pred(train_neg_g, h)[0]['score']
                        loss = compute_loss(pos_score, neg_score)

                        # backward
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()

                        # if e % 5 == 0:
                        # print('In epoch {}, loss: {}'.format(e, loss))
                # print('Training took: ', datetime.datetime.now()-start_t)
                #
                # # ----------- 5. check results ------------------------ #
                #
                start_t = datetime.datetime.now()
                plot_tests = False
                auc_scores = []
                precision_scores = []
                recall_scores = []
                f1_scores = []
                for batched_graph in test_graphs:
                    test_graph = copy.copy(batched_graph)
                    # print('Test graph', test_graph.ndata['feat'])
                    test_eids = test_graph.edges(form='eid')
                    test_graph.remove_edges(test_eids)
                    # print('Test graph', test_graph.num_nodes(), test_graph.num_edges())
                    # print(batched_graph.num_nodes(), batched_graph.num_edges())
                    # print(batched_graph.nodes())
                    u, v = batched_graph.edges()
                    u_t, v_t = test_graph.edges()
                    adj = sp.coo_matrix((np.ones(len(u)), (u.numpy(), v.numpy())))
                    # adj_t = sp.coo_matrix((np.ones(len(u_t)), (u_t.numpy(), v_t.numpy())))
                    try:
                        adj_neg = 1 - adj.todense() - np.eye(batched_graph.number_of_nodes())
                        adj_t_neg = 1 - np.eye(test_graph.number_of_nodes())
                    except ValueError:
                        continue
                    neg_u, neg_v = np.where(adj_neg != 0)
                    neg_t_u, neg_t_v = np.where(adj_t_neg != 0)
                    test_pos_u, test_pos_v = u, v
                    test_neg_u, test_neg_v = neg_u, neg_v
                    test_pos_g = dgl.graph((test_pos_u, test_pos_v), num_nodes=batched_graph.number_of_nodes())
                    test_neg_g = dgl.graph((test_neg_u, test_neg_v), num_nodes=batched_graph.number_of_nodes())
                    test_full_graph = dgl.graph((neg_t_u, neg_t_v), num_nodes=test_graph.number_of_nodes())
                    # test_full_graph.ndata['feat'] = test_graph.ndata['feat']
                    # print('Test graph negative stats', test_full_graph.num_nodes(), test_full_graph.num_edges())
                    with torch.no_grad():
                        pos_out, pos_graph_out = pred(test_pos_g, h)
                        neg_out, neg_graph_out = pred(test_neg_g, h)
                        test_out, test_graph_out = pred(test_full_graph, h)
                        pos_score = pos_out['score']
                        neg_score = neg_out['score']

                        pos_labels = pos_out['label']
                        neg_labels = neg_out['label']
                        test_labels = test_out['label']
                        # print('Test labels: ', len(test_labels), test_labels)
                        pred_labels = torch.cat([pos_labels, neg_labels]).numpy()
                        scores = torch.cat([pos_score, neg_score]).numpy()
                        labels = torch.cat(
                            [torch.ones(pos_score.shape[0]), torch.zeros(neg_score.shape[0])]).numpy()
                        auc = roc_auc_score(labels, scores)
                        # print(len(scores), '\n', pred_labels[:len(pos_labels)], '\n', pred_labels[len(pos_labels):])
                        # print('AUC', auc)
                        auc_scores.append(auc)

                    to_remove = []
                    for i in range(len(test_labels)):
                        if test_labels[i] == 0:
                            to_remove.append(i)
                    if plot_tests:
                        fig, (ax1, ax2) = plt.subplots(1, 2)
                    test_graph_out.remove_edges(to_remove)

                    original_nodec = batched_graph.num_nodes()
                    original_u, original_v = batched_graph.edges()
                    pred_nodec = test_graph_out.num_nodes()
                    pred_u, pred_v = test_graph_out.edges()
                    original_clusters = get_clusters(original_nodec, original_u, original_v)
                    pred_clusters = get_clusters(pred_nodec, pred_u, pred_v)
                    # pprint(original_clusters)
                    # pprint(pred_clusters)

                    swap_original_clusters = swap_clusters(original_clusters)
                    swap_pred_clusters = swap_clusters(pred_clusters)
                    # pprint(swap_original_clusters)
                    # pprint(swap_pred_clusters)

                    tp = 0
                    fp = 0
                    fn = 0
                    t = 2/3
                    t_ = 1-t
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
                        continue
                    else:
                        f1 = 2 * (precision * recall) / (precision + recall)
                        precision_scores.append(precision)
                        recall_scores.append(recall)
                        f1_scores.append(f1)

                    if plot_tests:
                        nx_g = test_graph_out.to_networkx().to_undirected()
                        # pos = nx.kamada_kawai_layout(nx_g)
                        ax1 = plt.subplot(1,2,1)
                        nx.draw(nx_g, pos, with_labels=True, node_color="#A0CBE2")
                        # ax1.margin(5)
                        ax2 = plt.subplot(1,2,2)
                        nx_g_original = batched_graph.to_networkx().to_undirected()
                        nx.draw(nx_g_original, pos, with_labels=True, node_color="#A0CBE2")

                        # ax2.margin(5)
                        # plt.show()
                        plt.close()
                        # if count == 3:
                        #     break
                # print('%d, %d\tTested on: %d, Avg AUC: %.4f, Stdev: %.4f' % (h_feats, epochs,
                #                                                                         len(auc_scores), np.mean(auc_scores),
                #                                                                         np.std(auc_scores)))
                # print('Testing took: ', datetime.datetime.now()-start_t)
                # print('Precision: %.4f, Recall: %.4f, F1: %.4f' % (np.mean(precision_scores), np.mean(recall_scores), np.mean(f1_scores)))
                if len(f1_scores) > 0:
                    iteration += 1
                    model_output_tracker = pd.DataFrame(
                        list(zip([datetime.datetime.now()], [h_feats], [epochs], [len(f1_scores)],
                                 [np.mean(precision_scores)], [np.mean(recall_scores)], [np.mean(f1_scores)])),
                        columns=['time', 'feature_count', 'epoch_count', 'test_length', 'mean_precision', 'mean_recall', 'mean_f1'])
                    if os.path.exists('model_output_tracker.csv'):
                        model_output_tracker.to_csv('model_output_tracker_f1.csv', mode='a', index=False, header=False)
                    else:
                        model_output_tracker.to_csv('model_output_tracker_f1.csv', mode='w', index=False, header=True)
            count += 1
            print('#%d of %d\t%d, %d' % (count, total, h_feats, epochs))
