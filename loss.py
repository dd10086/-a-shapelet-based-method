import torch
import math
import numpy
import sklearn
import random
from scipy import spatial
from collections import Counter
from sklearn.cluster import DBSCAN
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
import timeit
import networks
import slide
import gc


def normalize(A, symmetric=True):
    d = A.sum(1)
    if symmetric:
        D = torch.diag(torch.pow(d, -0.5))
        return D.mm(A).mm(D)
    else:
        D = torch.diag(torch.pow(d, -1))
        return D.mm(A)


def cal_adj2(points, node_number):
    Adj = torch.zeros(node_number, node_number, dtype=torch.float64)
    k = int(len(points) * 0.3)
    Adj_numpy = numpy.zeros((node_number, node_number))
    soft_max = torch.nn.Softmax(dim=0)
    for i in range(len(points)):
        for j in range(len(points)):
            if i != j:
                Adj[i, j] = numpy.sqrt(numpy.sum(numpy.square(points[i] - points[j])))
                Adj_numpy[i, j] = numpy.sqrt(numpy.sum(numpy.square(points[i] - points[j])))
        Adj[i, i] = 0
    Adj[:, 0:len(points)] = soft_max(-Adj[:, 0:len(points)])
    Adj = normalize(Adj)
    if node_number > len(points):
        Adj[len(points):node_number, :] = 0
    return Adj.data


def cal_adj1(points, node_number):
    Adj = torch.zeros(node_number, node_number, dtype=torch.float64)
    for i in range(len(points)):
        for j in range(len(points)):
            if i != j:
                Adj[i, j] = cosine_similarity([points[i], points[j]])[0, 1]
    Adj = normalize(Adj)
    return Adj.data


def cal_dis(points, sample_number):
    dis = torch.zeros(sample_number, sample_number,
                      dtype=torch.float64)
    for i in range(sample_number):
        for j in range(sample_number):
            dis[i, j] = torch.norm(points[i, :] - points[j, :])
    return dis


class PNTripletLoss(torch.nn.modules.loss._Loss):

    def __init__(self, compared_length):
        super(PNTripletLoss, self).__init__()
        self.compared_length = compared_length
        if self.compared_length is None:
            self.compared_length = numpy.inf

    def forward(self, alpha, Adj, epoch, current_epoch, batch, encoder, params, node_number, save_memory=False):

        loss = torch.DoubleTensor([1])
        Adj.data = torch.zeros(node_number, node_number).data
        slide_start = timeit.default_timer()
        batch_slide = slide.slide_MTS_tensor_step(batch, alpha)
        points = batch_slide.numpy()
        num_cluster = 2
        kmeans = KMeans(n_clusters=num_cluster)
        kmeans.fit(points)
        cluster_label = kmeans.labels_
        num_cluster_set = Counter(cluster_label)
        loss_cluster = torch.DoubleTensor([1])
        L2 = torch.DoubleTensor([1])
        for i in range(num_cluster):
            points_select = []
            points_num = []
            cluster_start = timeit.default_timer()
            if num_cluster_set[i] < 2:
                continue
            cluster_i = points[numpy.where(cluster_label == i)]
            distance_i = kmeans.transform(cluster_i)[:, i]
            dist_positive = torch.DoubleTensor([1])
            dist_intra_positive = torch.DoubleTensor([1])
            dist_intra_negative = torch.DoubleTensor([1])
            dist_negative = torch.DoubleTensor([1])
            if num_cluster_set[i] >= 250:
                num_positive = 50
            else:
                num_positive = int(num_cluster_set[i] / 5 + 1)
            anchor_positive = numpy.argpartition(distance_i, num_positive)[
                              :(num_positive + 1)]
            points_select.append(points[kmeans.labels_ == i][anchor_positive[0]])
            points_num.append(1)
            for l in range(1, num_positive + 1):
                points_select.append(points[kmeans.labels_ == i][anchor_positive[l]])
            points_num.append(num_positive)
            for k in range(num_cluster):
                if k == i:
                    continue
                else:
                    if num_cluster_set[k] >= 250:
                        num_negative_cluster_k = 50
                    else:
                        num_negative_cluster_k = int(num_cluster_set[k] / 5 + 1)
                    cluster_i = points[numpy.where(cluster_label == k)]
                    distance_k = kmeans.transform(cluster_i)[:, k]
                    if distance_k.shape[0] > 1:
                        anchor_negative = numpy.argpartition(distance_k, num_negative_cluster_k)[
                                          :(num_negative_cluster_k)]
                    else:
                        anchor_negative = [0]
                    for l in range(0, num_negative_cluster_k):
                        points_select.append(points[kmeans.labels_ == k][anchor_negative[l]])
                points_num.append(num_negative_cluster_k)
            Adj.data = cal_adj2(points_select, node_number)
            representation_points_select = torch.from_numpy(numpy.array(points_select))
            sample_num = len(representation_points_select)
            temp_representation = torch.zeros(node_number, representation_points_select.shape[1],
                                              dtype=torch.float64)
            temp_representation[0:sample_num, :] = representation_points_select
            eplsion = 0.000001
            original_dis = cal_dis(temp_representation, sample_num) + eplsion
            representation_selected = encoder(temp_representation.reshape(node_number, 1, -1), epoch, current_epoch)[
                                      0:sample_num]
            encoder_dis = cal_dis(representation_selected, sample_num)
            scale_original_encoder_dis = encoder_dis / original_dis
            mean_scale_original_encoder_dis = torch.mean(scale_original_encoder_dis)
            representation_anc = representation_selected[0, :]  #
            representation_pos = representation_selected[1:num_positive + 1, :]
            representation_negs = []
            L2 += torch.norm(
                (scale_original_encoder_dis - mean_scale_original_encoder_dis)[0:num_positive + 1, 0:num_positive + 1])
            for m in range(len(points_num) - 2):
                representation_negs.append(
                    representation_selected[sum(points_num[0:m + 2]):sum(points_num[0:m + 3])])
                L2 += torch.norm(
                    (scale_original_encoder_dis - mean_scale_original_encoder_dis)[sum(points_num[0:m + 2]):
                                                                                   sum(points_num[0:m + 3]),
                    sum(points_num[0:m + 2]):sum(points_num[0:m + 3])])
            for l in range(len(representation_pos)):
                anchor_minus_positive = representation_anc - representation_pos[l]
                dist_positive += torch.norm(anchor_minus_positive)
            dist_positive = dist_positive / len(representation_pos)
            if len(representation_pos) > 1:
                pos_dist_pos = float("-inf")
                for index in range(1, len(representation_pos)):
                    for j in range(index + 1, len(representation_pos) + 1):
                        pos_minus_pos = points_select[index] - points_select[j]
                        intra_pos = numpy.linalg.norm(pos_minus_pos)
                        if intra_pos > pos_dist_pos:
                            pos_dist_pos = intra_pos
                            first_index = index
                            second_index = j
                representation_intra_pos_0 = representation_pos[first_index - 1]
                representation_intra_pos_1 = representation_pos[second_index - 1]
                intra_minus_positive = representation_intra_pos_0 - representation_intra_pos_1
                dist_intra_positive = torch.norm(intra_minus_positive)
            pos_sum_anc_num = points_num[0] + points_num[1]
            neg_num = 0
            for k in range(len(representation_negs)):
                if k > 0:
                    neg_num += len(representation_negs[k - 1])
                dist_cluster_k_negative = torch.DoubleTensor([1])
                for j in range(len(representation_negs[k])):
                    anchor_minus_negative = representation_anc - representation_negs[k][j]
                    dist_cluster_k_negative += torch.norm(anchor_minus_negative)
                dist_cluster_k_negative = dist_cluster_k_negative / len(representation_negs[k])
                dist_negative += dist_cluster_k_negative
                if len(representation_negs[k]) > 1:
                    neg_dist_neg = float("-inf")
                    for index in range(0, len(representation_negs[k]) - 1):
                        for j in range(index + 1, len(representation_negs[k])):
                            neg_minus_neg = points_select[index + neg_num + pos_sum_anc_num] - points_select[
                                j + neg_num + pos_sum_anc_num]
                            intra_neg = numpy.linalg.norm(neg_minus_neg)
                            if intra_neg > neg_dist_neg:
                                neg_dist_neg = intra_neg
                                first_index = index
                                second_index = j
                    representation_intra_neg_0 = representation_negs[k][first_index]
                    representation_intra_neg_1 = representation_negs[k][second_index]
                    intra_minus_negative = representation_intra_neg_0 - representation_intra_neg_1
                    dist_intra_negative += torch.norm(intra_minus_negative)
            dist_negative = dist_negative / len(representation_negs)
            loss_cluster += torch.log((dist_positive + 0.2) / dist_negative)
            loss_cluster += dist_intra_positive
            loss_cluster += dist_intra_negative
            cluster_end = timeit.default_timer()
        del batch_slide, points, points_select, representation_points_select, temp_representation, representation_selected, representation_pos, representation_negs
        gc.collect()
        loss += loss_cluster
        loss = 0.9 * loss + 0.05 * torch.log(L2)
        slide_end = timeit.default_timer()
        return loss
