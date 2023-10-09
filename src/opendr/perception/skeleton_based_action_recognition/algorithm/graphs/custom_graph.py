"""
Modified based on: https://github.com/open-mmlab/mmskeleton
"""

import numpy as np

num_node = 24  #21 each hand + 2 shoulders + 1 elbows
self_link = [(i, i) for i in range(num_node)]
#in_edge_ori_index = [(1, 2), (2, 21), (3, 21), (4, 3), (5, 21), (6, 5), (7, 6),
#                     (8, 7), (9, 21), (10, 9), (11, 10), (12, 11), (13, 1),
#                     (14, 13), (15, 14), (16, 15), (17, 1), (18, 17), (19, 18),
#                     (20, 19), (22, 23), (23, 8), (24, 25), (25, 12)]


#num_node = 46  #21 each hand + 2 shoulders + 2 elbows
#self_link = [(i, i) for i in range(num_node)]
#in_edge_ori_index = [ #shoulders to elbows and left elbow to left wrist
#		      (1, 3), (2, 4), (3, 5),
#		      # Left hand
#		      (5, 6), (6, 7), (7, 8), (8, 9),
#		      (9, 10), (10, 11), (11, 12), (12, 13), (10,14), (14, 15),
#		      (15, 16), (16, 17), (14, 18), (18, 19), (19, 20), (20, 21),
#		      (18, 22), (22, 23), (23,24), (24,25), (5, 22),
#		      # Right elbow to right wrist
#		      (4,26),
#		      # Right hand
#		      (26, 27), (27, 28), (28, 29), (29, 30), 
#		      (26, 31), (31, 32), (32, 33), (33, 34), (31, 35), (35, 36),
#		      (36, 37), (37, 38), (35, 39), (39, 40), (40, 41), (41, 42),
#		      (39, 43), (43, 44), (44, 45), (45, 46), (26, 43)]


#1 left_shoulder
#2 right_shoulder
#3 right_elbow
#4 wrist
#5 thumb_cmc
#6 thumb_mcp
#7 thumb_ip
#8 thumb_tip
#9 index_finger_mcp
#10 index_finger_pip
#11 index_finger_dip
#12 index_finger_tip
#13 middle_finger_mcp
#14 middle_finger_pip
#15 middle_finger_dip
#16 middle_finger_tip
#17 ring_finger_mcp
#18 ring_finger_pip
#19 ring_finger_dip
#20 ring_finger_tip
#21 pinky_mcp
#22 pinky_pip
#23 pinky_dip
#24 pinky_tip

in_edge_ori_index = [ #shoulders to elbows and left elbow to left wrist
		      (1, 2), (2, 3), (3, 4),
		      # Left hand
		      (4, 5), (5, 6), (6, 7), (7, 8),
		      (4, 9), (9, 10), (10, 11), (11, 12), (9,13), (13, 14),
		      (14, 15), (15, 16), (13, 17), (17, 18), (18, 19), (19, 20),
		      (17, 21), (21, 22), (22,23), (23,24), (4, 21),]
		      
in_edge = [(i - 1, j - 1) for (i, j) in in_edge_ori_index]
out_edge = [(j, i) for (i, j) in in_edge]
neighbor = in_edge + out_edge


def get_hop(link, num_node):
    A = np.zeros((num_node, num_node))
    for i, j in link:
        A[j, i] = 1
    return A


def normalize_digraph(A):
    Dl = np.sum(A, 0)
    h, w = A.shape
    Dn = np.zeros((w, w))
    for i in range(w):
        if Dl[i] > 0:
            Dn[i, i] = Dl[i] ** (-1)
    AD = np.dot(A, Dn)
    return AD


def get_spatial_graph(num_node, self_link, in_edge, out_edge):
    I = get_hop(self_link, num_node)
    In = normalize_digraph(get_hop(in_edge, num_node))
    Out = normalize_digraph(get_hop(out_edge, num_node))
    A = np.stack((I, In, Out))
    return A


class CustomGraph:
    def __init__(self, labeling_mode='spatial'):
        self.A = self.get_adjacency_matrix(labeling_mode)
        self.num_node = num_node
        self.self_link = self_link
        self.in_edge = in_edge
        self.out_edge = out_edge
        self.neighbor = neighbor

    def get_adjacency_matrix(self, labeling_mode=None):
        if labeling_mode is None:
            return self.A
        if labeling_mode == 'spatial':
            A = get_spatial_graph(num_node, self_link, in_edge, out_edge)
        else:
            raise ValueError()
        return A
