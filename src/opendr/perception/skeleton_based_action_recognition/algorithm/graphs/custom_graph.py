"""
Modified based on: https://github.com/open-mmlab/mmskeleton
"""

import numpy as np

#num_node = 24  #21 each hand + 2 shoulders + 1 elbows
#self_link = [(i, i) for i in range(num_node)]
#in_edge_ori_index = [(1, 2), (2, 21), (3, 21), (4, 3), (5, 21), (6, 5), (7, 6),
#                     (8, 7), (9, 21), (10, 9), (11, 10), (12, 11), (13, 1),
#                     (14, 13), (15, 14), (16, 15), (17, 1), (18, 17), (19, 18),
#                     (20, 19), (22, 23), (23, 8), (24, 25), (25, 12)]


num_node = 46  #21 each hand + 2 shoulders + 2 elbows
self_link = [(i, i) for i in range(num_node)]
in_edge_ori_index = [ #shoulders to elbows and left elbow to left wrist
                    (1, 3), (2, 4), (3, 5), (1,2),
                    # Left hand
                    (5, 6), (6, 7), (7, 8), (8, 9),
                    (5, 10), (10, 11), (11, 12), (12, 13), (5,14), (14, 15),
                    (15, 16), (16, 17), (5, 18), (18, 19), (19, 20), (20, 21),
                    (22, 23), (23,24), (24,25), (5, 22),
                    # Right elbow to right wrist
                    (4,26),
                    # Right hand
                    (26, 27), (27, 28), (28, 29), (29, 30), 
                    (26, 31), (31, 32), (32, 33), (33, 34), (26, 35), (35, 36),
                    (36, 37), (37, 38), (26, 39), (39, 40), (40, 41), (41, 42),
                    (43, 44), (44, 45), (45, 46), (26, 43)]

#0 left_shoulder
#1 right_shoulder
#2 left elbow
#3 right_elbow
#4 left_wrist -> 4-5 5-6 6-7 7-8 
#5 left_thumb_cmc -> 4-9 9-10 10-11 11-12
#6 left_thumb_mcp -> 4-13 13-14 14-15 15-16
#7 left_thumb_ip -> 4-17 17-18-18-19 19-20
#8 left_thumb_tip -> 4-21 21-22 22-23 23-24
#9 left_index_finger_mcp
#10 left_index_finger_pip
#11 left_index_finger_dip
#12 left_index_finger_tip
#13 left_middle_finger_mcp
#14 left_middle_finger_pip
#15 left_middle_finger_dip
#16 left_middle_finger_tip
#17 left_ring_finger_mcp
#18 left_ring_finger_pip
#19 left_ring_finger_dip
#20 left_ring_finger_tip
#21 left_pinky_mcp
#22 left_pinky_pip
#23 left_pinky_dip
#24 left_pinky_tip
#25 right_wrist -> 25-26 26-27 27-28 28-29
#26 right_thumb_cmc -> 25-30 30-31 31-32 32-33
#27 right_thumb_mcp -> 25-34 34-35 35-36 36-37
#28 right_thumb_ip -> 25-38 38-39 39-40 40-41
#29 right_thumb_tip -> 25-42 42-43 43-44 44-45
#30 right_index_finger_mcp
#31 right_index_finger_pip
#32 right_index_finger_dip
#33 right_index_finger_tip
#34 right_middle_finger_mcp
#35 right_middle_finger_pip
#36 right_middle_finger_dip
#37 right_middle_finger_tip
#38 right_ring_finger_mcp
#39 right_ring_finger_pip
#40 right_ring_finger_dip
#41 right_ring_finger_tip
#42 right_pinky_mcp
#43 right_pinky_pip
#44 right_pinky_dip
#45 right_pinky_tip

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
