from lib import geom, graph as graph_helper

import numpy as np
import math
from PIL import Image
import random
import rtree
import sys
import time
import cv2 as cv
from skimage import measure


class Path(object):
    def __init__(self, idx, training, gc, tile_data, graph=None, road_seg=None, WINDOW_SIZE=256):
        """
        graph container contains the total graph of the region, not only the search rectangle,
        so it records the road segment info of the total graph, when reset the gc, be careful
        not to affect other path belonging to the same graph
        """

        self.idx = idx
        self.gc = gc
        self.tile_data = tile_data
        self.road_seg = road_seg
        self.road_seg_origin = self.tile_data['search_rect'].start
        self.ROADSEG_OVERWRITE_THICKNESS = 20
        self.ROADSEG_OVERWRITE_RADIUS = 20
        if graph is None:
            self.graph = graph_helper.Graph()
        else:
            self.graph = graph
            self.remove_graph_from_road_seg()
        self.tile_size = self.tile_data['search_rect'].lengths().x

        self.unmatched_vertices = 0

        self._load_edge_rtree()
        self._load_key_point_rtree()

        self.not_explored_starting_points = self.tile_data['starting_locations']['junction'].copy() + \
            self.tile_data['starting_locations']['middle'].copy()

        self.search_vertices = []

        self.anchor_point_rtree = rtree.index.Index()
        self.indexed_anchor_points = dict()
        self.is_training = training
        if self.is_training:
            self.rs_exploration = graph_helper.RoadSegmentExplorationDict(
                gc=gc, rect=tile_data['search_rect'])

    def remove_graph_from_road_seg(self):
        if self.graph is not None and self.road_seg is not None:
            for edge in self.graph.edges.values():
                src = edge.src(self.graph).point.sub(self.road_seg_origin)
                dst = edge.dst(self.graph).point.sub(self.road_seg_origin)
                cv.line(self.road_seg, (src.y, src.x), (dst.y, dst.x),
                        color=0, thickness=self.ROADSEG_OVERWRITE_THICKNESS)

    def _load_key_point_rtree(self):
        self.key_point_rtree = rtree.index.Index()
        self.indexed_key_points = dict()
        starting_locations = self.tile_data['starting_locations']['junction']
        for i, item in enumerate(starting_locations):
            pnt = item[0]['point']
            self.key_point_rtree.insert(i, (pnt.x, pnt.y, pnt.x, pnt.y))
            self.indexed_key_points[i] = pnt

    def _load_edge_rtree(self):
        self.indexed_edges = set()
        self.edge_rtree = rtree.index.Index()
        for edge in self.graph.edges.values():
            self._add_edge_to_rtree(edge)

    def _add_edge_to_rtree(self, edge):
        if edge.id in self.indexed_edges:
            return
        self.indexed_edges.add(edge.id)
        bounds = edge.segment(self.graph).bounds().add_tol(1)
        self.edge_rtree.insert(
            edge.id, (bounds.start.x, bounds.start.y, bounds.end.x, bounds.end.y))

    def _add_bidirectional_edge(self, src, dst, prob=1.0):
        edges = self.graph.add_bidirectional_edge(src.id, dst.id)
        edges[0].prob = prob
        edges[1].prob = prob
        self._add_edge_to_rtree(edges[0])
        self._add_edge_to_rtree(edges[1])

    def prepend_search_vertex(self, vertex, is_key_point):
        if self.tile_data['search_rect'].contains(vertex.point):
            self.search_vertices.append((vertex, is_key_point))
            return True
        else:
            return False

    def mark_rs_explored_part(self, rs, edge_pos):
        rs_exp = self.rs_exploration[rs.id]
        if rs_exp.is_explored():
            return
        edge = edge_pos.edge(self.gc.graph)
        curr_distance = rs.edge_distances[edge.id] + edge_pos.distance
        if curr_distance > rs_exp.explored_start_dis:
            if round(curr_distance)+1 >= round(rs_exp.explored_end_dis):
                self.mark_rs_explored(rs)
                return
            rs_exp.explored_start_dis = curr_distance
            rs_exp.explored_start_pos = edge_pos
            opposite_rs = rs.get_opposite_rs(
                self.gc.edge_id_to_rs_id, self.gc.road_segments, self.gc.graph)
            if opposite_rs is not None:
                opposite_rs_exp = self.rs_exploration[opposite_rs.id]
                opposite_rs_exp.explored_end_dis = opposite_rs.marked_length - curr_distance
                opposite_rs_exp.explored_end_pos = edge_pos.reverse(
                    self.gc.graph)
            else:
                print('err!')
        else:
            # meet a self-crossed overpass
            pass

    def mark_rs_explored(self, rs):
        self.rs_exploration[rs.id].explored = True
        oppo_rs = rs.get_opposite_rs(
            self.gc.edge_id_to_rs_id, self.gc.road_segments, self.gc.graph)
        if oppo_rs is not None:
            self.rs_exploration[oppo_rs.id].explored = True
        else:
            print('err!')

    def is_explored_edge_pos(self, edge_pos):
        edge = edge_pos.edge(self.gc.graph)
        rs = self.gc.edge_id_to_rs(edge.id)
        if self.rs_exploration[rs.id].is_explored():
            return True
        curr_distance = rs.edge_distances[edge.id] + edge_pos.distance
        if self.rs_exploration[rs.id].explored_start_dis <= curr_distance < self.rs_exploration[rs.id].explored_end_dis:
            return False
        return True

    def get_vertex_from_point_in_graph(self, pnt):
        lst = list(self.edge_rtree.intersection((pnt.x, pnt.y, pnt.x, pnt.y)))
        for edge_id in lst:
            edge = self.graph.edges[edge_id]
            for vertex in [edge.src(self.graph), edge.dst(self.graph)]:
                if vertex.point == pnt:
                    return vertex
        return None

    def get_rs_from_next_point(self, target_poses, next_point):
        target_poses_len = len(target_poses)
        assert target_poses_len >= 1
        if target_poses_len == 1:
            nearest_target_pos = target_poses[0]
        else:
            nearest_target_pos = None
            for pos in target_poses:
                if nearest_target_pos is None or \
                        pos.point(self.gc.graph).distance(next_point) < nearest_target_pos.point(self.gc.graph).distance(next_point):
                    nearest_target_pos = pos
        rs = self.gc.edge_id_to_rs(nearest_target_pos.edge_id)
        return rs, nearest_target_pos

    def generate_new_vertex(self, next_point):
        next_vertex = self.graph.add_vertex(next_point)
        next_vertex.edge_pos = None  # if TRAIN, set value below
        new_vertex_index = next_vertex.id
        self.anchor_point_rtree.insert(
            new_vertex_index, (next_point.x, next_point.y, next_point.x, next_point.y))
        self.indexed_anchor_points[new_vertex_index] = next_point
        return next_vertex

    # if at key point, target_poses contains position of next points in different road segment with one time step
    # [[.....], [], [], []]
    # else, target_poses contains position of next points in one road segment with num_targets time step
    # [[.], [.], [.], []] or [[.], [.....], [], []] (end with a junction)
    def push(self, extension_vertex, is_key_point, follow_mode, target_poses, output_points, RECT_RADIUS=10,
             road_segmentation=None, NUM_TARGETS=4, WINDOW_SIZE=256, STEP_LENGTH=20, AVG_CONFIDENCE_THRESHOLD=0.2):

        def proc(target_pos_lst, explored_points, curr_vertex, curr_pnt, next_point, next_pos, prepend_flag):
            new_vertex_flag, key_point_flag, end_flag, next_vertex, next_key_point = \
                self._follow_graph_one_step(
                    mode='push',
                    curr_pnt=curr_pnt, curr_rs=None,
                    next_point=next_point, next_pos=next_pos,
                    road_segmentation=road_segmentation,
                    origin_pnt=extension_vertex.point.sub(
                        geom.Point(WINDOW_SIZE//2, WINDOW_SIZE//2)),
                    explored_points=explored_points,
                    STEP_LENGTH=STEP_LENGTH, RECT_RADIUS=RECT_RADIUS, WINDOW_SIZE=WINDOW_SIZE)
            if next_vertex is not None and curr_vertex == next_vertex:
                return True, None
            # new_vertex_flag
            if new_vertex_flag == 0:
                next_vertex = self.generate_new_vertex(next_point)
            elif new_vertex_flag == 1:
                if next_vertex is None:
                    next_vertex = self.generate_new_vertex(next_key_point)
            elif new_vertex_flag == 2:
                pnt = next_vertex.point
                self.anchor_point_rtree.delete(
                    next_vertex.id, (pnt.x, pnt.y, pnt.x, pnt.y))
                self.anchor_point_rtree.insert(
                    next_vertex.id, (next_point.x, next_point.y, next_point.x, next_point.y))
                next_vertex.point = next_point
            # key_point_flag
            if prepend_flag is True or end_flag is True:
                self.prepend_search_vertex(
                    next_vertex, is_key_point=key_point_flag)
            if end_flag:
                key_pnts = get_points_from_rtree(point_rtree=self.key_point_rtree,
                                                 index2point=self.indexed_key_points, center_point=next_vertex.point, RECT_RADIUS=0)
                if len(key_pnts) == 0:
                    pnt = next_vertex.point
                    index = len(self.indexed_key_points)
                    self.key_point_rtree.insert(
                        index, (pnt.x, pnt.y, pnt.x, pnt.y))
                    self.indexed_key_points[index] = pnt
            # end_flag
            if self.is_training:
                rs, tp = self.get_rs_from_next_point(
                    target_pos_lst, next_point)
                if next_vertex.edge_pos is None:
                    next_vertex.edge_pos = tp
                if end_flag:
                    self.mark_rs_explored(rs=rs)
                else:
                    self.mark_rs_explored_part(rs=rs, edge_pos=tp)
            # add bidirectional edge
            self._add_bidirectional_edge(curr_vertex, next_vertex)
            if self.road_seg is not None:
                src = curr_vertex.point.sub(self.road_seg_origin)
                dst = next_vertex.point.sub(self.road_seg_origin)
                cv.line(self.road_seg, (src.y, src.x), (dst.y, dst.x),
                        color=0, thickness=self.ROADSEG_OVERWRITE_THICKNESS)
            explored_points.append(next_vertex.point)
            return end_flag, next_vertex

        if follow_mode == 'follow_target':
            next_points = [pos.point(
                self.gc.graph) for pos in target_poses.get_single_lst_without_junction_end()]
        elif follow_mode == 'follow_output':
            next_points = output_points
        else:
            raise NotImplementedError
        if len(next_points) == 0:
            return
        origin_point = extension_vertex.point.sub(
            geom.Point(WINDOW_SIZE//2, WINDOW_SIZE//2))

        if is_key_point:
            """
            check for loop pushing
            ---|---
               | /  <==
               |/   <==
            """
            if road_segmentation is not None:
                nearby_edge_segments = graph_helper.get_nearby_edge_segments(
                    extension_vertex, 1, self.graph)
                for pnt in next_points.copy():
                    for segment in nearby_edge_segments:
                        if segment.distance(pnt) < RECT_RADIUS and \
                                get_avg_between_pnts_in_map(
                                    im_map=road_segmentation,
                                    pnt1=segment.start.sub(origin_point),
                                    pnt2=segment.end.sub(origin_point),
                                    WINDOW_SIZE=WINDOW_SIZE
                        ) < AVG_CONFIDENCE_THRESHOLD:
                            next_points.remove(pnt)
                            break
            """
            check if end
            end_flag[]:
              1: not end;
              2: end at key points;
              3: end at anchor points
                (a)---+----
                      |
                   ==>
                      |
                      |
                (b)--------
                   ==>
                      |
            """
            explored_points = [x.point for x in graph_helper.get_nearby_vertices(
                extension_vertex, 2, self.graph)]
            for next_point in next_points:
                proc(target_pos_lst=target_poses[0] if target_poses is not None else None,
                     explored_points=explored_points,
                     curr_vertex=extension_vertex, curr_pnt=extension_vertex.point,
                     next_point=next_point, next_pos=None, prepend_flag=True)
        else:
            # cannot add edge between same vertex
            if len(next_points) == 1 and self.get_vertex_from_point_in_graph(next_points[0]) == extension_vertex:
                return
            # recurrent set curr_vertex
            curr_vertex = extension_vertex
            # -1: not end; >=0: end at key points; -2: end at anchor points
            # end_flag = False
            # explored_points = [extension_vertex.point]
            explored_points = [x.point for x in graph_helper.get_nearby_vertices(
                extension_vertex, 2, self.graph)]
            for i, next_point in enumerate(next_points):
                end_flag, next_vertex = proc(
                    target_pos_lst=target_poses[i] if target_poses is not None else None,
                    explored_points=explored_points,
                    curr_vertex=curr_vertex, curr_pnt=curr_vertex.point,
                    next_point=next_point, next_pos=None,
                    prepend_flag=True if i == len(next_points)-1 else False)
                if end_flag:
                    break
                # recurrent set curr_vertex
                curr_vertex = next_vertex

    def pop(self, follow_order=True, probs=[0.15, 0.8, 0.05], WINDOW_SIZE=256):
        """

        :param follow_order:
        :param probs: {
                "pop_unexplored_starting_point": 0.15,
                "pop_search_vertices": 0.8,
                "pop_random_starting_point": 0.05
            }
        :param WINDOW_SIZE:
        :return:
        """

        def _pop_search_vertices():
            if len(self.search_vertices) > 0:
                if follow_order:
                    _vertex, _is_key_point = self.search_vertices.pop()
                else:
                    _vertex, _is_key_point = self.search_vertices.pop(
                        random.randint(0, len(self.search_vertices)-1))
                return _vertex, _is_key_point
            return None, None

        def _pop_not_explored_starting_points():
            if len(self.not_explored_starting_points) > 0:
                index = random.randint(
                    0, len(self.not_explored_starting_points) - 1)
                start_loc = self.not_explored_starting_points.pop(index)
                if not self.is_training or start_loc[0]['key_point']:
                    _vertex = self.graph.add_vertex(start_loc[0]['point'])
                    _vertex.edge_pos = start_loc[0]['edge_pos']
                    _is_key_point = start_loc[0]['key_point']
                    return _vertex, True
                elif self.is_training:  # starting point at the middle of the road
                    split_point = start_loc[0]['point']
                    for edge_id in [start_loc[0]['edge_pos'].edge_id, start_loc[0]['edge_pos'].edge(self.gc.graph).get_opposite_edge(self.gc.graph).id]:
                        rs = self.gc.edge_id_to_rs(edge_id)
                        rs_exp = self.rs_exploration[rs.id]
                        rs_edges = rs.edges(self.gc.graph)
                        break_index = 0
                        for edge in rs_edges:
                            if edge.dst(self.gc.graph).point == split_point:
                                break_index = rs.edges_id.index(edge.id) + 1
                                break
                        if break_index == 0 or break_index == len(rs_edges):
                            return _pop_not_explored_starting_points()
                        if rs_exp.is_explored() or \
                                rs_exp.explored_start_dis + 1 >= \
                                rs.edge_distances[rs.edges_id[break_index]] \
                                or \
                                rs_exp.explored_end_dis - 1 <= \
                                rs.edge_distances[rs.edges_id[break_index]]:
                            return _pop_not_explored_starting_points()
                        new_rs = graph_helper.RoadSegment(
                            len(self.gc.road_segments))
                        new_rs.edges_id = rs.edges_id[break_index:]
                        rs.marked_length = rs.edge_distances[rs.edges_id[break_index]]
                        rs.edges_id = rs.edges_id[:break_index]
                        for k, v in rs.edge_distances.copy().items():
                            if k not in rs.edges_id:
                                del rs.edge_distances[k]
                        new_rs.compute_edge_distances(self.gc.graph)
                        for new_edge_id in new_rs.edges_id:
                            self.gc.edge_id_to_rs_id[new_edge_id] = new_rs.id
                        self.gc.road_segments.append(new_rs)
                        new_rs_exp = graph_helper.new_road_segment_exploration(
                            new_rs, self.gc.graph)
                        self.rs_exploration.data[new_rs.id] = new_rs_exp
                        new_rs_exp.explored_end_dis = new_rs.marked_length - \
                            (rs_exp.marked_length - rs_exp.explored_end_dis)
                        new_rs_exp.explored_end_pos = rs_exp.explored_end_pos
                        rs_exp.explored_end_pos = graph_helper.EdgePos(
                            rs.edges_id[-1], distance=self.gc.graph.edges[rs.edges_id[-1]].segment(self.gc.graph).length())
                        rs_exp.explored_end_dis = rs.marked_length
                        rs_exp.marked_length = rs.marked_length
                    _vertex = self.graph.add_vertex(start_loc[0]['point'])
                    _vertex.edge_pos = start_loc[0]['edge_pos']
                    return _vertex, True
            return None, None

        def _pop_road_seg_peak():
            if self.road_seg.max() > 0:
                peak = self.road_seg.argmax()
                peak = geom.Point(peak % self.tile_size, peak / self.tile_size)
                cv.circle(self.road_seg, (peak.x, peak.y),
                          radius=self.ROADSEG_OVERWRITE_RADIUS, color=0, thickness=-1)
                _vertex = graph_helper.Vertex(self.graph.vertices_index, peak.add(
                    self.tile_data['search_rect'].start))
                _vertex.edge_pos = None
                _vertex.from_road_seg = True
                _is_key_point = False
                return _vertex, _is_key_point
            return None, None

        def _pop_random_patch(max_iter=5):
            random_rect = get_random_rect_padding(
                self.tile_data['search_rect'], WINDOW_SIZE)
            small_rect = random_rect.add_tol(-WINDOW_SIZE//3)
            if len(self.gc.edge_index.search(small_rect)) > 0:
                if max_iter == 0:
                    return None, None
                return _pop_random_patch(max_iter-1)
            else:
                _vertex = graph_helper.Vertex(-1, random_rect.start.add(
                    geom.Point(WINDOW_SIZE//2, WINDOW_SIZE//2)))
                _vertex.edge_pos = None
                return _vertex, False

        vertex, is_key_point = None, None
        if follow_order:
            if len(self.search_vertices) > 0:
                vertex, is_key_point = _pop_search_vertices()
            elif len(self.not_explored_starting_points) > 0:
                vertex, is_key_point = _pop_not_explored_starting_points()
            elif self.road_seg is not None:
                vertex, is_key_point = _pop_road_seg_peak()
            return vertex, is_key_point
        else:
            choice = random_sample_given_probs(
                ["pop_unexplored_starting_point", "pop_search_vertices", "pop_random_starting_point"], probs=probs)
            if choice == "pop_unexplored_starting_point" and len(self.not_explored_starting_points) > 0:
                vertex, is_key_point = _pop_not_explored_starting_points()
            elif choice == "pop_search_vertices" and len(self.search_vertices) > 0:
                vertex, is_key_point = _pop_search_vertices()
            elif choice == "pop_random_starting_point":
                vertex, is_key_point = _pop_random_patch(max_iter=5)
            if vertex is None and (len(self.not_explored_starting_points) > 0 or len(self.search_vertices) > 0):
                if len(self.search_vertices) > 0:
                    vertex, is_key_point = _pop_search_vertices()
                if vertex is None:
                    vertex, is_key_point = _pop_not_explored_starting_points()
            return vertex, is_key_point

    def _follow_graph_one_step(self, mode, curr_pnt, curr_rs, next_point, next_pos, road_segmentation,
                               origin_pnt, explored_points=list(),
                               AVG_CONFIDENCE_THRESHOLD=0.2, STEP_LENGTH=20, RECT_RADIUS=10, WINDOW_SIZE=256):

        if mode == 'push':
            # 0: self.generate_new_vertex; 1: exist vertex; 2: move vertex
            new_vertex_flag = 0
            # False: anchor point; True: key point
            key_point_flag = False
            # False: part; True: end
            end_flag = False
            next_vertex = None
            next_key_point = None
        elif mode == 'pop':
            new_edge_pos = next_pos

        key_points = get_points_from_rtree(point_rtree=self.key_point_rtree,
                                           index2point=self.indexed_key_points, center_point=next_point, RECT_RADIUS=RECT_RADIUS)
        for pnt in explored_points:
            if pnt in key_points:
                del key_points[pnt]
        anchor_points = get_points_from_rtree(point_rtree=self.anchor_point_rtree,
                                              index2point=self.indexed_anchor_points, center_point=next_point, RECT_RADIUS=RECT_RADIUS)
        for pnt in explored_points:
            if pnt in anchor_points:
                del anchor_points[pnt]
        if len(anchor_points) == 0 and len(key_points) == 0:  # 0: not end
            if mode == 'push':
                new_vertex_flag = 0
                key_point_flag = False
                end_flag = False
            elif mode == 'pop':
                new_edge_pos = next_pos
        else:  # len(anchor_points) > 0 or len(key_points) > 0
            if len(anchor_points) > 0:
                anchor_pnt = get_nearest_end_point(
                    anchor_points.keys(), next_point)
                anchor_pnt_dis = anchor_pnt.distance(next_point)
            else:
                anchor_pnt_dis = math.inf
            if len(key_points) > 0:
                key_pnt = get_nearest_end_point(key_points.keys(), next_point)
                key_pnt_dis = key_pnt.distance(next_point)
            else:
                key_pnt_dis = math.inf
            if anchor_pnt_dis < key_pnt_dis:  # anchor_pnt
                avg_conf = get_avg_between_pnts_in_map(
                    im_map=road_segmentation,
                    pnt1=curr_pnt.sub(origin_pnt),
                    pnt2=anchor_pnt.sub(origin_pnt),
                    WINDOW_SIZE=WINDOW_SIZE)
                if anchor_pnt_dis > 8 and avg_conf < AVG_CONFIDENCE_THRESHOLD:  # do not select the anchor point
                    if mode == 'push':
                        new_vertex_flag = 0
                        key_point_flag = False
                        end_flag = False
                    elif mode == 'pop':
                        new_edge_pos = next_pos
                else:  # select the anchor point
                    end_flag = True
                    next_vertex = self.graph.vertices[anchor_points[anchor_pnt]]
                    if anchor_pnt_dis > 8 and len(next_vertex.in_edges_id) > 1:
                        if mode == 'push':
                            new_vertex_flag = 2
                            key_point_flag = True
                            end_flag = False
                        elif mode == 'pop':
                            new_edge_pos = curr_rs.closest_pos(
                                anchor_pnt, self.gc.graph)
                    else:  # case(a) or case(b)
                        if len(next_vertex.in_edges_id) > 1:
                            if mode == 'push':
                                new_vertex_flag = 1
                                key_point_flag = True
                                end_flag = False
                            elif mode == 'pop':
                                new_edge_pos = curr_rs.closest_pos(
                                    anchor_pnt, self.gc.graph)
                        else:
                            if mode == 'push':
                                new_vertex_flag = 1
                                key_point_flag = False
                                end_flag = False
                            elif mode == 'pop':
                                new_edge_pos = curr_rs.closest_pos(
                                    anchor_pnt, self.gc.graph)
            else:  # key_pnt
                avg_conf = get_avg_between_pnts_in_map(
                    im_map=road_segmentation,
                    pnt1=curr_pnt.sub(origin_pnt),
                    pnt2=key_pnt.sub(origin_pnt),
                    WINDOW_SIZE=WINDOW_SIZE)
                if key_pnt_dis > 8 and avg_conf < AVG_CONFIDENCE_THRESHOLD:  # do not select the key point
                    if mode == 'push':
                        new_vertex_flag = 0
                        key_point_flag = False
                        end_flag = False
                    elif mode == 'pop':
                        new_edge_pos = next_pos
                else:  # select the key point
                    if mode == 'push':
                        next_vertex = self.get_vertex_from_point_in_graph(
                            key_pnt)
                        if next_vertex is None:
                            new_vertex_flag = 0
                            key_point_flag = True
                            end_flag = True
                            next_key_point = key_pnt
                        else:
                            new_vertex_flag = 1
                            key_point_flag = True
                            end_flag = True
                    elif mode == 'pop':
                        # assert curr_rs.explored_end_pos.point() == key_pnt
                        new_edge_pos = curr_rs.closest_pos(
                            key_pnt, self.gc.graph)
        if mode == 'push':
            return new_vertex_flag, key_point_flag, end_flag, next_vertex, next_key_point
        elif mode == 'pop':
            return new_edge_pos

    def get_target_poses(self, extension_vertex, road_segmentation, STEP_LENGTH=20, is_key_point=False,
                         NUM_TARGETS=4, RECT_RADIUS=10, WINDOW_SIZE=256):
        """
        :return target_poses: [[], [], [], []]
        """

        def append_next_starting_pos(curr_rs, target_poses, curr_pnt, curr_vertex, explored_points, potential_rs_list):
            target_poses_len = len(target_poses)
            if target_poses_len >= NUM_TARGETS:
                return
            if potential_rs_list is None:
                potential_rs_list = []
                for edge in curr_vertex.out_edges(self.gc.graph):
                    next_rs = self.gc.edge_id_to_rs(edge.id)
                    if curr_rs is not None and (next_rs.id == curr_rs.id or next_rs.is_opposite(curr_rs, self.gc.graph)):
                        continue
                    next_rs_exp = self.rs_exploration[next_rs.id]
                    if next_rs_exp.is_explored() or next_rs_exp.explored_start_dis > 0:
                        continue
                    potential_rs_list.append(next_rs)
                # detect very short road segment
                for next_rs in potential_rs_list.copy():
                    if next_rs.marked_length < 5:
                        potential_rs_list.remove(next_rs)
                        for edge in next_rs.dst(self.gc.graph).out_edges(self.gc.graph):
                            next_next_rs = self.gc.edge_id_to_rs(edge.id)
                            if next_rs.id == next_next_rs.id or next_rs.is_opposite(next_next_rs, self.gc.graph):
                                continue
                            if self.rs_exploration[next_next_rs.id].is_explored():
                                continue
                            potential_rs_list.append(next_next_rs)
            for next_rs in potential_rs_list:
                next_rs_exp = self.rs_exploration[next_rs.id]
                # next_starting_pos = next_rs.closest_pos(curr_pnt)
                next_starting_pos = next_rs_exp.explored_start_pos
                # assert next_starting_pos.point(self.gc.graph) == curr_pnt  # TODO
                if self.is_explored_edge_pos(next_starting_pos):
                    continue
                if next_rs_exp.get_unexplored_dis() <= STEP_LENGTH * 1.5:
                    target_poses[target_poses_len].append(
                        next_rs_exp.explored_end_pos)
                    continue
                rs_follow_positions = self.gc.graph.follow_graph(
                    next_starting_pos, STEP_LENGTH)
                # assert len(rs_follow_positions) == 1 and \
                #     next_rs.id == self.gc.edge_id_to_rs_id[rs_follow_positions[0].edge.id].id
                next_pos = rs_follow_positions[0]
                new_edge_pos = self._follow_graph_one_step(
                    mode='pop',
                    curr_pnt=curr_pnt, curr_rs=next_rs,
                    next_point=next_pos.point(self.gc.graph), next_pos=next_pos,
                    road_segmentation=road_segmentation,
                    origin_pnt=extension_vertex.point.sub(
                        geom.Point(WINDOW_SIZE//2, WINDOW_SIZE//2)),
                    explored_points=explored_points,
                    STEP_LENGTH=STEP_LENGTH, RECT_RADIUS=RECT_RADIUS, WINDOW_SIZE=WINDOW_SIZE)
                if not self.tile_data['search_rect'].contains(new_edge_pos.point(self.gc.graph)):
                    continue
                if new_edge_pos.point(self.gc.graph) not in [x.point(self.gc.graph) for x in target_poses[target_poses_len]]:
                    # prevent same point but different rs, just waste one
                    target_poses[target_poses_len].append(new_edge_pos)
                explored_points.append(new_edge_pos.point(self.gc.graph))

        target_poses = TargetPosesContainer(NUM_TARGETS)
        if extension_vertex.edge_pos is None:
            return target_poses

        # avoid getting into another road
        if extension_vertex.edge_pos.point(self.gc.graph).distance(extension_vertex.point) > 2 * STEP_LENGTH:
            # map_match_pos()
            # if extension_vertex.edge_pos is None:
            #     return target_poses
            return target_poses

        curr_edge = extension_vertex.edge_pos.edge(self.gc.graph)
        curr_rs = self.gc.edge_id_to_rs(curr_edge.id)

        # if at key point, target_poses contains position of next points in different road segment with one time step
        # else, target_poses contains position of next points in one road segment with num_targets time step

        if is_key_point:  # more than one rs, just one time step
            extension_vertex_gt_graph = None
            potential_rs_list = None
            if extension_vertex.point == curr_edge.src(self.gc.graph).point:
                extension_vertex_gt_graph = curr_edge.src(self.gc.graph)
            elif extension_vertex.point == curr_edge.dst(self.gc.graph).point:
                extension_vertex_gt_graph = curr_edge.dst(self.gc.graph)
            else:
                # walk into a viaduct which do not cross but look like a key point in 2D
                potential_rs_list = [curr_rs]

            explored_points = [x.point for x in graph_helper.get_nearby_vertices(
                extension_vertex, 2, self.graph)]
            append_next_starting_pos(curr_rs=None, target_poses=target_poses,
                                     curr_pnt=extension_vertex.edge_pos.point(
                                         self.gc.graph),
                                     curr_vertex=extension_vertex_gt_graph,
                                     explored_points=explored_points, potential_rs_list=potential_rs_list)
        else:  # only one rs, more than one time step
            rs = curr_rs
            rs_exp = self.rs_exploration[rs.id]
            curr_pos = extension_vertex.edge_pos
            explored_points = [x.point for x in graph_helper.get_nearby_vertices(
                extension_vertex, 2, self.graph)]
            if not self.is_explored_edge_pos(curr_pos):
                for i in range(NUM_TARGETS):
                    if rs_exp.get_unexplored_dis() - i * STEP_LENGTH <= STEP_LENGTH * 1.5:
                        target_poses[i].append(rs_exp.explored_end_pos)
                        if rs_exp.explored_end_dis >= rs_exp.marked_length - 1:
                            append_next_starting_pos(
                                rs, target_poses, curr_pnt=rs.dst(
                                    self.gc.graph).point,
                                curr_vertex=rs.dst(self.gc.graph), explored_points=explored_points,
                                potential_rs_list=None)
                        break
                    rs_follow_positions = self.gc.graph.follow_graph(
                        extension_vertex.edge_pos, STEP_LENGTH * (i+1))
                    next_pos = rs_follow_positions[0]
                    new_edge_pos = self._follow_graph_one_step(
                        mode='pop',
                        curr_pnt=curr_pos.point(self.gc.graph), curr_rs=rs,
                        next_point=next_pos.point(self.gc.graph), next_pos=next_pos,
                        road_segmentation=road_segmentation,
                        origin_pnt=extension_vertex.point.sub(
                            geom.Point(WINDOW_SIZE//2, WINDOW_SIZE//2)),
                        explored_points=explored_points,
                        STEP_LENGTH=STEP_LENGTH, RECT_RADIUS=RECT_RADIUS, WINDOW_SIZE=WINDOW_SIZE)
                    target_poses[i].append(new_edge_pos)
                    explored_points.append(new_edge_pos.point(self.gc.graph))
                    curr_pos = new_edge_pos
        return target_poses

    def make_path_input(self, extension_vertex, fetch_list, is_key_point=False, WINDOW_SIZE=256):
        """
        :param extension_vertex: 
        :param fetch_list: 
            'aerial_image_chw':
            'aerial_image_hwc':
            'walked_path_small':
            'road_seg_small':
            'road_seg_thick3':
            'junc_seg_small':
        :param is_key_point: 
        :param WINDOW_SIZE:
        :return:
        """
        search_rect = self.tile_data['search_rect']
        big_origin = search_rect.start  # (0,0)
        big_img = self.tile_data['cache'].get(
            self.tile_data['region'], search_rect)
        # {'input': img.shape==(4096,4096,3)}

        if not search_rect.contains(extension_vertex.point):
            # (top_left:(128, 128), buttom_right:(1920, 1920))
            raise Exception('bad path {}'.format(self))
        origin = extension_vertex.point.sub(
            geom.Point(WINDOW_SIZE // 2, WINDOW_SIZE // 2))
        tile_origin = origin.sub(big_origin).add(
            geom.Point(WINDOW_SIZE // 2, WINDOW_SIZE // 2))
        rect = origin.bounds().extend(origin.add(geom.Point(WINDOW_SIZE, WINDOW_SIZE)))
        safe_rect = search_rect.add_tol(-WINDOW_SIZE // 2)

        ################ walked_path ################
        walked_path_small = None
        if 'walked_path_small' in fetch_list:
            walked_path_small = np.zeros(
                (WINDOW_SIZE // 4, WINDOW_SIZE // 4), dtype=np.float32)
            for edge_id in self.edge_rtree.intersection((rect.start.x, rect.start.y, rect.end.x, rect.end.y)):
                edge = self.graph.edges[edge_id]
                start = edge.src(self.graph).point.sub(origin)
                end = edge.dst(self.graph).point.sub(origin)
                cv.line(walked_path_small, (start.y // 4, start.x // 4),
                        (end.y // 4, end.x // 4), 1, 1)
            walked_path_small[WINDOW_SIZE // 8, WINDOW_SIZE // 8] = 1.0

        ################ road_seg & junc_seg ################
        road_seg_small = junc_seg_small = None
        if 'road_seg_small' in fetch_list or 'junc_seg_small' in fetch_list:
            seg_rect = rect
            seg_origin = seg_rect.start
            if self.is_training:
                road_seg_small = np.zeros(
                    (WINDOW_SIZE // 4, WINDOW_SIZE // 4), dtype=np.float32)
                junc_seg_small = np.zeros(
                    (WINDOW_SIZE // 4, WINDOW_SIZE // 4), dtype=np.float32)
                for edge in self.gc.edge_index.search(seg_rect):
                    if 'junc_seg_small' in fetch_list:
                        for vertex in [edge.src(self.gc.graph), edge.dst(self.gc.graph)]:
                            pnt = vertex.point
                            if len(vertex.out_edges(self.gc.graph)) > 2 and seg_rect.contains(pnt) and search_rect.contains(pnt):
                                pnt = pnt.sub(seg_origin)
                                cv.circle(
                                    junc_seg_small, (pnt.y // 4, pnt.x // 4), radius=2, color=1, thickness=-1)
                    if 'road_seg_small' in fetch_list:
                        start = edge.src(self.gc.graph).point
                        end = edge.dst(self.gc.graph).point
                        if search_rect.contains(start) or search_rect.contains(end):
                            start = start.sub(seg_origin)
                            end = end.sub(seg_origin)
                            cv.line(road_seg_small, (start.y // 4, start.x // 4),
                                    (end.y // 4, end.x // 4), color=1, thickness=1)
                if not safe_rect.contains(extension_vertex.point):
                    clip_rect = search_rect.clip_rect(seg_rect)
                    start, end = clip_rect.start.sub(
                        seg_origin), clip_rect.end.sub(seg_origin)
                    new_road_seg_small = np.zeros(
                        (WINDOW_SIZE // 4, WINDOW_SIZE // 4), dtype=np.float32)
                    new_road_seg_small[start.x // 4:end.x // 4, start.y // 4:end.y //
                                       4] = road_seg_small[start.x // 4:end.x // 4, start.y // 4:end.y // 4]
                    del road_seg_small
                    road_seg_small = new_road_seg_small

        aerial_image_hwc = big_img['input'][tile_origin.x:tile_origin.x + WINDOW_SIZE,
                                            tile_origin.y:tile_origin.y + WINDOW_SIZE, :].astype('float32') / 255.0
        aerial_image_chw = aerial_image_hwc.swapaxes(0, 2).swapaxes(1, 2)

        ret_dict = {
            'aerial_image_chw':  aerial_image_chw if 'aerial_image_chw' in fetch_list else None,
            'aerial_image_hwc':  aerial_image_hwc if 'aerial_image_hwc' in fetch_list else None,
            'walked_path_small': walked_path_small[np.newaxis, :, :] if 'walked_path_small' in fetch_list else None,
            'road_seg_small':    road_seg_small[np.newaxis, :, :] if 'road_seg_small' in fetch_list else None,
            'junc_seg_small':    junc_seg_small[np.newaxis, :, :] if 'junc_seg_small' in fetch_list else None,
        }
        return ret_dict

    def visualize_output(self, fname_prefix, extension_vertex, aerial_image, target_poses=None, pred_gt_pair_list=None,
                         WINDOW_SIZE=256):
        """
        :param
            aerial_image: PIL
            pred_gt_pair_list: list of tuples of predicted maps and ground truth maps
                [('anchor', anchor_output_map, anchor_target_map),  # anchor maps must be at the first index
                 ('road', road_segmentation_output_map, road_segmentation_target_map),
                 ('junc', junc_segmentation_output_map, junc_segmentation_output_map),
                 ('res_seg', residual_road_segment_segmentation_output_map, residual_road_segment_segmentation_target_map)]
            target_poses: TargetPosesContainer
        """

        ################ aerial_image ################
        if aerial_image is not None:
            aerial_image = aerial_image.swapaxes(0, 1)

        if self.gc is not None:
            if aerial_image is None:
                aerial_image = np.zeros((WINDOW_SIZE, WINDOW_SIZE, 3))

            aerial_image = np.ascontiguousarray(aerial_image)
            explored_edge = []
            origin = extension_vertex.point.sub(
                geom.Point(WINDOW_SIZE // 2, WINDOW_SIZE // 2))

            # draw road segments, green
            rect = origin.bounds().extend(origin.add(geom.Point(WINDOW_SIZE, WINDOW_SIZE)))
            for edge in self.gc.edge_index.search(rect):
                if edge in explored_edge:
                    continue
                explored_edge.append(edge)
                explored_edge.append(edge.get_opposite_edge(self.gc.graph))
                start = edge.src(self.gc.graph).point.sub(origin)
                end = edge.dst(self.gc.graph).point.sub(origin)
                cv.line(aerial_image, (start.x, start.y),
                        (end.x, end.y), color=(0., 1., 0.), thickness=3)

            # draw intersections
            for edge in self.gc.edge_index.search(rect):
                start = edge.src(self.gc.graph)
                end = edge.dst(self.gc.graph)
                for pnt in [start, end]:
                    # if len(pnt.in_edges) >= 3:
                    if True:  # TODO
                        pnt = pnt.point.sub(origin)
                        cv.circle(aerial_image, center=(pnt.x, pnt.y),
                                  radius=3, color=(1., 0., 1.), thickness=-1)

            # draw already walked path, red
            explored_edge = []
            for edge_id in self.edge_rtree.intersection((rect.start.x, rect.start.y, rect.end.x, rect.end.y)):
                edge = self.graph.edges[edge_id]
                if edge in explored_edge:
                    continue
                explored_edge.append(edge)
                explored_edge.append(edge.get_opposite_edge(self.graph))
                start = edge.src(self.graph).point.sub(origin)
                end = edge.dst(self.graph).point.sub(origin)
                cv.line(aerial_image, (start.x, start.y),
                        (end.x, end.y), color=(1., 0., 0.), thickness=1)

            # draw extension vertex point, half WINDOW_SIZE, blue
            cv.circle(aerial_image, center=(WINDOW_SIZE // 2, WINDOW_SIZE //
                                            2), radius=3, color=(0., 0., 1.), thickness=-1)

            # draw target position
            if target_poses is not None:
                if extension_vertex.edge_pos is not None:
                    pp = extension_vertex.edge_pos.point(
                        self.gc.graph).sub(origin)
                    # extension vertex's edge_pos, cyan, "青色"
                    cv.circle(aerial_image, (pp.x, pp.y), radius=2,
                              color=(0., 1., 1.), thickness=-1)
                for p in target_poses.get_single_lst():
                    pp = p.point(self.gc.graph).sub(origin)
                    # target points next step, white
                    cv.circle(aerial_image, (pp.x, pp.y), radius=2,
                              color=(1., 1., 1.), thickness=-1)

        if aerial_image is not None:
            Image.fromarray((aerial_image * 255.0).astype('uint8')
                            ).save(fname_prefix + 'ai.png')

        ################ anchor_output_map ################

        anchor_fname_suffix, anchor_output_map, anchor_target_map = pred_gt_pair_list.pop(
            0)
        anchor_output_map = np.sum(anchor_output_map, axis=0)
        if anchor_target_map is not None:
            anchor_target_map = np.sum(anchor_target_map, axis=0)
        pred_gt_pair_list.append(
            (anchor_fname_suffix, anchor_output_map, anchor_target_map))

        ################ pred_gt_pair_list ################

        for fname_suffix, output_map, target_map in pred_gt_pair_list:
            if target_map is not None:
                # print(output_map.shape, target_map.shape)
                res = np.stack(
                    [output_map, target_map, np.zeros(output_map.shape)], axis=-1)
                # print(res.shape)
                Image.fromarray((res * 255.0).swapaxes(0, 1).astype('uint8')
                                ).save(fname_prefix + fname_suffix + '.png')
            else:
                Image.fromarray((output_map * 255.0).swapaxes(0, 1).astype('uint8')).save(
                    fname_prefix + fname_suffix + '.png')

    def generate_target_maps(self, extension_vertex, target_poses, NUM_TARGETS=4, WINDOW_SIZE=224, is_key_point=False):
        """
        :param target_poses [[], [], [], []]
        :return: ndarray (NUM_TARGETS, WINDOW_SIZE, WINDOW_SIZE)
        """

        def generate_target(target_pnts, image_shape, target_shape):
            """
            :param joints:  [num_joints, 3]
            :return: target, target_weight(1: visible, 0: invisible)
            """
            sigma = 3
            num_joints = len(target_pnts)
            heatmap_size = np.array(target_shape)
            image_size = np.array(image_shape)

            target_weight = np.ones((num_joints, 1), dtype=np.float32)
            # target_weight[:, 0] = joints_vis[:, 0]

            target = np.zeros((num_joints,
                               heatmap_size[1],
                               heatmap_size[0]),
                              dtype=np.float32)

            tmp_size = sigma * 3

            for joint_id in range(num_joints):
                feat_stride = image_size / heatmap_size
                mu_x = int(target_pnts[joint_id].x / feat_stride[0] + 0.5)
                mu_y = int(target_pnts[joint_id].y / feat_stride[1] + 0.5)
                # Check that any part of the gaussian is in-bounds
                ul = [int(mu_x - tmp_size), int(mu_y - tmp_size)]
                br = [int(mu_x + tmp_size + 1), int(mu_y + tmp_size + 1)]
                if ul[0] >= heatmap_size[0] or ul[1] >= heatmap_size[1] \
                        or br[0] < 0 or br[1] < 0:
                    # If not, just return the image as is
                    target_weight[joint_id] = 0
                    continue

                # # Generate gaussian
                size = 2 * tmp_size + 1
                x = np.arange(0, size, 1, np.float32)
                y = x[:, np.newaxis]
                x0 = y0 = size // 2
                # The gaussian is not normalized, we want the center value to equal 1
                g = np.exp(- ((x - x0) ** 2 + (y - y0) ** 2) /
                           (2 * sigma ** 2))
                # print(g)

                # Usable gaussian range
                g_x = max(0, -ul[0]), min(br[0], heatmap_size[0]) - ul[0]
                g_y = max(0, -ul[1]), min(br[1], heatmap_size[1]) - ul[1]
                # Image range
                img_x = max(0, ul[0]), min(br[0], heatmap_size[0])
                img_y = max(0, ul[1]), min(br[1], heatmap_size[1])

                v = target_weight[joint_id]
                if v > 0.5:
                    target[joint_id][img_y[0]:img_y[1], img_x[0]:img_x[1]] = \
                        g[g_y[0]:g_y[1], g_x[0]:g_x[1]]

            return target, target_weight

        origin_point = extension_vertex.point.sub(
            geom.Point(WINDOW_SIZE // 2, WINDOW_SIZE // 2))
        target_maps = []
        for poses in target_poses:
            poses_len = len(poses)
            if poses_len == 1:
                lst = [pos.point(self.gc.graph).sub(origin_point)
                       for pos in poses]
                target, _ = generate_target(
                    lst, (WINDOW_SIZE, WINDOW_SIZE), (WINDOW_SIZE, WINDOW_SIZE))
                target_maps.append(target[0].swapaxes(0, 1))
            elif poses_len > 1:
                lst = [pos.point(self.gc.graph).sub(origin_point)
                       for pos in poses]
                target, _ = generate_target(
                    lst, (WINDOW_SIZE, WINDOW_SIZE), (WINDOW_SIZE, WINDOW_SIZE))
                target_map = np.sum(target, axis=0).swapaxes(0, 1)
                target_map[np.where(target_map > 1)] = 1
                target_maps.append(target_map)
            else:  # poses_len == 0:
                target_maps.append(np.zeros((WINDOW_SIZE, WINDOW_SIZE)))
        target_maps = np.stack(target_maps, axis=0)
        return target_maps


def random_sample_given_probs(seq, probs):
    sum_probs = sum(probs)
    if sum_probs != 1:
        probs = [x/sum_probs for x in probs]
    probs.insert(0, 0)
    for i in range(len(probs)-1):
        probs[i+1] = probs[i] + probs[i+1]
    rand = random.random()
    for i in range(len(probs)-1):
        if probs[i] < rand < probs[i+1]:
            break
    return seq[i]


def get_avg_between_pnts_in_map(im_map, pnt1, pnt2, WINDOW_SIZE=256):
    if im_map is None:
        return 0
    pnts = geom.draw_line(pnt1, pnt2, geom.Point(WINDOW_SIZE, WINDOW_SIZE))
    lst = [im_map[pnt.x, pnt.y] for pnt in pnts]
    return np.mean(lst) if len(lst) != 0 else 0


def get_points_from_rtree(point_rtree, index2point, center_point, RECT_RADIUS) -> dict:
    points = dict()
    start = geom.Point(center_point.x - RECT_RADIUS,
                       center_point.y - RECT_RADIUS)
    end = geom.Point(center_point.x + RECT_RADIUS,
                     center_point.y + RECT_RADIUS)
    for point_id in point_rtree.intersection((start.x, start.y, end.x, end.y)):
        points[index2point[point_id]] = point_id
    return points


def get_random_rect(big_rect, WINDOW_SIZE=256):
    x = random.randint(big_rect.start.x, big_rect.end.x-WINDOW_SIZE)
    y = random.randint(big_rect.start.y, big_rect.end.y-WINDOW_SIZE)
    return geom.Rectangle(geom.Point(x, y), geom.Point(x+WINDOW_SIZE, y+WINDOW_SIZE))


def get_random_rect_padding(big_rect, WINDOW_SIZE=256):
    x = random.randint(big_rect.start.x, big_rect.end.x-1)
    y = random.randint(big_rect.start.y, big_rect.end.y-1)
    sz = WINDOW_SIZE // 2
    return geom.Rectangle(geom.Point(x-sz, y-sz), geom.Point(x+sz, y+sz))


def get_nearest_end_point(key_points, point):
    if type(key_points) is not list:
        key_points = list(key_points)
    if len(key_points) == 1:
        return key_points[0]
    nearest_key_point = key_points[0]
    for pnt in key_points[1:]:
        if pnt.distance(point) < nearest_key_point.distance(point):
            nearest_key_point = pnt
    return nearest_key_point


class TargetPosesContainer:
    def __init__(self, NUM_TARGETS=4):
        self.target_poses = [[] for _ in range(NUM_TARGETS)]
        self.NUM_TARGETS = NUM_TARGETS

    def get_all_target_poses(self):
        res = []
        for poses in self.target_poses:
            res.extend(poses)
        return res

    def __getitem__(self, index):
        return self.target_poses[index]

    def __len__(self):
        for i, poses in enumerate(self.target_poses):
            if len(poses) == 0:
                return i
        return self.NUM_TARGETS  # == NUM_TARGETS

    def is_end_with_key_point(self):
        end_index = self.__len__()
        if end_index > 0 and len(self.target_poses[end_index - 1]) > 1:
            return True
        return False

    def get_single_lst(self):
        # to get target_poses without junction end
        res = []
        for poses in self.target_poses:
            if len(poses) == 0:
                break
            res.extend(poses)
        return res

    def get_single_lst_without_junction_end(self):
        # to get target_poses without junction end
        res = []
        for i, poses in enumerate(self.target_poses):
            if i == 0 and len(poses) > 1:
                return poses
            elif i > 0 and len(poses) > 1:
                return res
            else:
                res.extend(poses)
        return res

    def len_without_junction_end(self):
        for i, poses in enumerate(self.target_poses):
            if i != 0 and len(poses) > 1:
                return i
            if len(poses) == 0:
                return i
        return self.NUM_TARGETS

    def get_supervision_end_index(self):
        for i, poses in enumerate(self.target_poses):
            if len(poses) > 1:
                return i + 1
        return self.NUM_TARGETS

    def str(self, graph):
        string = "["
        for index, item in enumerate(self.target_poses):
            string += "["
            for i, x in enumerate(item):
                pnt = x.point(graph)
                string += "({},{})".format(pnt.x, pnt.y)
                if i != len(item)-1:
                    string += ", "
            string += "]"
            if index != len(self.target_poses)-1:
                string += ", "
        string += "]"
        return string


def map_to_coordinate(batch_output_maps, batch_is_key_point, batch_extension_vertices, ROAD_SEG_THRESHOLE=0.2,
                      STEP_LENGTH=20, JUNC_MAX_REGION_AREA=200):
    """
    :return:
        if is_key_point:
            res == [(x,y), ..., (x,y)]  # time_step == 1 # +
        else:
            res == [(x,y), ..., (x,y)]  # time_step  > 1 # ----
    """
    def _frame_to_coordinate(frame, origin_point, channel_index, previous_pnt=None):
        frame[np.where(frame < ROAD_SEG_THRESHOLE)] = 0
        frame[np.where(frame)] = 1
        labels = measure.label(frame, connectivity=2)
        props = measure.regionprops(labels)
        res = []
        for region in props:
            if region.area > JUNC_MAX_REGION_AREA:
                continue
            offset = geom.Point(
                int(region.centroid[0]), int(region.centroid[1]))
            distance = offset.distance(center_pnt)
            if distance > (channel_index + 2) * STEP_LENGTH:
                continue
            if previous_pnt is not None:
                distance = offset.distance(previous_pnt.sub(origin_point))
                if distance > 2 * STEP_LENGTH:
                    continue
            res.append(origin_point.add(offset))
        return res

    _, NUM_TARGETS, WINDOW_SIZE, _ = batch_output_maps.shape
    batch_size = len(batch_extension_vertices)
    batch_res = []
    center_pnt = geom.Point(WINDOW_SIZE // 2, WINDOW_SIZE // 2)
    for batch_idx in range(batch_size):
        origin_point = batch_extension_vertices[batch_idx].point.sub(
            geom.Point(WINDOW_SIZE // 2, WINDOW_SIZE // 2))
        previous_pnt = None
        res = []
        for i in range(NUM_TARGETS):
            frame = batch_output_maps[batch_idx, i, :, :]
            frame_coordinate = _frame_to_coordinate(
                frame, origin_point, i, previous_pnt)
            # judged by junction segmentation to be a keypoint
            if batch_is_key_point[batch_idx]:
                res.extend(frame_coordinate)
                break
            # batch_is_key_point[batch_idx]==False, judged by anchor to be a keypoint
            elif len(frame_coordinate) > 1:
                if i == 0:
                    batch_is_key_point[batch_idx] = True
                    res.extend(frame_coordinate)
                break
            elif len(frame_coordinate) == 0:
                break
            else:  # len(frame_coordinate) == 1:
                previous_pnt = frame_coordinate[0]
                res.extend(frame_coordinate)
        batch_res.append(res)
    return batch_res
