from . import geom

import math
import numpy
import rtree
import networkx as nx
from shapely.geometry import Point as GeometryPoint, LineString


class Vertex(object):
    def __init__(self, id: int, point: geom.Point):
        self.id = id
        self.point = point
        self.in_edges_id = []
        self.out_edges_id = []

    def in_edges(self, graph):
        return [graph.edges[x] for x in self.in_edges_id]

    def out_edges(self, graph):
        return [graph.edges[x] for x in self.out_edges_id]

    def _neighbors(self, graph):
        n = {}
        for edge_id in self.in_edges_id:
            edge = graph.edges[edge_id]
            n[edge.src_id] = edge_id
        for edge_id in self.out_edges_id:
            edge = graph.edges[edge_id]
            n[edge.dst_id] = edge_id
        return n

    def neighbors(self, graph):
        return self._neighbors(graph).keys()

    def __repr__(self):
        return "Vertex:{" + str(self.point) + ", in:" + str(self.in_edges_id) + ", out:" + str(self.out_edges_id) + "}"


class Edge(object):
    def __init__(self, id: int, src_id: int, dst_id: int):
        self.id = id
        assert type(src_id) is int and type(dst_id) is int
        self.src_id = src_id
        self.dst_id = dst_id

    def src(self, graph):
        return graph.vertices[self.src_id]

    def dst(self, graph):
        return graph.vertices[self.dst_id]

    def bounds(self, graph):
        return self.src(graph).point.bounds().extend(self.dst(graph).point)

    def segment(self, graph):
        return geom.Segment(self.src(graph).point, self.dst(graph).point)

    def closest_pos(self, point, graph):
        p = self.segment(graph).project(point)
        return EdgePos(self.id, p.distance(self.src(graph).point))

    def is_opposite(self, edge):
        return edge.src_id == self.dst_id and edge.dst_id == self.src_id

    def get_opposite_edge(self, graph):
        for edge in self.dst(graph).out_edges(graph):
            if self.is_opposite(edge):
                return edge
        return None

    def is_adjacent(self, edge):
        return edge.src_id == self.src_id or edge.src_id == self.dst_id \
               or edge.dst_id == self.src_id or edge.dst_id == self.dst_id

    def __repr__(self):
        return "Edge:(" + str(self.src_id) + ", " + str(self.dst_id) + ")"


class EdgePos(object):
    def __init__(self, edge_id: int, distance: float):
        self.edge_id = edge_id
        self.distance = distance

    def edge(self, graph):
        return graph.edges[self.edge_id]

    def point(self, graph):
        segment = self.edge(graph).segment(graph)
        vector = segment.vector()  # point
        if vector.magnitude() < 1:
            return segment.start
        else:
            return segment.start.add(vector.scale(self.distance / vector.magnitude()))

    def reverse(self, graph):
        return EdgePos(self.edge(graph).get_opposite_edge(graph).id,
                       self.edge(graph).segment(graph).length() - self.distance)


class Index(object):
    def __init__(self, graph, index):
        self.graph = graph
        self.index = index

    def search(self, rect):
        edge_ids = self.index.intersection((rect.start.x, rect.start.y, rect.end.x, rect.end.y))
        return [self.graph.edges[edge_id] for edge_id in edge_ids]

    def subgraph(self, rect):
        ng = Graph()
        vertex_map = {}
        for edge in self.search(rect):
            src = edge.src(self.graph)
            dst = edge.dst(self.graph)
            if src.point == dst.point:
                continue
            rect_contains_src_point = rect.contains(src.point)
            rect_contains_dst_point = rect.contains(dst.point)
            if not rect_contains_src_point and not rect_contains_dst_point:
                continue
            elif not rect_contains_src_point or not rect_contains_dst_point:
                if rect_contains_dst_point:
                    inner_point, outer_point = dst.point, src.point
                else:
                    inner_point, outer_point = src.point, dst.point
                if rect.at_bounds(inner_point):
                    continue
                while True:
                    middle_point = inner_point.add(outer_point).scale(0.5)
                    if rect.at_bounds(middle_point):
                        break
                    if rect.contains(middle_point):
                        inner_point = middle_point
                    else:
                        outer_point = middle_point
                    # if middle_point == inner_point.add(outer_point).scale(0.5):
                    #     middle_point = inner_point
                    #     break
                if rect_contains_dst_point:
                    src_point, dst_point = middle_point, dst.point
                else:
                    src_point, dst_point = src.point, middle_point
                for point in [src_point, dst_point]:
                    if point not in vertex_map:
                        vertex_map[point] = ng.add_vertex(point)
                ng.add_edge(vertex_map[src_point].id, vertex_map[dst_point].id)
            else:
                for vertex in [src, dst]:
                    if vertex.point not in vertex_map:
                        vertex_map[vertex.point] = ng.add_vertex(vertex.point)
                ng.add_edge(vertex_map[src.point].id, vertex_map[dst.point].id)
        return ng


class Graph(object):

    def __init__(self):
        self.vertices = {}
        self.vertices_index = 0
        self.edges = {}
        self.edges_index = 0

    def add_vertex(self, point, vertex_id=None):
        if vertex_id is None:
            vertex = Vertex(self.vertices_index, point)
            self.vertices[self.vertices_index] = vertex
            self.vertices_index += 1
        else:
            vertex = Vertex(vertex_id, point)
            self.vertices[vertex_id] = vertex
            if vertex_id >= self.vertices_index:
                self.vertices_index = vertex_id + 1
        return vertex

    def add_edge(self, src_id: int, dst_id: int, edge_id=None) -> Edge:
        if src_id == dst_id:
            raise Exception('cannot add edge between same vertex')
        if edge_id is None:
            edge = Edge(self.edges_index, src_id, dst_id)
            self.edges[self.edges_index] = edge
            edge.src(self).out_edges_id.append(edge.id)
            edge.dst(self).in_edges_id.append(edge.id)
            self.edges_index += 1
        else:
            edge = Edge(edge_id, src_id, dst_id)
            self.edges[edge_id] = edge
            edge.src(self).out_edges_id.append(edge.id)
            edge.dst(self).in_edges_id.append(edge.id)
            if edge_id >= self.edges_index:
                self.edges_index = edge_id + 1
        return edge

    def add_bidirectional_edge(self, src: int, dst: int):
        return (
            self.add_edge(src, dst),
            self.add_edge(dst, src),
        )

    def edgeIndex(self, file_name=None):
        rt = rtree.index.Index(file_name)
        for edge in self.edges.values():
            bounds = edge.bounds(self)
            rt.insert(edge.id, (bounds.start.x, bounds.start.y, bounds.end.x, bounds.end.y))
        return Index(self, rt)

    def bounds(self):
        r = None
        for vertex in self.vertices.values():
            if r is None:
                r = geom.Rectangle(vertex.point, vertex.point)
            else:
                r = r.extend(vertex.point)
        return r

    def save(self, file_name, with_index=False, clear_self=True):
        with open(file_name, 'w') as f:
            if with_index:
                for vertex in self.vertices.values():
                    f.write("{} {} {}\n".format(vertex.id, vertex.point.x, vertex.point.y))
                f.write("\n")
                for edge in self.edges.values():
                    f.write("{} {} {}\n".format(edge.id, edge.src_id, edge.dst_id))
            else:
                if clear_self:
                    ng = self.clear_self()
                else:
                    ng = self
                for vertex in ng.vertices.values():
                    f.write("{} {}\n".format(vertex.point.x, vertex.point.y))
                f.write("\n")
                for edge in ng.edges.values():
                    f.write("{} {}\n".format(edge.src_id, edge.dst_id))

    def convert_to_networkx(self):
        graph = nx.Graph()
        for vertex in self.vertices.values():
            graph.add_node(vertex.id, x_pix=vertex.point.x, y_pix=vertex.point.y)
        for edge in self.edges.values():
            graph.add_edge(edge.src_id, edge.dst_id)
        return graph

    def clear_self(self):
        ng = Graph()
        vertex_map = {}
        edge_lst = []
        for edge in self.edges.values():
            src = (self.vertices[edge.src_id].point.x, self.vertices[edge.src_id].point.y)
            dst = (self.vertices[edge.dst_id].point.x, self.vertices[edge.dst_id].point.y)
            if src == dst:
                # print("self link")
                continue
            edge_repr = (src, dst)
            if edge_repr in edge_lst:
                # print("redundant edge {} {}".format(src, dst))
                continue
            for pnt in [src, dst]:
                if pnt not in vertex_map:
                    vertex_map[pnt] = ng.add_vertex(geom.Point(pnt[0], pnt[1]))
            edge_lst.append((src, dst))
            ng.add_edge(vertex_map[src].id, vertex_map[dst].id)
        return ng

    def convert_rs_to_wkt(self):
        linestring = "LINESTRING ({})"
        wkt = []
        road_segments, edge_id_to_rs_id = get_graph_road_segments(self)
        seen_rs_linestring = []
        seen_rs_id = []
        seen_edge_id_lst = []
        for rs in road_segments:
            if len(rs.edges_id) == 0:
                continue

            oppo_rs = rs.get_opposite_rs(edge_id_to_rs_id, road_segments, self)
            if oppo_rs is not None:
                if oppo_rs.id in seen_rs_id:
                    continue
                oppo_lst = []
                for edge in oppo_rs.edges(self):
                    src = edge.src(self)
                    dst = edge.dst(self)
                    if src.point == dst.point:
                        continue
                    oppo_lst.append("{} {}".format(src.point.x, src.point.y))
                oppo_lst.append("{} {}".format(dst.point.x, dst.point.y))
                line = linestring.format(', '.join(oppo_lst))
                if line in seen_rs_linestring:
                    continue
                else:
                    seen_rs_linestring.append(line)

            lst = []
            if rs.edges_id[0] in seen_edge_id_lst:  # loop several times
                continue
            for edge in rs.edges(self):
                src = edge.src(self)
                dst = edge.dst(self)
                if src.point == dst.point:
                    continue
                lst.append("{} {}".format(src.point.x, src.point.y))
            lst.append("{} {}".format(dst.point.x, dst.point.y))
            line = linestring.format(', '.join(lst))
            if line in seen_rs_linestring:
                continue
            else:
                seen_rs_linestring.append(line)
                wkt.append(line)
                seen_edge_id_lst.extend(rs.edges_id)
                seen_edge_id_lst.extend([x.get_opposite_edge(self).id for x in rs.edges(self)])
        return wkt

    def clone(self):
        ng = Graph()
        for vertex in self.vertices.values():
            v = ng.add_vertex(vertex.point)
            if hasattr(vertex, 'edge_pos'):
                v.edge_pos = vertex.edge_pos
        for edge in self.edges.values():
            e = ng.add_edge(edge.src.id, edge.dst.id)
        return ng

    def filter_edges(self, filter_edges):
        graph = Graph()
        vertex_map = {}
        for edge in self.edges.values():
            if edge in filter_edges:
                continue
            for vertex_id in [edge.src_id, edge.dst_id]:
                if vertex_id not in vertex_map:
                    vertex_map[vertex_id] = graph.add_vertex(graph.vertices[vertex_id].point)
            graph.add_edge(edge.src_id, edge.dst_id)
        return graph

    def follow_graph(self, edge_pos, distance, explored_node_pairs=None):
        if explored_node_pairs:
            explored_node_pairs = set(explored_node_pairs)
        else:
            explored_node_pairs = set()
        explored_node_pairs.add((edge_pos.edge(self).src_id, edge_pos.edge(self).dst_id))
        positions = []

        def search_edge(edge, remaining):
            l = edge.segment(self).length()
            if remaining > l:
                search_vertex(edge.dst(self), remaining - l)
            else:
                pos = EdgePos(edge.id, remaining)
                positions.append(pos)

        def search_vertex(vertex, remaining):
            for edge in vertex.out_edges(self):
                if (edge.src_id, edge.dst_id) in explored_node_pairs or \
                        (edge.dst_id, edge.src_id) in explored_node_pairs:
                    continue
                explored_node_pairs.add((edge.src_id, edge.dst_id))
                search_edge(edge, remaining)

        remaining = distance
        l = edge_pos.edge(self).segment(self).length() - edge_pos.distance
        if remaining > l:
            search_vertex(edge_pos.edge(self).dst(self), remaining - l)
        else:
            positions = [EdgePos(edge_pos.edge_id, edge_pos.distance + remaining)]

        return positions


def read_graph(file_name, merge_duplicates=True):
    graph = Graph()
    with open(file_name, 'r') as f:
        vertex_section = True
        vertices = {}
        next_vertex_id = 0
        seen_points = {}
        for line in f:
            parts = line.strip().split(' ')
            if vertex_section:
                if len(parts) == 2:
                    point = geom.Point(float(parts[0]), float(parts[1]))
                    if point in seen_points and merge_duplicates:
                        #print('merging duplicate vertex at {}'.format(point))
                        vertices[next_vertex_id] = seen_points[point]
                    else:
                        vertex = graph.add_vertex(point)
                        vertices[next_vertex_id] = vertex
                        seen_points[point] = vertex
                    next_vertex_id += 1
                elif len(parts) == 3:
                    point = geom.Point(float(parts[1]), float(parts[2]))
                    if point in seen_points and merge_duplicates:
                        #print('merging duplicate vertex at {}'.format(point))
                        vertices[next_vertex_id] = seen_points[point]
                    else:
                        vertex = graph.add_vertex(point, int(parts[0]))
                        vertices[next_vertex_id] = vertex
                        seen_points[point] = vertex
                    next_vertex_id += 1
                else:
                    vertex_section = False
            elif len(parts) == 2:
                src = int(parts[0])
                dst = int(parts[1])
                if vertices[src].point == vertices[dst].point and merge_duplicates:
                    #print('ignoring self edge at {}'.format(vertices[src].point))
                    continue
                graph.add_edge(vertices[src].id, vertices[dst].id)
            elif len(parts) == 3:
                src = int(parts[1])
                dst = int(parts[2])
                if vertices[src].point == vertices[dst].point and merge_duplicates:
                    #print('ignoring self edge at {}'.format(vertices[src].point))
                    continue
                graph.add_edge(vertices[src].id, vertices[dst].id, int(parts[0]))
    for vertex in graph.vertices.values():
        if len(vertex.in_edges_id) != len(vertex.out_edges_id):
            print(vertex.in_edges_id, vertex.out_edges_id, vertex.point)
    return graph


class RoadSegment(object):
    def __init__(self, id):
        self.id = id
        self.edges_id = []
        self.marked_length = None
        self.is_loop = False

    def edges(self, graph):
        return [graph.edges[x] for x in self.edges_id]

    def add_edge(self, edge_id: int, direction: str):
        if direction == 'forwards':
            self.edges_id.append(edge_id)
        elif direction == 'backwards':
            self.edges_id = [edge_id] + self.edges_id
        else:
            raise Exception('bad edge')

    def compute_edge_distances(self, graph):
        if hasattr(self, 'edge_distances') and len(self.edge_distances) != 0:
            return
        l = 0
        self.edge_distances = {}
        for edge_id in self.edges_id:
            self.edge_distances[edge_id] = l
            l += graph.edges[edge_id].segment(graph).length()
        self.marked_length = l

    def distance_to_edge(self, distance, graph, return_idx=False):
        for i in range(len(self.edges_id)):
            edge_id = self.edges_id[i]
            edge = graph.edges[edge_id]
            distance -= edge.segment(graph).length()
            if distance <= 0:
                if return_idx:
                    return i
                else:
                    return edge
        if return_idx:
            return len(self.edges_id) - 1
        else:
            return graph.edges[self.edges_id[-1]]

    def src(self, graph):
        return graph.vertices[graph.edges[self.edges_id[0]].src_id]

    def dst(self, graph):
        return graph.vertices[graph.edges[self.edges_id[-1]].dst_id]

    def src_id(self, graph):
        return graph.edges[self.edges_id[0]].src_id

    def dst_id(self, graph):
        return graph.edges[self.edges_id[-1]].dst_id

    def is_opposite(self, rs, graph):
        return self.src_id(graph) == rs.dst_id(graph) and self.dst_id(graph) == rs.src_id(graph) \
               and graph.edges[self.edges_id[0]].is_opposite(graph.edges[rs.edges_id[-1]])

    def in_rs(self, edge_id_to_rs_id, road_segments, graph):
        rs_set = {}
        for edge_id in self.src(graph).in_edges_id:
            rs_id = edge_id_to_rs_id[edge_id]
            if rs_id != self.id and rs_id not in rs_set:
                rs_set[rs_id] = road_segments[rs_id]
        return rs_set.values()

    def out_rs(self, edge_id_to_rs_id, road_segments, graph):
        rs_set = {}
        for edge_id in self.dst(graph).out_edges_id:
            rs_id = edge_id_to_rs_id[edge_id]
            if rs_id != self.id and rs_id not in rs_set:
                rs_set[rs_id] = road_segments[rs_id]
        return rs_set.values()

    def get_opposite_rs(self, edge_id_to_rs_id, road_segments, graph):
        for rs in self.out_rs(edge_id_to_rs_id, road_segments, graph):
            if self.is_opposite(rs, graph):
                return rs
        return None

    def length(self, graph):
        return sum([graph.edges[edge_id].segment(graph).length() for edge_id in self.edges_id])

    # def length(self):
    #     return self.marked_length

    def closest_pos(self, point, graph):
        best_edge_pos = None
        best_distance = None
        for edge in self.edges(graph):
            edge_pos = edge.closest_pos(point, graph)
            distance = edge_pos.point(graph).distance(point)
            if best_edge_pos is None or distance < best_distance:
                best_edge_pos = edge_pos
                best_distance = distance
        return best_edge_pos

    def point_at_factor(self, t, graph):
        edge = self.distance_to_edge(t, graph)
        return edge.segment(graph).point_at_factor(t - self.edge_distances[edge.id])

    def get_unexplored_rs(self, rs_exp, graph):

        def get_forward_edge(vertex, explored_edge):
            # assert len(vertex.out_edges) == 2
            for edge_id in vertex.out_edges_id:
                if edge_id not in explored_edge:
                    return graph.edges[edge_id]

        edge = rs_exp.explored_start_pos.edge(graph)
        assert edge.id in self.edges_id
        lst = [rs_exp.explored_start_pos.point(graph)]
        explored_edge = {edge.id, edge.get_opposite_edge(graph).id}
        next_vertex = edge.dst(graph)
        while True:
            if len(next_vertex.out_edges_id) != 2:
                break
            edge = get_forward_edge(next_vertex, explored_edge)
            if self.edge_distances[edge.id] >= rs_exp.explored_end_dis:
                break
            lst.append(edge.src(graph).point)
            explored_edge.add(edge.id)
            explored_edge.add(edge.get_opposite_edge(graph).id)
            next_vertex = edge.dst(graph)
        lst.append(rs_exp.explored_end_pos.point(graph))
        return lst

    def clone(self):
        new_rs = RoadSegment(self.id)
        new_rs.edges_id = self.edges_id.copy()
        new_rs.marked_length = self.marked_length
        if hasattr(self, 'edge_distances') and len(self.edge_distances) != 0:
            new_rs.edge_distances = self.edge_distances.copy()
        return new_rs

    def __repr(self):
        return "RS:" + str(self.edges_id)


class RoadSegmentExploration(object):
    def __init__(self, id, explored_end_dis, explored_start_pos, explored_end_pos, marked_length):
        self.id = id
        self.explored = False
        self.explored_start_dis = 0
        self.explored_end_dis = explored_end_dis
        self.explored_start_pos = explored_start_pos
        self.explored_end_pos = explored_end_pos
        self.marked_length = marked_length

    def is_explored(self):
        return self.explored or self.explored_start_dis == self.explored_end_dis

    def get_unexplored_dis(self):
        return 0 if self.is_explored() else self.explored_end_dis - self.explored_start_dis


def new_road_segment_exploration(rs, graph):
    return RoadSegmentExploration(
        id=rs.id,
        explored_end_dis=rs.marked_length,
        explored_start_pos=EdgePos(rs.edges_id[0], distance=0),
        explored_end_pos=EdgePos(rs.edges_id[-1], distance=graph.edges[rs.edges_id[-1]].segment(graph).length()),
        marked_length=rs.marked_length
    )


class RoadSegmentExplorationDict(object):
    def __init__(self, gc, rect):
        self.data = dict()
        self.road_segments = gc.road_segments
        self.graph = gc.graph
        for rs in self.road_segments:
            if rect.contains(rs.src(gc.graph).point) or rect.contains(rs.dst(gc.graph).point):
                self.data[rs.id] = new_road_segment_exploration(rs, gc.graph)

    def __getitem__(self, key):
        if key not in self.data:
            self.data[key] = new_road_segment_exploration(self.road_segments[key], self.graph)
        return self.data[key]


def get_graph_road_segments(graph):
    road_segments = []
    edge_id_to_rs_id = {}

    def search_from_edge(rs, edge, direction):
        cur_edge = edge
        while True:
            if direction == 'forwards':
                vertex = cur_edge.dst(graph)
                edges = vertex.out_edges(graph)
            elif direction == 'backwards':
                vertex = cur_edge.src(graph)
                edges = vertex.in_edges(graph)

            edges = [next_edge for next_edge in edges if not next_edge.is_opposite(cur_edge)]

            if len(edges) != 1:
                # we have hit intersection vertex or a dead end
                return 0

            next_edge = edges[0]

            if next_edge.id in edge_id_to_rs_id:
                # this should only happen when we run in a segment that is actually a loop
                # although really it shouldn't happen in that case either, since loops should start/end at an intersection
                # TODO: think about this more
                return -1

            rs.add_edge(next_edge.id, direction)
            edge_id_to_rs_id[next_edge.id] = rs.id
            cur_edge = next_edge

    for edge in graph.edges.values():
        if edge.id in edge_id_to_rs_id:
            continue

        rs = RoadSegment(len(road_segments))
        rs.add_edge(edge.id, 'forwards')
        edge_id_to_rs_id[edge.id] = rs.id
        res_code = search_from_edge(rs, edge, 'forwards')
        if res_code == 0:
            search_from_edge(rs, edge, 'backwards')
            rs.compute_edge_distances(graph)
            road_segments.append(rs)
        else:
            rs.compute_edge_distances(graph)
            rs.is_loop = True
            road_segments.append(rs)
            oppo_rs = RoadSegment(len(road_segments)+1)
            oppo_rs.edges_id = [graph.edges[edges_id].get_opposite_edge(graph).id for edges_id in rs.edges_id][::-1]
            oppo_rs.compute_edge_distances(graph)
            oppo_rs.is_loop = True
            road_segments.append(oppo_rs)

    return road_segments, edge_id_to_rs_id


class GraphContainer(object):
    def __init__(self, graph, edge_index=None, road_segments=None, edge_id_to_rs_id=None, file_name=None):
        self.graph = graph
        if edge_index is None:
            self.edge_index = graph.edgeIndex(file_name)
        else:
            self.edge_index = edge_index
        if road_segments is None and edge_id_to_rs_id is None:
            self.road_segments, self.edge_id_to_rs_id = get_graph_road_segments(graph)
        else:
            self.road_segments = road_segments
            self.edge_id_to_rs_id = edge_id_to_rs_id
            
    def edge_id_to_rs(self, edge_id):
        return self.road_segments[self.edge_id_to_rs_id[edge_id]]

    def reset(self):
        for rs in self.road_segments:
            rs.reset()
        return self

    def clone(self):
        new_rss = list()
        for rs in self.road_segments:
            new_rss.append(rs.clone())
        new_edge_to_rs = dict()
        for edge_id, rs_id in self.edge_id_to_rs_id.items():
            new_edge_to_rs[edge_id] = rs_id
        new_gc = GraphContainer(graph=self.graph, edge_index=self.edge_index, road_segments=new_rss,
                                edge_id_to_rs_id=new_edge_to_rs)
        return new_gc


def get_nearby_vertices(vertex, n, graph):
    nearby_vertices = set()

    def search(vertex, remaining):
        if vertex in nearby_vertices:
            return
        nearby_vertices.add(vertex)
        if remaining == 0:
            return
        for edge in vertex.in_edges(graph):
            search(edge.src(graph), remaining - 1)
            search(edge.dst(graph), remaining - 1)

    search(vertex, n)
    return nearby_vertices


def get_nearby_edge_segments(vertex, n, graph):
    nearby_edges = set()
    nearby_vertices = get_nearby_vertices(vertex, n, graph)
    for vertex in nearby_vertices:
        for edge in vertex.in_edges(graph):
            flag = True
            for e in nearby_edges:
                if edge == e or edge.is_opposite(e):
                    flag = False
                    break
            if flag:
                nearby_edges.add(edge)
    nearby_edge_segments = [edge.segment(graph) for edge in nearby_edges]
    return nearby_edge_segments
