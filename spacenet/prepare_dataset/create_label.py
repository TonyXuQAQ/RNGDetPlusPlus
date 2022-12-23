import json 
import os
from PIL import Image, ImageDraw
import numpy as np 
import json 
from tqdm import tqdm
import shutil
import pickle

IMAGE_SIZE = 400
INTER_P_RADIUS = 3
SEGMENT_WIDTH = 3

def create_directory(dir,delete=False):
    if os.path.isdir(dir) and delete:
        shutil.rmtree(dir)
    os.makedirs(dir,exist_ok=True)

create_directory('../data/vis',delete=True)
create_directory('../data/graph',delete=True)
create_directory('../data/intersection',delete=True)
create_directory('../data/segment',delete=True)

class Graph():
    def __init__(self):
        self.vertices = {}
        self.edges = []
        self.edge_counter = 0
        self.v_id = 0
        self.nidmap = None
    
    def add_v(self,v_list):
        if f'{v_list[0]}_{v_list[1]}' in self.vertices.keys():
            return self.vertices[f'{v_list[0]}_{v_list[1]}']
        else:
            self.vertices[f'{v_list[0]}_{v_list[1]}'] = Vertex(v_list[0],v_list[1],self.v_id)
            self.v_id += 1
            return self.vertices[f'{v_list[0]}_{v_list[1]}']
    
    def add_e(self,v_list1,v_list2,vertices=None,orientation=None,circle=False):
        v_list1 = [int(x) for x in v_list1]
        v_list2 = [int(x) for x in v_list2]
        v1 = self.add_v(v_list1)
        v2 = self.add_v(v_list2)
        if v1 not in v2.neighbor_vertices:
            v1.add_neighbor(v2)
            v2.add_neighbor(v1)
            new_edge = Edge(v1,v2,self.edge_counter,vertices=vertices,orientation=orientation,circle=circle)
            self.edge_counter += 1
            v1.add_neighbor(new_edge)
            v2.add_neighbor(new_edge)
            self.edges.append(new_edge)
            return v1, v2, new_edge
        return v1, v2, None
    

    def merge(self,edge_list):
        e1 = edge_list[0]
        e2 = edge_list[1]
        if e1.id == e2.id:
            return
        if (e1.dst == e2.src and e1.src == e2.dst) or (e1.dst == e2.dst and e1.src == e2.src):
            # circle
            circle = True
            if len(e1.src.neighbor_edges)==2 and len(e1.dst.neighbor_edges)==2:
                if e1.src == e2.src:
                    e1.reverse()
            else:
                if len(e1.src.neighbor_edges)==2:
                    e1.reverse()
                if len(e2.dst.neighbor_edges)==2:
                    e2.reverse()
            new_e = Edge(e1.src,e2.dst,self.edge_counter,vertices=(e1.vertices[:-1]+e2.vertices),circle=circle)
            self.edge_counter += 1
            e1.src.remove_neighbor(e1)
            e1.src.add_neighbor(new_e)
            e1.dst.removed = True
            e2.dst.remove_neighbor(e2)
            self.edges.remove(e1)
            self.edges.remove(e2)
            self.edges.append(new_e)
        else:
            # no circle
            if e1.dst == e2.src:
                pass
            elif e1.src == e2.src:
                e1.reverse()
            elif e1.dst == e2.dst:
                e2.reverse()
            elif e1.src == e2.dst:
                e1, e2 = e2, e1
            else:
                raise Exception('Error edge vertices...')
            circle = False
            # _,_,new_e = self.add_e([e1.src.x,e1.src.y],[e2.dst.x,e2.dst.y],vertices=(e1.vertices[:-1]+e2.vertices),circle=circle)
            new_e = Edge(e1.src,e2.dst,self.edge_counter,vertices=(e1.vertices[:-1]+e2.vertices),circle=circle)
            self.edge_counter += 1
            e1.src.remove_neighbor(e1)
            e1.src.add_neighbor(new_e)
            e1.dst.removed = True
            e2.dst.remove_neighbor(e2)
            e2.dst.add_neighbor(new_e)
            self.edges.remove(e1)
            self.edges.remove(e2)
            self.edges.append(new_e)

class Vertex():
    def __init__(self,x,y,id):
        self.x = x
        self.y = y
        self.neighbor_vertices = []
        self.neighbor_edges = []
        self.id = id
        self.removed = False
    
    def add_neighbor(self,X):
        if isinstance(X,Vertex):
            if X not in self.neighbor_vertices:
                self.neighbor_vertices.append(X)
        elif isinstance(X,Edge):
            if X not in self.neighbor_edges:
                self.neighbor_edges.append(X)
    
    def remove_neighbor(self,X):
        if isinstance(X,Vertex):
            if X in self.neighbor_vertices:
                self.neighbor_vertices.remove(X)
        elif isinstance(X,Edge):
            if X in self.neighbor_edges:
                self.neighbor_edges.remove(X)

class Edge():
    def __init__(self,src,dst,id,vertices=None,orientation=None,circle=False):
        self.id = id
        self.src = src
        self.dst = dst
        self.circle = circle
        if not vertices:
            self.vertices = [[src.x,src.y],[dst.x,dst.y]]
        else:
            self.vertices = vertices
        self.orientation = orientation
    
    def reverse(self):
        self.src, self.dst = self.dst, self.src
        self.vertices = self.vertices[::-1]


def get_orientation_angle(vector):
    norm = np.linalg.norm(vector)
    theta = 0
    if norm:
        vector = vector / norm
        theta = np.arccos(vector[0])
        if vector[1] > 0:
            theta = 2*np.pi - theta
        theta = (theta//(np.pi/32))%64 + 1
    return int(theta)     


def get_dense_edge(src,dst):
    start_vertex = np.array([int(src[0]),int(src[1])])
    end_vertex = np.array([int(dst[0]),int(dst[1])])
    p = start_vertex
    d = end_vertex - start_vertex
    N = np.max(np.abs(d))
    output_list = [[int(x) for x in start_vertex.tolist()]]
    if N:
        s = d / (N)
        for i in range(0,N):
            p = p + s
            p_list = [int(x+0.5) for x in p]
            if p_list not in output_list:
                output_list.append(p_list)
    return output_list

def whether_inside(v):
    if not isinstance(v, list):
        v = [v.x,v.y]
    if v[0]<IMAGE_SIZE  and v[0]>=0 and v[1]<IMAGE_SIZE  and v[1]>=0:
        return True
    return False

def interpolation(v_out,v_in):
    if (v_out[0])*(v_in[0]) < 0: 
        v_out[0] = 0
        v_out[1] = v_out[1]-abs(v_out[0])*abs(v_in[1]-v_out[1])/abs(v_out[0]-v_in[0])
    elif (v_out[0]-IMAGE_SIZE )*(v_in[0]-IMAGE_SIZE ) < 0:
        v_out[0] = IMAGE_SIZE - 1 
        v_out[1] = v_out[1]-abs(v_out[0]-IMAGE_SIZE )*abs(v_in[1]-v_out[1])/abs(v_out[0]-v_in[0])
    elif (v_out[1])*(v_in[1]) < 0:
        v_out[1] = 0
        v_out[0] = v_out[0]-abs(v_out[1])*abs(v_in[0]-v_out[0])/abs(v_out[1]-v_in[1])
    elif (v_out[1]-IMAGE_SIZE )*(v_in[1]-IMAGE_SIZE ) < 0:
        v_out[1] = IMAGE_SIZE - 1 
        v_out[0] = v_out[0]-abs(v_out[1]-IMAGE_SIZE )*abs(v_in[0]-v_out[0])/abs(v_out[1]-v_in[1])
    return [int(x) for x in v_out]

with open('../data/data_split.json','r') as jf:
    data_list = json.load(jf)
    data_list = data_list['test'] + data_list['validation'] + data_list['train']
# data_list = ['AOI_2_Vegas_1104']
for data_index, tile_index in enumerate(data_list):
    vertices = []
    edges = []
    vertex_flag = True

    # =============================================================================================
    # Data load part
    # =============================================================================================
    
    gt_graph = pickle.load(open(f"../data/RGB_1.0_meter/{tile_index}__gt_graph.p",'rb'))
    graph = Graph()
    for n, v in gt_graph.items():
        for nei in v:
            graph.add_e([int(n[1]), IMAGE_SIZE-int(n[0])],[int(nei[1]), IMAGE_SIZE-int(nei[0])])

    # =============================================================================================
    # Processing graph label
    # =============================================================================================
    # remove segments without intersections (usually very short roads)
    temp_edges = []
    for e in graph.edges:
        src = e.src
        dst = e.dst
        # # remove too short road segments without intersections
        # if (len(src.neighbor_edges)<=1 and len(dst.neighbor_edges)<=1) or \
        #         (not whether_inside(src) and not whether_inside(dst)):
        #     src.remove_neighbor(e)
        #     dst.remove_neighbor(e)
        # remove road segments that have two same vertices
        if len(e.vertices)<=2 and src.id==dst.id:
            src.remove_neighbor(e)
        else:
            temp_edges.append(e)
    graph.edges = temp_edges    

    # merge incident edges whose degree is 2
    for k,v in graph.vertices.items():
        if len(v.neighbor_edges)==2 and not v.removed and not len([e for e in v.neighbor_edges if e.circle]): 
            graph.merge(v.neighbor_edges)
    
    temp_vertices = {}
    for k,v in graph.vertices.items():
        if not v.removed:
            temp_vertices[k] = v
    graph.vertices = temp_vertices
            
    # densify edge pixels
    output_edges = []
    edge_names = []
    output_vertices = []
    # clean vertices, re-process
    graph_edges_copy = graph.edges.copy()
    for e_idx, e in enumerate(graph_edges_copy):
        new_vertices = []
        orientation = []
        for i in range(len(e.vertices)-1):
            dense_segment = get_dense_edge(e.vertices[i],e.vertices[i+1])
            new_vertices.extend(dense_segment)
            orientation_segment = get_orientation_angle(np.array(e.vertices[i+1])-np.array(e.vertices[i]))
            orientation.extend([orientation_segment for _ in range(len(dense_segment))])
        
        # add circle edges
        if e.circle is True:
            output_edges.append({'id':e.id,'src':e.src.id,'dst':e.dst.id,'circle':True,'vertices':new_vertices,'orientation':orientation})
            continue    
            
        # remove pixels outside the image
        vertex_state_list = [whether_inside(v) for v in new_vertices]
        locate_indexs = [i for i, x in enumerate(vertex_state_list[1:]) if vertex_state_list[i]!=x]
        
        # When the road instance cross the image edge for multiple times (larger than 2),
        # we only keep the part of the road instance that is connected with either e.src or e.dst 
        if len(locate_indexs) >=2:
            if whether_inside([e.src.x,e.src.y]) and len(new_vertices[:locate_indexs[0]+1])>5:
                src, dst, new_edge = graph.add_e([e.src.x,e.src.y],new_vertices[locate_indexs[0]],new_vertices[:locate_indexs[0]+1])
                e.src.remove_neighbor(e)
                output_edges.append({'id':new_edge.id,'src':new_edge.src.id,'dst':new_edge.dst.id,'circle':False,'vertices':new_edge.vertices,'orientation':orientation[:locate_indexs[0]+1]})
                
            if whether_inside([e.dst.x,e.dst.y]) and len(new_vertices[locate_indexs[-1]+1:])>5:
                src, dst, new_edge = graph.add_e(new_vertices[locate_indexs[-1]+1],[e.dst.x,e.dst.y],new_vertices[locate_indexs[-1]+1:])
                e.dst.remove_neighbor(e)
                output_edges.append({'id':new_edge.id,'src':new_edge.src.id,'dst':new_edge.dst.id,'circle':False,'vertices':new_edge.vertices,'orientation':orientation[locate_indexs[-1]+1:]})
                
        # When the road instance cross the image edge for one or zero time,
        # we directly remove the part of the road instance that is outside the image
        elif len([x for x in vertex_state_list if not x]):
            new_vertices = [v for i, v in enumerate(new_vertices) if vertex_state_list[i]]
            orientation = [o for i, o in enumerate(orientation) if vertex_state_list[i]]
            # update end vertices
            if len(new_vertices)>5:
                src, dst, new_edge = graph.add_e(new_vertices[0],new_vertices[-1])
                src.remove_neighbor(e)
                dst.remove_neighbor(e)
                output_edges.append({'id':new_edge.id,'src':src.id,'dst':dst.id,'circle':False,'vertices':new_vertices,'orientation':orientation})
        else:
            output_edges.append({'id':e.id,'src':e.src.id,'dst':e.dst.id,'circle':False,'vertices':new_vertices,'orientation':orientation})
             
                
    # vertices to json
    for k,v in graph.vertices.items():
        output_vertices.append({'id':v.id,'x':v.x,'y':v.y,'neighbors':[x.id for x in v.neighbor_edges]})

    # vis
    with open(f'../data/graph/{tile_index}.json','w') as jf:
        json.dump({'edges':output_edges,'vertices':output_vertices},jf)

    vis_map = Image.fromarray(np.zeros((IMAGE_SIZE,IMAGE_SIZE,3)).astype(np.uint8)).convert('RGB')
    draw = ImageDraw.Draw(vis_map)
    p = vis_map.load()
    for e in output_edges:
        for v in e['vertices']:
            try:
                p[int(v[0]),int(v[1])] = (255,255,255)
            except:
                pass

    for k,v in graph.vertices.items():
        if len([e for e in v.neighbor_edges if e.circle]):
            draw.ellipse([v.x-INTER_P_RADIUS,v.y-INTER_P_RADIUS,v.x+INTER_P_RADIUS,v.y+INTER_P_RADIUS],fill='purple')
        elif len(v.neighbor_edges) == 1:
            draw.ellipse([v.x-INTER_P_RADIUS,v.y-INTER_P_RADIUS,v.x+INTER_P_RADIUS,v.y+INTER_P_RADIUS],fill='pink')
        elif len(v.neighbor_edges) == 2:
            print(tile_index,v.id,[e.id for e in v.neighbor_edges])
            draw.ellipse([v.x-INTER_P_RADIUS*2,v.y-INTER_P_RADIUS*2,v.x+INTER_P_RADIUS*2,v.y+INTER_P_RADIUS*2],fill='red')
        elif len(v.neighbor_edges) > 2:
            draw.ellipse([v.x-INTER_P_RADIUS,v.y-INTER_P_RADIUS,v.x+INTER_P_RADIUS,v.y+INTER_P_RADIUS],fill='orange')

    vis_map.save(f'../data/vis/{tile_index}.png')

    # =============================================================================================
    # Processing segmentation label
    # =============================================================================================
    
    # 
    global_mask = Image.fromarray(np.zeros((IMAGE_SIZE,IMAGE_SIZE))).convert('RGB')
    draw = ImageDraw.Draw(global_mask)
    p = global_mask.load()

    for e in output_edges:
        for i, v in enumerate(e['vertices'][1:]):
            draw.line([e['vertices'][i][0],e['vertices'][i][1],v[0],v[1]],width=SEGMENT_WIDTH,fill=(255,255,255))
                
    global_mask.save(f'../data/segment/{tile_index}.png')

    # 
    global_keypoint_map = Image.fromarray(np.zeros((IMAGE_SIZE,IMAGE_SIZE))).convert('RGB')
    draw = ImageDraw.Draw(global_keypoint_map)
    for v in output_vertices:
        draw.ellipse([v['x']-3,v['y']-3,v['x']+3,v['y']+3],fill='white')
    
    global_keypoint_map.save(f'../data/intersection/{tile_index}.png')

    print(f'tile: {tile_index}/{data_index}/{len(data_list)}')