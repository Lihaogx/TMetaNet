import math
import pickle
import itertools
from collections import defaultdict
from tqdm import tqdm
from multiprocessing import Pool
import numpy as np
import torch
import networkx as nx
import dionysus as d
import matplotlib.pyplot as plt
import scipy.sparse as sp
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import shortest_path
from scipy.spatial.distance import squareform
import random
import time



## load graph

def relabel_graphs(graphs):
    all_nodes = set()
    for G in graphs:
        all_nodes.update(G.nodes)
        
    node_mapping = {old_node: new_node for new_node, old_node in enumerate(sorted(all_nodes))}

    relabeled_graphs = []
    for G in graphs:
        relabeled_G = nx.relabel_nodes(G, node_mapping)
        relabeled_graphs.append(relabeled_G)
    return relabeled_graphs , len(all_nodes) 


def load_data(edges,is_directed):
    """
    将边列表转换为图
    
    Args:
        edges: 边的张量/数组,需要重新整理成正确的边列表格式
    """
    if isinstance(edges, torch.Tensor):
        edges = edges.numpy()  # 如果是tensor先转换为numpy数组
    
    # 如果是(2, n)格式，转为为(n, 2)格式
    if len(edges.shape) == 2 and edges.shape[0] == 2:
        edges = edges.T
    
    # 将edges转换为边列表
    edge_list = [(int(src), int(dst)) for src, dst in edges]

    if is_directed:
        G = nx.DiGraph()
    else:
        G = nx.Graph()
    G = nx.Graph()
    G.add_edges_from(edge_list)
    # 移除自环
    G.remove_edges_from([(u, v) for u, v in G.edges() if u == v])
    return G

def load_graphs(graph_edges_list,remove_edge,is_directed=False):
    graph_list_ori = []

    for edges in graph_edges_list:
        G = load_data(edges, is_directed)
        if 'kshell' in remove_edge:
            k = int(remove_edge.split('-')[1])
            k_shell = nx.core_number(G)
            k_shell_nodes = [node for node, core in k_shell.items() if core <= k]
            nodes_to_remove = random.sample(k_shell_nodes, int(len(k_shell_nodes)))
            G.remove_nodes_from(nodes_to_remove)
        elif 'degree' in remove_edge:
            degree_threshold = int(remove_edge.split('-')[1])
            degree_nodes = [node for node, degree in G.degree() if degree <= degree_threshold]
            nodes_to_remove = random.sample(degree_nodes, int(len(degree_nodes)))
            G.remove_nodes_from(nodes_to_remove)
            
        graph_list_ori.append(G)  

    graph_list, Nnode = relabel_graphs(graph_list_ori)
        
    return graph_list, Nnode

## Witness and Landmark Selection
def calculate_distances_bfs(adj_matrix,is_directed=False):
    """Calculate shortest path matrix between all nodes in graph using scipy's shortest_path function"""
    dist_matrix = shortest_path(adj_matrix, directed=is_directed, return_predecessors=False)
    return sp.csr_matrix(dist_matrix)  # Return distance matrix in sparse format

def construct_epsilon_net(dist_matrix, epsilon, nodes ,mapped_landmark_seed):
    """Build Epsilon-net based on shortest path matrix and epsilon value, prioritizing nodes with higher degrees and only selecting nodes with edges"""
    epsilon_net = []
    covered_nodes = set()

    node_degrees = {}
    for node in nodes:
        column = dist_matrix[:, node].toarray().flatten()
        degree = sum(1 for i in range(len(column)) if column[i] <= epsilon and i != node)
        node_degrees[node] = degree

    # Sort nodes by degree in descending order
    sorted_nodes = sorted(nodes, key=lambda node: node_degrees[node], reverse=True)
    # Process seed nodes first
    if len(mapped_landmark_seed)>0:
        sorted_seed = sorted(mapped_landmark_seed, key=lambda node: node_degrees[node], reverse=True)
        for start_node in sorted_seed:
            epsilon_net.append(start_node)
            covered_nodes.add(start_node)
            break 

        for node in sorted_seed:
            if node not in covered_nodes:
                if all(dist_matrix[node, selected] > epsilon for selected in epsilon_net):
                    epsilon_net.append(node)
                    covered_nodes.add(node)

    for node in sorted_nodes:
        if node not in covered_nodes:
            if all(dist_matrix[node, selected] > epsilon for selected in epsilon_net):
                epsilon_net.append(node)
                covered_nodes.add(node)

    return epsilon_net



def select_landmarks_and_witnesses(nxgraph, epsilon ,seed_original,is_directed):
    original_nodes = list(nxgraph.nodes())
    # Select nodes that exist in both seed nodes and graph nodes
    Landmark_seed=[]
    for seed in seed_original:
        if seed in original_nodes:
            Landmark_seed.append(seed)
    
    
    n_nodes=len(original_nodes)
    node_mapping = {node: idx for idx, node in enumerate(original_nodes)}
    reverse_mapping = {v: k for k, v in node_mapping.items()}
    mapped_graph = nx.relabel_nodes(nxgraph, node_mapping)
    matrix = nx.adjacency_matrix(mapped_graph, nodelist=range(0, n_nodes)).toarray()
    sparse_matrix= sp.csr_matrix(matrix)
    
    connected_nodes = [i for i in range(n_nodes)]
    mapped_landmark_seed = [node_mapping[node] for node in Landmark_seed if node in node_mapping]

    dist_matrix = calculate_distances_bfs(sparse_matrix,is_directed)

    L = construct_epsilon_net(dist_matrix, epsilon, connected_nodes,mapped_landmark_seed)
    W = [node for node in connected_nodes if node not in L]
    return reverse_mapping,L, W, dist_matrix


## Computing Dowker Complex
def build_dowker_complex(dist_matrix, L, W, D):
    # 为每个witness节点生成邻接表，记录其与L节点的连接情况
    WL_D_neighbors = {}
    dowker_edge = []

    for w_node in W:
        # 查找w_node与L节点的邻接关系
        w_neighbors = []
        for l_node in L:
            if dist_matrix[w_node, l_node] <= D :
                w_neighbors.append(l_node)
        WL_D_neighbors[w_node] = w_neighbors

    for node, neighbors in WL_D_neighbors.items():
        for comb in itertools.combinations(neighbors, 2):
            dowker_edge.append(comb)
    return dowker_edge


def compute_dowker(nxgraph,epsilon,alpha,seed_original,is_directed):
    
    reverse_mapping,L_mapped, W_mapped, dist_matrix = select_landmarks_and_witnesses(nxgraph, epsilon,seed_original,is_directed)
    edges = build_dowker_complex(dist_matrix, L_mapped, W_mapped, alpha)
    dowker_edges = [[reverse_mapping[u], reverse_mapping[v]] for u, v in edges]
    L_original = [reverse_mapping[node] for node in L_mapped]
    return dowker_edges,L_original


## Computing Zigzag Homology
def construct_union_edges(dowkre_graph, j, Nnode):
    A_edges = list(dowkre_graph[j].edges())  
    B_edges = list(dowkre_graph[j+1].edges())
    A_edges += [(v, u) for u, v in A_edges if (v, u) not in A_edges]
    B_edges += [(v, u) for u, v in B_edges if (v, u) not in B_edges]
    C = nx.Graph()
    C.add_edges_from(A_edges)
    C.add_edges_from(B_edges)
    C_edges = list(C.edges())
    C_edges += [(v, u) for u, v in C_edges if (v, u) not in C_edges]
    MDisAux_edges = []
    for (i, j) in A_edges:
        MDisAux_edges.append((i, j))

    for (i, j) in B_edges:
        MDisAux_edges.append((i + Nnode, j + Nnode))
    for (i, j) in C_edges:
        MDisAux_edges.append((i, j + Nnode))  
        MDisAux_edges.append((i + Nnode, j))  
    return MDisAux_edges

def shift_filtration(rips,n):
    "Take a Dionysus filtration and increase the name of all of the vertices by n."
    f = d.Filtration()
    for s in rips:
        dim = s.dimension()
        temp = []
        for i in range(0,dim+1):
            temp.append(s[i]+n)
        f.append(d.Simplex(temp,s.data))
    return f

def complex_union(f,g):
    "Takes two filtrations and builds their union simplicial complex."
    union = d.Filtration()
    for s in f:
        union.append(s)
    for s in g:
        union.append(s)
    return union

def build_zigzag_times(rips,n,numbins):
    times = [[] for x in range(0,rips.__len__())]
    i=0
    for x in tqdm(rips):
       dim = x.dimension()
       t = []
       for k in range(0,dim+1):
          t.append(x[k])
       xmin = math.floor(min(t)/n)
       xmax = math.floor(max(t)/n)
       if xmax == 0:
          bd = [0,1]
       elif xmin == numbins-1:
          bd = [2*xmin-1,2*xmin]
       elif xmax == xmin:
          bd = [2*xmin-1,2*xmin+1]
       elif xmax > xmin:
          bd = [2*xmax-1,2*xmax-1]
       else:
          print("Something has gone horribly wrong!")
       times[i] = bd
       i = i+1
    return times

## Computing Topological Diagram for each window
def compute_topo_diagram(edges_list,windowsize,epsilon,delta,remove_edge,is_directed=False):
    graph_list,Nnode=load_graphs(edges_list,remove_edge,is_directed)

    T = len(graph_list)
    graph_Union_list=[]
    dowker_edges_list=[]
    windowsize = min(T, windowsize)
    Landmarks_seed_list = []
    Landmarks_seed_list.append([])

    print('Begin Dowker Complex computation')
    for k in tqdm(range(T), disable=False):
        if len(Landmarks_seed_list)>windowsize:
            Landmarks_seeds  = list(set(element for sublist in Landmarks_seed_list[k-windowsize+2:k+1] for element in sublist))
        else:
            Landmarks_seeds = list(set(element for sublist in Landmarks_seed_list[0:k+1] for element in sublist))
        dowker_edges, L_original=compute_dowker(graph_list[k],epsilon,delta,Landmarks_seeds,is_directed)
        dowker_edges_list.append(dowker_edges)
        Landmarks_seed_list.append(L_original)
    print('End Dowker Complex computation')


    dowker_graph=[]
    for i in range(T):
        G=nx.Graph()
        G.add_edges_from(dowker_edges_list[i]) 
        G.add_nodes_from([j for j in range(0,Nnode)])
        dowker_graph.append(G)

    # #union
    for j in range(T):
        if j < T-1:
            union_edges=construct_union_edges(dowker_graph, j, Nnode)
            graph_Union_list.append(union_edges)
            
    print('Begin simplices computation')
    simplices_list = []
    for k in range(len(graph_Union_list)):
        edges = graph_Union_list[k]  
        cpl = d.Filtration()
        # 使用 defaultdict 创建邻接列表
        adj_list = defaultdict(set)
        for u, v in edges:
            adj_list[u].add(v)
            adj_list[v].add(u)

        # 添加每单纯形
        for u, v in edges:
            cpl.append(d.Simplex([u]))  # 节点 u
            cpl.append(d.Simplex([v]))  # 节点 v
            cpl.append(d.Simplex([u, v]))  # 边 [u, v]
            common_neighbors = adj_list[u].intersection(adj_list[v])
            for w in common_neighbors:
                cpl.append(d.Simplex([u, v, w])) 

        simplices_list.append(cpl)
    print("  --- End simplices computation")  # Ending

    
    print("Shifting filtrations...")  # Beginning
    G_shift = []
    G_shift.append(simplices_list[0])  # Shift 0... original rips01
    for kk in tqdm(range(1, T - 1)):
        shiftAux = shift_filtration(simplices_list[kk], Nnode * kk)
        G_shift.append(shiftAux)
    print("  --- End shifting...")  # Ending

    print("Combining complexes...") # Beginning
    completecpl = complex_union(G_shift[0], G_shift[1]) 
    for i in tqdm(range(2,T-1)):
        completecpl = complex_union(completecpl, G_shift[i]) 

    print("  --- End combining") # Ending

    print("Determining time intervals...")  # Beginning
    times_list = build_zigzag_times(completecpl, Nnode, T)
    print("  --- End time")  # Beginning

    print("computing zigzag homology ...")
    dgms_list=[]
    time_begin = time.time()
    G_zz, G_dgms, G_cells = d.zigzag_homology_persistence(completecpl, times_list,progress=True)
    time_end = time.time()
    print(f"zigzag homology persistence time taken: {time_end - time_begin} seconds")
    complete_dgms = []
    # Personalized plot
    for vv, dgm in enumerate(G_dgms):
        if (vv < 2):
            matBarcode = np.zeros((len(dgm), 2))
            k = 0
            for p in dgm:
                matBarcode[k, 0] = p.birth
                matBarcode[k, 1] = p.death
                k = k + 1
            complete_dgms.append(matBarcode)
    print("success")
    if len(complete_dgms)==0:
        dim1_windowed_dgms = [[] for _ in range(T)]
        dim1_windowed_dgms = [[] for _ in range(T)]
    elif len(complete_dgms)==1:
        dim0_windowed_dgms = sliding_window_persistence_filtered(complete_dgms[0], windowsize,T)
        dim1_windowed_dgms = [[] for _ in range(T)]
    else:
        dim0_windowed_dgms = sliding_window_persistence_filtered(complete_dgms[0], windowsize,T)
        dim1_windowed_dgms = sliding_window_persistence_filtered(complete_dgms[1], windowsize,T)
        
    dgms_list= []
    for i in range(T):
        window_dgms=[]
        window_dgms.append(dim0_windowed_dgms[i])
        window_dgms.append(dim1_windowed_dgms[i])
        dgms_list.append(window_dgms)
    time_end = time.time()
    return dgms_list

def sliding_window_persistence_filtered(dgm, windowsize,T):
    windowed_dgms = []
    birth = dgm[:, 0]
    death = dgm[:, 1]
    if windowsize==T:
        s=0
    else:
        s=1
    for i in range(0, T):  
        window_start = i - windowsize+1
        window_end = i
        filtered_dgm = []
        shift_N = (i-windowsize +1)*s
    
        for b, d in zip(birth, death):
            if b >= window_start and d <= window_end:
                filtered_dgm.append([b-shift_N, d-shift_N])
            elif (b < window_start and d > window_start and d <= window_end) or (b >= window_start and b < window_end and d > window_end):
                filtered_dgm.append([max(b, window_start)-shift_N, min(d, window_end)-shift_N])
            elif b < window_start and d > window_end:
                filtered_dgm.append([window_start-shift_N, window_end-shift_N])

        windowed_dgms.append(np.array(filtered_dgm))

    return windowed_dgms
## Computing persistence image
def compute_persistence_image(i,topo_diagram, resolution, windowsize, bandwidth, power):
    zpi_list = []
    zpi = np.zeros(resolution)
    zpi_list.append(zpi)

    if windowsize>=len(topo_diagram):
        for j in tqdm(range(1,len(topo_diagram))):
            if len(topo_diagram[j][i])>0:
                windowsize = j+1
                zpi = zigzag_persistence_images(topo_diagram[j], resolution, return_raw = False, normalization = True, bandwidth = bandwidth, power = power,windowsize=windowsize,dimensional = i)
            else:
                zpi = zpi_list[j-1]
            zpi_list.append(zpi)
    else:
        for j in tqdm(range(1,len(topo_diagram))):
            if len(topo_diagram[j][i])>0:
                zpi = zigzag_persistence_images(topo_diagram[j], resolution, return_raw = False, normalization = True, bandwidth = bandwidth, power = power,windowsize=windowsize,dimensional = i)
            else:
                zpi = zpi_list[j-1]
            zpi_list.append(zpi)  

    return zpi_list


def zigzag_persistence_images(dgms, resolution = [50,50], return_raw = False, normalization = True, bandwidth = 1., power = 1.,windowsize = False, dimensional = 0):

    x = np.linspace(0, windowsize-1, resolution[0])
    y = np.linspace(0, windowsize-1, resolution[1])

    X, Y = np.meshgrid(x, y)
    Zfinal = np.zeros(X.shape)
    X, Y = X[:, :, np.newaxis], Y[:, :, np.newaxis]

    P0, P1 = np.reshape(dgms[int(dimensional)][:, 0], [1, 1, -1]), np.reshape(dgms[int(dimensional)][:, 1], [1, 1, -1])

    weight = np.abs(P1 - P0)

    distpts = np.sqrt((X - P0) ** 2 + (Y - P1) ** 2)

    if return_raw:
        lw = [weight[0, 0, pt] for pt in range(weight.shape[2])]
        lsum = [distpts[:, :, pt] for pt in range(distpts.shape[2])]
    else:
        weight = weight ** power
        Zfinal = (np.multiply(weight, np.exp(-distpts ** 2 / bandwidth))).sum(axis=2)

    output = [lw, lsum] if return_raw else Zfinal

    if normalization:
        norm_output = (output - np.min(output))/(np.max(output) - np.min(output) + 1e-10)
    else:
        norm_output = output
    
    return norm_output