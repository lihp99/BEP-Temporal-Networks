from collections import defaultdict, deque
from pandas import Timestamp
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor
import numpy as np

def temporal_degree(G):
    temporal_degree_dict = defaultdict(int)
    for u, v, data in G.edges(data=True):
        temporal_degree_dict[u] += 1
        if not G.is_directed():
            temporal_degree_dict[v] += 1
    return temporal_degree_dict

def find_temporal_shortest_paths(G, source, target, time_attr): #Used for Betweenness Calculation
    shortest_paths = []
    min_length = float('inf')
    queue = deque([(source, [source], -float('inf'))])
    visited = set()
    
    while queue:
        current, path, last_time = queue.popleft()
        
        if current == target:
            if len(path) < min_length:
                shortest_paths = [path]
                min_length = len(path)
            elif len(path) == min_length:
                shortest_paths.append(path)
            continue
        
        if (current, last_time) in visited:
            continue
        visited.add((current, last_time))
        
        neighbors = G.successors(current) if G.is_directed() else G.neighbors(current)
        
        for neighbor in neighbors:
            for key, edge_data in G[current][neighbor].items():
                edge_time = edge_data[time_attr]
                if isinstance(edge_time, datetime):
                    edge_time = edge_time.timestamp()  # Convert Timestamp to seconds
                if edge_time > last_time:
                    queue.append((neighbor, path + [neighbor], edge_time))
    
    return shortest_paths

def temporal_betweenness(G, time_attr='time'):
    nodes = list(G.nodes())
    N = len(nodes)
    betweenness = {node: 0 for node in nodes}
    
    def process_pair(s, t):
        if s == t:
            return
        
        paths = find_temporal_shortest_paths(G, s, t, time_attr)
        if not paths:
            return
        
        sigma_st = len(paths)
        path_count = defaultdict(int)
        
        for path in paths:
            for i in range(1, len(path) - 1):
                path_count[path[i]] += 1
        
        for node in path_count:
            betweenness[node] += path_count[node] / sigma_st
    
    with ThreadPoolExecutor() as executor:
        for s in nodes:
            for t in nodes:
                executor.submit(process_pair, s, t)
    
    scale = 1 / ((N - 1) * (N - 2))
    for node in betweenness:
        betweenness[node] *= scale
    
    return betweenness

def temporal_closeness(G, time_attr='time'):
    closeness = {}
    nodes = list(G.nodes())
    N = len(nodes)
    
    for s in nodes:
        distance = {node: float('inf') for node in nodes}
        distance[s] = 0
        Q = deque([(s, 0)])  # (node, current_time)
        
        while Q:
            current_node, current_time = Q.popleft()
            neighbors = G.successors(current_node) if G.is_directed() else G.neighbors(current_node)
            
            for neighbor in neighbors:
                for key in G[current_node][neighbor]:
                    edge_time = G[current_node][neighbor][key][time_attr]
                    if isinstance(edge_time, datetime):
                        edge_time = edge_time.timestamp()  # Convert Timestamp to seconds
                    if current_time <= edge_time < distance[neighbor]: 
                        distance[neighbor] = edge_time
                        Q.append((neighbor, edge_time))
        
        total_reciprocal_distance = sum([1/d for d in distance.values() if d != float('inf') and d != 0])
        reachable_nodes = len([d for d in distance.values() if d != float('inf') and d != 0])
        
        if reachable_nodes > 0:
            closeness[s] = total_reciprocal_distance / (N - 1)
        else:
            closeness[s] = 0
    
    return closeness

def find_reachable_nodes(G, source, time_attr):
    reachable = set()
    queue = deque([(source, -float('inf'))])
    visited = set()
    
    while queue:
        current, last_time = queue.popleft()
        
        if (current, last_time) in visited:
            continue
        visited.add((current, last_time))
        reachable.add(current)
        
        neighbors = G.successors(current) if G.is_directed() else G.neighbors(current)
        
        for neighbor in neighbors:
            for key, edge_data in G[current][neighbor].items():
                edge_time = edge_data[time_attr]
                if isinstance(edge_time, datetime):
                    edge_time = edge_time.timestamp()  # Convert Timestamp to seconds
                if edge_time > last_time:
                    queue.append((neighbor, edge_time))
    
    return reachable

def calculate_reachability_ratio(G, time_attr):
    total_nodes = len(G.nodes)
    total_pairs = total_nodes * (total_nodes - 1)
    reachable_pairs = 0
    
    for node in G.nodes:
        reachable_nodes = find_reachable_nodes(G, node, time_attr)
        reachable_pairs += len(reachable_nodes) - 1  # Exclude the node itself
    
    reachability_ratio = reachable_pairs / total_pairs
    return reachability_ratio

def reachability_latency(G, time_attr, r):
    T = len(set(data[time_attr] for u, v, data in G.edges(data=True)))
    N = len(G.nodes)
    
    d_t_i = np.zeros((T, N))
    
    for t, time in enumerate(sorted(set(data[time_attr] for u, v, data in G.edges(data=True)))):
        for i, node in enumerate(G.nodes):
            path_lengths = []
            for target in G.nodes:
                if node != target:
                    shortest_paths = find_temporal_shortest_paths(G, node, target, time_attr)
                    if shortest_paths:
                        path_lengths.append(len(shortest_paths[0]) - 1)
            if path_lengths:
                d_t_i[t, i] = np.mean(path_lengths)
    
    k = int(np.floor(r * N))
    
    R_r = (1 / (T * N)) * np.sum(np.sort(d_t_i, axis=1)[:, k])
    
    return R_r


    