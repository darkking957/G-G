import networkx as nx
from collections import deque
import random

def build_graph(graph: list) -> nx.Graph:
    G = nx.Graph()
    for triplet in graph:
        h, r, t = triplet
        G.add_edge(h, t, relation=r.strip())
    return G

# 定义一个函数来进行宽度优先搜索
def bfs_with_rule(graph, start_node, target_rule, max_p = 10):
    result_paths = []
    queue = deque([(start_node, [])])  # 使用队列存储待探索节点和对应路径
    while queue:
        current_node, current_path = queue.popleft()

        # 如果当前路径符合规则，将其添加到结果列表中
        if len(current_path) == len(target_rule):
            result_paths.append(current_path)
            # if len(result_paths) >= max_p:
            #     break
            
        # 如果当前路径长度小于规则长度，继续探索
        if len(current_path) < len(target_rule):
            if current_node not in graph:
                continue
            for neighbor in graph.neighbors(current_node):
                # 剪枝：如果当前边类型与规则中的对应位置不匹配，不继续探索该路径
                rel = graph[current_node][neighbor]['relation']
                if rel != target_rule[len(current_path)] or len(current_path) > len(target_rule):
                    continue
                queue.append((neighbor, current_path + [(current_node, rel,neighbor)]))
    
    return result_paths

def get_truth_paths(q_entity: list, a_entity: list, graph: nx.Graph) -> list:
    '''
    Get shortest paths connecting question and answer entities.
    '''
    # Select paths
    paths = []
    for h in q_entity:
        if h not in graph:
            continue
        for t in a_entity:
            if t not in graph:
                continue
            try:
                for p in nx.all_shortest_paths(graph, h, t):
                    paths.append(p)
            except:
                pass
    # Add relation to paths
    result_paths = []
    for p in paths:
        tmp = []
        for i in range(len(p)-1):
            u = p[i]
            v = p[i+1]
            tmp.append((u, graph[u][v]['relation'], v))
        result_paths.append(tmp)
    return result_paths
    
def get_simple_paths(q_entity: list, a_entity: list, graph: nx.Graph, hop=2) -> list:
    '''
    Get all simple paths connecting question and answer entities within given hop
    '''
    # Select paths
    paths = []
    for h in q_entity:
        if h not in graph:
            continue
        for t in a_entity:
            if t not in graph:
                continue
            try:
                for p in nx.all_simple_edge_paths(graph, h, t, cutoff=hop):
                    paths.append(p)
            except:
                pass
    # Add relation to paths
    result_paths = []
    for p in paths:
        result_paths.append([(e[0], graph[e[0]][e[1]]['relation'], e[1]) for e in p])
    return result_paths

def get_negative_paths(q_entity: list, a_entity: list, graph: nx.Graph, n_neg: int, hop=2) -> list:
    '''
    Get negative paths for question witin hop
    '''
    # Get random paths starting from question entities
    paths = []
    for _ in range(n_neg):
        for start_node in q_entity:
            if start_node not in graph:
                continue
            path = simple_random_walk(graph, start_node, hop)
            if path and path[-1][2] not in a_entity:  # Ensure it doesn't end at answer entity
                paths.append(path)
    return paths

def simple_random_walk(graph: nx.Graph, start_node: str, walk_len: int) -> list:
    '''
    Perform a simple random walk on the graph
    '''
    path = []
    current_node = start_node
    
    for _ in range(walk_len):
        neighbors = list(graph.neighbors(current_node))
        if not neighbors:
            break
        next_node = random.choice(neighbors)
        relation = graph[current_node][next_node]['relation']
        path.append((current_node, relation, next_node))
        current_node = next_node
    
    return path

def get_random_paths(q_entity: list, graph: nx.Graph, n=3, hop=2) -> tuple[list, list]:
    '''
    Get random paths for question within hop
    '''
    # sample paths
    result_paths = []
    rules = []
    
    for _ in range(n):
        for start_node in q_entity:
            if start_node not in graph:
                continue
            path = simple_random_walk(graph, start_node, hop)
            if path:
                result_paths.append(path)
                rules.append([p[1] for p in path])  # Extract relations
    
    return result_paths, rules

def get_paths_from_entity(graph: nx.Graph, start_entity: str, max_hop: int = 2, max_paths: int = 10) -> list:
    '''
    Get paths starting from an entity up to max_hop
    '''
    paths = []
    queue = deque([(start_entity, [])])
    visited_paths = set()
    
    while queue and len(paths) < max_paths:
        current_node, current_path = queue.popleft()
        
        # If we've reached the desired hop length, add to results
        if len(current_path) == max_hop:
            path_tuple = tuple((p[0], p[1], p[2]) for p in current_path)
            if path_tuple not in visited_paths:
                paths.append(current_path)
                visited_paths.add(path_tuple)
            continue
        
        # If we haven't reached max hop, continue exploring
        if len(current_path) < max_hop:
            if current_node not in graph:
                continue
            neighbors = list(graph.neighbors(current_node))
            random.shuffle(neighbors)  # Randomize to get diverse paths
            
            for neighbor in neighbors[:5]:  # Limit branching factor
                rel = graph[current_node][neighbor]['relation']
                new_path = current_path + [(current_node, rel, neighbor)]
                queue.append((neighbor, new_path))
    
    return paths

# Add these functions to the existing graph_utils.py

import random
from typing import List, Tuple, Set

def validate_relation_path(graph: nx.Graph, relation_path: List[str], max_attempts: int = 100, min_instances: int = 1) -> bool:
    """
    Validate if a relation path exists in the knowledge graph.
    
    Args:
        graph: NetworkX graph representing the KG
        relation_path: List of relations forming the path
        max_attempts: Maximum number of starting entities to try
        min_instances: Minimum number of valid instances required
        
    Returns:
        True if the path is valid (exists in KG), False otherwise
    """
    if not relation_path:
        return False
        
    nodes = list(graph.nodes())
    if not nodes:
        return False
        
    valid_instances = 0
    attempts = 0
    
    # Try random starting nodes
    random.shuffle(nodes)
    
    for start_node in nodes:
        if attempts >= max_attempts:
            break
            
        attempts += 1
        current_nodes = {start_node}
        
        # Try to traverse the path
        for rel in relation_path:
            next_nodes = set()
            for node in current_nodes:
                if node in graph:
                    for neighbor in graph.neighbors(node):
                        if graph[node][neighbor].get('relation') == rel:
                            next_nodes.add(neighbor)
            
            if not next_nodes:
                break
            current_nodes = next_nodes
        else:
            # Successfully traversed the entire path
            valid_instances += 1
            if valid_instances >= min_instances:
                return True
    
    return False


def find_path_instances(graph: nx.Graph, relation_path: List[str], max_instances: int = 10) -> List[List[Tuple[str, str, str]]]:
    """
    Find actual instances of a relation path in the knowledge graph.
    
    Args:
        graph: NetworkX graph
        relation_path: List of relations
        max_instances: Maximum number of instances to return
        
    Returns:
        List of path instances, each as a list of (head, relation, tail) tuples
    """
    instances = []
    nodes = list(graph.nodes())
    random.shuffle(nodes)
    
    for start_node in nodes:
        if len(instances) >= max_instances:
            break
            
        current_paths = [[(start_node, None, None)]]
        
        for rel in relation_path:
            next_paths = []
            for path in current_paths:
                current_node = path[-1][0] if path[-1][2] is None else path[-1][2]
                
                if current_node in graph:
                    for neighbor in graph.neighbors(current_node):
                        if graph[current_node][neighbor].get('relation') == rel:
                            new_path = path[:-1] if path[-1][2] is None else path
                            new_path = new_path + [(current_node, rel, neighbor)]
                            next_paths.append(new_path)
            
            current_paths = next_paths
            if not current_paths:
                break
        
        for path in current_paths:
            if len(instances) < max_instances:
                instances.append(path)
    
    return instances


def score_relation_path(graph: nx.Graph, relation_path: List[str], question_entities: List[str]) -> float:
    """
    Score a relation path based on its connectivity to question entities.
    
    Args:
        graph: NetworkX graph
        relation_path: List of relations
        question_entities: List of entities mentioned in the question
        
    Returns:
        Score between 0 and 1
    """
    if not relation_path:
        return 0.0
        
    # Check if path starts from any question entity
    connectivity_score = 0.0
    total_attempts = 0
    
    for q_entity in question_entities:
        if q_entity not in graph:
            continue
            
        total_attempts += 1
        current_nodes = {q_entity}
        
        for i, rel in enumerate(relation_path):
            next_nodes = set()
            for node in current_nodes:
                if node in graph:
                    for neighbor in graph.neighbors(node):
                        if graph[node][neighbor].get('relation') == rel:
                            next_nodes.add(neighbor)
            
            if not next_nodes:
                # Path breaks at position i
                connectivity_score += (i / len(relation_path))
                break
            current_nodes = next_nodes
        else:
            # Complete path found
            connectivity_score += 1.0
    
    return connectivity_score / max(total_attempts, 1)