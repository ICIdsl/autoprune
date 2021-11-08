import os
import sys
import copy
import logging

import torch
import numpy as np

def find_all_convs(graph):
    return graph.findAllNodes('aten::conv2d')

def find_all_bn(graph):
    return graph.findAllNodes('aten::batch_norm')

def find_all_fcs(graph):
    return graph.findAllNodes('aten::matmul')

def find_all_add_nodes(graph):
    return graph.findAllNodes('aten::add_')

def translate_sequential_dependencies(deps, conv_translate):
    new_deps = {conv_translate[k] : [conv_translate[x] for x in v] for k,v in deps.items()}
    return new_deps

def translate_add_dependencies(deps, conv_translate):
    remove_idx = []
    new_deps = [[conv_translate[x] for x in v] for k,v in deps.items()]
    for i,deps in enumerate(new_deps):
        if any(set(deps) < set(x) for j,x in enumerate(new_deps) if i != j):
            remove_idx.append(i)
    no_redundency_deps = [x for i,x in enumerate(new_deps) if i not in remove_idx]
    
    duplicates = []
    for i,deps in enumerate(no_redundency_deps):
        _dup = [j for j,x in enumerate(no_redundency_deps) if i<j and set(deps) == set(x)]
        if len(_dup) != 0:
            duplicates += _dup
    no_redundency_deps = [x for i,x in enumerate(no_redundency_deps) if i not in duplicates]
    return no_redundency_deps

def get_idom_tree(root, nodes, stop_at=None):
    if not root.hasMultipleOutputs():
        node = root.output()
        for user in node.uses():
            if user.user.kind() == stop_at:
                nodes.append(user.user)
            else:
                if user.user.hasUses():
                    nodes.append(user.user)
                    get_idom_tree(user.user, nodes, stop_at)
    else:
        logging.warning(f"Node {root} has multiple outputs!")
        sys.exit()

def find_nodes_idom_by(root):
    nodes = []
    get_idom_tree(root, nodes, stop_at='aten::conv2d')
    return nodes

def get_sequentially_connected_deps(idom_trees):
    linked_convs = {c:[] for c in idom_trees.keys()}
    for conv, tree in idom_trees.items():
        for node in tree:
            if node.kind() == 'aten::conv2d':
                linked_convs[conv].append(node)
            elif node.kind() == 'aten::batch_norm':
                linked_convs[conv].append(node)
            elif node.kind() == 'aten::matmul':
                linked_convs[conv].append(node)
    return linked_convs

def get_add_connected_deps(idom_trees, add_nodes):
    connected_convs = {n:[] for n in add_nodes}
    for node in add_nodes:
        for inp in node.inputs():
            for conv, tree in idom_trees.items():
                if conv.output().debugName() == inp.debugName():
                    connected_convs[node].append(conv)
                else:
                    for idom_node in tree:
                        if idom_node.kind() != 'aten::conv2d':
                            if idom_node.output().debugName() == inp.debugName():
                                connected_convs[node].append(conv)

    keys_to_remove = []
    for k,v in connected_convs.items():
        if len(v) == 1:
            keys_to_remove.append(k)
    [connected_convs.pop(k) for k in keys_to_remove]
    
    return connected_convs
            
def identify_dependencies(graph):
    all_convs = find_all_convs(graph)         
    add_nodes = find_all_add_nodes(graph)
    
    conv_idom_trees = {}
    for conv in all_convs:
        nodes = find_nodes_idom_by(conv)
        conv_idom_trees[conv] = nodes

    add_idom_trees = {}
    for add in add_nodes:
        nodes = find_nodes_idom_by(add)
        add_idom_trees[add] = nodes

    add_deps = get_add_connected_deps(conv_idom_trees, add_nodes)
    add_node_connections = get_sequentially_connected_deps(add_idom_trees) 
    conv_node_connections = get_sequentially_connected_deps(conv_idom_trees)

    return {'seq_deps': conv_node_connections, 
            'add_deps': add_deps,
            'add_node_connections': add_node_connections}

def get_dependencies(model):
    scripted_model = torch.jit.script(model)
    network_graph = scripted_model.inlined_graph
    dependencies = identify_dependencies(network_graph)
    
    model_convs = [n for n,m in model.named_modules() if isinstance(m, torch.nn.Conv2d)]
    graph_convs = find_all_convs(network_graph)
    conv_translate = {g:m for g,m in zip(graph_convs, model_convs)}
    
    model_bns = [n for n,m in model.named_modules() if isinstance(m, torch.nn.BatchNorm2d)]
    graph_bns = find_all_bn(network_graph)
    for m_bn, g_bn in zip(model_bns, graph_bns):
        conv_translate[g_bn] = m_bn
    
    first_model_fc = [n for n,m in model.named_modules() if isinstance(m, torch.nn.Linear)][0]
    first_graph_fc = find_all_fcs(network_graph)[0]
    conv_translate[first_graph_fc] = first_model_fc

    model_deps = {}
    for dep_type, deps in dependencies.items():
        if dep_type == 'seq_deps':
            model_deps['layer_connectivity'] = translate_sequential_dependencies(deps, conv_translate)
            breakpoint()
        elif dep_type == 'add_deps':
            model_deps['add_dependencies'] = translate_add_dependencies(deps, conv_translate)
        elif dep_type == 'add_node_connections':
            model_deps['add_node_connections'] = translate_add_dependencies(deps, conv_translate)
    
    return model_deps

def categorise_dependencies(dependencies):
    local_dep_types = ['dw_dependencies']
    global_dep_types = ['add_dependencies']
    local_deps = [x for k,l in dependencies.items() for x in l if k in local_dep_types]
    global_deps = [x for k,l in dependencies.items() for x in l if k in global_dep_types]
    
    connectivity = dependencies['layer_connectivity']
    
    join_nodes = {}
    if 'add_node_connections' in dependencies.keys():
        join_nodes['aten::add_'] = dependencies['add_node_connections']
    
    return connectivity, join_nodes, local_deps, global_deps

