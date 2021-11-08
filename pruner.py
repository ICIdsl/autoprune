import os
import sys
import copy
import math
import logging

import torch
import numpy as np

import utils
import channel_ranking
import dependency_extractor as de
from pruning_estimator import NetworkSizeTracker

def rank_filters(ranking_type, model, ignore_kws, custom_ranker):
    if custom_ranker is None:
        logging.info(f"Performing {ranking_type} ranking of filters")
        if ranking_type == 'l1-norm':
            channels_per_layer, global_ranking = channel_ranking.l1_norm(model, ignore=ignore_kws)
        else:
            raise ArgumentError(f"Ranking Type {ranking_type}, unsupported by default")
    else:
        logging.info(f"Performing custom ranking of filters")
        channels_per_layer, global_ranking = custom_ranker(model, ignore=ignore_kws) 

    return channels_per_layer, global_ranking

def limit_global_deps_pruning(pl, prn_limits, channels_per_layer, global_deps):
    for dep_layers in global_deps:
        for layer in dep_layers:
            channels = len(channels_per_layer[layer])
            if pl >= 0.85:
                prn_limits[layer] = int(math.ceil(channels * 0.5))
            else:
            # This strategy ensures that at as many pruning levels as possible we maintain wider 
            # networks. At very high pruning levels (those in the if case above), we can't apply the 
            # same strategy as it limits the total amount of pruning possible (memory reduction) --> 
            # this is capped at at most pruning 50% of the layer which allows desired memory reduction 
            # with some width maintenance. This is enforced only for external dependencies, 
            # but internally within a block the internal layers can be pruned heavily
                if pl <= 0.5: 
                    prn_limits[layer] = int(math.ceil(channels * (1.0 - pl)))
                else:
                    prn_limits[layer] = int(math.ceil(channels * pl))
    return prn_limits

def identify_filters_to_prune(pl,
                             _model,
                             channels_per_layer,
                             global_ranking,
                             connectivity,
                             join_nodes,
                             dependencies,
                             prn_limits):
    logging.info(f"Identifying Filters to Prune")
    param_calc = lambda x : sum(np.prod(p.shape) for p in x.parameters())
    model = _model.copy()
    unpruned_model_params = param_calc(model) 
    _channels_per_layer = channels_per_layer.copy()
    network_size_tracker = NetworkSizeTracker(model)
    
    curr_pr = 0
    filter_idx = 0
    pruned_model_params = unpruned_model_params
    while (curr_pr < pl and filter_idx < len(global_ranking)):
        layer, filter_num, _ = global_ranking[filter_idx]
        dep_layers = [x for l in dependencies for x in l if layer in l]
        dep_layers = [layer] if dep_layers == [] else dep_layers 
        for layer in dep_layers:
            if len(_channels_per_layer[layer]) <= prn_limits[layer]:
                # prevent overpruning of certain layers
                continue
            
            if filter_num not in [x[0] for x in _channels_per_layer[layer]]:
                # layer has already been pruned (due to dependencies)
                continue

            _channels_per_layer[layer].pop([i for i,x in enumerate(_channels_per_layer[layer])\
                                        if x[0] == filter_num][0])
            
            pruned_params = network_size_tracker.prune_single_filter(layer, connectivity,
                                                                     join_nodes)
            pruned_model_params -= pruned_params
            curr_pr = 1. - (pruned_model_params / unpruned_model_params)
        filter_idx += 1
    return model, _channels_per_layer

def perform_pruning(pruned_model, filters_to_keep, connectivity):
    logging.info(f"Removing Filters")
    model_dict = dict(pruned_model.named_modules())
    for n,m in pruned_model.named_modules():
        if isinstance(m, torch.nn.Conv2d):
            op_filters = [x for x,y in filters_to_keep[n]]
            utils.reshape_conv_layer(m, op_filters, ofm=True)
            for layer in connectivity[n]:
                module = model_dict[layer]
                if isinstance(module, torch.nn.Conv2d):
                    utils.reshape_conv_layer(module, op_filters, ofm=False)
                elif isinstance(module, torch.nn.BatchNorm2d):
                    utils.reshape_bn_layer(module, op_filters)
                elif isinstance(module, torch.nn.Linear):
                    utils.reshape_linear_layer(module, m, op_filters)
    check_pruning(pruned_model)
    return pruned_model

def check_pruning(model):
    logging.info(f"Checking prunining process")
    for n,m in model.named_modules():
        if isinstance(m, torch.nn.Conv2d):
            assert m.out_channels == m.weight.shape[0], f"Layer {n} pruned incorrectly"
            assert m.in_channels == m.weight.shape[1], f"Layer {n} pruned incorrectly"
        elif isinstance(m, torch.nn.BatchNorm2d):
            assert m.num_features == m.weight.shape[0], f"Layer {n} pruned incorrectly"
        elif isinstance(m, torch.nn.Linear):
            assert m.in_features == m.weight.shape[1], f"Layer {n} pruned incorrectly"

def prune_network(pl,
                  model,
                  ignore_kws=None,
                  min_filters_kept=2,
                  custom_ranker=None,
                  ranking_type='l1-norm',
                  maintain_network_width=True):
    
    assert 0 <= pl < 1, "Pruning level must be value in range [0,1)" 
    
    dependencies = de.get_dependencies(model)
    connectivity, join_nodes, local_deps, global_deps = de.categorise_dependencies(dependencies)
    
    channels_per_layer, global_ranking = rank_filters(ranking_type, model, ignore_kws,\
                                                      custom_ranker)
    
    prn_limits = {k:min_filters_kept for k in channels_per_layer.keys()}
    if maintain_network_width:
        prn_limits = limit_global_deps_pruning(pl, prn_limits, channels_per_layer, global_deps)
    # After this step, "pruned_model" will only have the description parameters set correctly, but
    # not the weights themselves. "filters_to_keep" will have the channels to keep for each layer 
    pruned_model, filters_to_keep = identify_filters_to_prune(pl, model, channels_per_layer,\
                                                              global_ranking, connectivity,\
                                                              join_nodes, local_deps+global_deps,\
                                                              prn_limits)
    pruned_model = perform_pruning(pruned_model, filters_to_keep, connectivity)
    return pruned_model, filters_to_keep
