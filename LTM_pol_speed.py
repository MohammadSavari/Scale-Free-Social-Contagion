'''
The code goes through the graphs for LTM and generate bunch of poliarization files based on the percentage and selector variables
'''

import numpy as np
import networkx as nx
import scipy as sp
from pathlib import Path
from scipy import sparse
import graph_tool.clustering
import graph_tool.spectral
import graph_tool.topology
import matplotlib.pyplot as plt
import graph_tool as gt
import glob
import pandas as pd
import os
from scipy import linalg
from scipy.sparse import coo_array
import time
from matplotlib.colors import to_hex
import matplotlib
import sys
import warnings
import random
warnings.filterwarnings("ignore")
plt.rc('text', usetex=True)
plt.rc('text.latex', preamble = r'\usepackage{mathptmx}')


def linear_threshold_model(G,threshold,seed_nodes=None,init_spread=True,max_iter=None):
    """
    Runs the linear threshold model on a grap_tool graph G. S_i(t+1) = {1 if <s_j>i~j > T otherwise 0. If the average state of neighbours of i is more than the threshold T, switch the state from 0 to 1

    Takes inputs:
       G: Graph_tool graph
       threshold: float or list of the thresholds as a fraction of neighbors that need to be active to transmit
       seed_nodes: int or list of nodes to start infection from, if None a single random agent is choosen. If int, a number of random nodes is choosen, if a list of nodes those are the initial seeds.
       init_spread: Bool, Wether to spread the infection to the neighbors of seed nodes
       max_iter = maximum number of iterations before stopping. If all nodes in the graph are active the model will stop
    Returnd:
       infected_step: Graph_tool VertexPropertyMap with an interger denoting the step of activation for a given vertex, None if never activated. Velues are stored as vectors, with each index corresponding to a given threshold.
       seed_nodes: Graph_tool GraphProperty with the initail seed nodes for the model run

    """

    if seed_nodes == None:
        [seed_nodes for x in np.random.choice(G.get_vertices(),1)]

    if not type(seed_nodes) is list:
        seed_nodes = np.random.choice(G.get_vertices(),seed_nodes)

    if max_iter is None:
        max_iter = G.num_vertices()

    if not type(threshold) is list:
        [threshold]

    infections = []
    degree_dist = G.get_out_degrees(G.get_vertices())

    T = np.array((graph_tool.spectral.adjacency(G).T.toarray() / degree_dist).T)  

    for th in threshold:
        # Choose the initial infected nodes
        infected = np.zeros(G.num_vertices(),dtype=int)
        infection_step = np.full(G.num_vertices(),np.inf,dtype=float)
        node_list = np.arange(G.num_vertices(),dtype=int)

        #Infect the seed nodes
        infected[seed_nodes] = 1
        #Record seed nodes infected at t=-1
        infection_step[seed_nodes] = -1

        #Initial spread, if choosen
        if init_spread:
            infected[T.dot(infected) > 0] = 1
            infection_step[np.logical_and(infected > 0, np.isinf(infection_step))] = 0
            i = 1
        else:
            i = 0
        while (not all(infected) and (i < max_iter) and i-1 in infection_step):
            infected[T.dot(infected) >= th] = 1
            infection_step[np.logical_and(infected > 0, np.isinf(infection_step))] = i
            i += 1
        infected_step = G.new_vp(value_type='int',vals=infection_step)
        infections.append(infected_step)

    infected_vectormap = gt.group_vector_property(infections)
    # G.vp['infected_step'] = infected_vectormap    
    # G.gp['threshold_vector'] = G.new_gp(value_type='vector<double>',val=threshold)
    # G.gp['cascades'] = G.new_gp(value_type='vector<double>',val=cascades)
    # G.gp['seed_nodes'] = G.new_gp(value_type='vector<double>',val=[seed_nodes])
    
    return infected_vectormap, seed_nodes

def get_laplacian_eigenvalues(G):
    """
    """

    if not G.vertex_properties.get('eig_laplacian',False):
        eig_lap = np.linalg.eigvalsh(gt.spectral.laplacian(G,norm=False).todense())
        G.vp['eig_laplacian'] =  G.new_vertex_property('double',vals=eig_lap)
    return G

def get_kirchhoff_index(G):
    G = get_laplacian_eigenvalues(G)
    G.graph_properties['kirchhoff'] = G.new_graph_property('int64_t',sum(1/np.sort(G.vp.eig_laplacian.get_array())[1:]))
    return G

def get_recursive_graph_paths(root):
    paths = Path(Path(root)).rglob('*.gt')
    return paths

def get_local_clutsering(G):
    if not G.vertex_properties.get('local_clustering',False):
        G.vertex_properties['local_clustering'] = graph_tool.clustering.local_clustering(G)
    return G

def get_transitivity(G):
    if not G.gp.get('transitivity',False):
        trans = G.new_graph_property('double',val=graph_tool.clustering.global_clustering(G)[0])
        G.graph_properties['transitivity'] =  trans
    return G

def get_ave_shortest_path(G):
    if not G.gp.get('shortest_path',False):
        G.gp['shortest_path'] =  G.new_graph_property('double',val=np.sum( graph_tool.topology.shortest_distance(G).get_2d_array(range(G.num_vertices())))/(G.num_vertices()*(G.num_vertices()-1)))
    return G




model = 'LTM'
n = 1000
ks = [8,16,32,200]
network_type = 'mhk' #['mhk','ws','ke']

cascades = np.round(np.linspace(0.1,0.9,9),1)
threshold = np.linspace(0.01,0.5,16) 



selector = ['top','top','rand','bot','bot'] # centrality selector
percentage = [int(n/20),int(n/10),0,-int(n/20),-int(n/10)] # number of nodes to choose

for k in ks:
    for centrality,percen in zip(selector,percentage):
        print(centrality,percen)
        cols=['ID', 'network', 'p','th', 'seed']+ cascades.astype('str').tolist()
        df = pd.DataFrame(columns = cols)
        polarization_file = f'networks/{model}/{n}/{network_type}/{k}_{centrality}{percen}.csv'
        network_path = f'networks/{model}/{n}/{network_type}/{k}'
        count = 0
        for graph_loc in glob.glob(network_path+'/*.gt'):
            G = gt.load_graph(graph_loc)


            degrees = dict(nx.from_numpy_array(gt.spectral.adjacency(G).T.toarray()).degree())
            if percen > 0:
                sorted_nodes = sorted(degrees, key=degrees.get, reverse=True)
                seed_nodes = sorted_nodes[:percen]
            elif percen < 0:
                sorted_nodes = sorted(degrees, key=degrees.get, reverse=True)
                seed_nodes = sorted_nodes[percen:]
            else:
                #randomly select 5% of the nodes
                all_nodes = list(degrees.keys())
                num_nodes_to_choose = int(len(all_nodes) * 5 / 100)
                seed_nodes = random.sample(all_nodes, num_nodes_to_choose)

            for seed in seed_nodes:
                infected_vectormap, selected_seed = linear_threshold_model(G,threshold,seed_nodes=[seed])

                spread = gt.ungroup_vector_property(infected_vectormap,range(len(threshold)))
                data = np.empty((len(threshold),len(cascades),)) * np.nan
                for idx,th in enumerate(threshold):
                    speeds = []
                    # speeds = np.empty((1,len(cascades),)) * np.nan
                    cascade_sizes = cascades ## Copy where we can shorten list on iterations in while loop
                    infected = 0             ## Reset on loop start

                    ## Find nodes infected at given step
                    val,counts = np.unique(spread[idx].a,return_counts=True)
                    ## Refactor to fraction of nodes
                    counts = counts / G.num_vertices()
                    ## For each step in LTM add newly infected to total
                    for i,new in enumerate(counts[val>-2]):
                        infected += new
                        ## Current number of infected are used to find polarization speed if larger than cascade size, and the step is after initialization, and all cascade sizes has not been exceeded yet.
                        while len(cascade_sizes) > 0 and infected > cascade_sizes[0] and val[i] > 0:
                            speeds.append(infected/val[i])
                            cascade_sizes = cascade_sizes[1:]
                    # print('th:',th)
                    nan_padding = len(cascades) - len(speeds)
                    speeds = np.pad(speeds, (0, nan_padding), constant_values=np.nan)
                    # print(speeds)
                    data[idx,:] = speeds
                    df.loc[count] = [G.gp.ID] + [G.gp.ntype] + [G.gp.probability] + [th] + [seed] + list(speeds)
                    count += 1
        print('\r' + f'{polarization_file} saved.', end='', flush=True)
        df.to_csv(polarization_file, sep='\t',index = False)