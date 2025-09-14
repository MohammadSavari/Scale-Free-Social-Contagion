'''
LFC goes over the graphs and gets the gains with for a specific degree and a specific centrality
Make sure to adjust the gain function in the loop
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
import random
from graph_tool.clustering import local_clustering  

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

def get_gain(graph,w,N, centrality = 240, base = 'degree'):
    '''
    graph : G is a graph tool network
    w : array of the frequancies
    N : number of nodes
    centrality : choosing the top (positive) or bottom (negative) nodes with selector
    base = choose the the node selector 'degree' or 'clustering'
    '''
    L = gt.spectral.laplacian(graph,norm=False) #the build in normalized gives the symetric normalized laplacian, but we want the random walk normalized laplacian
    
    L = (L/L.diagonal()).T  ## Random walk normalization  D^-1 L = LD^-1 because L is symetric

    L = L.toarray()
    h2 = G.new_vertex_property('vector<double>')
    
    
    if base == 'degree':
        degrees = dict(nx.from_numpy_array(gt.spectral.adjacency(G).T.toarray()).degree())
        if centrality == N:
            Nodes = G.vertices()
        elif centrality > 0:
            sorted_nodes = sorted(degrees, key=degrees.get, reverse=True)
            seed_nodes = sorted_nodes[:centrality]
            Nodes = seed_nodes
        else:
            # harmonic_centrality = nx.harmonic_centrality(nx.from_numpy_array(gt.spectral.adjacency(G).T.toarray()))
            # seed_nodes = sorted(harmonic_centrality, key=harmonic_centrality.get, reverse=True)[centrality:]
            sorted_nodes = sorted(degrees, key=degrees.get, reverse=True)
            seed_nodes = sorted_nodes[centrality:]
            Nodes = seed_nodes
    elif base == 'clustering':
        clustering_coeffs = local_clustering(G)
        sorted_nodes = sorted(G.vertices(), key=lambda v: clustering_coeffs[v], reverse=True)
        if centrality == N:
            Nodes = G.vertices()
        elif centrality > 0:
            seed_nodes =  [int(v) for v in sorted_nodes[:centrality]]  
            Nodes = seed_nodes
        else:
            # harmonic_centrality = nx.harmonic_centrality(nx.from_numpy_array(gt.spectral.adjacency(G).T.toarray()))
            # seed_nodes = sorted(harmonic_centrality, key=harmonic_centrality.get, reverse=True)[centrality:]
            seed_nodes =  [int(v) for v in sorted_nodes[centrality:]]  
            Nodes = seed_nodes
    
        
        
    for g in Nodes:
        ida = np.arange(N) != g
        idb = np.arange(N) == g
        A = L[np.ix_(ida,ida)].astype(complex)
        B = L[np.ix_(ida,idb)]

        H2 = []    
        for f in w:
            np.fill_diagonal(A,1.0+1j*f)
            #A.setdiag(f*1j-1)
            h = linalg.solve(A,-B)
            
            H2.append(linalg.norm(h)**2)

        h2[g] = H2

    return h2






network_type = ['mhk','ws','ke'] # select the network type
nodes=[240]
n = 240
average_degree = [16] 
# selector = ['top','top','all','bot','bot'] # centrality selector
# centrality_nodes = [int(n/20),int(n/10),240,-int(n/20),-int(n/10)] # number of nodes to choose
# percentages = [5,10,100,-5,-10]

selector = ['top'] # centrality selector
centrality_nodes = [int(n/20)] # number of nodes to choose
percentages = [5]

model = 'LFC'
w = np.logspace(-4,1,100)

b = {'ID':[],'CC':[],'T':[],'p':[],'SP':[]}
d = {'ID':[],'freq':[],'p':[]}

for n_type in network_type:
    for n in nodes:
        for k in average_degree:
            for th, centrality in enumerate(selector):
                network_path = f'networks/{model}/{n}/{n_type}/{k}'
                file_path = f'networks_/{model}/{n}/{n_type}/{k}'
                if not os.path.exists(file_path):
                    os.makedirs(file_path)
                gain_file =  f'{file_path}_{centrality}_{abs(percentages[th])}_corr_gains_degree.csv'
                network_gains = pd.DataFrame(data=d)
                network_gains.set_index(['ID','freq','p'],inplace=True)
                for graph_loc in glob.glob(network_path+'/*.gt'):
                    G = gt.load_graph(graph_loc)

                    frequencies = G.new_graph_property('vector<double>',val=w)
                    G.graph_properties['frequencies'] = frequencies
                    G.vertex_properties['gains'] = get_gain(G,w,n,centrality_nodes[th],base = 'degree') # adjust the second n for selecting top or bot
                    my_gain_data = pd.DataFrame(data=d)
                    my_gain_data.set_index(['ID','freq','p'],inplace=True)

                    for f,g in zip(G.graph_properties['frequencies'],np.mean(G.vp['gains'].get_2d_array(range(len(G.graph_properties['frequencies']))),1)): 
                        my_gain_data.loc[(G.graph_properties['ID'],f,G.gp['probability']),'H2'] = g 

                    network_gains = pd.concat([network_gains, my_gain_data])

                network_gains.to_csv(gain_file,sep='\t',mode='w',header=True)
                
                print('\r' + f'{gain_file}\tSaved.', end='', flush=True)
