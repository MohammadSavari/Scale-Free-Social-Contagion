'''
Generates the gt files and then generates CSVs
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



def ws_network(N,k,p,seed=None):
    """
    Function for creating a Modified Holme-Kim network
    Takes inputs:
       N: int, Number of nodes
       k: integer, The mean degree of nodes
       p: Probability of rewireing of edges on the graph
       seed: Integer for the numpy random seed function
    Returnd:
       G: a graphtool graph

    """ 
    if seed is None:
        seed=np.random.randint(2**63)

    G = gt.Graph(directed=False)
    G.add_edge_list(np.transpose(sp.sparse.tril(nx.adjacency_matrix(nx.connected_watts_strogatz_graph(N,k,p,tries=1000000,seed=seed))).nonzero()))

    return G


def mhk_network(N,k,p,seed=None):
    """
    Function for creating a Modified Holme-Kim network
    Takes inputs:
       N: int, Number of nodes
       k: integer, The mean degree of nodes
       p: Probability of rewireing of edges on the graph
       seed: Integer for the numpy random seed function
    Returnd:
       G: a graphtool graph

    """ 
    if seed is None:
        seed=np.random.randint(2**63)
    G = nx.connected_watts_strogatz_graph(k,2,0)

    
    for n in range(k,N):
        anchor = np.random.choice(list(G.nodes()))   
        anchor_neigh = list(G.neighbors(anchor))

        for i in range(int(k)):

            if np.random.random() < p:     
                G.add_edge(np.random.choice(anchor_neigh),n)

            else: 
                try:
                    temp = np.random.choice(np.setdiff1d(G.nodes(),np.append(anchor_neigh,[anchor, n])))
                    G.add_edge(temp,n)
                except:
                    temp = np.random.choice(anchor_neigh)
                    G.add_edge(temp,n)
        G.add_edge(anchor,n)
    
    T = gt.Graph(directed=False)
    T.add_edge_list(np.transpose(sp.sparse.tril(nx.adjacency_matrix(G)).nonzero()))

    return T

def ke_network(n,m):

    #https://rf.mokslasplius.lt/acchieving-high-clustering-in-scale-free-networks/
    G = nx.connected_watts_strogatz_graph(m,m,0)
    active_nodes = list(G.nodes())
    for i in range(m,n):
        for k in active_nodes:
            G.add_edge(k,i)
        active_nodes.append(i)
        active_nodes.remove(random.choice(active_nodes))
        
    T = gt.Graph(directed=False)
    T.add_edge_list(np.transpose(sp.sparse.tril(nx.adjacency_matrix(G)).nonzero()))
    
    return T


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


b = {'ID':[],'CC':[],'T':[],'p':[],'SP':[]}
d = {'ID':[],'freq':[],'p':[]}

network_type = ['mhk','ws']
nodes=[240]
average_degree= range(4,32,2)
p = np.linspace(0.001,1,50)
model = 'LFC'
w = np.logspace(-4,1,100)
for n_type in network_type:
    for n in nodes:
        for k in average_degree:
            network_path = f"networks/{model}/{n}/{n_type}/{k}"
            spec_file =  f"networks/{model}/{n}/{n_type}/{k}_props.csv'
            network_props = pd.DataFrame(data=b)
            network_props.set_index(['ID','p'],inplace=True)
            if not os.path.exists(network_path):
                os.makedirs(network_path)
            for prob in p:
                if n_type=="mhk":
                    G = mhk_network(n,k,prob)
                else:
                    G = ws_network(n,k,prob)
                G.graph_properties['ID'] = G.new_graph_property('int64_t',val=int(time.time()*1000))
                G.graph_properties['ntype'] = G.new_graph_property('string',val=network_type)
                G.graph_properties['probability'] = G.new_graph_property('double',prob)
                frequencies = G.new_graph_property('vector<double>',val=w)
                G.graph_properties['frequencies'] = frequencies
                G.vertex_properties['gains'] = get_gain(G,w,n)
                G = get_local_clutsering(G)
                G = get_transitivity(G)
                G = get_ave_shortest_path(G)
                G = get_laplacian_eigenvalues(G)
                G.save(f'{network_path}/{G.gp.ID}.gt')
                
                network_props.loc[(G.gp['ID'],G.gp['probability']),'CC'] = sum((G.vp.local_clustering.get_array()))/len(G.get_vertices())
                network_props.loc[(G.gp['ID'],G.gp['probability']),'T'] = G.gp.transitivity 
                network_props.loc[(G.gp['ID'],G.gp['probability']),'SP'] = G.gp.get('shortest_path')  
                network_props.loc[(G.gp['ID'],G.gp['probability']),'l2'] = np.sort(G.vp.eig_laplacian.a)[1]
                network_props.loc[(G.gp['ID'],G.gp['probability']),'lmax_l2'] =  np.max(G.vp.eig_laplacian.a) / np.sort(G.vp.eig_laplacian.a)[1]
                network_props.loc[(G.gp['ID'],G.gp['probability']),'Rg'] =  n*np.sum(1/np.sort(G.vp.eig_laplacian.a)[1:])    
            network_props.to_csv(spec_file,sep='\t',mode='w',header=True)
            print(spec_file)
            
print('Done all!')
