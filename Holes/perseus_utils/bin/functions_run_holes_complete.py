### needed functions to run holes/javaplex
import sys
import os
import pickle as pk
import networkx as nx
import numpy as np
import scipy.io
computer = os.getlogin()
sys.path.append('/home/%s/Software/'%computer) #IMPORTANT (path holes)
import Holes as ho

def relabel_to_int(G):
    """
    relabel str to int (scaffolds are labeled by str)
    """
    relabel = {}
    for n in G.nodes():
        relabel[n] = eval(n);
    G = nx.relabel_nodes(G,relabel)
    return(G)



def read_original_subject_graph(files,num_nodes_data,select_matrix):
    """
    Given list of files that contains a matrix we return the averaged matrix
    INPUT:
        - files: list of files
        - num_nodes_data: shape square matrix
        - select_matrix: auxiliar name to read matrices in files
    OUTPUT:
        - return an averaged matrix 
    """
    M = np.zeros((num_nodes_data,num_nodes_data))
    for file_matrix in files:
        if(select_matrix!=None):
            mat = np.matrix(scipy.io.loadmat(file_matrix)[select_matrix])
        else:
            mat = np.matrix(scipy.io.loadmat(file_matrix))
        M = M+mat
    M=M/len(files)
    return(M)


def read_generators_scaff(mat,G,type_analysis,output_folder,name_output_holes,index_graph=None):
    """
    Given parameters to read output file of holes/javaplex and the initial graph from where we have computed homology (after "bin" the graph, return bined graph (input to holes) and dictionary of cycles for dim 0 and 1 obtained from holes.
    Input:
        - mat (read correlation matrix)
        - G: binned network and converted to distance network
        - output_folder,name_output_holes: path to be able to load results obtained from holes
        - type_analysis: by subject or averaged
        - index_graph: if we are in type_analysis by subject we need to indicate in which index we are
    Output:
        - G (networkx graph)
        - wgen1 (dict) as keys dim 0 and dim 1 in homology that contains lists of Holes.classes.cycle.Cycle, contains output holes
    """
    print 'type_analysis ',type_analysis
    if(type_analysis == 'by_subject' and index_graph!=None):
        # print '1'
        file_gen = output_folder+'gen/generators_'+name_output_holes+'_%i_.pck'%index_graph
    if(type_analysis == 'by_subject' and index_graph==None):
        # print '2'
        file_gen = output_folder+'gen/generators_'+name_output_holes+'_.pck'
    if(type_analysis == 'averaged'):
        # print '3'
        file_gen = output_folder+'gen/generators_'+name_output_holes+'_.pck'
    gen1 = pk.load(open(file_gen))
    wgen1 = ho.weight_label_generators(gen1,G,ascending=True)
    return(G,wgen1)




def read_generators_scaff_adhd(mat,G,type_analysis,output_folder,name_output_holes):
    """
    Given parameters to read output file of holes/javaplex and the initial graph from where we have computed homology (after "bin" the graph, return bined graph (input to holes) and dictionary of cycles for dim 0 and 1 obtained from holes.
    Input:
        - mat (read correlation matrix)
        - G: binned network and converted to distance network
        - output_folder,name_output_holes: path to be able to load results obtained from holes
        - type_analysis: by subject or averaged
        - index_graph: if we are in type_analysis by subject we need to indicate in which index we are
    Output:
        - G (networkx graph)
        - wgen1 (dict) as keys dim 0 and dim 1 in homology that contains lists of Holes.classes.cycle.Cycle, contains output holes
    """
    print 'type_analysis ',type_analysis
    if(type_analysis == 'by_subject'):
        file_gen = output_folder+'gen/generators_'+name_output_holes+'_.pck'
    if(type_analysis == 'averaged'):
        file_gen = output_folder+'gen/generators_'+name_output_holes+'_.pck'
    gen1 = pk.load(open(file_gen))
    wgen1 = ho.weight_label_generators(gen1,G,ascending=True)
    return(G,wgen1)


def create_graph_bined(mat,thresholds):
    """
    Given a mat file, convert to a graph, remove selfloops and "bine" weights in order to have a smaller filtration
    Input: 
        - mat: correlation matrix from matlab
        - thresholds: bins to bined weights graph (list)
    Output: 
        - bined graph (metworkx graph)
    """
    G=nx.from_numpy_matrix(mat)
    # remove possible selfloops in the original graph
    sl = G.selfloop_edges() ## selfloop edges
    G.remove_edges_from(sl)

    ### convert to distance graph and agrouping thresholds
    for e in G.edges():
        w = abs(G[e[0]][e[1]]['weight'])
        index = search_index(thresholds,np.array(w))
        if(1-thresholds[index]>0):
            G[e[0]][e[1]]['weight'] = 1-thresholds[index]
        else:
            G[e[0]][e[1]]['weight'] = 0.000000000001
    return(G)

###### filter: group weights
def search_index(thresholds,point):
    """
    Given a threshold (th_i) list and a point (p) select index (i) such that th_i < p < th_{i+1}
    Input:
        - thresholds (list)
        - point (number)
    Output:
        - Index (i) thresholds st th_i < p < th_{i+1}
    """
    a = list(point <= thresholds)
    index = a.index(True)
    if(index == 0):
        return(index)
    else:
        return(index-1)
#####################################
### for density ###
#####################################
def create_graph_bined_density(mat,thresholds):
    """
    Given a mat file, convert to a graph, remove selfloops and "bine" weights in order to have a smaller filtration
    Input: 
        - mat: correlation matrix from matlab
        - thresholds: bins to bined weights graph (list)
    Output: 
        - bined graph (metworkx graph)
    """
    G=nx.from_numpy_matrix(mat)
    # remove possible selfloops in the original graph
    sl = G.selfloop_edges() ## selfloop edges
    G.remove_edges_from(sl)

    ### convert to distance graph and agrouping thresholds
    for e in G.edges():
        w = abs(G[e[0]][e[1]]['weight'])
        index = search_index_density(thresholds,np.array(w))
        if(1-thresholds[index]>0):
            G[e[0]][e[1]]['weight'] = 1-thresholds[index]
        else:
            G[e[0]][e[1]]['weight'] = 0.000000000001
    return(G)

###### filter: group weights
def search_index_density(thresholds,point):
    """
    Given a threshold (th_i) list and a point (p) select index (i) such that th_i < p < th_{i+1}
    Input:
        - thresholds (list)
        - point (number)
    Output:
        - Index (i) thresholds st th_i < p < th_{i+1}
    """
    a = list(point <= thresholds)
    index = a.index(True)
    return(index)
#####################################

def create_persistent_scaffold(cycles):
    """
    Given output from javaplex/holes and applied 'ho.weight_label_generators' of homology 1 we construct the graph that has as edge if it appears in cycles obtained by javaplex with a weight equal as times that it appears
    Input:
        - cycles: list of Holes.classes.cycle.Cycle (output from holes)
    Output:
        - Frequency Scaffold graph (networkx): edges that appear in cycles weighted by their frequency
    """
    S = nx.Graph()
    for i in range(len(cycles)):
        c = cycles[i]
        p = c.persistence_interval()
        edges = c.composition
        for e in edges:
            if(S.has_edge(*e)):
                S[e[0]][e[1]]['weight'] = S[e[0]][e[1]]['weight'] + p
            else:
                S.add_edge(*e)
                S[e[0]][e[1]]['weight'] =p
    return(S)

def create_freq_scaffold(cycles):
    """
    Given output from javaplex/holes and applied 'ho.weight_label_generators' of homology 1 we construct the graph that has as edge if it appears in cycles obtained by javaplex with a weight equal as sum of its persistence (for each time that it appears in a cycle)
    Input:
        - cycles: list of Holes.classes.cycle.Cycle (output from holes)
    Output:
        - Persistent Scaffold graph (networkx): edges that appear in cycles weighted by sum of their persistence
    """
    S = nx.Graph()
    for i in range(len(cycles)):
        c = cycles[i]
        edges = c.composition
        for e in edges:
            if(S.has_edge(*e)):
                S[e[0]][e[1]]['weight'] = S[e[0]][e[1]]['weight'] + 1
            else:
                S.add_edge(*e)
                S[e[0]][e[1]]['weight'] =1
    return(S)


def create_scaff_matrix(cycles,scaff,num_nodes,with_scaff=False):
    """
    Given cycles (H1) obtained from holes/javaplex and type of scaffold that we want make, we return the corresponding scaffold. Fixed shape matrix scaff to 165 nodes.
    Input:
        - cycles = wgen1[1] (list of Holes.classes.cycle.Cycle)
        - scaff ('freq','persistent') (freq: weights = frequency edge, persistent: weights = sum persistence edge)
    Output:
        - M: half-matrix, indices are scaffold nodes, values are weights in its corresponding scaffold matrix
        - M_completa:  complete matrix (just M but with all entries, M_completa is symmetric)
    """
    if(scaff=='persistent'):
        S = create_persistent_scaffold(cycles)
    else:
        S = create_freq_scaffold(cycles)
    M = np.zeros((num_nodes,num_nodes))
    M_completa = np.zeros((num_nodes,num_nodes))
    for e in S.edges():
        e1 = eval(e[0]); e2 = eval(e[1])
        M_completa[e1,e2] = S[e[0]][e[1]]['weight']
        M_completa[e2,e1] = S[e[0]][e[1]]['weight']
        if(e1 <= e2):
            M[e1,e2] = S[e[0]][e[1]]['weight']
        else:
            M[e2,e1] = S[e[0]][e[1]]['weight']
    if(with_scaff):
        return(M,M_completa,S)
    else:
        return(M,M_completa)


def nodal_strength(G):
    """
    Equivalent computation to PSS (at persistent scaff)
    """
    aa =nx.adjacency_matrix(G)
    if(type(aa) == scipy.sparse.csr.csr_matrix):
        aa = aa.todense()
    s = sum(np.array(aa))
    nodes = G.nodes()
    node_strength = dict()
    for i in range(len(nodes)):
        node_strength[int(nodes[i])] = s[i]
    return(node_strength)

## for hcp, sleep redo
def create_thresholds_from_zcorr(thresholds_aux,mat):
    """
    given a zcorr matrix create bins using percentiles
    """
    thresholds = []
    for th in thresholds_aux:
        thresholds.append(np.percentile(mat,th*100))
    return(thresholds)

def create_graph_bined_z(mat,thresholds,max_a):
    """
    Given a mat file, convert to a graph, remove selfloops and "bine" weights in order to have a smaller filtration
    Input: 
        - mat: matrix from matlab
        - thresholds: bins to bined weights graph (list)
        - max_a: max zcorr matrix
    Output: 
        - bined graph (networkx graph) with "distances" instead of "correlations" as weights
    """
    G=nx.from_numpy_matrix(mat)
    # remove possible selfloops in the original graph
    sl = G.selfloop_edges() ## selfloop edges
    G.remove_edges_from(sl)

    ### convert to distance graph and agrouping thresholds
    for e in G.edges():
        w = G[e[0]][e[1]]['weight']
        index = search_index(thresholds,np.array(w))
        if(max_a - thresholds[index]>0):
            G[e[0]][e[1]]['weight'] = max_a - thresholds[index]
        else:
            G[e[0]][e[1]]['weight'] = 0.000000000001
    return(G)
