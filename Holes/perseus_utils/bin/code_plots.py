import os, sys
computer = os.getlogin() # get current username
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx 
import itertools
from scipy.stats import linregress,f_oneway,kruskal
import seaborn as sns ## import or not depending the plot
import pandas as pd
import pickle as pk
import matplotlib.lines as mlines
from functions_plot_persistence_diagram import * 



folder_gen = '/hypnosis-high_low/gen/' ## 18 Highs, 19 Lows
folder_scaff = '/hypnosis-high_low/scaffolds/' ## 18 Highs, 19 Lows

groups = ['Highs','Lows']
files = os.listdir(folder_gen)
filtered_files = {'Highs':[],'Lows':[]}
tasks = []
ids = {'Highs':[],'Lows':[]}

for f in files:
    g = f.split('_')[2]
    id_subj = f.split('_')[3]
    task = f.split('_')[1]
    if(id_subj!=''):
        filtered_files[g].append(f)
        ids[g].append(id_subj)
        tasks.append(task)

for g in ids:
    ids[g]=set(ids[g])

attributes = {k:[] for k in itertools.product(groups,set(tasks))}
## we have as many as samples as subjects
for k in itertools.product(groups,set(tasks)):
    g = k[0] ## group 
    task = k[1]
    p_start = [];p_end = []; pers = []; len_cycle = []
    for i in ids[g]:
        f ='generators_%s_%s_%s_.pck'%(task,g,i)
        cycles = nx.read_gpickle('%s/%s'%(folder_gen,f))[1]
        for c in cycles:
            p_start.append(float(c.start))
            p_end.append(float(c.end))
            pers.append(c.persistence_interval())
            len_cycle.append(len(c.composition))
    attributes[(g,task),'birth'] = p_start
    attributes[(g,task),'death'] = p_end
    attributes[(g,task),'persistence'] = pers
    attributes[(g,task),'len_cycle'] = len_cycle

## how many nodes we have?
num_nodes_data = 32

a = scipy.io.loadmat('/hypnosis-high_low/newchanlocs_short.mat')['chanlocs']
channels = []
X = []; Y = []; Z = []
## there is an error in the data, and X-> Y, Y->X
for i in range(32):
    channels.append(str(a[0][i][0][0]))
    ### new x->y, y -> x
    Y.append(a[0][i][4][0][0])
    X.append(a[0][i][5][0][0])
    Z.append(a[0][i][6][0][0])

## channel classification by region:
channels_region=dict()
channels_region['frontal']=['FP1','FP2','F7','F3','F4','F8']
channels_region['med-anterior']=['FC3','FC4','FT7','FT8','T3','T4']
channels_region['med-posterior']=[ 'C3','C4','CP4','CP3','TP7','TP8']
channels_region['occipital']=['T6','T5','P3','P4','PO1','PO2','O1','O2']
channels_region['central'] = ['FZ','FCZ','CPZ','OZ','CZ','PZ']

num_regions = len(channels_region.keys())
regions_names = channels_region.keys()

## new dict
regions = dict()
for k in channels_region.keys():
    for j in channels_region[k]:
        regions[j] = k


######################################### scaffolds ############################
groups = ['Highs','Lows']
files = os.listdir(folder_scaff)
with_zeros =True## if false, we only consider !=0 values in scaff, if true we add zeros when the sacff is empty or some nodes dont have pss because are not in the scaff.

num_nodes = 32
for g in ids:
    ids[g]=set(ids[g])

labels = list(itertools.product(groups,['brt','rt']))
regions_names_extended = list(itertools.combinations(regions_names,2))+regions_names

G = nx.complete_graph(32)
e_keys = G.edges()

# dict_edges = {e:[] for e in e_keys}
dict_scaff = dict()
dict_nodal_strenght = dict()
# import os.path
for k in labels:
    dict_edges = {e:[] for e in e_keys}
    dict_aux_nodal = {n:[] for n in range(num_nodes)}
    g = k[0] ## group 
    task = k[1]
    for i in ids[g]:
        # name = folder_scaff+'%s_%s_scaff_persistent_%s.gpickle'%(task,g,i)
        # print os.path.isfile(name) 
        scaff = relabel_to_int(nx.read_gpickle(folder_scaff+'%s_%s_scaff_persistent_%s.gpickle'%(task,g,i)))
        if(scaff.nodes()==[]):
            if(with_zeros):
                aux_n ={ss:0 for ss in range(num_nodes)} ## to add dict of zeros for each node in the case that scaff is null
            else:
                continue
        else:
            aux_n = nodal_strength(scaff)
            if(with_zeros):
                ## complete nodal strenght equal to zero in the case that a node doesnt appear in a scaff
                for ss in range(num_nodes):
                    if aux_n.has_key(ss) == False:
                        aux_n[ss]=0

        for n in aux_n:
            dict_aux_nodal[n].append(aux_n[n])
        for e in scaff.edges(data=True):
            if(e[0]>e[1]):
                dict_edges[(e[1],e[0])].append(e[2]['weight'])
            else:
                dict_edges[(e[0],e[1])].append(e[2]['weight'])
    dict_scaff[k] = dict_edges
    dict_nodal_strenght[k] = dict_aux_nodal


###################################################################################################
### grouping by regions ###
## nodes 
dict_nodal_strenght_region = dict()
for k in dict_nodal_strenght.keys():
    aux_dict = {kk:[] for kk in channels_region.keys()}
    aux = dict_nodal_strenght[k]
    for i in aux:
        aux_dict[regions[channels[i]]] = aux_dict[regions[channels[i]]] + aux[i]
    dict_nodal_strenght_region[k] = aux_dict

## edges (intra and extra regions)
dict_pers = {}
for k in labels:
    scaff = dict_scaff[k]
    regions_persistence = {k: [] for k in regions_names_extended} ## create dict new for each netwroks / scaff 
    for e in scaff.keys():
        r1 = regions[channels[e[0]]]; r2 = regions[channels[e[1]]]
        if(r1 == r2):
            regions_persistence[r1] = regions_persistence[r1] +scaff[e]
        else:
            if(regions_persistence.has_key((r1,r2))):
                regions_persistence[(r1,r2)] = regions_persistence[(r1,r2)] + scaff[e]
            else:
                regions_persistence[(r2,r1)] = regions_persistence[(r2,r1)] +scaff[e]
    dict_pers[k] = regions_persistence

distance_rr_brr = []
for j in range(num_nodes):
    data = []
    for k in dict_nodal_strenght:
        data.append(dict_nodal_strenght[k][j])
    data = map(np.array,data)


    dist_h = np.linalg.norm(data[0]-data[1])
    dist_l = np.linalg.norm(data[2]-data[3])
    distance_rr_brr.append((dist_h,dist_l))


################### PAPER PLOT cutting ylim and legend #########################################
## FIGURE 2
import matplotlib.patches as mpatches
plt.style.use('default')
regions_names_order_enrica = ['frontal', 'med-anterior', 'med-posterior',  'occipital', 'central']

plt.figure(figsize=(18,7))
labels = list(itertools.product(groups,['rh','brh'])) ## rt -> rh

for i,j in enumerate(regions_names_order_enrica):
    data = []
    for k in dict_nodal_strenght_region.keys():
        data.append(dict_nodal_strenght_region[k][j])
        print k
    plt.subplot(1,5,i+1)
    if(i==0):
        plt.ylabel('nodal strength',fontsize=15)
    ax = sns.boxplot(data= data, palette="Set3", showfliers=False) ## to hide outliers
    # plt.xticks(plt.xticks()[0],labels,fontsize=10,rotation=40)
    plt.xticks([])
    plt.title('region %s'%(j),fontsize=15)
    plt.ylim((0,.5))


from matplotlib import cm
lab = ['Highs: rh', 'Highs: brh', 'Lows: rh', 'Lows: brh']


colors = [cm.Set3(0),cm.Set3(128),cm.Set3(192),cm.Set3(64)]
p = []
for i in range(4):
    patch = mpatches.Patch(color=colors[i], label=lab[i])
    p.append(patch)
plt.legend(handles=p)

plt.suptitle('Hyp HL dataset Nodal strength by regions',fontsize=20)
# plt.savefig('Hyp_HL_nodal_strength_by_region_with_zeros1_legend.pdf')
plt.show()

## FIGURE 3
############ PAPER ENRICA (2nd) plot distance, same than above but colours in node 6,25 (H), 22,30 (L) ############
plt.style.use('default')
for label, x, y in zip(channels, [d[0] for d in distance_rr_brr], [d[1] for d in distance_rr_brr]):
    p1 = (x,y)
    p2 = ((x+y)/2.,(x+y)/2.)
    dist = np.sqrt( (p2[0] - p1[0])**2 + (p2[1] - p1[1])**2 )## distance to the diagonal
    if(label in ['F8','P4']):
        plt.plot(x,y,'or',markersize=10+dist*30,alpha=.5)
    if(label in ['T5','PO1']):
        plt.plot(x,y,'ob',markersize=10+dist*30,alpha=.5)
    else:
        plt.plot(x,y,'o',markersize=10+dist*30,alpha=.5,markeredgewidth=1,markeredgecolor='g',markerfacecolor='None')

    print label, dist
    plt.annotate(
        label,
        xy=(x, y))
plt.plot(([0,0],[1,1]),'k.-')
plt.xlabel(r'$d_{euc}(brh,rh)$ Highs',fontsize=18)
plt.ylabel(r'$d_{euc}(brh,rh)$ Lows',fontsize=18)
plt.title('Euclidean distance between brh and rh: H vs L by channel',fontsize=15)
## add legend 
blue_line = mlines.Line2D([], [], color='blue', marker='o', markersize=15, alpha=.5,label='L significant')
red_line = mlines.Line2D([], [], color='red', marker='o', markersize=15,alpha=.5, label='H significant')
plt.legend(handles=[blue_line,red_line],loc=0,fontsize=10,frameon=False) ## frameon= False when style = default

# plt.savefig('distance_brh_rh_Highs_vs_Lows_by_channel_sizepoints.png')
# plt.savefig('distance_brh_rh_Highs_vs_Lows_by_channel_sizepoints.pdf')
plt.show()