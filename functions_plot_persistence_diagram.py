import sys
import os
computer = os.getlogin()
import pickle as pk
import networkx as nx
import numpy as np
import pylab as plt
import scipy.io
import seaborn as sns
sys.path.append('/home/%s/Software/'%computer) #IMPORTANT (path holes)
import Holes as ho
from functions_run_holes_complete import *

def plot_barcode(intervals=[(2,3), (5,6), (3,4),(2,-1)], ax=None,tit=None):
    """
    intervals: give list intervals (zip(h1_start,h1_end))
    ax: figure axis (fig.plt.figure(); ax = fig.add_subplot(1,1,1))
    tit: add title

    """
    import matplotlib.patches as patches
    max_time = max([s[0] for s in intervals]+[s[1] for s in intervals])
    total_height = 10.0
    delta_y = total_height/(len(intervals))

    for i in range(len(intervals)):
        s = intervals[i]
        if s[1]<0:
            s=(s[0],2*max_time)

        width = s[1]-s[0]
        height = delta_y
        y = i*delta_y
        ax.add_patch(patches.Rectangle((s[0],y), width, height))

    ax.plot((max_time,max_time),(0,total_height), color='red', linewidth=2.0)
    ax.set_ylim((0,total_height))
    ax.set_xlim((0,max_time+2))
    ax.set_xlabel('step')
    if(tit!=None):
        plt.title(tit)

def points_persistence_diagram(type_analysis,mode,dir_results,short_name,id_subject=None):
    """
    given a set of parameters to read generators return list of points of the persistence diagram
    id_subject : identify subject if we are in the subject analysis case
    """
    ## by subject
    if(type_analysis == 'subject'):
        p_start = [];p_end = [] 
        ### exception for lsd data ###
        if(short_name[:3]=='lsd'):
            # short_name[9:] equvilaent to lsd_type
            if(short_name[4]=='P'):## different pipelines
                cycles = nx.read_gpickle('%s/generators_%s_%i_.pck'%(dir_results,mode,id_subject))[1]
            else:
                cycles = nx.read_gpickle('%s/generators_%s_%s_%i_.pck'%(dir_results,mode,short_name[9:],id_subject))[1]
        elif(short_name[:3]=='hcp'):
            if(short_name[4]=='1'):
                cycles = nx.read_gpickle('%s/generators_HCP_ica100_%i_.pck'%(dir_results,id_subject))[1]
            else:
                cycles = nx.read_gpickle('%s/generators_HCP_ica50_%i_.pck'%(dir_results,id_subject))[1]
        elif(short_name[:3]=='bip'):
            cycles = nx.read_gpickle('%s/%s_holes/gen/generators_gen%i_1_.pck'%(dir_results,mode,id_subject))[1]
        else:            
            cycles = nx.read_gpickle('%s/generators_%s_%i_.pck'%(dir_results,mode,id_subject))[1]
        for c in cycles:
            p_start.append(float(c.start))
            p_end.append(float(c.end))
        return(zip(p_start,p_end))

    ## average
    if(type_analysis=='averaged'):
        p_start = [];p_end = []   
        ### exception for lsd data ###
        if(short_name[:3]=='lsd'):
            # short_name[9:] equvilaent to lsd_type        
            cycles = nx.read_gpickle('%s/generators_averaged-%s_%s_.pck'%(dir_results,mode,short_name[9:]))[1]
        else:
            # generators_bc1_Highs__
            if(os.path.exists('%sgenerators_%s__.pck'%(dir_results,mode))):
                cycles = nx.read_gpickle('%sgenerators_%s__.pck'%(dir_results,mode))[1] ## oju
            else:
                cycles = nx.read_gpickle('%sgenerators_%s_averaged_.pck'%(dir_results,mode))[1]
        for c in cycles:
            p_start.append(float(c.start))
            p_end.append(float(c.end))
        return(zip(p_start,p_end))
    return()


def points_persistence_diagram_h0(type_analysis,mode,dir_results,short_name,id_subject=None,with_start=False):
    """
    given a set of parameters to read generators return list of points of the persistence diagram
    id_subject : identify subject if we are in the subject analysis case
    with_start: False, we only return end points (death) because in H0 all start at step 0. with_start = True, we moreover return p_start filled by zeros (initial step for connected components)
    """
    ## by subject
    if(type_analysis == 'subject'):
        p_end = [] 
        ### exception for lsd data ###
        if(short_name[:3]=='lsd'):
            # short_name[9:] equvilaent to lsd_type
            cycles = nx.read_gpickle('%s/generators_%s_%s_%i_.pck'%(dir_results,mode,short_name[9:],id_subject))[0]
        elif(short_name[:3]=='hcp'):
            if(short_name[4]=='1'):
                cycles = nx.read_gpickle('%s/generators_HCP_ica100_%i_.pck'%(dir_results,id_subject))[0]
            else:
                cycles = nx.read_gpickle('%s/generators_HCP_ica50_%i_.pck'%(dir_results,id_subject))[0]
        else:            
            cycles = nx.read_gpickle('%s/generators_%s_%i_.pck'%(dir_results,mode,id_subject))[0]
        for c in cycles:
            p_end.append(float(c.end))
        if(with_start):
            p_start = [0]*len(p_end)
            return(zip(p_start,p_end))
        else:
            return(p_end)

    ## average
    if(type_analysis=='averaged'):
        p_end = []   
        ### exception for lsd data ###
        if(short_name[:3]=='lsd'):
            # short_name[9:] equvilaent to lsd_type        
            cycles = nx.read_gpickle('%s/generators_averaged-%s_%s_.pck'%(dir_results,mode,short_name[9:]))[0]
        else:
            if(os.path.exists('%sgenerators_%s__.pck'%(dir_results,mode))):
                cycles = nx.read_gpickle('%sgenerators_%s__.pck'%(dir_results,mode))[0] ## oju
            else:
                cycles = nx.read_gpickle('%sgenerators_%s_averaged_.pck'%(dir_results,mode))[0]
        for c in cycles:
            p_end.append(float(c.end))
        if(with_start):
            p_start = [0]*len(p_end)
            return(zip(p_start,p_end))
        else:
            return(p_end)
    return()

def create_diff_persistence_heatmaps(M_norm1,M_norm2,val_abs=True):
    """
    Given two heatmaps compute the difference. If abs = true -> absolute value, otherwise M_norm1-M_norm2
    """
    ### do difference 
    if(M_norm1.shape[0] != M_norm2.shape[0]):
        m1 = M_norm1.shape[0]; m2 = M_norm2.shape[1]
        if(m1<m2):
            M_new = np.zeros((m2,m2))
            M_new[:m1,:m1]=M_norm1
            if(val_abs):
                M_diff = abs(M_new-M_norm2)
            else:
                M_diff = M_new-M_norm2
        else:
            M_new = np.zeros((m1,m1))
            M_new[:m2,:m2]=M_norm2
            M_diff = abs(M_new-M_norm1)
    else:
        if(val_abs):
            M_diff = abs(M_norm1-M_norm2)
        else:
            M_diff = M_norm1-M_norm2
    return(M_diff)

def plot_and_create_persistence_diagram(type_analysis,num_subjects,modes,dir_results,short_name):
    """
    """
    if(type(num_subjects)==int):
        num_subjects = range(num_subjects)
    ## by subject
    if(type_analysis == 'subject'):
        plt.figure(figsize=(16,8))
        pmax = 0
        for mode in modes:
            p_start = [];p_end = []
            for i in num_subjects:
                cycles = nx.read_gpickle('%s/generators_%s_%i_.pck'%(dir_results,mode,i))[1]

                for c in cycles:
                    p_start.append(float(c.start))
                    p_end.append(float(c.end))

            if(mode==modes[0]):
                plt.subplot(121)          
                plt.plot(p_start,p_end,'or',label=modes[0],alpha = 0.5)
                plt.title(modes[0])
                pmax = max(pmax,max(p_end)+1)
                # plt.xlim((0,max(p_end)+1))
                # plt.ylim((0,max(p_end)+1))
            else:
                plt.subplot(122) 
                plt.plot(p_start,p_end,'ob',label=modes[1],alpha = 0.5)
                plt.title(modes[1])
                pmax = max(pmax,max(p_end)+1)
                # plt.xlim((0,max(p_end)+1))
                # plt.ylim((0,max(p_end)+1))
        plt.subplot(121)
        plt.xlim((0,pmax))
        plt.ylim((0,pmax))
        plt.subplot(122)
        plt.xlim((0,pmax))
        plt.ylim((0,pmax))
        
        plt.suptitle('Persistence diagram %s %s'%(short_name,type_analysis))
        plt.savefig('persistence_diagram_%s_%s.png'%(short_name,type_analysis))
    ## average
    if(type_analysis=='averaged'):
        plt.figure(figsize=(16,8))
        pmax=0
        for mode in modes:
            p_start = [];p_end = []
            if(os.path.exists('%sgenerators_%s__.pck'%(dir_results,mode))):
                cycles = nx.read_gpickle('%sgenerators_%s__.pck'%(dir_results,mode))[1] ## oju
            else:
                cycles = nx.read_gpickle('%sgenerators_%s_averaged_.pck'%(dir_results,mode))[1]

            for c in cycles:
                p_start.append(float(c.start))
                p_end.append(float(c.end))

            if(mode==modes[0]):
                plt.subplot(121)          
                plt.plot(p_start,p_end,'or',label=modes[0],alpha = 0.5)
                plt.title(modes[0])
                # plt.xlim((0,max(p_end)+1))
                # plt.ylim((0,max(p_end)+1))
                pmax = max(pmax,max(p_end)+1)
            else:
                plt.subplot(122) 
                plt.plot(p_start,p_end,'ob',label=modes[1],alpha = 0.5)
                plt.title(modes[1])
                # plt.xlim((0,max(p_end)+1))
                # plt.ylim((0,max(p_end)+1))
                pmax = max(pmax,max(p_end)+1)

        plt.subplot(121)
        plt.xlim((0,pmax))
        plt.ylim((0,pmax))
        plt.subplot(122)
        plt.xlim((0,pmax))
        plt.ylim((0,pmax))
        plt.suptitle('Persistence diagram %s %s'%(short_name,type_analysis))
        plt.savefig('persistence_diagram_%s_%s.png'%(short_name,type_analysis))
    print 'fig saved in '+str(os.getcwd())+'persistence_diagram_%s_%s.png'%(short_name,type_analysis)
    plt.show()
    return()

def persistence_heat_matrix(type_analysis,modes,num_subjects,dir_results,val_abs):
    """
    Assuming 2 modes (file1, file2)
    INPUT:
        file1,file2: file (with path) with holes cycles generators
    OUTPUT:
        persistence diagram converted to heat matrix for file1, file2 and the difference between these diagrams (diff between matrices)
    """
    if(type(num_subjects)==int):
        num_subjects = range(num_subjects)
    if(type_analysis=='subject'):
        print 'sub'
        mode = modes[0]
        p_start = [];p_end = []

        for i in num_subjects:
            cycles = nx.read_gpickle('%s/generators_%s_%i_.pck'%(dir_results,mode,i))[1]
            for c in cycles:
                p_start.append(float(c.start))
                p_end.append(float(c.end))
        M = np.zeros(((int(max(p_end))+1,int(max(p_end))+1)))
        for i in range(len(p_end)):
            M[int(p_start[i]),int(p_end[i])] = M[int(p_start[i]),int(p_end[i])] +1

        M_norm1 = M/sum(sum(M))
        mode = modes[1]
        p_start = [];p_end = []
        for i in num_subjects:
            cycles = nx.read_gpickle('%s/generators_%s_%i_.pck'%(dir_results,mode,i))[1]
            for c in cycles:
                p_start.append(float(c.start))
                p_end.append(float(c.end))
        M = np.zeros(((int(max(p_end))+1,int(max(p_end))+1)))
        for i in range(len(p_end)):
            M[int(p_start[i]),int(p_end[i])] = M[int(p_start[i]),int(p_end[i])] +1

        M_norm2 = M/sum(sum(M))

        ### difference with or without absolute value
        M_diff = create_diff_persistence_heatmaps(M_norm1,M_norm2,val_abs) 
    if(type_analysis=='averaged'):
        mode = modes[0]
        p_start = [];p_end = []
        if(os.path.exists('%sgenerators_%s__.pck'%(dir_results,mode))):
            cycles = nx.read_gpickle('%sgenerators_%s__.pck'%(dir_results,mode))[1] ## oju
        else:
            cycles = nx.read_gpickle('%sgenerators_%s_averaged_.pck'%(dir_results,mode))[1]
        for c in cycles:
            p_start.append(float(c.start))
            p_end.append(float(c.end))
        M = np.zeros(((int(max(p_end))+1,int(max(p_end))+1)))
        for i in range(len(p_end)):
            M[int(p_start[i]),int(p_end[i])] = M[int(p_start[i]),int(p_end[i])] +1

        M_norm1 = M/sum(sum(M))
        mode = modes[1]
        p_start = [];p_end = []
        if(os.path.exists('%sgenerators_%s__.pck'%(dir_results,mode))):
            cycles = nx.read_gpickle('%sgenerators_%s__.pck'%(dir_results,mode))[1] ## oju
        else:
            cycles = nx.read_gpickle('%sgenerators_%s_averaged_.pck'%(dir_results,mode))[1]
        for c in cycles:
            p_start.append(float(c.start))
            p_end.append(float(c.end))
        M = np.zeros(((int(max(p_end))+1,int(max(p_end))+1)))
        for i in range(len(p_end)):
            M[int(p_start[i]),int(p_end[i])] = M[int(p_start[i]),int(p_end[i])] +1

        M_norm2 = M/sum(sum(M))
        ### difference with or without absolute value
        M_diff = create_diff_persistence_heatmaps(M_norm1,M_norm2,val_abs) 

    return(M_norm1,M_norm2,M_diff)

def plot_persistence_heat_matrix(type_analysis,M_norm1,M_norm2,M_diff,val_abs,modes,short_name):
    """
    Given heat persistence diagram matrices, plot them.
    INPUT:
        M_norm1, M_norm2, M_diff: 3 heat persistence diagram matrices,
        title: end of the title to distinguish data
        end_savefile: end of the name to savefile to distinguish data
    """
    max_m = np.max(np.max(M_norm1),np.max(M_norm2))
    fig = plt.figure(figsize=(18,6))
    ax = fig.add_subplot(1,3,1)
    cax = ax.matshow(M_norm1,vmin=0,vmax=max_m,cmap='Oranges',interpolation='none')
    fig.colorbar(cax)
    ax.set_title('%s'%(modes[0]))
    ax = fig.add_subplot(1,3,2)
    cax = ax.matshow(M_norm2,vmin=0,vmax=max_m,cmap='Oranges',interpolation='none')
    fig.colorbar(cax)
    ax.set_title('%s'%(modes[1]))
    ax = fig.add_subplot(1,3,3)
    if(val_abs):
        cax = ax.matshow(M_diff,vmin=0,cmap='Blues',interpolation='none')
        ax.set_title('Difference (abs)') ## WITH VAL ABS
        fig.colorbar(cax)
        if(type_analysis=='subject'):
            fig.suptitle('Persistence diagram by subj %s'%short_name)
            fig.savefig('persistence_diagram_by_subj_heatmap-%s.png'%short_name)
            print 'fig saved in '+str(os.getcwd())+'persistence_diagram_by_subj_heatmap-%s.png'%short_name
        if(type_analysis=='averaged'):
            fig.suptitle('Persistence diagram average %s'%short_name)
            fig.savefig('persistence_diagram_average_heatmap-%s.png'%short_name)
            print 'fig saved in '+str(os.getcwd())+'persistence_diagram_average_heatmap-%s.png'%short_name
    else:
        print 'no abs'
        mm = max(map(abs,[np.max(M_diff),np.min(M_diff)]))
        cax = ax.matshow(M_diff,vmin=-mm,vmax=mm,cmap='bwr',interpolation='none')
        ax.set_title('Difference %s - %s'%(modes[0],modes[1]))
        fig.colorbar(cax)
        if(type_analysis=='subject'):
            fig.suptitle('Persistence diagram by subj %s'%short_name)
            fig.savefig('persistence_diagram_by_subj_heatmap1-%s.png'%short_name)
            print 'fig saved in '+str(os.getcwd())+'persistence_diagram_by_subj_heatmap1-%s.png'%short_name
        if(type_analysis=='averaged'):
            fig.suptitle('Persistence diagram average %s'%short_name)
            fig.savefig('persistence_diagram_average_heatmap1-%s.png'%short_name)
            print 'fig saved in '+str(os.getcwd())+'persistence_diagram_average_heatmap1-%s.png'%short_name
    plt.show()
    # return()

####################################################################################################
# betti number across filtration

def betti_numbers_filtration(type_analysis,mode,dir_results,short_name,id_subject=None):
    """
    given a set of parameters to read generators return list of points of the persistence diagram
    id_subject : identify subject if we are in the subject analysis case
    """
    ## by subject
    if(type_analysis == 'subject'):
        p_start = [];p_end = [] 
        ### exception for lsd data ###
        if(short_name[:3]=='lsd'):
            # short_name[9:] equvilaent to lsd_type
            cycles = nx.read_gpickle('%s/generators_%s_%s_%i_.pck'%(dir_results,mode,short_name[9:],id_subject))[1]
        elif(short_name[:3]=='hcp'):
            if(short_name[4]=='1'):
                cycles = nx.read_gpickle('%s/generators_HCP_ica100_%i_.pck'%(dir_results,id_subject))[1]
            else:
                cycles = nx.read_gpickle('%s/generators_HCP_ica50_%i_.pck'%(dir_results,id_subject))[1]
        else:            
            cycles = nx.read_gpickle('%s/generators_%s_%i_.pck'%(dir_results,mode,id_subject))[1]
        if(cycles != []):
            for c in cycles:
                p_start.append(int(float(c.start)))
                p_end.append(int(float(c.end)))

            max_step = max(p_end)
            betti1 = np.zeros(max_step+1)
            for i in range(len(cycles)):
                betti1[p_start[i]:p_end[i]+1] = betti1[p_start[i]:p_end[i]+1] +1 
        else:
            betti1 = np.zeros(1)
        p_start = [];p_end = [] 
        ### exception for lsd data ###
        if(short_name[:3]=='lsd'):
            # short_name[9:] equvilaent to lsd_type
            cycles = nx.read_gpickle('%s/generators_%s_%s_%i_.pck'%(dir_results,mode,short_name[9:],id_subject))[0]
        elif(short_name[:3]=='hcp'):
            if(short_name[4]=='1'):
                cycles = nx.read_gpickle('%s/generators_HCP_ica100_%i_.pck'%(dir_results,id_subject))[0]
            else:
                cycles = nx.read_gpickle('%s/generators_HCP_ica50_%i_.pck'%(dir_results,id_subject))[0]
        else:            
            cycles = nx.read_gpickle('%s/generators_%s_%i_.pck'%(dir_results,mode,id_subject))[0]
        if(cycles != []):
            for c in cycles:
                p_start.append(int(float(c.start)))
                p_end.append(int(float(c.end)))

            max_step = max(p_end)
            betti0 = np.zeros(max_step+1)
            for i in range(len(cycles)):
                betti0[p_start[i]:p_end[i]+1] = betti0[p_start[i]:p_end[i]+1] +1 
        else:
            betti0 = np.zeros(1)
        return(betti0,betti1)

    ## average
    if(type_analysis=='averaged'):
        p_start = [];p_end = []   
        ### exception for lsd data ###
        if(short_name[:3]=='lsd'):
            # short_name[9:] equvilaent to lsd_type        
            cycles = nx.read_gpickle('%s/generators_averaged-%s_%s_.pck'%(dir_results,mode,short_name[9:]))[1]
        else:
            if(os.path.exists('%sgenerators_%s__.pck'%(dir_results,mode))):
                cycles = nx.read_gpickle('%sgenerators_%s__.pck'%(dir_results,mode))[1] ## oju
            else:
                cycles = nx.read_gpickle('%sgenerators_%s_averaged_.pck'%(dir_results,mode))[1]
        for c in cycles:
            p_start.append(int(float(c.start)))
            p_end.append(int(float(c.end)))

        max_step = max(p_end)
        betti1 = np.zeros(max_step+1)
        for i in range(len(cycles)):
            betti1[p_start[i]:p_end[i]+1] = betti1[p_start[i]:p_end[i]+1] +1 

        p_start = [];p_end = []   
        ### exception for lsd data ###
        if(short_name[:3]=='lsd'):
            # short_name[9:] equvilaent to lsd_type        
            cycles = nx.read_gpickle('%s/generators_averaged-%s_%s_.pck'%(dir_results,mode,short_name[9:]))[0]
        else:
            if(os.path.exists('%sgenerators_%s__.pck'%(dir_results,mode))):
                cycles = nx.read_gpickle('%sgenerators_%s__.pck'%(dir_results,mode))[0] ## oju
            else:
                cycles = nx.read_gpickle('%sgenerators_%s_averaged_.pck'%(dir_results,mode))[0]
        for c in cycles:
            p_start.append(int(float(c.start)))
            p_end.append(int(float(c.end)))

        max_step = max(p_end)
        betti0 = np.zeros(max_step+1)
        for i in range(len(cycles)):
            betti0[p_start[i]:p_end[i]+1] = betti0[p_start[i]:p_end[i]+1] +1 
        return(betti0,betti1)

    return()


# cycle persistence

def cycle_persistence(type_analysis,mode,dir_results,short_name,id_subject=None):
    """
    given a set of parameters to read generators return list of points of the persistence diagram
    id_subject : identify subject if we are in the subject analysis case
    return persistence cycles by subject (for holes in H0 and holes in H1)
    """
    pers0 = [];pers1 = [] ### persistences for H0, persistences for H1
    ## by subject
    if(type_analysis == 'subject'):
        ### exception for lsd data ###
        if(short_name[:3]=='lsd'):
            # short_name[9:] equvilaent to lsd_type
            cycles = nx.read_gpickle('%s/generators_%s_%s_%i_.pck'%(dir_results,mode,short_name[9:],id_subject))[1]
        elif(short_name[:3]=='hcp'):
            if(short_name[4]=='1'):
                cycles = nx.read_gpickle('%s/generators_HCP_ica100_%i_.pck'%(dir_results,id_subject))[1]
            else:
                cycles = nx.read_gpickle('%s/generators_HCP_ica50_%i_.pck'%(dir_results,id_subject))[1]
        else:            
            cycles = nx.read_gpickle('%s/generators_%s_%i_.pck'%(dir_results,mode,id_subject))[1]
        for c in cycles:
            pers1.append(c.persistence_interval())

        ### exception for lsd data ###
        if(short_name[:3]=='lsd'):
            # short_name[9:] equvilaent to lsd_type
            cycles = nx.read_gpickle('%s/generators_%s_%s_%i_.pck'%(dir_results,mode,short_name[9:],id_subject))[0]
        elif(short_name[:3]=='hcp'):
            if(short_name[4]=='1'):
                cycles = nx.read_gpickle('%s/generators_HCP_ica100_%i_.pck'%(dir_results,id_subject))[0]
            else:
                cycles = nx.read_gpickle('%s/generators_HCP_ica50_%i_.pck'%(dir_results,id_subject))[0]
        else:            
            cycles = nx.read_gpickle('%s/generators_%s_%i_.pck'%(dir_results,mode,id_subject))[0]
        for c in cycles:
            pers0.append(c.persistence_interval())

        return(pers0,pers1)

    ## average
    if(type_analysis=='averaged'):  
        ### exception for lsd data ###
        if(short_name[:3]=='lsd'):
            # short_name[9:] equvilaent to lsd_type        
            cycles = nx.read_gpickle('%s/generators_averaged-%s_%s_.pck'%(dir_results,mode,short_name[9:]))[1]
        else:
            if(os.path.exists('%sgenerators_%s__.pck'%(dir_results,mode))):
                cycles = nx.read_gpickle('%sgenerators_%s__.pck'%(dir_results,mode))[1] ## oju
            else:
                cycles = nx.read_gpickle('%sgenerators_%s_averaged_.pck'%(dir_results,mode))[1]
        for c in cycles:
            pers1.append(c.persistence_interval())

        ### exception for lsd data ###
        if(short_name[:3]=='lsd'):
            # short_name[9:] equvilaent to lsd_type        
            cycles = nx.read_gpickle('%s/generators_averaged-%s_%s_.pck'%(dir_results,mode,short_name[9:]))[0]
        else:
            if(os.path.exists('%sgenerators_%s__.pck'%(dir_results,mode))):
                cycles = nx.read_gpickle('%sgenerators_%s__.pck'%(dir_results,mode))[0] ## oju
            else:
                cycles = nx.read_gpickle('%sgenerators_%s_averaged_.pck'%(dir_results,mode))[0]
        for c in cycles:
            pers0.append(c.persistence_interval())

        return(pers0,pers1)

    return()
