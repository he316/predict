# -*- coding: utf-8 -*-
"""
## weight: weight of edge /2
##training: module genes
#pbmc3k
"""

import os
from os import listdir
import pandas as pd
import scipy.stats as sc
import numpy as np
import seaborn as sns
from sklearn.neural_network import MLPClassifier
from sklearn import svm
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.linear_model  import LogisticRegression
from sklearn.linear_model import LinearRegression
from sklearn import ensemble, preprocessing, metrics
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
plt.switch_backend('agg')
from sklearn.ensemble import RandomForestClassifier
from random import sample
import csv
from sklearn.model_selection import cross_val_score
import matplotlib.ticker as ticker
from scipy.stats import norm
import sklearn
from sklearn.metrics import average_precision_score, auc, roc_curve, precision_recall_curve
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import plot_precision_recall_curve
from sklearn.metrics import plot_roc_curve
import networkx as nx
import re
import anndata 
from copy import copy
import scvelo as scv
from sklearn.neighbors import NearestNeighbors
from scipy.stats import norm as normal
import gc
import scanpy as scp

#from memory_profiler import profile
#暫時保留,再加入cross validation就是randomforest_cross 
def randomforest_predict(posX,negX,data):       
    train_X = posX+negX
    train_Y = [1]*len(posX)+[0]*len(negX)
        
    clf = RandomForestClassifier(n_estimators=100, max_depth=2, random_state=0)
    clf.fit(train_X, train_Y)
    predict_list = list(clf.predict(data))
    print (clf.score(train_X, train_Y))
    return predict_list

#keep
#[[]] to [] 

def Dimensionality_reduction(data):
    dist = []
    for i in data:
        for j in i:
            dist.append(round(j, 3))
    return dist

#keep
def mapping(data, lable):
    lable_list = []
    WT = data.index.tolist()
    for i in WT:
        lable_list.append(lable['Cluster'][i])
    return lable_list

#keep
def change_color(predict_list, color0, color1):
    new_list = []
    for i in predict_list:
        if i == 0:
            new_list.append(color0)
        elif i == 1:
            new_list.append(color1)
    return new_list

#keep

def draw_cor_heat_map(pos_data, neg_data, col_name_list, pos_figure_name, neg_figure_name):
    #positive組的圖
    sns.set(font_scale=2)
    sns.set_style("ticks")
    pos_data = pos_data[col_name_list]
    pos_gene = []
    #
    for col1 in col_name_list:
        gene_gene = []
        X = pos_data[col1].tolist()
        for col2 in col_name_list:
            try :
                Y = pos_data[col2].tolist()
            except:
                print(col2)
                return
            gene_gene.append(sc.pearsonr(X, Y)[0])
        gene_gene = pd.Series(gene_gene)
        gene_gene.fillna(0, inplace=True)
        pos_gene.append(gene_gene)
    gene = pd.DataFrame(pos_gene)
    gene.index=col_name_list
    gene.columns=col_name_list
    plt.figure(figsize=(15,15))
    cmap = sns.diverging_palette(220, 10, sep=10, n=40)
    pos_figure = sns.heatmap(gene, cbar_kws={'ticks': [-1, -0.8, -0.6, -0.4, -0.2, 0.0, 0.2, 0.4, 0.6, 0.8, 1.0]}, vmin=-1, vmax=1, cmap=cmap)
    plt.savefig(pos_figure_name,   # 儲存圖檔
                bbox_inches='tight',               # 去除座標軸占用的空間
                pad_inches=0.0)                    # 去除所有白邊
    fig=plt.gcf()
    plt.close(fig)
    #negative組的圖，軸由positive決定    
    
    neg_data = neg_data[col_name_list]
    neg_gene = []
    for col1 in col_name_list:
        gene_gene = []
        X = neg_data[col1].tolist()
        for col2 in col_name_list:
            try :
                Y = neg_data[col2].tolist()
            except:
                print(col2)
                return
            gene_gene.append(sc.pearsonr(X, Y)[0])
        gene_gene = pd.Series(gene_gene)
        gene_gene.fillna(0, inplace=True)
        neg_gene.append(gene_gene)
    gene = pd.DataFrame(neg_gene)
    gene.index=col_name_list
    gene.columns=col_name_list
    plt.figure(figsize=(15,15))
    sns.set(font_scale=2)
    sns.set_style("ticks")
    cmap = sns.diverging_palette(220, 10, sep=10, n=40)
    neg_figure = sns.heatmap(gene, cbar_kws={'ticks': [-1, -0.8, -0.6, -0.4, -0.2, 0.0, 0.2, 0.4, 0.6, 0.8, 1.0]}, vmin=-1, vmax=1, cmap=cmap)
    plt.savefig(neg_figure_name,   # 儲存圖檔
                bbox_inches='tight',               # 去除座標軸占用的空間
                pad_inches=0.0)
    fig=plt.gcf()
    plt.close(fig)
    return [pos_gene, neg_gene]
### random forest
def predict_package(module_gene_list, pos_sample, neg_sample, data, count_cluster_list, preservation_255_list,  title, save_name_255):
    pos_m = pos_sample[module_gene_list]
    neg_m = neg_sample[module_gene_list]
    train_pos = pos_m.values.tolist()
    train_neg = neg_m.values.tolist()
    data_rd = data[module_gene_list]
    data_X = data_rd.values.tolist()
    predict_list = randomforest_cross(train_pos, train_neg, data_X)
    
    #算所有cluster的pos率
    num_list = []
    for i in count_cluster_list:
        num = count_number(i, predict_list, count_cluster_list)
        pri = num[0]/num[1]
        num_list.append(pri)
        
    plt.figure()
    plt.title(title, fontsize=20)
    plt.xlabel('preservation Z score', fontsize=20)
    plt.ylabel('positive', fontsize=20)
    plt.scatter(preservation_255_list, num_list)
#    g = sns.regplot(preservation_255_list, num_list)
    plt.ylim(0, 1)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    ax = plt.gca()
    ax.spines['top'].set_linewidth(2)
    ax.spines['bottom'].set_linewidth(2)
    ax.spines['left'].set_linewidth(2)
    ax.spines['right'].set_linewidth(2)
    plt.savefig(save_name_255, bbox_inches = 'tight')
    
#    plt.figure()
#    plt.title(title)
#    plt.xlabel('preservation Z score')
#    plt.ylabel('positive')
#    plt.scatter(preservation_ALL_list, num_list)
#    plt.savefig(save_name_ALL)
    #改成dictionary或二維list,一個裝predict_list,另一個裝num_list
    #num_list
    return predict_list

def randomforest_cross(posX,negX,data):    
    train_X = posX+negX
    train_Y = [1]*len(posX)+[0]*len(negX)
        
    clf = RandomForestClassifier(n_estimators=100, max_depth=2, random_state=0)
    clf.fit(train_X, train_Y)
    predict_list = list(clf.predict(data))
    scores = cross_val_score(clf, train_X, train_Y, cv=5)
    print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() ))#* 2))
#    print (clf.score(train_X, train_Y))
    return predict_list
#算指定cluster內準確度

def count_number(cluster_number, preservation_Z_score, cell_cluster,column_name):
    all_num = 0
    pos_num = 0
    #從cell_cluster中去找每個細胞對應的cluster,再從preservation_Z_score中去尋找對應的prediction結果
    #才能計算positive率
    for cell_count in range(0,len(cell_cluster)):        
        #第cell_count個細胞的'leiden'存的是cluster,所以要比較一次
        if cell_cluster[column_name][cell_count]== cluster_number:
            all_num+=1
            if preservation_Z_score[cell_count] == 1:#是positive的次數
                pos_num+=1
    return (pos_num, all_num)#每個細胞去查看
        
        

### random forest
def predict_packagecopy1(module_gene_list,
                         pos_sample,
                         neg_sample, 
                         cell_data, 
                         cell_cluster, 
                         Cluster, 
                         preservation_Z_score, 
                         modulename, 
                         save_name_255):
    pos_m = pos_sample[module_gene_list]
    neg_m = neg_sample[module_gene_list]
    train_pos = pos_m.values.tolist()
    train_neg = neg_m.values.tolist()
    data_rd = cell_data[module_gene_list]
    data_X = data_rd.values.tolist()
    result = randomforest_crosscopy1(train_pos, train_neg, data_X)
    
    #算所有cluster的pos率
    positive_rate = []
    for i in Cluster:
        num = count_number(i, result['predict_list'], cell_cluster) #count_number
        pri = num[0]/num[1]
        positive_rate.append(pri)
    
    pltdata={"preservation_Z_score": preservation_Z_score,
            "positive_rate":positive_rate,
            "Cluster":Cluster}
    #print( predict_list)
    #print(positive_rate)
    #print(Cluster)
    pltdata=pd.DataFrame(pltdata)
    ## pd.dataframe    
    pltdata["ClusterStr"]=list(map(str, pltdata["Cluster"]))
    sns.lmplot(data=pltdata,x='preservation_Z_score', y='positive_rate', hue='Cluster',
                   fit_reg=False, legend=True, legend_out=True,size=9)
    plt.title(modulename+" Accuracy: %0.2f (+/- %0.2f)" % (result['ACC_mean_std'][0], result['ACC_mean_std'][1] * 2), fontsize=16)
    for i, label in enumerate(Cluster):
        
    #loop through data points and plot each point 
        for l, row in pltdata.loc[pltdata['Cluster']==label,:].iterrows():
        
            #add the data point as text
            plt.annotate(str(row['ClusterStr']), 
                         (row['preservation_Z_score'], row['positive_rate']),
                         horizontalalignment='center',
                         verticalalignment='center',
                         size=11,
                         )
    
    plt.savefig(save_name_255,bbox_inches='tight',               # 去除座標軸占用的空間
                pad_inches=0.0)
    ##回傳值已改成dictionary,一個裝 predict_list,另一個裝ACC的mean跟std
    #後面需要num_list
    #predict_list會標註說哪一些細胞被判斷成positive,哪一些細胞被判斷成negtive,
    #copy1 ver:num_list已經改成positive_rate
    #          preservation_255_list已經改成 preservation_Z_score
    #
    return result

### 在 predictpackage裡面 #randomforest
def randomforest_crosscopy1(posX,negX,data):
    train_X = posX+negX
    train_Y = [1]*len(posX)+[0]*len(negX)
        
    clf = RandomForestClassifier(n_estimators=100, max_depth=2, random_state=0)
    clf.fit(train_X, train_Y)
    predict_list = list(clf.predict(data))
    scores = cross_val_score(clf, train_X, train_Y, cv=5)
    #print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
    #0
    result={
    'predict_list':predict_list,
    'ACC_mean_std':[scores.mean(),(scores.std() * 2)]
    }
#    print (clf.score(train_X, train_Y))
    return  result

def drawlmplot_annotation(pltdata,cluster,Xaxis,Yaxis,Hue,savepath):
    sns.lmplot(data=pltdata, x=Xaxis, y=Yaxis, hue=Hue,fit_reg=False, legend=True, legend_out=True,size=14)
    for i, label in enumerate(range(0,len(cluster))):
            
        #loop through data points and plot each point 
            for l, row in pltdata.loc[pltdata[Hue]==label,:].iterrows():
            
                #add the data point as text
                plt.annotate(int(row[Hue]), 
                             (row[Xaxis], row[Yaxis]),
                             horizontalalignment='center',
                             verticalalignment='center',
                             size=11,
                             )
    plt.savefig(savepath, bbox_inches='tight',pad_inches=0.0)
    fig=plt.gcf()
    plt.close(fig)


def positive_rate(Cluster,Predictlist,cell_UMAP,column_name):
    pr=[]
    for i in Cluster:
        num = count_number(i, Predictlist, cell_UMAP,column_name) #由於在count_number中指定傳入的dataframe的第一行為cluster,所以只能用另一個dataframe
        pri = num[0]/num[1]
        pr.append(pri)
    return pr


def find_boundary_cluster(cell_UMAP_cluster,Predictlist,Cluster,column_name,save_name_barplot):
    sns.set(font_scale=2)
    sns.set_style("ticks")
    pr=positive_rate(Cluster, Predictlist, cell_UMAP_cluster,column_name)
    pltdata={"positive_rate":pr,
        column_name:Cluster}    
    sns.set(font_scale=2)
    sns.set_style("ticks")
    plt.figure(figsize=(20,6))
    splot = sns.barplot(data=pltdata, x = column_name, y = 'positive_rate', ci = None)
    for p in splot.patches:
        splot.annotate(format(p.get_height(), '.2f'), (p.get_x() + p.get_width() / 2., p.get_height()), ha = 'center', va = 'center', xytext = (0, 10), textcoords = 'offset points')
    plt.xlabel(column_name, size=14)
    plt.ylabel("Positive_Rate", size=14)
    plt.savefig(save_name_barplot,bbox_inches='tight',               # 去除座標軸占用的空間
                pad_inches=0.0)
    fig=plt.gcf()
    plt.close(fig)
    bdc=[]
    for index,element in enumerate(pr):
        if element<0.9 and element>0.1:
            bdc.append(index)
    return bdc


def prediction_and_ploting(Adata,#adata
                           cell_data,#cell_data
                           cell_UMAP_cluster,#cell_UMAP_cluster
                           cell_leiden_cluster,#cell_leiden_cluster
                           Clustermethod,#Clustermethod
                           clustering,#clustering
                           POS_cluster,#POS_cluster
                           NEG_cluster,#NEG_cluster#module_savedir
                           moduleColor,
                           savedir,
                           module_savedir):
    sns.set(font_scale=2)
    sns.set_style("ticks")
    Module_cluster_Zscore = pd.read_csv(os.path.join(savedir+str(POS_cluster), 'module_preservation_Zscore.csv'),index_col=0)
    Module_cluster_Zscore = Module_cluster_Zscore.drop(columns=['gold', 'grey'])

    POS=pd.read_csv(savedir+str(POS_cluster)+'.csv',index_col=0)
    NEG=pd.read_csv(savedir+str(NEG_cluster)+'.csv',index_col=0)

    target_Module=pd.read_csv(os.path.join(savedir+str(POS_cluster), 'modules', moduleColor+'.csv'),index_col=0)
    target_Module_genes=list(target_Module.index)
    target_module_genes_set= set(target_Module_genes)
    edges = pd.read_table(os.path.join(savedir+str(POS_cluster), 'modules', Clustermethod+'_cluster_'+str(POS_cluster)+'edges.txt'))
    
    
    #nodes = pd.read_table(savedir+str(POS_cluster)+'/modules/'+'Leiden_cluster_'+str(POS_cluster)+'nodes.txt')
    #target_module_genes_set=set()
    #for i in range(len(nodes.index)):
    #    if nodes['nodeAttr[nodesPresent, ]'][i] == moduleColor:
    #        target_module_genes_set.add(nodes['nodeName'][i])
    #target_Module_genes=list(target_module_genes_set)    
    #weightthreshold=edges.weight.median()
    filtered_edges=pd.DataFrame(columns=['fromNode','toNode','weight'])
    
    for i in range(len(edges.index)):
        if edges.fromNode[i] in target_module_genes_set and edges.toNode[i] in target_module_genes_set:
            filtered_edges.loc[len(filtered_edges.index)]=[edges.fromNode[i],edges.toNode[i],edges.weight[i]]
    target_module_genes_set
    target_Module_c_list=set()    
    #篩選edge權重
    for i in range(len(filtered_edges.index)):    
        if (filtered_edges.weight[i] >= (filtered_edges.weight.max()/2)) :#權重在這裡
            target_Module_c_list.add(filtered_edges.fromNode[i])
            target_Module_c_list.add(filtered_edges.toNode[i])
    target_Module_c_list=list(target_Module_c_list)
    #del(edges,filtered_edges)
        
    if len(POS.index) >= len(NEG.index):
        min_sample_count=len(NEG.index)
    else:
        min_sample_count=len(POS.index)

    POS_training=POS.sample(n=min_sample_count, axis=0)
    NEG_training=NEG.sample(n=min_sample_count, axis=0)
                                #第一個引數輸入要訓練的feature
    result = predict_packagecopy2(target_Module_genes,
                               POS_training, 
                               NEG_training, 
                               cell_data, #cell data, after preprocessing!!,cell-gene matrix
                               cell_UMAP_cluster,#細胞對應cluster
                               list(range(0,len(Module_cluster_Zscore.index))),#cluster列表 
                               list(Module_cluster_Zscore[moduleColor]), 
                               str(moduleColor),
                               os.path.join(module_savedir, moduleColor+'_PC'+str(POS_cluster)+'_NC'+str(NEG_cluster)+'.png'),
                               cell_leiden_cluster,
                               os.path.join(module_savedir, "barplot_of_pos_rate_on_leiden_res2.png")
                              )
    predict_list=result['predict_list']    
    #作圖loop區域,後續的圖
    pos_cluster_neg_cluster_cell_list=[]
    for i in range(len(cell_UMAP_cluster)):
        if cell_UMAP_cluster[clustering][i] == POS_cluster:
            pos_cluster_neg_cluster_cell_list.append('positive_cluster_'+str(POS_cluster))
        elif cell_UMAP_cluster[clustering][i] == NEG_cluster:
            pos_cluster_neg_cluster_cell_list.append('negative_cluster_'+str(NEG_cluster))
        else:
            pos_cluster_neg_cluster_cell_list.append('other')
    
    cell_UMAP_cluster['training_clusters']=pos_cluster_neg_cluster_cell_list
        
    sns.lmplot(data=cell_UMAP_cluster, x='UMAP1', y='UMAP2', hue='training_clusters',fit_reg=False, legend=True, legend_out=True,size=14,hue_order=(sorted(list(set(cell_UMAP_cluster['training_clusters'])))))
    plt.savefig(os.path.join(module_savedir, 'training_cluster_UMAP.png'), bbox_inches='tight',pad_inches=0.0)
    fig=plt.gcf()
    plt.close(fig)
    #predict_listtest_to_color = change_color(predict_listtest, '#808080', '#FF0000')
    cell_UMAP_cluster['prediction']=predict_list
    #標示出predict為pos和neg的細胞分布 UMAP
    #SAVE!
    sns.lmplot(data=cell_UMAP_cluster, x='UMAP1', y='UMAP2', hue='prediction',fit_reg=False, legend=True, legend_out=True,size=14)
    plt.savefig(os.path.join(module_savedir, 'positive_UMAP.png'), bbox_inches='tight',               # 去除座標軸占用的空間
                pad_inches=0.0)
    fig=plt.gcf()
    plt.close(fig)
    boundaryC=find_boundary_cluster(cell_UMAP_cluster, predict_list, cell_leiden_cluster, 'leiden', os.path.join(module_savedir, "barplot_of_pos_rate_on_leiden_res2.png"))
    boundaryUMAP=cell_UMAP_cluster[cell_UMAP_cluster.leiden.isin(boundaryC)]
    boundaryCstr=[str(int) for int in boundaryC]
    bdata=Adata.copy()
    bdata=bdata[bdata.obs.leiden.isin(boundaryC)]
    boundaryUMAP=cell_UMAP_cluster[cell_UMAP_cluster.leiden.isin(boundaryC)]
    
    try:
        velocity_stream_scatterplt(adata=bdata, pltdata=boundaryUMAP, x='UMAP1', y='UMAP2', hue='latent_time', palette='turbo', style='prediction',
                                   style_order=[1,0],density=1,bbox_to_anchor_x=1.3,savename=os.path.join(module_savedir, 'positive_UMAP_boundary_annotated.png'))
    except:
        pass
    velocity_stream_scatterplt(adata=Adata, pltdata=boundaryUMAP, x='UMAP1', y='UMAP2', hue='latent_time', palette='turbo', style='prediction',
                               style_order=[1,0],density=1,bbox_to_anchor_x=1.3,savename=os.path.join(module_savedir, 'global_stream_positive_UMAP_boundary_annotated.png'))
    try:
        g=plt.figure(figsize=(10,10),dpi=150)
        g=sns.violinplot(boundaryUMAP['prediction'],boundaryUMAP['latent_time'])
        plt.savefig(os.path.join(module_savedir, 'leiden_positive_boxplot.png'), bbox_inches='tight',pad_inches=0.0)
        plt.close()
    except:
        pass
    #del(boundaryC,boundaryCstr,boundaryUMAP)
    
    boundaryC=find_boundary_cluster(cell_UMAP_cluster, predict_list, list(set(cell_UMAP_cluster['fixed_time_latent_time_group'])),'fixed_time_latent_time_group',os.path.join(module_savedir, "barplot_of_pos_rate_on_fixed_time_latent_time_group.png"))
    boundaryUMAP=cell_UMAP_cluster[cell_UMAP_cluster.fixed_time_latent_time_group.isin(boundaryC)]
    boundaryCstr=[str(int) for int in boundaryC]
    bdata=Adata.copy()
    bdata=bdata[bdata.obs.fixed_time_latent_time_group.isin(boundaryC)]
    boundaryUMAP=cell_UMAP_cluster[cell_UMAP_cluster.fixed_time_latent_time_group.isin(boundaryC)]
    
    try:
        velocity_stream_scatterplt(adata=bdata,pltdata=boundaryUMAP,x='UMAP1',y='UMAP2',hue='latent_time',palette='turbo',
                                   style='prediction',style_order=[1,0],density=1,bbox_to_anchor_x=1.3,savename=os.path.join(module_savedir, 'fixed_time_latent_time_group_positive_UMAP_boundary_annotated.png'))
    except:
        pass
    velocity_stream_scatterplt(adata=Adata,pltdata=boundaryUMAP,x='UMAP1',y='UMAP2',hue='latent_time',palette='turbo',
                               style='prediction',style_order=[1,0],density=1,bbox_to_anchor_x=1.3,savename=os.path.join(module_savedir, 'fixed_time_latent_time_group_global_stream_positive_UMAP_boundary_annotated.png'))
    try:
        g=plt.figure(figsize=(10,10),dpi=150)
        g=sns.violinplot(boundaryUMAP['prediction'],boundaryUMAP['latent_time'])
        plt.savefig(os.path.join(module_savedir, 'fixed_time_latent_time_group_positive_boxplot.png'), bbox_inches='tight',pad_inches=0.0)
        plt.close()
    except:
        pass
    #del(boundaryC,boundaryCstr,boundaryUMAP)
    
    
    boundaryC=find_boundary_cluster(cell_UMAP_cluster, predict_list, list(set(cell_UMAP_cluster['fixed_cell_number_latent_time_group'])),'fixed_cell_number_latent_time_group',os.path.join(module_savedir, "barplot_of_pos_rate_on_fixed_cell_number_latent_time_group.png"))
    boundaryUMAP=cell_UMAP_cluster[cell_UMAP_cluster.fixed_cell_number_latent_time_group.isin(boundaryC)]
    boundaryCstr=[str(int) for int in boundaryC]
    bdata=Adata.copy()
    bdata=bdata[bdata.obs.fixed_cell_number_latent_time_group.isin(boundaryC)]
    boundaryUMAP=cell_UMAP_cluster[cell_UMAP_cluster.fixed_cell_number_latent_time_group.isin(boundaryC)]
    
    try:
        velocity_stream_scatterplt(adata=bdata,pltdata=boundaryUMAP,x='UMAP1',y='UMAP2',hue='latent_time',palette='turbo',
                                   style='prediction',style_order=[1,0],density=1,bbox_to_anchor_x=1.3,savename=os.path.join(module_savedir, 'fixed_cell_number_latent_time_group_positive_UMAP_boundary_annotated.png'))
    except:
        pass
    velocity_stream_scatterplt(adata=Adata,pltdata=boundaryUMAP,x='UMAP1',y='UMAP2',hue='latent_time',palette='turbo',
                               style='prediction',style_order=[1,0],density=1,bbox_to_anchor_x=1.3,savename=os.path.join(module_savedir, 'fixed_cell_number_latent_time_group_global_stream_positive_UMAP_boundary_annotated.png'))
    try:
        g=plt.figure(figsize=(10,10),dpi=150)
        g=sns.violinplot(boundaryUMAP['prediction'],boundaryUMAP['latent_time'])
        plt.savefig(os.path.join(module_savedir, 'fixed_cell_number_latent_time_group_positive_boxplot.png'), bbox_inches='tight',pad_inches=0.0)
        plt.close()
    except:
        pass
    #del(boundaryC,boundaryCstr,boundaryUMAP)
    #gc.collect()
    
    #POS cluster module gene expression (hub gene) heatmap 
    #plt.figure(figsize=(10,10))
    #sns.heatmap(POS_training[target_Module_c_list], vmax=6)
    
    ####whole module genes heatmap
    moduleGenePosHeatmap=sns.clustermap(data=(POS_training[target_Module_genes].T),xticklabels=False,yticklabels=True,
               figsize=((10+16*len(target_Module_genes)/40),(10+9*len(target_Module_genes)/30)),method='ward')
    fixed_gene_list = list(moduleGenePosHeatmap.data2d.index)
    plt.savefig(os.path.join(module_savedir, 'pos_module_gene_heatmap.png'), bbox_inches='tight',pad_inches=0.0)
    fig=plt.gcf()
    plt.close(fig)    
    NEG_moduleGeneOrdered=NEG_training[target_Module_genes]
    NEG_moduleGeneOrdered=NEG_moduleGeneOrdered.reindex(columns=fixed_gene_list)
    
    plt.figure(figsize=((10+16*len(target_Module_genes)/40),(10+9*len(target_Module_genes)/30)))
    moduleGeneNegHeatmap=sns.heatmap(data=NEG_moduleGeneOrdered.T,xticklabels=False,yticklabels=True)
    plt.savefig(os.path.join(module_savedir, 'neg_module_gene_heatmap.png'), bbox_inches='tight',pad_inches=0.0)    
    fig=plt.gcf()
    plt.close(fig)
    #訓練資料
 
    #training data (whole module gene) PCC map
    pos_neg_cor = draw_cor_heat_map(POS_training, NEG_training, fixed_gene_list,
                               os.path.join(module_savedir, 'training_module_PCC_heatmap_PC'+str(POS_cluster)+'module_'+moduleColor+'.png'),
                               os.path.join(module_savedir, 'training_module_PCC_heatmap_NC'+str(NEG_cluster)+'module_'+moduleColor+'.png'))
    pos_pcc_list = Dimensionality_reduction(pos_neg_cor[0])
    neg_pcc_list = Dimensionality_reduction(pos_neg_cor[1])
    
    
    jcounter=0
    for listresult in pos_pcc_list:
        if listresult == 1:
            jcounter+=1
         
    for i in range(jcounter):
        pos_pcc_list.remove(1)
        
    jcounter=0
    for listresult in neg_pcc_list:
        if listresult == 1:
            jcounter+=1
            
    for i in range(jcounter):
        neg_pcc_list.remove(1)
    
    plt.figure(figsize=(10,10))
    sns.set(font_scale=1.5)
    sns.set_style("ticks")
    sns.distplot(neg_pcc_list, label='negative training')
    sns.distplot(pos_pcc_list, label='positive training')
    plt.xlabel('PCC', fontsize=20)
    plt.ylabel('Density', fontsize=20)
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(module_savedir, 'training_module_PCC.png'),   # 儲存圖檔
                bbox_inches='tight',               # 去除座標軸占用的空間
                pad_inches=0.0)
    fig=plt.gcf()
    plt.close(fig)    
    gene_data = cell_data[target_Module_genes]
    
    pos = gene_data[gene_data.index.isin(cell_UMAP_cluster[cell_UMAP_cluster['prediction']==1].index)]
    neg = gene_data[gene_data.index.isin(cell_UMAP_cluster[cell_UMAP_cluster['prediction']==0].index)]
    
    #breakpoint()
    #提醒!要把檔名的cluster拿掉
    
    
    
    
    ##predict結果的PCC
    #prediction result data PCC map
    
    pos_neg_cor = draw_cor_heat_map(pos, neg, fixed_gene_list,
                               os.path.join(module_savedir, 'prediction_module_PCC_heatmap_pos'+str(POS_cluster)+'.png'),
                               os.path.join(module_savedir, 'prediction_module_PCC_heatmap_neg'+str(NEG_cluster)+'.png'))
    
    #breakpoint()
    pos_pcc_list = Dimensionality_reduction(pos_neg_cor[0])
    neg_pcc_list = Dimensionality_reduction(pos_neg_cor[1])
    
    ## 
    jcounter=0
    for listresult in pos_pcc_list:
        if listresult == 1:
            jcounter+=1     
    for i in range(jcounter):
        pos_pcc_list.remove(1)
        
    jcounter=0
    for listresult in neg_pcc_list:
        if listresult == 1:
            jcounter+=1        
    for i in range(jcounter):
        neg_pcc_list.remove(1)
    
    plt.figure(figsize=(10,10))
    sns.set(font_scale=1.5)
    sns.set_style("ticks")
    sns.distplot(neg_pcc_list, label='negative prediction')
    sns.distplot(pos_pcc_list, label='positive prediction')
    plt.xlabel('PCC', fontsize=20)
    plt.ylabel('Density', fontsize=20)
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(module_savedir, 'prediction_module_PCC.png'),   # 儲存圖檔
                bbox_inches='tight',               # 去除座標軸占用的空間
                pad_inches=0.0)
    fig=plt.gcf()
    plt.close(fig)
    #del(gene_data,moduleGeneNegHeatmap,moduleGenePosHeatmap,neg,NEG_moduleGeneOrdered,neg_pcc_list,NEG_training,pos,pos_cluster_neg_cluster_cell_list,pos_neg_cor,pos_pcc_list,POS_training,target_Module)
    #gc.collect()
    #breakpoint()
    ACC_PCC=result['ACC_mean_std']
    ACC_PCC.append(result['PCC(positive_rate,preservation_Z_score)'])
    
    return ACC_PCC


def prediction_and_ploting_semi(Adata,#adata
                           cell_data,#cell_data
                           cell_UMAP_cluster,#cell_UMAP_cluster
                           cell_leiden_cluster,#cell_leiden_cluster
                           Clustermethod,#Clustermethod
                           clustering,#clustering
                           POS_cluster,#POS_cluster
                           NEG_cluster,#NEG_cluster#module_savedir
                           moduleColor,
                           savedir,
                           module_savedir):
    sns.set(font_scale=2)
    sns.set_style("ticks")
    Module_cluster_Zscore = pd.read_csv(os.path.join(savedir+str(POS_cluster), 'module_preservation_Zscore.csv'),index_col=0)
    Module_cluster_Zscore = Module_cluster_Zscore.drop(columns=['gold', 'grey'])

    POS=pd.read_csv(savedir+str(POS_cluster)+'.csv',index_col=0)
    NEG=pd.read_csv(savedir+str(NEG_cluster)+'.csv',index_col=0)

    target_Module=pd.read_csv(os.path.join(savedir+str(POS_cluster), 'modules', moduleColor+'.csv'),index_col=0)
    target_Module_genes=list(target_Module.index)
    target_module_genes_set= set(target_Module_genes)
    edges = pd.read_table(os.path.join(savedir+str(POS_cluster), 'modules', Clustermethod+'_cluster_'+str(POS_cluster)+'edges.txt'))
    
    
    #nodes = pd.read_table(savedir+str(POS_cluster)+'/modules/'+'Leiden_cluster_'+str(POS_cluster)+'nodes.txt')
    #target_module_genes_set=set()
    #for i in range(len(nodes.index)):
    #    if nodes['nodeAttr[nodesPresent, ]'][i] == moduleColor:
    #        target_module_genes_set.add(nodes['nodeName'][i])
    #target_Module_genes=list(target_module_genes_set)    
    #weightthreshold=edges.weight.median()
    filtered_edges=pd.DataFrame(columns=['fromNode','toNode','weight'])
    
    for i in range(len(edges.index)):
        if edges.fromNode[i] in target_module_genes_set and edges.toNode[i] in target_module_genes_set:
            filtered_edges.loc[len(filtered_edges.index)]=[edges.fromNode[i],edges.toNode[i],edges.weight[i]]
    target_module_genes_set
    target_Module_c_list=set()    
    #篩選edge權重
    for i in range(len(filtered_edges.index)):    
        if (filtered_edges.weight[i] >= (filtered_edges.weight.max()/2)) :#權重在這裡
            target_Module_c_list.add(filtered_edges.fromNode[i])
            target_Module_c_list.add(filtered_edges.toNode[i])
    target_Module_c_list=list(target_Module_c_list)
    #del(edges,filtered_edges)
        
    if len(POS.index) >= len(NEG.index):
        min_sample_count=len(NEG.index)
    else:
        min_sample_count=len(POS.index)

    POS_training=POS.sample(n=min_sample_count, axis=0)
    NEG_training=NEG.sample(n=min_sample_count, axis=0)
                                #第一個引數輸入要訓練的feature
    result = predict_packagecopy2(target_Module_genes,
                               POS_training, 
                               NEG_training, 
                               cell_data, #cell data, after preprocessing!!,cell-gene matrix
                               cell_UMAP_cluster,#細胞對應cluster
                               list(range(0,len(Module_cluster_Zscore.index))),#cluster列表 
                               list(Module_cluster_Zscore[moduleColor]), 
                               str(moduleColor),
                               os.path.join(module_savedir, moduleColor+'_PC'+str(POS_cluster)+'_NC'+str(NEG_cluster)+'.png'),
                               cell_leiden_cluster,
                               os.path.join(module_savedir, "barplot_of_pos_rate_on_leiden_res2.png")
                              )
    predict_list=result['predict_list']    
    #作圖loop區域,後續的圖
    pos_cluster_neg_cluster_cell_list=[]
    for i in range(len(cell_UMAP_cluster)):
        if cell_UMAP_cluster[clustering][i] == POS_cluster:
            pos_cluster_neg_cluster_cell_list.append('positive_cluster_'+str(POS_cluster))
        elif cell_UMAP_cluster[clustering][i] == NEG_cluster:
            pos_cluster_neg_cluster_cell_list.append('negative_cluster_'+str(NEG_cluster))
        else:
            pos_cluster_neg_cluster_cell_list.append('other')
    
    cell_UMAP_cluster['training_clusters']=pos_cluster_neg_cluster_cell_list
        
    sns.lmplot(data=cell_UMAP_cluster, x='UMAP1', y='UMAP2', hue='training_clusters',fit_reg=False, legend=True, legend_out=True,size=14,hue_order=(sorted(list(set(cell_UMAP_cluster['training_clusters'])))))
    plt.savefig(os.path.join(module_savedir, 'training_cluster_UMAP.png'), bbox_inches='tight',pad_inches=0.0)
    fig=plt.gcf()
    plt.close(fig)
    #predict_listtest_to_color = change_color(predict_listtest, '#808080', '#FF0000')
    cell_UMAP_cluster['prediction']=predict_list
    #標示出predict為pos和neg的細胞分布 UMAP
    #SAVE!
    sns.lmplot(data=cell_UMAP_cluster, x='UMAP1', y='UMAP2', hue='prediction',fit_reg=False, legend=True, legend_out=True,size=14)
    plt.savefig(os.path.join(module_savedir, 'positive_UMAP.png'), bbox_inches='tight',               # 去除座標軸占用的空間
                pad_inches=0.0)
    fig=plt.gcf()
    plt.close(fig)
    functionOption_boundary=''
    while functionOption_boundary!=0:
        functionOption_boundary=int(input('all cell type and certain latent time group activation scatterplot:1\n, '+
                                      'certain cell type activation and latent time groups activation  scatterplot:2\n'+
                                      'quit:0\n'))
        if functionOption_boundary==1:            
            boundaryC=find_boundary_cluster(cell_UMAP_cluster, predict_list, list(set(cell_UMAP_cluster['fixed_time_latent_time_group'])),'fixed_time_latent_time_group',os.path.join(module_savedir, "barplot_of_pos_rate_on_fixed_time_latent_time_group.png"))
            print('enter fixed time latent time groups for display:\n')
            try:
                boundaryC = []
              
                while True:
                    boundaryC.append(int(input()))
                      
                # if the input is not-integer, just print the list
            except:
                print(boundaryC)
            boundaryUMAP=cell_UMAP_cluster[cell_UMAP_cluster.fixed_time_latent_time_group.isin(boundaryC)]
            boundaryCstr=[str(int) for int in boundaryC]
            bdata=Adata.copy()
            bdata=bdata[bdata.obs.fixed_time_latent_time_group.isin(boundaryC)]
            boundaryUMAP=cell_UMAP_cluster[cell_UMAP_cluster.fixed_time_latent_time_group.isin(boundaryC)]    
            try:
                velocity_stream_scatterplt(adata=bdata,pltdata=boundaryUMAP,x='UMAP1',y='UMAP2',hue='latent_time',palette='turbo',
                                           style='prediction',style_order=[1,0],density=1,bbox_to_anchor_x=1.3,savename=os.path.join(module_savedir, 'fixed_time_latent_time_group_positive_UMAP_boundary_annotated.png'))
            except:
                pass
            velocity_stream_scatterplt(adata=Adata,pltdata=boundaryUMAP,x='UMAP1',y='UMAP2',hue='latent_time',palette='turbo',
                                       style='prediction',style_order=[1,0],density=1,bbox_to_anchor_x=1.3,savename=os.path.join(module_savedir, 'fixed_time_latent_time_group_global_stream_positive_UMAP_boundary_annotated.png'))
            try:
                g=plt.figure(figsize=(10,10),dpi=150)
                g=sns.violinplot(boundaryUMAP['prediction'],boundaryUMAP['latent_time'])
                plt.savefig(os.path.join(module_savedir, 'fixed_time_latent_time_group_positive_boxplot.png'), bbox_inches='tight',pad_inches=0.0)
                plt.close()
            except:
                pass
            
        if functionOption_boundary==2:
            
            boundaryC=find_boundary_cluster(cell_UMAP_cluster, predict_list, list(set(cell_UMAP_cluster['fixed_time_latent_time_group'])),'fixed_time_latent_time_group',os.path.join(module_savedir, "barplot_of_pos_rate_on_fixed_time_latent_time_group.png"))
            print('enter fixed time latent time groups for display:\n')
            try:
                boundaryC = []
              
                while True:
                    boundaryC.append(int(input()))
                      
                # if the input is not-integer, just print the list
            except:
                print(boundaryC)
            celltypeC=find_boundary_cluster(cell_UMAP_cluster, predict_list, list(set(cell_UMAP_cluster['cell_type'])),'cell_type',os.path.join(module_savedir, "barplot_of_pos_rate_on_fixed_time_latent_time_group.png"))
            print('enter cell types for display:\n')
            try:
                celltypeC = []
              
                while True:
                    celltypeC.append(int(input()))
                      
                # if the input is not-integer, just print the list
            except:
                print(celltypeC)
            
            boundaryUMAP=cell_UMAP_cluster[cell_UMAP_cluster.cell_type.isin(celltypeC)]
            boundaryCstr=[str(int) for int in boundaryC]
            bdata=Adata.copy()
            bdata=bdata[bdata.obs.cell_type.isin(celltypeC)]
            bdata=bdata[bdata.obs.fixed_time_latent_time_group.isin(boundaryC)]
            boundaryUMAP=boundaryUMAP[boundaryUMAP.fixed_time_latent_time_group.isin(boundaryC)]    
            try:
                velocity_stream_scatterplt(adata=bdata,pltdata=boundaryUMAP,x='UMAP1',y='UMAP2',hue='latent_time',palette='turbo',
                                           style='prediction',style_order=[1,0],density=1,bbox_to_anchor_x=1.3,savename=os.path.join(module_savedir, 'fixed_time_latent_time_group_positive_UMAP_boundary_annotated.png'))
            except:
                pass
            velocity_stream_scatterplt(adata=Adata,pltdata=boundaryUMAP,x='UMAP1',y='UMAP2',hue='latent_time',palette='turbo',
                                       style='prediction',style_order=[1,0],density=1,bbox_to_anchor_x=1.3,savename=os.path.join(module_savedir, 'fixed_time_latent_time_group_global_stream_positive_UMAP_boundary_annotated.png'))
            try:
                g=plt.figure(figsize=(10,10),dpi=150)
                g=sns.violinplot(boundaryUMAP['prediction'],boundaryUMAP['latent_time'])
                plt.savefig(os.path.join(module_savedir, 'fixed_time_latent_time_group_positive_boxplot.png'), bbox_inches='tight',pad_inches=0.0)
                plt.close()
            except:
                pass
    if 1>0:  #Draw heatmap
        ####whole module genes heatmap
        moduleGenePosHeatmap=sns.clustermap(data=(POS_training[target_Module_genes].T),xticklabels=False,yticklabels=True,
                   figsize=((10+16*len(target_Module_genes)/40),(10+9*len(target_Module_genes)/30)),method='ward')
        fixed_gene_list = list(moduleGenePosHeatmap.data2d.index)
        plt.savefig(os.path.join(module_savedir, 'pos_module_gene_heatmap.png'), bbox_inches='tight',pad_inches=0.0)
        fig=plt.gcf()
        plt.close(fig)    
        NEG_moduleGeneOrdered=NEG_training[target_Module_genes]
        NEG_moduleGeneOrdered=NEG_moduleGeneOrdered.reindex(columns=fixed_gene_list)
        
        plt.figure(figsize=((10+16*len(target_Module_genes)/40),(10+9*len(target_Module_genes)/30)))
        moduleGeneNegHeatmap=sns.heatmap(data=NEG_moduleGeneOrdered.T,xticklabels=False,yticklabels=True)
        plt.savefig(os.path.join(module_savedir, 'neg_module_gene_heatmap.png'), bbox_inches='tight',pad_inches=0.0)    
        fig=plt.gcf()
        plt.close(fig)
        #訓練資料
     
        #training data (whole module gene) PCC map
        pos_neg_cor = draw_cor_heat_map(POS_training, NEG_training, fixed_gene_list,
                                   os.path.join(module_savedir, 'training_module_PCC_heatmap_PC'+str(POS_cluster)+'module_'+moduleColor+'.png'),
                                   os.path.join(module_savedir, 'training_module_PCC_heatmap_NC'+str(NEG_cluster)+'module_'+moduleColor+'.png'))
        pos_pcc_list = Dimensionality_reduction(pos_neg_cor[0])
        neg_pcc_list = Dimensionality_reduction(pos_neg_cor[1])
        
        
        jcounter=0
        for listresult in pos_pcc_list:
            if listresult == 1:
                jcounter+=1
             
        for i in range(jcounter):
            pos_pcc_list.remove(1)
            
        jcounter=0
        for listresult in neg_pcc_list:
            if listresult == 1:
                jcounter+=1
                
        for i in range(jcounter):
            neg_pcc_list.remove(1)
        
        plt.figure(figsize=(10,10))
        sns.set(font_scale=1.5)
        sns.set_style("ticks")
        sns.distplot(neg_pcc_list, label='negative training')
        sns.distplot(pos_pcc_list, label='positive training')
        plt.xlabel('PCC', fontsize=20)
        plt.ylabel('Density', fontsize=20)
        plt.xticks(fontsize=15)
        plt.yticks(fontsize=15)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(module_savedir, 'training_module_PCC.png'),   # 儲存圖檔
                    bbox_inches='tight',               # 去除座標軸占用的空間
                    pad_inches=0.0)
        fig=plt.gcf()
        plt.close(fig)    
        gene_data = cell_data[target_Module_genes]
        
        pos = gene_data[gene_data.index.isin(cell_UMAP_cluster[cell_UMAP_cluster['prediction']==1].index)]
        neg = gene_data[gene_data.index.isin(cell_UMAP_cluster[cell_UMAP_cluster['prediction']==0].index)]
        
        #breakpoint()
        #提醒!要把檔名的cluster拿掉
        
        
        
        
        ##predict結果的PCC
        #prediction result data PCC map
        
        pos_neg_cor = draw_cor_heat_map(pos, neg, fixed_gene_list,
                                   os.path.join(module_savedir, 'prediction_module_PCC_heatmap_pos'+str(POS_cluster)+'.png'),
                                   os.path.join(module_savedir, 'prediction_module_PCC_heatmap_neg'+str(NEG_cluster)+'.png'))
        
        #breakpoint()
        pos_pcc_list = Dimensionality_reduction(pos_neg_cor[0])
        neg_pcc_list = Dimensionality_reduction(pos_neg_cor[1])
        
        ## 
        jcounter=0
        for listresult in pos_pcc_list:
            if listresult == 1:
                jcounter+=1     
        for i in range(jcounter):
            pos_pcc_list.remove(1)
            
        jcounter=0
        for listresult in neg_pcc_list:
            if listresult == 1:
                jcounter+=1        
        for i in range(jcounter):
            neg_pcc_list.remove(1)
        
        plt.figure(figsize=(10,10))
        sns.set(font_scale=1.5)
        sns.set_style("ticks")
        sns.distplot(neg_pcc_list, label='negative prediction')
        sns.distplot(pos_pcc_list, label='positive prediction')
        plt.xlabel('PCC', fontsize=20)
        plt.ylabel('Density', fontsize=20)
        plt.xticks(fontsize=15)
        plt.yticks(fontsize=15)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(module_savedir, 'prediction_module_PCC.png'),   # 儲存圖檔
                    bbox_inches='tight',               # 去除座標軸占用的空間
                    pad_inches=0.0)
        fig=plt.gcf()
        plt.close(fig)
    #del(gene_data,moduleGeneNegHeatmap,moduleGenePosHeatmap,neg,NEG_moduleGeneOrdered,neg_pcc_list,NEG_training,pos,pos_cluster_neg_cluster_cell_list,pos_neg_cor,pos_pcc_list,POS_training,target_Module)
    #gc.collect()
    #breakpoint()
    ACC_PCC=result['ACC_mean_std']
    ACC_PCC.append(result['PCC(positive_rate,preservation_Z_score)'])
    
    return ACC_PCC

    
def predict_packagecopy2(module_gene_list, 
                         pos_sample, 
                         neg_sample, 
                         cell_data,
                         cell_cluster, 
                         Cluster,#細胞對應cluster,Cluster列表
                         preservation_Z_score, 
                         modulename, 
                         save_name_255,
                         cell_leiden_cluster,#cluster列表
                         save_name_barplot):
    pos_m = pos_sample[module_gene_list]
    neg_m = neg_sample[module_gene_list]
    train_pos = pos_m.values.tolist()
    train_neg = neg_m.values.tolist()
    data_rd = cell_data[module_gene_list]
    data_X = data_rd.values.tolist()
    result = randomforest_crosscopy1(train_pos, train_neg, data_X)
    
    #算所有cluster的pos率
    pr=positive_rate(Cluster,result['predict_list'],cell_cluster,'clusters2num')
    
    pltdata={"preservation_Z_score": preservation_Z_score,
            "positive_rate":pr,
            "Cluster":Cluster}
    #print( predict_list)
    #print(positive_rate)
    #print(Cluster)
    pltdata=pd.DataFrame(pltdata)
    ## pd.dataframe
    pltdata["ClusterStr"]=list(map(str, pltdata["Cluster"]))
    sns.lmplot(data=pltdata,x='preservation_Z_score', y='positive_rate', hue='Cluster',
                   fit_reg=False, legend=True, legend_out=True,size=9,scatter_kws={'alpha':0.65,'s':196})
    plt.title(modulename+" Accuracy: %0.2f (+/- %0.2f)" % (result['ACC_mean_std'][0], result['ACC_mean_std'][1] * 2), fontsize=16)
    
    
    plt.savefig(save_name_255,bbox_inches='tight',               # 去除座標軸占用的空間
                pad_inches=0.0)
    fig=plt.gcf()
    plt.close(fig)
    result['PCC(positive_rate,preservation_Z_score)']=sc.pearsonr(pr, preservation_Z_score)[0]

    ##回傳值已改成dictionary,一個裝 predict_list,另一個裝ACC的mean跟std
    #後面需要num_list
    #predict_list會標註說哪一些細胞被判斷成positive,哪一些細胞被判斷成negtive,
    #copy1 ver:num_list已經改成positive_rate
    #          preservation_255_list已經改成 preservation_Z_score
    #
    return result
#=========from scVelo Start===========    
def quiver_autoscale(X_emb, V_emb):
    import matplotlib.pyplot as pl

    scale_factor = np.abs(X_emb).max()  # just so that it handles very large values
    fig, ax = pl.subplots()
    Q = ax.quiver(
        X_emb[:, 0] / scale_factor,
        X_emb[:, 1] / scale_factor,
        V_emb[:, 0],
        V_emb[:, 1],
        angles="xy",
        scale_units="xy",
        scale=None,
    )
    Q._init()
    fig.close()
    pl.close(fig)
    return Q.scale / scale_factor

def compute_velocity_on_grid(
    X_emb,
    V_emb,
    density=None,
    smooth=None,
    n_neighbors=None,
    min_mass=None,
    autoscale=True,
    adjust_for_stream=False,
    cutoff_perc=None,
):
    # remove invalid cells
    idx_valid = np.isfinite(X_emb.sum(1) + V_emb.sum(1))
    X_emb = X_emb[idx_valid]
    V_emb = V_emb[idx_valid]

    # prepare grid
    n_obs, n_dim = X_emb.shape
    density = 1 if density is None else density
    smooth = 0.5 if smooth is None else smooth

    grs = []
    for dim_i in range(n_dim):
        m, M = np.min(X_emb[:, dim_i]), np.max(X_emb[:, dim_i])
        m = m - 0.01 * np.abs(M - m)
        M = M + 0.01 * np.abs(M - m)
        gr = np.linspace(m, M, int(50 * density))
        grs.append(gr)

    meshes_tuple = np.meshgrid(*grs)
    X_grid = np.vstack([i.flat for i in meshes_tuple]).T

    # estimate grid velocities
    if n_neighbors is None:
        n_neighbors = int(n_obs / 50)
    nn = NearestNeighbors(n_neighbors=n_neighbors, n_jobs=-1)
    nn.fit(X_emb)
    dists, neighs = nn.kneighbors(X_grid)

    scale = np.mean([(g[1] - g[0]) for g in grs]) * smooth
    weight = normal.pdf(x=dists, scale=scale)
    p_mass = weight.sum(1)

    V_grid = (V_emb[neighs] * weight[:, :, None]).sum(1)
    V_grid /= np.maximum(1, p_mass)[:, None]
    if min_mass is None:
        min_mass = 1

    if adjust_for_stream:
        X_grid = np.stack([np.unique(X_grid[:, 0]), np.unique(X_grid[:, 1])])
        ns = int(np.sqrt(len(V_grid[:, 0])))
        V_grid = V_grid.T.reshape(2, ns, ns)

        mass = np.sqrt((V_grid ** 2).sum(0))
        min_mass = 10 ** (min_mass - 6)  # default min_mass = 1e-5
        min_mass = np.clip(min_mass, None, np.max(mass) * 0.9)
        cutoff = mass.reshape(V_grid[0].shape) < min_mass

        if cutoff_perc is None:
            cutoff_perc = 5
        length = np.sum(np.mean(np.abs(V_emb[neighs]), axis=1), axis=1).T
        length = length.reshape(ns, ns)
        cutoff |= length < np.percentile(length, cutoff_perc)

        V_grid[0][cutoff] = np.nan
    else:
        min_mass *= np.percentile(p_mass, 99) / 100
        X_grid, V_grid = X_grid[p_mass > min_mass], V_grid[p_mass > min_mass]

        if autoscale:
            V_grid /= 3 * quiver_autoscale(X_grid, V_grid)

    return X_grid, V_grid

def get_emb(adata):
    X_emb = np.array(adata.obsm["X_umap"][:, [0,1]])
    V_emb = np.array(adata.obsm["velocity_umap"][:, [0,1]])
    return X_emb, V_emb


def velocity_stream_scatterplt(adata,pltdata,
                                      x,y,hue,palette,style,style_order,
                                      density,bbox_to_anchor_x,savename):
    X_emb,V_emb=get_emb(adata)
    X_grid, V_grid = compute_velocity_on_grid(
        X_emb=X_emb,
        V_emb=V_emb,
        density=1,
        autoscale=False,
        adjust_for_stream=True
    )
    lengths = np.sqrt((V_grid ** 2).sum(0))
    linewidth=None
    linewidth = 1 if linewidth is None else linewidth
    linewidth *= 2 * lengths / lengths[~np.isnan(lengths)].max()
    stream_kwargs = {
                "linewidth": linewidth,
                "density": density ,
                "zorder": 3,
                "color": "k" 
    }
    sns.set(font_scale=2)
    sns.set_style("ticks")
    fig=plt.figure(figsize=(16,12),dpi=150)
    g=plt.streamplot(X_grid[0], X_grid[1], V_grid[0], V_grid[1], **stream_kwargs)
    g=sns.scatterplot(data=pltdata, x=x, y=y, hue=hue,palette=palette,style=style,style_order=style_order,s=196,alpha=0.65,markers=('o','$\u043E$'),ax=fig.gca())
    g.legend(loc='right', bbox_to_anchor=(bbox_to_anchor_x, 0.5), ncol=1)
    plt.savefig(savename, bbox_inches='tight',pad_inches=0.0)
    fig=plt.gcf()
    plt.close(fig)


#========from scVelo End==============


def main():
    #自動化的時候用得上的路徑        savedir="D:/DS100rounds/-"+str(DSpct)+"0pct/round"+str(rounds)+"/"
    #data資料夾位置的根目錄
    os.chdir("C:/Users/user/Desktop/test/scanpy")
    
    foldername = input('Enter foldername:') #"scv_pancreas_prep" or "scv_pancreas_impute"## datafolder
    Clustermethod = "celltype"
    clustering = "cell_type"
    
    adata = anndata.read_h5ad(os.path.join('.', foldername, foldername+'.h5ad'))
    
    resultfolder = "preservation_result"
    savedir = os.path.join(".", foldername, Clustermethod+"_cluster_")
    result_savedir = os.path.join(".", foldername, resultfolder, "")
    
    clustering_size = pd.read_csv(os.path.join(".", foldername, Clustermethod+"_clustering_size.csv"))
    cell_data = pd.read_csv(os.path.join(".", foldername, "preprocessed_cell.csv"),index_col=0)
    cell_UMAP_cluster = pd.read_csv(os.path.join(".", foldername, "UMAP_cell_embeddings_to_"+Clustermethod+"_clusters_and_coordinates.csv"),index_col=0)
    
    cell_UMAP_cluster['leiden'] = pd.to_numeric(adata.obs.leiden,downcast='unsigned')#uint64
    cell_UMAP_cluster['cell_type'] = pd.to_numeric(adata.obs.clusters2num)#int64
    cell_leiden_cluster = list(set(cell_UMAP_cluster['leiden']))
    adata.obs['cell_type'] = pd.Categorical(list(adata.obs.clusters2num))
    cell_UMAP_cluster['latent_time'] = adata.obs.latent_time
    scp.tl.rank_genes_groups(adata, 'cell_type', method='wilcoxon')
    
    DEGinGroups=pd.DataFrame(adata.uns['rank_genes_groups']['names'][0:50])
    
    if not os.path.isdir(result_savedir):
        os.mkdir(result_savedir,755)
        

    os.chdir("./"+foldername)
    scp.pl.rank_genes_groups(adata,n_genes=50,shery=False,save="wilcoxon_DEG.png",dpi=150)
    sns.set(font_scale=2,)
    sns.set_style("ticks")
    scv.pl.velocity_embedding_stream(adata, basis='umap',save="celltype_stream.png",dpi=150)
    scv.pl.velocity_embedding_stream(adata, basis='umap',legend_loc="right",save="celltype_stream_legend_out.png",dpi=150)
    scv.pl.velocity_embedding_stream(adata, basis='umap',color=clustering,legend_loc="right",save="clusters2num_stream_legend_out.png",dpi=150)
    scv.pl.velocity_embedding_stream(adata, basis='umap',color=clustering,save="clusters2num_stream.png",dpi=150)
    scv.pl.velocity_embedding_stream(adata, basis='umap',color='leiden',legend_loc="right",save="leiden_stream_legend_out.png",dpi=150)
    scv.pl.velocity_embedding_stream(adata, basis='umap',color='leiden',save="leiden_stream.png",dpi=150)
    scv.pl.scatter(adata, color='latent_time', color_map='turbo', size=80, colorbar=True,save="latent_time_scatterplt.png",figsize=(10,10),dpi=150)
    os.chdir('../')
    
    #drawlmplot_annotation(cell_UMAP_cluster, clustering_size, 'UMAP1', 'UMAP2', clustering,
    #                      result_savedir+"clustering_UMAP_annotation.png")
    
    sns.lmplot(data=cell_UMAP_cluster, x='UMAP1', y='UMAP2', hue=clustering,fit_reg=False, legend=True, legend_out=True,size=14)
    plt.savefig(os.path.join(result_savedir, "clustering_UMAP.png"), bbox_inches='tight',pad_inches=0.0)# 去除座標軸占用的空間
    fig=plt.gcf()
    plt.close(fig)
    #drawlmplot_annotation(cell_UMAP_cluster, cell_leiden_cluster, 'UMAP1', 'UMAP2', 'leiden',
    #                      result_savedir+"leiden_res2_clustering_UMAP_annotation.png")
    
    sns.lmplot(data=cell_UMAP_cluster, x='UMAP1', y='UMAP2', hue='leiden',fit_reg=False, legend=True, legend_out=True,size=14)
    plt.savefig(os.path.join(result_savedir, "leiden_res2_clustering_UMAP.png"), bbox_inches='tight',pad_inches=0.0)# 去除座標軸占用的空間
    fig=plt.gcf()
    plt.close(fig)
                
    plt.figure(figsize=(10,10))
    sns.distplot(cell_UMAP_cluster['latent_time'])
    plt.ylabel('Density')
    plt.savefig(os.path.join(result_savedir, "latent_time_distribution.png"), bbox_inches='tight',pad_inches=0.0)# 去除座標軸占用的空間
    fig=plt.gcf()
    plt.close(fig)
    
    adata.obs['fixed_time_latent_time_group']=adata.obs.latent_time
    for i in range(0,20,1):
        if i<19:
            adata.obs.loc[adata[np.where((adata.obs.latent_time>=(i*0.05))&(adata.obs.latent_time<((i*0.05)+0.05)))].obs.fixed_time_latent_time_group.index, 'fixed_time_latent_time_group'] = pd.to_numeric(i,downcast='unsigned')
        else:
            adata.obs.loc[adata[np.where((adata.obs.latent_time>=(i*0.05))&(adata.obs.latent_time<=((i*0.05)+0.05)))].obs.fixed_time_latent_time_group.index, 'fixed_time_latent_time_group'] = pd.to_numeric(i,downcast='unsigned')
    cell_UMAP_cluster['fixed_time_latent_time_group']=adata.obs.fixed_time_latent_time_group
    
    adata.obs['fixed_cell_number_latent_time_group']=adata.obs.latent_time
    sorted_latent_time=np.sort(cell_UMAP_cluster['latent_time'])
    for i in range(0,20,1):
        a=len(adata.obs['fixed_cell_number_latent_time_group'])
        floor=sorted_latent_time[int(a*i/20)]
        if i<19:
            celling=sorted_latent_time[int(a*(i+1)/20)]
            adata.obs.loc[adata[np.where((adata.obs.latent_time>=(floor))&
                                         (adata.obs.latent_time<(celling)))].obs.fixed_cell_number_latent_time_group.index, 'fixed_cell_number_latent_time_group'] = pd.to_numeric(i,downcast='unsigned')
        else:
            celling=sorted_latent_time[int(a*(i+1)/20)-1]
            adata.obs.loc[adata[np.where((adata.obs.latent_time>=(floor))&
                                         (adata.obs.latent_time<=(celling)))].obs.fixed_cell_number_latent_time_group.index, 'fixed_cell_number_latent_time_group'] = pd.to_numeric(i,downcast='unsigned')
    cell_UMAP_cluster['fixed_cell_number_latent_time_group']=adata.obs.fixed_cell_number_latent_time_group
    velocity_stream_scatterplt(adata=adata,pltdata=cell_UMAP_cluster,x='UMAP1',y='UMAP2',hue='fixed_cell_number_latent_time_group',palette='tab20',style= None,
                                       style_order=None,density=1,bbox_to_anchor_x=1.5,savename=os.path.join(result_savedir, 'fixed_cell_number_latent_time_group.png'))
    velocity_stream_scatterplt(adata=adata,pltdata=cell_UMAP_cluster,x='UMAP1',y='UMAP2',hue='fixed_time_latent_time_group',palette='tab20',style= None,
                                       style_order=None,density=1,bbox_to_anchor_x=1.5,savename=os.path.join(result_savedir, 'fixed_time_latent_time_group.png'))
    
    del(sorted_latent_time)
    
    #loop區域,尋找可以測試的組合,並建立資料夾
    training_group_DF=pd.DataFrame(columns=['POS','NEG','Color','moduleName'])
    for POS_cluster in range(0,len(clustering_size.index)):
        try:
            Module_cluster_Zscore = pd.read_csv(os.path.join(savedir+str(POS_cluster), 'module_preservation_Zscore.csv'),index_col=0)
            Module_cluster_Zscore = Module_cluster_Zscore.drop(columns=['gold', 'grey'])
            # module_preservation_Zscore:所有module在所有cluster的Zscore
            if len(Module_cluster_Zscore.columns)>0 :
                for moduleColor in Module_cluster_Zscore.columns:
                    #每個module
                    if Module_cluster_Zscore[moduleColor][POS_cluster]>10:
                        for NEG_cluster in range(0,len(clustering_size.index)):
                            if POS_cluster!=NEG_cluster:
                                if Module_cluster_Zscore[moduleColor][NEG_cluster]<2:
                                    if not os.path.isdir(os.path.join(result_savedir, "PC"+str(POS_cluster)+"NC"+str(NEG_cluster)+"_"+moduleColor, "")):
                                        os.mkdir(os.path.join(result_savedir, "PC"+str(POS_cluster)+"NC"+str(NEG_cluster)+"_"+moduleColor, ""),755)                            
                                    print('找到測試組!:\nPOS:cluster'+str(POS_cluster)+'\n'+'NEG:cluster'+str(NEG_cluster)+'\n'+'module:'+moduleColor+'\n')
                                    training_group_DF.loc[len(training_group_DF.index)]=[POS_cluster,NEG_cluster,moduleColor,'cluster'+str(POS_cluster)+'_'+moduleColor]
        except: 
            pass
    #作圖loop區域,前段 #提醒! 要再加入後續的做圖code
    ACC_mean_std_DF=pd.DataFrame(columns=['mean','std','PCC_of_positive_rate_and_preservation_Z_score'])
    
    functionOption=''
    while functionOption!=0:
        functionOption=int(input('1)auto\n2)semi\n0)quit:'))
        if functionOption==1:
            for i in training_group_DF.index:
                POS_cluster=training_group_DF.loc[i]['POS']
                NEG_cluster=training_group_DF.loc[i]['NEG']
                moduleColor=training_group_DF.loc[i]['Color']
                
                module_savedir=os.path.join(result_savedir+"PC"+str(POS_cluster)+"NC"+str(NEG_cluster)+"_"+moduleColor, "")
                
                result_acc_mean_std=prediction_and_ploting(Adata=adata,
                                                           cell_data=cell_data,
                                                           cell_UMAP_cluster=cell_UMAP_cluster,
                                                           cell_leiden_cluster=cell_leiden_cluster,
                                                           Clustermethod=Clustermethod,
                                                           clustering=clustering,
                                                           POS_cluster=POS_cluster, NEG_cluster=NEG_cluster,
                                                           moduleColor=moduleColor,savedir=savedir,
                                                           module_savedir=module_savedir)
                ACC_mean_std_DF.loc[len(ACC_mean_std_DF.index)]=result_acc_mean_std#準確率平均[0]跟標準差[1]
                #breakpoint()
                gc.collect()
            training_group_DF=pd.concat([training_group_DF, ACC_mean_std_DF], axis=1)#把測試組合與其準確率跟準確率標準差合併
            
            fig=plt.figure(figsize=(10,10),dpi=150)
            fig= sns.swarmplot(training_group_DF['PCC_of_positive_rate_and_preservation_Z_score'],color=".2",size=12)
            fig= sns.boxplot(training_group_DF['PCC_of_positive_rate_and_preservation_Z_score'])
            plt.savefig(os.path.join(result_savedir, "PCC_of_positive_rate_and_preservation_Z_score.png"), bbox_inches='tight',pad_inches=0.0)# 去除座標軸占用的空間
            fig=plt.gcf()
            plt.close(fig)
            #breakpoint()
            FEA_geneModule=pd.DataFrame(columns=['POS','Color'])
            for i in list(set(training_group_DF['moduleName'])):
                for j in training_group_DF.index:
                    if training_group_DF['moduleName'][j] == i :
                        FEA_geneModule.loc[len(FEA_geneModule.index)]=[training_group_DF['POS'][j],training_group_DF['Color'][j]]
                        break
            moduleGenes=[]
            for i in FEA_geneModule.index:
                    target_Module=pd.read_csv(os.path.join(savedir+str(FEA_geneModule['POS'][i]), 'modules', FEA_geneModule['Color'][i]+'.csv'),index_col=0)
                    target_Module_genes=list(target_Module.index)
                    #target_module_genes_set= set(target_Module_genes)
                    moduleGenes.append(target_Module_genes)
            moduleGenesDF=pd.DataFrame(moduleGenes).T
            moduleGenesDF.columns=list(set(training_group_DF['moduleName']))
            moduleGenesDF.to_csv(os.path.join(result_savedir, "moduleGenesDF.csv"),sep='\t')
            
            
        if functionOption==2:
            POS_cluster=int(input('POS: '))
            NEG_cluster=int(input('NEG: '))
            moduleColor=str(input('moduleColor: '))
            module_savedir=os.path.join(result_savedir+"PC"+str(POS_cluster)+"NC"+str(NEG_cluster)+"_"+moduleColor, "")
                
            result_acc_mean_std=prediction_and_ploting_semi(Adata=adata,
                                                        cell_data=cell_data,
                                                        cell_UMAP_cluster=cell_UMAP_cluster,
                                                        cell_leiden_cluster=cell_leiden_cluster,
                                                        Clustermethod=Clustermethod,
                                                        clustering=clustering,
                                                        POS_cluster=POS_cluster, NEG_cluster=NEG_cluster,
                                                        moduleColor=moduleColor,savedir=savedir,
                                                        module_savedir=module_savedir)
            
            
        DEGinGroups.to_csv(os.path.join(result_savedir, "DEG_50inGroups.csv"),sep='\t')
if __name__=="__main__":
    main()