# -*- coding: utf-8 -*-
"""
## weight: weight of edge /2
##training: module genes
#pbmc3k
"""

import os
from os import listdir
import pandas as pd
import umap
from tqdm import tqdm_notebook
import tqdm
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
def draw_cor_map(pos_data, neg_data, col_name_list, pos_figure_name, neg_figure_name):
    #positive組的圖
    sns.set(font_scale=1)
    pos_data = pos_data[col_name_list]
    pos_gene = []
    #
    for col1 in tqdm.tqdm_notebook(col_name_list):
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
    plt.figure(figsize=(15,15))
    cmap = sns.diverging_palette(220, 10, sep=10, n=40)
    pos_figure = sns.clustermap(pos_gene, cbar_kws={'ticks': [-1, -0.8, -0.6, -0.4, -0.2, 0.0, 0.2, 0.4, 0.6, 0.8, 1.0]}, vmin=-1, vmax=1, cmap=cmap)
    plt.savefig(pos_figure_name,   # 儲存圖檔
                bbox_inches='tight',               # 去除座標軸占用的空間
                pad_inches=0.0)                    # 去除所有白邊
    plt.close()
    #negative組的圖，軸由positive決定    
    column_list = list(pos_figure.data2d.columns)
    
    neg_data = neg_data[col_name_list]
    neg_gene = []
    for col1 in tqdm.tqdm_notebook(col_name_list):
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
    gene = gene.reindex(index=column_list, columns=column_list)
    plt.figure(figsize=(15,15))
    sns.set(font_scale=2)
    cmap = sns.diverging_palette(220, 10, sep=10, n=40)
    neg_figure = sns.heatmap(gene, cbar_kws={'ticks': [-1, -0.8, -0.6, -0.4, -0.2, 0.0, 0.2, 0.4, 0.6, 0.8, 1.0]}, vmin=-1, vmax=1, cmap=cmap)
    plt.savefig(neg_figure_name,   # 儲存圖檔
                bbox_inches='tight',               # 去除座標軸占用的空間
                pad_inches=0.0)
    plt.close()
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
def count_number(cluster_number, preservation_Z_score, cell_cluster):
    all_num = 0
    pos_num = 0
    #從cell_cluster中去找每個細胞對應的cluster,再從preservation_Z_score中去尋找對應的prediction結果
    #才能計算positive率
    for cell_count in range(0,len(cell_cluster)):        
        #第cell_count個細胞的'leiden'存的是cluster,所以要比較一次
        if cell_cluster[cell_cluster.columns[0]][cell_count]== cluster_number:
            all_num+=1
            if preservation_Z_score[cell_count] == 1:#是positive的次數
                pos_num+=1
    return (pos_num, all_num)#每個細胞去查看
        
        

### random forest
def predict_packagecopy1(module_gene_list, pos_sample, neg_sample, cell_data, cell_cluster, Cluster, preservation_Z_score, modulename, save_name_255):
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
def positive_rate(Cluster,Predictlist,cell_UMAP):
    pr=[]
    for i in Cluster:
        num = count_number(i, Predictlist, cell_UMAP) #由於在count_number中指定傳入的dataframe的第一行為cluster,所以只能用另一個dataframe
        pri = num[0]/num[1]
        pr.append(pri)
    return pr

def find_boundary_cluster(cell_UMAP_cluster,Predictlist,Cluster):
    pr=positive_rate(Cluster, Predictlist, cell_UMAP_cluster)
    bdc=[]
    for index,element in enumerate(pr):
        if element<0.9 and element>0.1:
            bdc.append(index)
    return bdc
def predict_packagecopy2(module_gene_list, pos_sample, neg_sample, cell_data,
                         cell_cluster, Cluster,#細胞對應cluster,Cluster列表
                         preservation_Z_score, modulename, save_name_255,
                         cell_leiden_UMAP_cluster,cell_leiden_cluster,#
                         save_name_barplot):
    pos_m = pos_sample[module_gene_list]
    neg_m = neg_sample[module_gene_list]
    train_pos = pos_m.values.tolist()
    train_neg = neg_m.values.tolist()
    data_rd = cell_data[module_gene_list]
    data_X = data_rd.values.tolist()
    result = randomforest_crosscopy1(train_pos, train_neg, data_X)
    
    #算所有cluster的pos率
    pr=positive_rate(Cluster,result['predict_list'],cell_cluster)
    
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
    
    pr = positive_rate(cell_leiden_cluster,result['predict_list'],cell_leiden_UMAP_cluster)
        
    pltdata={"positive_rate":pr,
        "Cluster":cell_leiden_cluster}    
    plt.figure(figsize=(10,6))
    splot = sns.barplot(data=pltdata, x = 'Cluster', y = 'positive_rate', ci = None)
    for p in splot.patches:
        splot.annotate(format(p.get_height(), '.2f'), (p.get_x() + p.get_width() / 2., p.get_height()), ha = 'center', va = 'center', xytext = (0, 10), textcoords = 'offset points')
    plt.xlabel("Clusters", size=14)
    plt.ylabel("Positive_Rate", size=14)
    plt.savefig(save_name_barplot,bbox_inches='tight',               # 去除座標軸占用的空間
                pad_inches=0.0)
    return result
#自動化的時候用得上的路徑        savedir="D:/DS100rounds/-"+str(DSpct)+"0pct/round"+str(rounds)+"/"
#data資料夾位置的根目錄
os.chdir("C:/Users/user/Desktop/test/scanpy")
foldername="scv_pancreas_prep"## datafolder
Clustermethod="celltype"
clustering="clusters2num"
resultfolder="preservation_result"
savedir="./"+foldername+"/"+Clustermethod+"_cluster_"
clustering_size = pd.read_csv("./"+foldername+"/"+Clustermethod+"_clustering_size.csv")
cell_data = pd.read_csv("./"+foldername+"/preprocessed_cell.csv",index_col=0)
cell_UMAP_cluster = pd.read_csv("./"+foldername+"/UMAP_cell_embeddings_to_"+Clustermethod+"_clusters_and_coordinates.csv",index_col=0)
result_savedir="./"+foldername+"/"+resultfolder+"/"
cell_leiden_UMAP_cluster = pd.read_csv("./"+foldername+"_leiden2/UMAP_cell_embeddings_to_leiden_clusters_and_coordinates.csv",index_col=0)
cell_UMAP_cluster['leiden']=cell_leiden_UMAP_cluster['leiden']
cell_leiden_cluster=list(set(cell_UMAP_cluster['leiden']))
try:
    os.mkdir(result_savedir,755)
except:
    pass

drawlmplot_annotation(cell_UMAP_cluster, clustering_size, 'UMAP1', 'UMAP2', clustering,
                      result_savedir+"clustering_UMAP_annotation.png")

sns.lmplot(data=cell_UMAP_cluster, x='UMAP1', y='UMAP2', hue=clustering,fit_reg=False, legend=True, legend_out=True,size=14)
plt.savefig(result_savedir+"clustering_UMAP.png", bbox_inches='tight',pad_inches=0.0)# 去除座標軸占用的空間
drawlmplot_annotation(cell_leiden_UMAP_cluster, cell_leiden_cluster, 'UMAP1', 'UMAP2', 'leiden',
                      result_savedir+"leiden_res2_clustering_UMAP_annotation.png")

sns.lmplot(data=cell_leiden_UMAP_cluster, x='UMAP1', y='UMAP2', hue='leiden',fit_reg=False, legend=True, legend_out=True,size=14)
plt.savefig(result_savedir+"leiden_res2_clustering_UMAP.png", bbox_inches='tight',pad_inches=0.0)# 去除座標軸占用的空間
            
#loop區域,尋找可以測試的組合,並建立資料夾
training_group_DF=pd.DataFrame(columns=['POS','NEG','Color','moduleName'])
for POS_cluster in range(0,len(clustering_size.index)):
    try:
        Module_cluster_Zscore = pd.read_csv(savedir+str(POS_cluster)+'/module_preservation_Zscore.csv',index_col=0)
        Module_cluster_Zscore = Module_cluster_Zscore.drop(columns=['gold', 'grey'])
        # module_preservation_Zscore:所有module在所有cluster的Zscore
        if len(Module_cluster_Zscore.columns)>0 :
            for moduleColor in Module_cluster_Zscore.columns:
                #每個module
                if Module_cluster_Zscore[moduleColor][POS_cluster]>10:
                    for NEG_cluster in range(0,len(clustering_size.index)):
                        if POS_cluster!=NEG_cluster:
                            if Module_cluster_Zscore[moduleColor][NEG_cluster]<2:
                                try:
                                    #os.mkdir(savedir+str(POS_cluster)+"/modules/PC"+str(POS_cluster)+"NC"+str(NEG_cluster)+"_"+moduleColor+"/",755)
                                    os.mkdir(result_savedir+"PC"+str(POS_cluster)+"NC"+str(NEG_cluster)+"_"+moduleColor+"/",755)                            
                                except:
                                    pass
                                print('找到測試組!:\nPOS:cluster'+str(POS_cluster)+'\n'+'NEG:cluster'+str(NEG_cluster)+'\n'+'module:'+moduleColor+'\n')
                                training_group_DF.loc[len(training_group_DF.index)]=[POS_cluster,NEG_cluster,moduleColor,'cluster'+str(POS_cluster)+'_'+moduleColor]
    except: 
        pass
#作圖loop區域,前段 #提醒! 要再加入後續的做圖code
ACC_mean_std_DF=pd.DataFrame(columns=['mean','std'])
for i in training_group_DF.index:
    POS_cluster=training_group_DF.loc[i]['POS']
    NEG_cluster=training_group_DF.loc[i]['NEG']
    moduleColor=training_group_DF.loc[i]['Color']
    
    module_savedir=result_savedir+"PC"+str(POS_cluster)+"NC"+str(NEG_cluster)+"_"+moduleColor+"/"
    
    Module_cluster_Zscore = pd.read_csv(savedir+str(POS_cluster)+'/module_preservation_Zscore.csv',index_col=0)
    Module_cluster_Zscore = Module_cluster_Zscore.drop(columns=['gold', 'grey'])

    POS=pd.read_csv(savedir+str(POS_cluster)+'.csv',index_col=0)
    NEG=pd.read_csv(savedir+str(NEG_cluster)+'.csv',index_col=0)

    target_Module=pd.read_csv(savedir+str(POS_cluster)+'/modules/'+moduleColor+'.csv',index_col=0)
    target_Module_genes=list(target_Module.index)
    target_module_genes_set= set(target_Module_genes)
    edges = pd.read_table(savedir+str(POS_cluster)+'/modules/'+Clustermethod+'_cluster_'+str(POS_cluster)+'edges.txt')
    
    
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
    del(edges,filtered_edges)
        
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
                               module_savedir+moduleColor+'_PC'+str(POS_cluster)+'_NC'+str(NEG_cluster)+'.png',
                               cell_leiden_UMAP_cluster,
                               cell_leiden_cluster,
                               module_savedir+"barplot_of_pos_rate_on_leiden_res2.png"
                              )
    predict_list=result['predict_list']
    ACC_mean_std_DF.loc[len(ACC_mean_std_DF.index)]=result['ACC_mean_std']#準確率平均[0]跟標準差[1]
    
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
        
    sns.lmplot(data=cell_UMAP_cluster, x='UMAP1', y='UMAP2', hue='training_clusters',fit_reg=False, legend=True, legend_out=True,size=14)
    plt.savefig(module_savedir+'/training_cluster_UMAP.png', bbox_inches='tight',pad_inches=0.0)
    
    #predict_listtest_to_color = change_color(predict_listtest, '#808080', '#FF0000')
    cell_UMAP_cluster['prediction']=predict_list
    #標示出predict為pos和neg的細胞分布 UMAP
    #SAVE!
    sns.lmplot(data=cell_UMAP_cluster, x='UMAP1', y='UMAP2', hue='prediction',fit_reg=False, legend=True, legend_out=True,size=14)
    plt.savefig(module_savedir+'/positive_UMAP.png', bbox_inches='tight',               # 去除座標軸占用的空間
                pad_inches=0.0)
    boundaryC=find_boundary_cluster(cell_leiden_UMAP_cluster, predict_list, cell_leiden_cluster)
    boundaryUMAP=cell_UMAP_cluster[cell_UMAP_cluster.leiden.isin(boundaryC)]
    sns.lmplot(data=boundaryUMAP, x='UMAP1', y='UMAP2', hue='leiden',fit_reg=False, legend=True, legend_out=True,size=14)
    for i, label in enumerate(range(0,len(cell_leiden_cluster))):
            
        #loop through data points and plot each point 
            for l, row in boundaryUMAP.loc[boundaryUMAP['prediction']==label,:].iterrows():
                #add the data point as text
                if row['prediction']==0:
                    plt.annotate(int(row['prediction']), 
                                 (row['UMAP1'], row['UMAP2']),
                                 horizontalalignment='center',
                                 verticalalignment='center',
                                 size=11,
                                 )
    plt.savefig(module_savedir+'/positive_UMAP_boundary_annotated.png', bbox_inches='tight',pad_inches=0.0)
    del(boundaryC,boundaryUMAP)
    #POS cluster module gene expression (hub gene) heatmap 
    #plt.figure(figsize=(10,10))
    #sns.heatmap(POS_training[target_Module_c_list], vmax=6)
    
    ####whole module genes heatmap
    moduleGenePosHeatmap=sns.clustermap(data=(POS_training[target_Module_genes].T),xticklabels=False,yticklabels=True,
               figsize=((10+16*len(target_Module_genes)/40),(10+9*len(target_Module_genes)/30)),method='ward')
    fixed_gene_list = list(moduleGenePosHeatmap.data2d.index)
    plt.savefig(module_savedir+'pos_module_gene_heatmap.png', bbox_inches='tight',pad_inches=0.0)

    
    NEG_moduleGeneOrdered=NEG_training[target_Module_genes]
    NEG_moduleGeneOrdered=NEG_moduleGeneOrdered.reindex(columns=fixed_gene_list)
    
    plt.figure(figsize=((10+16*len(target_Module_genes)/40),(10+9*len(target_Module_genes)/30)))
    moduleGeneNegHeatmap=sns.heatmap(data=NEG_moduleGeneOrdered.T,xticklabels=False,yticklabels=True)
    plt.savefig(module_savedir+'/neg_module_gene_heatmap.png', bbox_inches='tight',pad_inches=0.0)    

    #訓練資料
 
    #training data (whole module gene) PCC map
    pos_neg_cor = draw_cor_map(POS_training, NEG_training, target_Module_genes,
                               module_savedir+'training_module_PCC_heatmap_PC'+str(POS_cluster)+'module_'+moduleColor+'.png',
                               module_savedir+'training_module_PCC_heatmap_NC'+str(NEG_cluster)+'module_'+moduleColor+'.png')
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
    sns.distplot(neg_pcc_list, label='negative training')
    sns.distplot(pos_pcc_list, label='positive training')
    plt.xlabel('PCC', fontsize=20)
    plt.ylabel('Density', fontsize=20)
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.legend()
    plt.tight_layout()
    plt.savefig(module_savedir+'training_module_PCC.png',   # 儲存圖檔
                bbox_inches='tight',               # 去除座標軸占用的空間
                pad_inches=0.0)
    
    
    gene_data = cell_data[target_Module_genes].values.tolist()
    #pos_list就是被判斷為Positive的細胞中特定module gene的表現量
    #neg_list就是negitive
    pos_list = []
    neg_list = []
    for result in enumerate(predict_list):
        if result[1] == 1:
            pos_list.append(gene_data[result[0]])
        elif result[1] == 0:
            neg_list.append(gene_data[result[0]])
    
    pos = pd.DataFrame(pos_list)
    neg = pd.DataFrame(neg_list)
    
    #breakpoint()
    #提醒!要把檔名的cluster拿掉
    
    
    
    
    ##predict結果的PCC
    #prediction result data PCC map
    
    pos_neg_cor = draw_cor_map(pos, neg, pos.columns.tolist(),
                               module_savedir+'prediction_module_PCC_heatmap_pos'+str(POS_cluster)+'.png',
                               module_savedir+'prediction_module_PCC_heatmap_neg'+str(NEG_cluster)+'.png')
    
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
    sns.distplot(neg_pcc_list, label='negative prediction')
    sns.distplot(pos_pcc_list, label='positive prediction')
    plt.xlabel('PCC', fontsize=20)
    plt.ylabel('Density', fontsize=20)
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.legend()
    plt.tight_layout()
    plt.savefig(module_savedir+'prediction_module_PCC.png',   # 儲存圖檔
                bbox_inches='tight',               # 去除座標軸占用的空間
                pad_inches=0.0)
    
training_group_DF=pd.concat([training_group_DF, ACC_mean_std_DF], axis=1)#把測試組合與其準確率跟準確率標準差合併
