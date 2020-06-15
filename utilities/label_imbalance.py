#-------------------------------FUNCTIONS-------------------------------------#
#
# 1.observe_imbalance_cols(labels)
#     input : a dataframe containing the labels' votes ("1" and "0")
#     output: a dataframe containing the number of votes for each label
#
# 2.observe_imbalance_rows(labels)
#     input : a dataframe containing the labels' votes ("1" and "0")
#     output: a dataframe containing the number of votes for each instance
#
# 3.find_label_cardinality(labels)
#     input : a dataframe containing the labels' votes ("1" and "0")
#     output: label cardinality
#
# 4.find_label_cardinality2(labels_count_rows)
#     input : labels_count_rows (output from observe_imbalance_rows)
#     output: label cardinality
#
# 5.find_label_density(labels,label_cardinality)
#     input : label cardinality and length of labels
#     output: label density
#
# 6.positive_votes_per_label(labels,title_given)
#     input : dataframe labels and a string for the plot title
#     output: plot with the positive votes
#
# 7.negative_votes_per_label(labels,title_given)
#     input : dataframe labels and a string for the plot title
#     output: plot with the negative votes
#
# 8.find_multilabel_instances(labels,title_given)
#     input : dataframe labels and a string for the plot title
#     output: plot with the number of votes versus number of labels selected
#
# 9.observe_imbalance(labels,labels_dictionary,categories,title_name)
#     input : labels,labels_dictionary,categories,title_name
#     output: label_cardinality,label_density,IRLbls,mean_IR,scumbles,scumble_metric,scumble_metric_cv,IRLblis (+plots)
#
# 10.find_IRLbl(labels_count_cols,categories)
#     input : the number of positive and negative votes for each label and label names
#     output: IRLbls of each label, mean_IRLbl of the dataset
#
# 11.find_scumble(labels_dictionary,IRLbls,categories)
#     input : dictionary of instances each having its positive labels, IRLbls and label names
#     output: scumbles,scumble_metric, new IRLblis
#
# 12.find_scumble_cv(scumbles,scumble_metric)
#     input : scumbles,scumble_metric 
#     output: scumble_metric_cv (the corresponding coefficient of variation of scumble_metric)
#
# 13.apply_remedial_algorithm(data_with_labels,scumbles,scumble_metric,IRLbls,mean_IR,threshold,dataset_features)
#     input : dataframe of instances with features and labels,scumbles,scumble_metric,IRLbls,mean_IR, a threshold parameter(0.5 default),number of features
#     output: dataframe of instances and cloned instances with features and labels
#
# 14.observe_imbalance_after_remedial(labels_and_clones,labels_dictionary,categories,threshold,title_name)
#     input : labels_and_clones,labels_dictionary,categories,threshold,title_name
#     output: label_cardinality_remedial,label_density_remedial,IRLbls_remedial,mean_IR_remedial,scumble_metric_remedial,scumble_metric_cv_remedial,IRLblis_remedial
#
# 15.find_IRLblis_complete(IRLblis,labels_train)
#     input : IRLblis,labels_train
#     output: IRLblis_complete
#
# 16.newSample(sample_features,sample_labels,r_neighbors_features,r_neighbors_labels)
#     input : sample_features,sample_labels,r_neighbors_features,r_neighbors_labels
#     output: synthetic_sample_features,synthetic_sample_labels
#
# 17.apply_mlsmote_algorithm(IRLblis,mean_IR,k_neighbors,features_train,labels_train,categories)
#     input : IRLblis,mean_IR,k_neighbors,features_train,labels_train,categories
#     output: features_mlsmote,labels_mlsmote
#    
# 18.observe_imbalance_after_mlsmote(labels_and_clones,labels_dictionary,categories,title_name)    
#     input : labels_and_clones,labels_dictionary,categories,title_name
#     output: label_cardinality_mlsmote,label_density_mlsmote,IRLbls_mlsmote,mean_IR_mlsmote,scumble_metric_mlsmote,scumble_metric_cv_mlsmote,IRLblis_mlsmote
#    
# 19.print_metrics(labels_test, prediction, name)
#     input : the test labels, the prediction of the classifier, a name as the title
#     output: print the metrics in console (uncomment to keep the values as well)
#
#-----------------------------------------------------------------------------#
#--------------------------IMPORT LIBRARIES-----------------------------------#
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt 
import numpy as np
import functools,random
#from utilities.data_formulation import transform_dataframe_to_dictionary
from utilities.data_visualization import data_visualization_control,cooccurrence_matrix,find_feature_and_label_dist,find_best_distribution
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
import sklearn.metrics as metrics
from sklearn.neighbors import NearestNeighbors
#-----------------------------------------------------------------------------#
#-------------------------------FUNCTIONS-------------------------------------#
def observe_imbalance_cols(labels):
    imb_cols=pd.DataFrame()
    for i in range(labels.shape[1]):
        for x in labels[labels.columns[i]]:  
            imb_cols[labels.columns[i]]=labels[labels.columns[i]].value_counts().tolist()
    return(imb_cols)
    
def observe_imbalance_rows(labels):
    imb_rows=pd.DataFrame()
    for i in range(labels.shape[0]):
        for x in labels.iloc[i]:  
            imb_rows[i]=pd.Series(labels.iloc[i].value_counts().tolist())
    return(imb_rows)   
    
def find_label_cardinality(labels):
    # Cardinality
    n = float(len(labels))
    lc = 0
    for instance, active_labels in labels.iterrows():
        active_labels = active_labels.iloc[active_labels.to_numpy().nonzero()]
        lc += len(active_labels)
    lc /= n
    return(lc)

def find_label_cardinality2(labels_count_rows):
    lc=labels_count_rows.iloc[1].values.mean()
    return(lc)
    
def find_label_density(labels,label_cardinality):
    label_density = label_cardinality / labels.shape[1]
    return(label_density)    
    
def positive_votes_per_label(labels,title_given):    
    categories = list(labels.columns.values)
    sns.set(font_scale = 2)
    plt.figure(figsize=(15,8))
    ax= sns.barplot(categories, labels.sum().values)
    plt.title(title_given, fontsize=24)
    plt.ylabel('Number of votes', fontsize=18)
    plt.xlabel('Labels ', fontsize=18)
    plt.xticks(np.arange(len(categories)),categories)
    #adding the text labels
    rects = ax.patches
    lab = labels.sum().values
    for rect, label in zip(rects, lab):
        height = rect.get_height()
        #ax.text(rect.get_x() + rect.get_width()/2, height + 5, label, ha='center', va='bottom', fontsize=18)
        ax.text(rect.get_x() + rect.get_width()/2, height -10, label, ha='center', va='bottom', fontsize=18)
    plt.show()

def negative_votes_per_label(labels,title_given): 
    categories = (labels.columns).tolist()
    sns.set(font_scale = 2)
    plt.figure(figsize=(15,8))
    ax= sns.barplot(categories, len(labels)-labels.sum())
    plt.title(title_given, fontsize=24)
    plt.ylabel('Number of votes', fontsize=18)
    plt.xlabel('Labels ', fontsize=18)
    plt.xticks(np.arange(len(categories)),categories)
    #adding the text labels
    rects = ax.patches
    lab = len(labels)-labels.sum()
    for rect, label in zip(rects, lab):
        height = rect.get_height()
        #ax.text(rect.get_x() + rect.get_width()/2, height -5, label, ha='center', va='bottom', fontsize=18)
        ax.text(rect.get_x() + rect.get_width()/2, height -10, label, ha='center', va='bottom', fontsize=18)
    plt.show()
    
def find_multilabel_instances(labels,title_given):    
    rowSums = labels.sum(axis=1)
    multiLabel_counts = rowSums.value_counts()
    multiLabel_counts = multiLabel_counts.sort_index()
    sns.set(font_scale = 2)
    plt.figure(figsize=(15,8))
    ax = sns.barplot(multiLabel_counts.index, multiLabel_counts.values)
    plt.title(title_given)
    plt.ylabel('Number of instances', fontsize=18)
    plt.xlabel('Number of labels', fontsize=18)
    #adding the text labels
    rects = ax.patches
    lab = multiLabel_counts.values
    for rect, label in zip(rects, lab):
        height = rect.get_height()
        ax.text(rect.get_x() + rect.get_width()/2, height, label, ha='center', va='bottom')
    plt.show()

def observe_imbalance(labels,labels_dictionary,categories,title_name):
    # find the number of positive and negative votes for each label
    labels_count_cols=observe_imbalance_cols(labels) 
    # find the number of positive and negative votes for each instance
    labels_count_rows=observe_imbalance_rows(labels)
    # find label cardinality
    label_cardinality=find_label_cardinality(labels)
    # find label density
    label_density=find_label_density(labels,label_cardinality)
    # find the number of negative votes
    sum_zeros=labels_count_rows.iloc[0].values.sum()
    # find the number of positive votes
    sum_ones=labels_count_rows.iloc[1].values.sum()
    # find the positive votes for each label 
    positive_votes_per_label(labels,"Positive votes for each label "+title_name)
    # find the negative votes for each label 
    negative_votes_per_label(labels,"Negative votes for each label "+title_name)
    # Counting the number of instances that have multiple labels.
    find_multilabel_instances(labels,"Instances having multiple labels "+title_name+"\n"+'cardinality: '+str(label_cardinality)+' - density: '+str(label_density))
    #find irlbls   
    IRLbls,mean_IR=find_IRLbl(labels_count_cols,categories)
    # find scuble
    scumbles,scumble_metric,IRLblis=find_scumble(labels_dictionary,IRLbls,categories)
    # find scuble cv
    scumble_metric_cv=find_scumble_cv(scumbles,scumble_metric)
    
    # concurrence_builder = LabelCooccurrenceGraphBuilder(weighted=True, include_self_edges=False)
    # concurrence = concurrence_builder.transform(labels)
    # print(concurrence)
    
    # plot the co-occurrence matrix and the co-occurrence matrix percentage
    cooccurrence_matrix(labels,categories,title_name)
    print('')
    print(title_name+'properties :')
    print('label cardinality  :',label_cardinality)
    print('mean IR            :',mean_IR)
    print('scumble metric     :',scumble_metric)
    print('scumbles_metric_cv :',scumble_metric_cv)
    print('')
    return(label_cardinality,label_density,IRLbls,mean_IR,scumbles,scumble_metric,scumble_metric_cv,IRLblis)

def find_IRLbl(labels_count_cols,categories):
    '''
    IRLbl is a measure calculated individually for each label. 
    The higher is the IRLbl the larger would be the imbalance, allowing to know which labels are in minority or majority. 
    MeanIR is the average IRLbl for an MLD. 
    It is useful to estimate the global imbalance level.
    '''
    max_of_labels_count_cols=labels_count_cols.iloc[1].values.max()
    irlbls=[]
    for i in range(labels_count_cols.shape[1]):
        irlbls.append(max_of_labels_count_cols/labels_count_cols.iloc[1].values[i])
        mean_IR=sum(irlbls)/len(irlbls)
#    # plot the frequency in which the instances select each label 
#    plt.figure()
#    plt.title('Selection frequency for each label')
#    plt.plot(labels_count_cols.iloc[1].values)
#    plt.xticks(np.arange(len(categories)),categories, rotation=0)
#    plt.grid()
#    plt.show()    
    return(irlbls,mean_IR)

def find_scumble(labels_dictionary,IRLbls,categories):
    '''
    The SCUMBLE metric aims to quantify the imbalance variance 
    among the labels present in each data sample.
    '''
    # SCUMBLE
    scumbles=[]
    IRLblis=[]
    for instance, active_labels in labels_dictionary.items():
        get_IRLbl = lambda x: IRLbls[categories.index(x)]
        IRLbli = list(map(get_IRLbl, active_labels)) # lista me orous pou tha pol/stoun
        if len(IRLbli)==0:
            IRLbli_bar=0
            IRLbli_prod=0
            scumblei=0
        else:
            IRLbli_bar = sum(IRLbli)/len(IRLbli)
            IRLbli_prod = (functools.reduce(lambda x,y:x*y, IRLbli))
            scumblei = 1 - (IRLbli_prod ** (1/len(IRLbli))) / IRLbli_bar
        IRLblis.append(IRLbli)
        scumbles.append(scumblei)
    scumble_metric = sum(scumbles)/len(scumbles)
    return(scumbles,scumble_metric,IRLblis)  
    
def find_scumble_cv(scumbles,scumble_metric):
    '''
    Since SCUMBLE is computed as an average of concurrence by instance, it could be influenced by extreme values. 
    A few instances with a very high SCUMBLE_ins value would introduce a certain deviation into the global SCUMBLE measure. 
    To estimate the importance of this deviation, the SCUMBLE.CV metric provides the corresponding coefficient of variation. 
    The higher is the SCUMBLE.CV, the larger would be the differences in concurrence among instances.
    '''
    scumble_metric_cv=np.std(scumbles)/scumble_metric
    return(scumble_metric_cv)
    
def apply_remedial_algorithm(data_with_labels,scumbles,scumble_metric,IRLbls,mean_IR,threshold,dataset_features):    
    '''
    REMEDIAL (REsampling MultilabEl datasets by Decoupling highly ImbAlanced Labels) 
    is a method specifically designed for MLDs that suffer from concurrence between imbalanced labels. 
    In this context, highly imbalanced labels has to be understood as labels with large differences in their IRLbls. 
    This is a fact assessed with the SCUMBLE measure, thus REMEDIAL is directed to MLDs with a high SCUMBLE level.
    
    Most of the differences are not statistically significant. However, Precision and MacroFM show important differences in half or more 
    of the cases. The former metric reveals statistically significant improvements with BR, CLR, ECC and HOMER. On the contrary, MacroFM 
    indicates that the worsening of results is remarkable for CLR, HOMER and IBLR.
    
    Overall, REMEDIAL would be a recommended resampling for MLDs with high SCUMBLE levels and when BR or LP based classifiers 
    are going to be used. In these cases the prediction of minority labels would be improved, and the global performance of the 
    classifiers would be better. MLDs such as genbase, medical and tmc2007, as their intrinsic traits have demonstrated, should not 
    be processed with REMEDIAL. The same would be applicable to classifiers such as IBLR, as putting two data samples at the same 
    location but having disjoint labelsets tend to confuse this kind of algorithms. Excluding these cases, the global evaluation of 
    the results produced by REMEDIAL would be much more positive.
    
    The proposed hybrid method will firstly decouple imbalanced labels, then will look for instances linked to minority labels and
    will produce clones from them. These new samples will increase the frequency of rare labels without also implying a grow in those 
    linked to majority labels. As a result, the MLDs would have a more balanced label distribution and would be easier to process 
    by MLC algorithms.

    '''
    mean_IR=2*mean_IR*threshold
    list_of_indices_instances=[]
    data_with_labels_and_clones=pd.DataFrame()
    # number of instances in the dataset                                 
    for i in range(data_with_labels.shape[0]):       
        if scumbles[i] > scumble_metric:
            list_of_indices_instances.append(i)
        # clone the affected instances and add them (at the end) into the original dataset
        #data_with_labels_and_clones=pd.concat([data_with_labels,data_with_labels.iloc[list_of_indices_instances]],ignore_index=True)
        # clone the affected instances
        data_clones=data_with_labels.iloc[list_of_indices_instances]
        data_clones=data_clones.reset_index(drop=True)
    ind=0
    for i in list_of_indices_instances:   
        for j,k in enumerate(IRLbls):
            # maintain minority labels
            if  k>mean_IR:
                data_with_labels.iloc[i,dataset_features+j]=0
            # maintain majority labels
            if  k<=mean_IR:
                data_clones.iloc[ind,dataset_features+j]=0
        ind+=1       
    data_with_labels_and_clones=pd.concat([data_with_labels,data_clones],ignore_index=True)
    #----------------------print and plot-------------------------------------#
#    # find labels with the new instances (clones)
#    labels_and_clones=data_with_labels_and_clones.iloc[:,dataset_features:]
#    categories=(labels_and_clones.columns).tolist()
#
#    # find if metrics have improved    
#    labels_and_clones_cols=observe_imbalance_cols(labels_and_clones)
#    labels_and_clones_rows=observe_imbalance_rows(labels_and_clones)
#    label_cardinality_clones=find_label_cardinality(labels_and_clones)
#    label_density_clones=find_label_density(labels_and_clones,label_cardinality_clones)
#    #old plot
#    #labels_and_clones_cols.plot(kind='bar', title='Count (labels with clones)')
#    # find the positive votes for each label 
#    positive_votes_per_label(labels_and_clones,"Positive votes for each label\n"
#                                                         +'threshold: '+str(threshold)
#                                                         +' - cardinality: '+str(label_cardinality_clones)
#                                                         +' - density: '+str(label_density_clones))
#    # find the negative votes for each label 
#    negative_votes_per_label(labels_and_clones,'Negative votes for each label\n'
#                                                                      +'threshold: '+str(threshold)
#                                                                      +' - cardinality: '+str(label_cardinality_clones)
#                                                                      +' - density: '+str(label_density_clones))
#    # Counting the number of instances that have multiple labels.
#    find_multilabel_instances(labels_and_clones,'Instances having multiple labels\n'
#                                               +'threshold:'+str(threshold)
#                                               +' - cardinality: '+str(label_cardinality_clones)
#                                               +' - density: '+str(label_density_clones))
#    
#    IRLbls_clones,mean_IR_clones=find_IRLbl(labels_and_clones_cols,categories)
#    labels_dictionary_clones=transform_dataframe_to_dictionary(labels_and_clones)
#    scumbles_clones,scumble_metric_clones,IRLblis_clones=find_scumble(labels_dictionary_clones,IRLbls_clones,categories)
#    scumble_clones_metric_cv=find_scumble_cv(scumbles_clones,scumble_metric_clones) 
#    print('Data properties after REMEDIAL algorithm - threshold='+str(threshold))
#    #print('labels and_clones cols   :',labels_and_clones_cols)
#    print('label cardinality clones :',label_cardinality_clones)
#    #print('IRLbls clones            :',IRLbls_clones)
#    print('mean IR clones           :',mean_IR_clones)
#    #print('scumbles clones          :',scumbles_clones)
#    print('scumble metric clones    :',scumble_metric_clones)
#    #print('IRLblis clones           :',IRLblis_clones)
#    print('scumble_clones_metric_cv :',scumble_clones_metric_cv)
#    print('')
    #-------------------------------------------------------------------------#
    return(data_with_labels_and_clones)

def observe_imbalance_after_remedial(labels_and_clones,labels_dictionary,categories,threshold,title_name):
    # find if metrics have improved    
    labels_and_clones_cols=observe_imbalance_cols(labels_and_clones)
    labels_and_clones_rows=observe_imbalance_rows(labels_and_clones)
    label_cardinality_clones=find_label_cardinality(labels_and_clones)
    label_density_clones=find_label_density(labels_and_clones,label_cardinality_clones)
    # find the positive votes for each label 
    positive_votes_per_label(labels_and_clones,"Positive votes for each label "+title_name+"\n"
                                                   +'threshold: '+str(threshold)
                                                   +' - cardinality: '+str(label_cardinality_clones)
                                                   +' - density: '+str(label_density_clones))
    # find the negative votes for each label 
    negative_votes_per_label(labels_and_clones,"Negative votes for each label "+title_name+"\n"
                                                   +'threshold: '+str(threshold)
                                                   +' - cardinality: '+str(label_cardinality_clones)
                                                   +' - density: '+str(label_density_clones))
    # Counting the number of instances that have multiple labels.
    find_multilabel_instances(labels_and_clones,"Instances having multiple labels "+title_name+"\n"
                                                    +'threshold:'+str(threshold)
                                                    +' - cardinality: '+str(label_cardinality_clones)
                                                    +' - density: '+str(label_density_clones))
    
    IRLbls_clones,mean_IR_clones=find_IRLbl(labels_and_clones_cols,categories)
    labels_dictionary_clones=transform_dataframe_to_dictionary(labels_and_clones)
    scumbles_clones,scumble_metric_clones,IRLblis_clones=find_scumble(labels_dictionary_clones,IRLbls_clones,categories)
    scumble_clones_metric_cv=find_scumble_cv(scumbles_clones,scumble_metric_clones) 
    #concurrence_builder = LabelCooccurrenceGraphBuilder(weighted=True, include_self_edges=False)
    #concurrence = concurrence_builder.transform(labels_and_clones)
    #print(concurrence)
    
    # plot the co-occurrence matrix and the co-occurrence matrix percentage
    cooccurrence_matrix(labels_and_clones,categories,'data '+title_name)
    print('')
    print('Data properties '+title_name+' algorithm - threshold='+str(threshold))
    print('label cardinality clones :',label_cardinality_clones)
    print('mean IR clones           :',mean_IR_clones)
    print('scumble metric clones    :',scumble_metric_clones)
    print('scumble_clones_metric_cv :',scumble_clones_metric_cv)
    print('')
    return(label_cardinality_clones,label_density_clones,IRLbls_clones,mean_IR_clones,scumble_metric_clones,scumble_clones_metric_cv,IRLblis_clones)

# def get_IRLbl(IRLbl_l,L,x): 
#     return IRLbl_l[L.index(x)] 
# def REMEDIAL(D,SCUMBLE_ins,SCUMBLE,IRMean):
#         # Calculate SCUMBLE
#         #SCUMBLE_ins, SCUMBLE = self.calculate_SCUMBLES()
        
#         # Edit dataset
#         #new_D = copy.D[:] # copy dataset
#         new_D = D.copy()
#         for i in range(len(new_D)):
#             if SCUMBLE_ins[i] > SCUMBLE:
#                 instance, labels = new_D[i]
#                 # Maintain minority labels
#                 min_labels = filter(lambda x: get_IRLbl(x) > IRMean, labels)
#                 maj_labels = filter(lambda x: get_IRLbl(x) <= IRMean, labels)

#                 new_D[i] = (instance, min_labels)
#                 Di = (instance, maj_labels)
#                 new_D.append(Di)

#         return new_D

def find_IRLblis_complete(IRLblis,labels_train):
    '''
    Extension of IRLblis lsit, with the addition of zeros for the empty labels.
    '''
    IRLblis_complete=[]        
    for index in range(len(labels_train)):
        tmp_list=[]
        var=0
        for label in labels_train.iloc[index]:
            if label == 0:
                tmp_list.append(0)
            else:
                tmp_list.append(IRLblis[index][var])
                var +=1
        IRLblis_complete.append(tmp_list)
    return(IRLblis_complete)

def newSample(sample_features,sample_labels,r_neighbors_features,r_neighbors_labels,neighbors_labels,label_generation_method):
    synthetic_sample_features=sample_features.copy()
    synthetic_sample_labels=sample_labels.copy()
    ones=0
    zeros=0
    if label_generation_method=="Ranking":
        for index,feat in enumerate(sample_features):
            diff=abs(feat-r_neighbors_features.iloc[index])
            rng=np.random.uniform(0, 1)
            new_feat=diff*rng
            synthetic_sample_features.iloc[index]=new_feat
        for ind,label in enumerate(sample_labels):
            if label==1:
                ones+=1
            else:
                zeros+=1
            for l in neighbors_labels.iloc[:,ind]:  
                if l==1:
                    ones+=1
                else:
                    zeros+=1  
            if(ones*2>=(ones+zeros)):
                synthetic_sample_labels.iloc[ind]=1
            else:
                synthetic_sample_labels.iloc[ind]=0
                
    elif label_generation_method=="Union":
        for index,feat in enumerate(sample_features):
            diff=abs(feat-r_neighbors_features.iloc[index])
            rng=np.random.uniform(0, 1)
            new_feat=diff*rng
            synthetic_sample_features.iloc[index]=new_feat
        for ind,label in enumerate(sample_labels):
            if label==1:
                ones+=1
            else:
                zeros+=1
            for l in neighbors_labels.iloc[:,ind]:  
                if l==1:
                    ones+=1
                else:
                    zeros+=1  
            if ones>0:
                synthetic_sample_labels.iloc[ind]=1
            else:
                synthetic_sample_labels.iloc[ind]=0
                
    elif label_generation_method=="Intersection":
        for index,feat in enumerate(sample_features):
            diff=abs(feat-r_neighbors_features.iloc[index])
            rng=np.random.uniform(0, 1)
            new_feat=diff*rng
            synthetic_sample_features.iloc[index]=new_feat
        for ind,label in enumerate(sample_labels):
            if label==1:
                ones+=1
            else:
                zeros+=1
            for l in neighbors_labels.iloc[:,ind]:  
                if l==1:
                    ones+=1
                else:
                    zeros+=1  
            if zeros==0:
                synthetic_sample_labels.iloc[ind]=1
            else:
                synthetic_sample_labels.iloc[ind]=0            
    return(synthetic_sample_features,synthetic_sample_labels)

def apply_mlsmote_algorithm(IRLblis_complete,mean_IR,k_neighbors,features_train,labels_train,categories,label_generation_method="Ranking"):
    new_features=pd.DataFrame(columns=features_train.columns)
    new_labels=pd.DataFrame(columns=labels_train.columns)
    for i in range(len(categories)):
        min_bag=[]
        for j in range(len(features_train)):
            if IRLblis_complete[j][i]<mean_IR:
                pass
            else:
                min_bag.append(j)
        if len(min_bag)==0:
            continue
        data_features=features_train.iloc[min_bag]
        data_labels=labels_train.iloc[min_bag]
        
        neigh =NearestNeighbors(n_neighbors=k_neighbors+1)
        neigh.fit(data_features)
        all_neighbors=neigh.kneighbors(data_features, return_distance=False)[:,1:]
        
        for idx in range(len(data_features)):
            sample_features=data_features.iloc[idx]
            sample_labels=data_labels.iloc[idx]
            
            index_neighbors=all_neighbors[idx]
            neighbors_labels=data_labels.iloc[index_neighbors]
            
            index_r_neighbor=index_neighbors[random.randint(0,k_neighbors-1)]
            
            r_neighbors_labels=data_labels.iloc[index_r_neighbor]
            r_neighbors_features=data_features.iloc[index_r_neighbor]
            r_neighbors_features.columns=range(r_neighbors_features.shape[0])
            
            synthetic_features,synthetic_labels=newSample(sample_features,sample_labels,r_neighbors_features,r_neighbors_labels,neighbors_labels,label_generation_method)
            
            new_features=new_features.append(synthetic_features,ignore_index=True)
            new_labels=new_labels.append(synthetic_labels,ignore_index=True)
    return(new_features,new_labels)

def observe_imbalance_after_mlsmote(labels_and_clones,labels_dictionary,categories,title_name):
    # find if metrics have improved    
    labels_and_clones_cols=observe_imbalance_cols(labels_and_clones)
    labels_and_clones_rows=observe_imbalance_rows(labels_and_clones)
    label_cardinality_clones=find_label_cardinality(labels_and_clones)
    label_density_clones=find_label_density(labels_and_clones,label_cardinality_clones)
    # find the positive votes for each label 
    positive_votes_per_label(labels_and_clones,"Positive votes for each label "+title_name+"\n"
                                                   +' - cardinality: '+str(label_cardinality_clones)
                                                   +' - density: '+str(label_density_clones))
    # find the negative votes for each label 
    negative_votes_per_label(labels_and_clones,"Negative votes for each label "+title_name+"\n"
                                                   +' - cardinality: '+str(label_cardinality_clones)
                                                   +' - density: '+str(label_density_clones))
    # Counting the number of instances that have multiple labels.
    find_multilabel_instances(labels_and_clones,"Instances having multiple labels "+title_name+"\n"
                                                    +' - cardinality: '+str(label_cardinality_clones)
                                                    +' - density: '+str(label_density_clones))
    
    IRLbls_clones,mean_IR_clones=find_IRLbl(labels_and_clones_cols,categories)
    labels_dictionary_clones=transform_dataframe_to_dictionary(labels_and_clones)
    scumbles_clones,scumble_metric_clones,IRLblis_clones=find_scumble(labels_dictionary_clones,IRLbls_clones,categories)
    scumble_clones_metric_cv=find_scumble_cv(scumbles_clones,scumble_metric_clones) 
    #concurrence_builder = LabelCooccurrenceGraphBuilder(weighted=True, include_self_edges=False)
    #concurrence = concurrence_builder.transform(labels_and_clones)
    #print(concurrence)
    
    # plot the co-occurrence matrix and the co-occurrence matrix percentage
    #cooccurrence_matrix(labels_and_clones,categories,'data '+title_name)   # NEED TO FIX THIS !!
    print('')
    print('Data properties '+title_name+' algorithm')
    print('label cardinality clones :',label_cardinality_clones)
    print('mean IR clones           :',mean_IR_clones)
    print('scumble metric clones    :',scumble_metric_clones)
    print('scumble_clones_metric_cv :',scumble_clones_metric_cv)
    print('')
    return(label_cardinality_clones,label_density_clones,IRLbls_clones,mean_IR_clones,scumble_metric_clones,scumble_clones_metric_cv,IRLblis_clones)


def print_metrics(labels_test, prediction,name):
    '''
    Hamming-Loss is the fraction of labels that are incorrectly predicted, i.e., 
    the fraction of the wrong labels to the total number of labels.
    
    Exact Match Ratio (Subset accuracy) is the most strict metric, 
    indicating the percentage of samples that have all their labels classified correctly.
    The disadvantage of this measure is that multi-class classification problems have a chance 
    of being partially correct, but here we ignore those partially correct matches.
   
    F1-Score is the harmonic mean between precision and recall.
    Macro-averaging method can be used when you want to know how the system performs overall across the sets of data.
    You should not come up with any specific decision with this average. 
    Micro-averaging can be a useful measure when your dataset varies in size.
    '''
    HL=metrics.hamming_loss(labels_test, prediction)
    ACC=metrics.accuracy_score(labels_test, prediction)
    F1_micro=f1_score(labels_test, prediction, average='micro')
    F1_macro=f1_score(labels_test, prediction, average='macro')
    precision_micro=precision_score(labels_test, prediction, average='micro')
    precision_macro=precision_score(labels_test, prediction, average='macro')
    recall_micro=recall_score(labels_test, prediction, average='micro')
    recall_macro=recall_score(labels_test, prediction, average='macro')
    print(name)
    print("Hamming loss   :", HL)
    print("Accuracy       :", ACC)
    print("f1 score micro :", F1_micro)
    print("f1 score macro :", F1_macro)
    print("precision micro:", precision_micro)
    print("precision macro:", precision_macro)
    print("recall micro   :", recall_micro)
    print("recall macro   :", recall_macro)
    print(metrics.classification_report(labels_test, prediction,zero_division=0))
    print('')
    #return(HL,ACC,F1_micro,F1_macro,precision_micro,precision_macro,recall_micro,recall_macro)
