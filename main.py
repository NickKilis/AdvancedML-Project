#-----------------------------------------------------------------------------#
# LIST OF CONTENTS :
#
# 0.PRIOR INSTALLATIONS
# 1.IMPORT LIBRARIES
# 2.CHOOSE DATASET
# 3.DATA LOAD
# 4.GET LABELS AND TRANSFORM THEM FROM BYTES
# 5.TRANSFORM LABELS INTO A DICTIONARY
# 6.DATA DESCRIPTION              
# 7.DATA VISUALISATION  
# 8.OBSERVE IMBALANCE INITIAL DATA
#   8a.Cardinality/Label Density
#   8b.IRLbl (Imbalance ratio of each label)
#   8c.SCUMBLE metric (Score of ConcUrrence among iMBalanced LabEls)
#   8d.SCUMBLE_CV metric (coefficient of variation of SCUMBLE)
# 9.REMEDIAL ALGORITHM
#   8a.IMBALANCE COMPARISON REMEDIAL
# 10.MLSMOTE ALGORITHM
#   10a.IMBALANCE COMPARISON MLSMOTE
# 11.GRID SEARCH FOR 4 DATASETS (original,original-remedial,original-mlsmote,remedial-mlsmote)
#   11a.BINARY RELEVANCE
#   11b.MULTI LABEL KNN
#   11c.BEST FIT FOR 4 DATASETS- BINARY RELEVANCE
#   11d.BEST FIT FOR 4 DATASETS- MULTI LABEL KNN
# 12.BUILD A CLASSIFIER FROM THE BEST PARAMETERS
#   12a.Binary Relevance (BR)
#   12b.Multi-label KNN (MLKNN)
# 13.EVALUATION
#   13a.Metrics (HL,ACC,F1_micro,F1_macro,precision_micro,precision_macro,recall_micro,recall_macro)
#   13b.Confusion Matrix
# 14.INTERPRETATION
#   14a.Local Interpretable Model-agnostic Explanations (LIME)
#   14b.Multi-label Rule learning (MLRL) - via an application written in JAVA
#
#-----------------------------------------------------------------------------#
#--------------------------PRIOR INSTALLATIONS--------------------------------#
# pip install imblearn
# pip install pactools
# pip install LIME
#--------------------------IMPORT LIBRARIES-----------------------------------#
import time,collections,operator
import numpy as np 
import pandas as pd
from scipy.io import arff
from utilities.data_visualization import data_visualization_control,cooccurrence_matrix,find_feature_and_label_dist,find_best_distribution
from utilities.data_formulation import dataframe_bytes_to_int,transform_dataframe_to_dictionary,find_dataset_in_folder,grid_search_mlknn,grid_search_br
from utilities.data_interpretation import hyperparameters_control,lime_interpreter
import utilities.label_imbalance as imba
from sklearn.metrics import multilabel_confusion_matrix,classification_report
import warnings
warnings.filterwarnings("ignore")
#-----------------------------------------------------------------------------#
#----------------------------START THE OVERALL TIMER--------------------------#
start_of_program = time.perf_counter()
#-------------------------------CHOOSE DATASET--------------------------------#
dataset_name='emotions'
#dataset_name='yeast'
#dataset_name='scene'
#dataset_name='corel5k'
print('You have selected the dataset :',dataset_name)
print('')
#-----------------------------------------------------------------------------#
#-------------------------------DATA LOAD-------------------------------------#
# open the dataset you selected (check if it exists) and store the number of features and labels
file_to_open_data,file_to_open_train,file_to_open_test,dataset_instances,dataset_features,dataset_labels=find_dataset_in_folder(dataset_name)
data = arff.loadarff(file_to_open_data)
data_train = arff.loadarff(file_to_open_train)
data_test = arff.loadarff(file_to_open_test)
# convert them into dataframes
data = pd.DataFrame(data[0])
data_train = pd.DataFrame(data_train[0])
data_test = pd.DataFrame(data_test[0])
#-----------------------------------------------------------------------------#
#----------------GET LABELS AND TRANSFORM THEM FROM BYTES---------------------#
# take only the training set to work with
data=data_train
# extract labels from data
labels=pd.DataFrame(data.iloc[0:data.shape[0],dataset_features:data.shape[1]])
# transform datatype "bytes" into integers
labels=dataframe_bytes_to_int(labels)
# split into data with and without labels
features=data.iloc[:,:dataset_features]
dataset_original=pd.concat([features, labels],axis=1)
# keep label names in a list
categories=(labels.columns).tolist()
#-----------------------------------------------------------------------------#
#----------------TRANSFORM LABELS INTO A DICTIONARY---------------------------#
labels_dictionary=transform_dataframe_to_dictionary(labels)
#----------------------------DATA DESCRIPTION---------------------------------# 
#data_description(dataset_original,features,data)
#---------------------------DATA VISUALISATION--------------------------------# 
#df=data_test
#data_visualization_control(df)
# find the feature distributions
destination_folder_plots="feature_distributions"
feature_distributions=find_feature_and_label_dist(dataset_original,dataset_features,destination_folder_plots)
if feature_distributions != None:
    feature_distributions_dict=collections.Counter(feature_distributions)
    feature_distributions_sorted = sorted(feature_distributions_dict.items(),reverse=True,key=operator.itemgetter(1))
    feature_distributions_sorted_df = pd.DataFrame(feature_distributions_sorted,columns =['Dist.Name','Frequency']).T
    print("")
    print(feature_distributions_sorted_df)
    print("")
#-----------------------------------------------------------------------------#
#--------------------OBSERVE IMBALANCE INITIAL DATA---------------------------#
title_name="Initial data"
label_cardinality,label_density,IRLbls,mean_IR,scumbles,scumble_metric,scumble_metric_cv,IRLblis=imba.observe_imbalance(labels,labels_dictionary,categories,title_name)
#-----------------------------------------------------------------------------#
#------------------------REMEDIAL ALGORITHM-----------------------------------#
#threshold_values=np.array([0.25,0.37,0.5,0.62,0.75])
threshold_values=np.array([0.5]) # threshold=0.5 (original REMEDIAL paper)
for threshold in threshold_values:
    tmp=dataset_original.copy()
    dataset_original_remedial=imba.apply_remedial_algorithm(tmp,scumbles,scumble_metric,IRLbls,mean_IR,threshold,dataset_features)
dataset_original_remedial.columns=dataset_original.columns    
#-----------------------------------------------------------------------------#
#------------------------IMBALANCE COMPARISON REMEDIAL------------------------#
# find labels with the new instances (clones)
labels_original_remedial=dataset_original_remedial.iloc[:,dataset_features:]
labels_original_remedial.columns=labels.columns
title_name_remedial="after REMEDIAL"
label_cardinality_remedial,label_density_remedial,IRLbls_remedial,mean_IR_remedial,scumble_metric_remedial,scumble_metric_cv_remedial,IRLblis_remedial=imba.observe_imbalance_after_remedial(labels_original_remedial,labels_dictionary,categories,threshold_values[0],title_name_remedial)
#-----------------------------------------------------------------------------#
#--------------------------MLSMOTE ALGORITHM----------------------------------#
features_train=data_train.iloc[:,:dataset_features]
labels_train=data_train.iloc[:,dataset_features:]
labels_train=dataframe_bytes_to_int(labels_train)

features_test=data_test.iloc[:,:dataset_features]
labels_test=data_test.iloc[:,dataset_features:]
labels_test=dataframe_bytes_to_int(labels_test)
k_neighbors=3
# mlsmote's new samples
IRLblis_complete=imba.find_IRLblis_complete(IRLblis,labels_train)
label_generation_method="Ranking"
#label_generation_method="Union"
#label_generation_method="Intersection" 
features_mlsmote,labels_mlsmote=imba.apply_mlsmote_algorithm(IRLblis_complete,mean_IR,k_neighbors,features_train,labels_train,categories,label_generation_method)

# create new dataset from the original data
features_original_mlsmote=pd.concat([features, features_mlsmote], ignore_index=True)
labels_original_mlsmote=pd.concat([labels, labels_mlsmote], ignore_index=True)
dataset_original_mlsmote = pd.concat([features_original_mlsmote, labels_original_mlsmote], axis=1, ignore_index=True)

# create new dataset from the data after REMEDIAL algorithm
dataset_mlsmote = pd.concat([features_mlsmote, labels_mlsmote], axis=1, ignore_index=True)
dataset_original_remedial.columns =range(0,dataset_features+dataset_labels)

features_train_rem=dataset_original_remedial.iloc[:,:dataset_features]
labels_train_rem=dataset_original_remedial.iloc[:,dataset_features:]
IRLblis_complete_rem=imba.find_IRLblis_complete(IRLblis_remedial,labels_train_rem)
mean_IR_rem=mean_IR_remedial
features_rem_mlsmote,labels_rem_mlsmote=imba.apply_mlsmote_algorithm(IRLblis_complete_rem,mean_IR_rem,k_neighbors,features_train_rem,labels_train_rem,categories,label_generation_method)
dataset_rem_mlsmote = pd.concat([features_rem_mlsmote, labels_rem_mlsmote], axis=1, ignore_index=True)
dataset_rem_mlsmote.columns=data.columns
dataset_original_remedial.columns=data.columns
dataset_remedial_mlsmote=pd.concat([dataset_original_remedial, dataset_rem_mlsmote], axis=0, ignore_index=True)
#-----------------------------------------------------------------------------#
#------------------------IMBALANCE COMPARISON MLSMOTE-------------------------#
title_name_mlsmote="after MLSMOTE"
label_cardinality_mlsmote,label_density_mlsmote,IRLbls_mlsmote,mean_IR_mlsmote,scumble_metric_mlsmote,scumble_metric_cv_mlsmote,IRLblis_mlsmote=imba.observe_imbalance_after_mlsmote(labels_mlsmote,labels_dictionary,categories,title_name_mlsmote)
#-----------------------------------------------------------------------------#
#----------------------GRID SEARCH FOR 4 DATASETS-----------------------------#
#-------------------GRID SEARCH - BINARY RELEVANCE----------------------------#
toggle_grid_search_br=False
# choose a dataset to perform grid search
select_dataset="original"
#select_dataset="original-remedial"
#select_dataset="original-mlsmote"
#select_dataset="remedial-mlsmote"
if toggle_grid_search_br:
    since = time.time()
    grid_search_br(select_dataset,dataset_original,dataset_original_remedial,dataset_original_mlsmote,dataset_remedial_mlsmote,features_test,labels_test,dataset_features)
    time_elapsed = time.time() - since
    print('Grid search for BR-RF completed in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print("")
#-----------------------------------------------------------------------------#
#----------------BEST FIT FOR 4 DATASETS- BINARY RELEVANCE--------------------#
print("RESULTS AFTER GRID SEARCH - BRRF-100-entropy :")
print("The best model parameters for dataset original are : ")
print("RobustScaler ,StandardScaler(with_mean=True,with_std=True)")
print("Training score=0.65440164847305 ,Test score=0.6695035460992907")
'''
Hamming loss   : 0.19224422442244224
Accuracy       : 0.28217821782178215
f1 score micro : 0.6695035460992907
f1 score macro : 0.6458767877194348
precision micro: 0.7712418300653595
precision macro: 0.7886896020836346
recall micro   : 0.5914786967418546
recall macro   : 0.5767868551180902
'''
print("")
print("RESULTS AFTER GRID SEARCH - BRRF-50-gini :")
print("The best model parameters for dataset original-remedial are : ")
print("QuantileTransformer(n_quantiles=len(dataset)), StandardScaler(with_mean=True,with_std=True)")
print("Training score=0.2612727439281964 ,Test score=0.4280442804428044")
'''
Hamming loss   : 0.25577557755775576
Accuracy       : 0.17326732673267325
f1 score micro : 0.4280442804428044
f1 score macro : 0.41073333898393
precision micro: 0.8111888111888111
precision macro: 0.7957343807088053
recall micro   : 0.2907268170426065
recall macro   : 0.3146019397472191
'''
print("")
print("RESULTS AFTER GRID SEARCH - BRRF-50-entropy :")
print("The best model parameters for dataset original-mlsmote are : ")
print("RobustScaler, StandardScaler(with_mean=True,with_std=True)")
print("Training score=0.39838994700914265 ,Test score=0.6732954545454546")
'''
Hamming loss   : 0.18976897689768976
Accuracy       : 0.27722772277227725
f1 score micro : 0.6732954545454546
f1 score macro : 0.6551419009366987
precision micro: 0.7770491803278688
precision macro: 0.8046823935842627
recall micro   : 0.5939849624060151
recall macro   : 0.5803896030109489
'''
print("")
print("RESULTS AFTER GRID SEARCH - BRRF-50-gini :")
print('The best model parameters for dataset remedial-mlsmote are : ')
print("PowerTransformer(method='yeo-johnson') ,StandardScaler(with_mean=True,with_std=True)")
print("Training score=0.16809269270584465 ,Test score=0.39999999999999997")
'''
Hamming loss   : 0.2599009900990099
Accuracy       : 0.14356435643564355
f1 score micro : 0.39999999999999997
f1 score macro : 0.37979303164821593
precision micro: 0.8333333333333334
precision macro: 0.8646464646464646
recall micro   : 0.2631578947368421
recall macro   : 0.29160842393944614
'''
print("")
#-----------------------------------------------------------------------------#
#-------------------GRID SEARCH - MULTI LABEL KNN-----------------------------#
toggle_grid_search_mlknn=False  
# choose a dataset to perform grid search
select_dataset="original"
#select_dataset="original-remedial"
#select_dataset="original-mlsmote"
#select_dataset="remedial-mlsmote"
if toggle_grid_search_mlknn: 
    since = time.time()       
    grid_search_mlknn(select_dataset,dataset_original,dataset_original_remedial,dataset_original_mlsmote,dataset_remedial_mlsmote,features_test,labels_test,dataset_features)    
    time_elapsed = time.time() - since
    print('Grid search for BR-RF completed in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print("")
#-----------------------------------------------------------------------------#
#----------------BEST FIT FOR 4 DATASETS- MULTI LABEL KNN---------------------#
print("RESULTS AFTER GRID SEARCH - MLKNN :")
print("The best model parameters for dataset original are : ")
print("PowerTransformer(method='yeo-johnson') ,StandardScaler(with_mean='passthrough',with_std=False) ,MLKNN(k=8 ,s=0.1)")
print("Training score=0.6407355047080833 ,Test score=0.7037974683544304")
'''
Hamming loss   : 0.19306930693069307
Accuracy       : 0.31683168316831684
f1 score micro : 0.7037974683544304
f1 score macro : 0.6934647528658621
precision micro: 0.710997442455243
precision macro: 0.7262745644371184
recall micro   : 0.6967418546365914
recall macro   : 0.6952434626307052
'''
print("")
print("The best model parameters for dataset original-remedial are : ")
print("QuantileTransformer(n_quantiles=len(dataset)) ,StandardScaler(with_mean='passthrough',with_std=False) ,MLKNN(k=10 ,s=0.1)")
print("Training score=0.3292094032082703 ,Test score=0.5584415584415584")
'''
Hamming loss   : 0.22442244224422442
Accuracy       : 0.22772277227722773
f1 score micro : 0.5584415584415584
f1 score macro : 0.5236396377543627
precision micro: 0.7926267281105991
precision macro: 0.8461309998203627
recall micro   : 0.43107769423558895
recall macro   : 0.43579912456510694
'''
print("")
print("The best model parameters for dataset original-mlsmote are : ")
print("PowerTransformer(method='yeo-johnson') ,'passthrough' ,MLKNN(k=4 ,s=0.1)")
print("Training score=0.4726897157669262 ,Test score=0.6752577319587629")
'''
Hamming loss   : 0.21122112211221122
Accuracy       : 0.2524752475247525
f1 score micro : 0.6775818639798489
f1 score macro : 0.6520425288450847
precision micro: 0.6810126582278481
precision macro: 0.6856636812057074
recall micro   : 0.6741854636591479
recall macro   : 0.6537124351706801
'''
print("")
print('The best model parameters for dataset remedial-mlsmote are : ')
print("QuantileTransformer(n_quantiles=len(dataset)) ,StandardScaler(with_mean='passthrough',with_std=False) ,MLKNN(k=4 ,s=0.1)")
print("Training score=0.31740728257804435 ,Test score=0.5540334855403348")
'''
Hamming loss   : 0.24174917491749176
Accuracy       : 0.16336633663366337
f1 score micro : 0.5540334855403348
f1 score macro : 0.5481796577350997
precision micro: 0.7054263565891473
precision macro: 0.7501945575206103
recall micro   : 0.45614035087719296
recall macro   : 0.5015480793960311
'''
print("")
#-----------------------------------------------------------------------------#
#--------------BUILD A CLASSIFIER FROM THE BEST PARAMETERS--------------------#
# choose a dataset
#select_dataset="original"
#select_dataset="original-remedial"
select_dataset="original-mlsmote"
#select_dataset="remedial-mlsmote"

# choose a scaler
# select_scaler="minmax"
#select_scaler="quantile"
# select_scaler="robust"
select_scaler="power"

# choose an extra scaler
select_extra_scaler="standard"

# choose a classifier (Binary-relevance with RF or MLKNN)
#select_classifier="randomforest"
select_classifier="mlknn"

# # choose a the classifiers parameters
classifier_params={"with_mean":'passthrough',"with_std":'passthrough',"n_estimators":50,"criterion":"entropy","k":4,"s":0.1}
x_train,y_train,x_test,y_test,classifier,y_pred=hyperparameters_control(select_dataset,select_scaler,select_extra_scaler,select_classifier,classifier_params,
                     dataset_original,dataset_original_remedial,dataset_original_mlsmote,dataset_remedial_mlsmote,features_test,labels_test,dataset_features)

model_name=f"model-dataset-{select_dataset}-scaler-{select_scaler}-extra_scaler-{select_extra_scaler}-clf-{select_classifier}-{int(time.time())}"
print("")
print(model_name)
print("You have selected the dataset       : ",select_dataset)
print("You have selected the scaler        : ",select_scaler)
print("You have selected an extra scaler   : ",select_extra_scaler)
print("You have selected the classifier    : ",select_classifier)
print("")
#-----------------------------------------------------------------------------#
#------------------------------EVALUATION-------------------------------------#
imba.print_metrics(y_test,y_pred,select_classifier)
print("")
print(multilabel_confusion_matrix(y_test, y_pred))
print("")
print(classification_report(y_test, y_pred))
#-----------------------------------------------------------------------------#
#---------------------------INTERPRETATION------------------------------------#
# use lime to explain individual predictions
rng=True
if rng:
    print("You have selected a random instance to be interpreted!")
    instance=None
else:
    instance=100
    print("You have selected the instance : ",instance," to be interpreted!")
    
lime_interpreter(dataset_features,x_train,x_test,classifier,model_name,rng,instance)
#-----------------------------------------------------------------------------#
# FIND HOW MUCH TIME HAS ELAPSED
end_of_program = time.perf_counter()
elapsed_time_of_program = end_of_program - start_of_program
print('The program finished in : '+ str(elapsed_time_of_program)+' seconds.')
