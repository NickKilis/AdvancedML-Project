#-------------------------------FUNCTIONS-------------------------------------#
#
# 1.dataframe_bytes_to_int(labels) 
#     input : a dataframe containing datatype bytes
#     output: a dataframe containing datatype integers
#
# 2.transform_dataframe_to_dictionary(labels)
#     input : a dataframe
#     output: a dictionary of the input dataframe
#
# 3.return_string_from_list(mylist)
#     input : a list
#     output: a list containing only the input's strings 
#
# 4.find_dataset_in_folder(dataset_name)
#     input : string of the name of the dataset
#     output: prints out if it exists and gives it's features, #instances,#labels
#
# 5.grid_search_mlknn(select_dataset,dataset_original,dataset_original_remedial,dataset_original_mlsmote,dataset_remedial_mlsmote,features_test,labels_test,dataset_features)
#     input : the dataset's name, the 4 datasets, the test features and labels, the number of features
#     output: best scores and metrics
#
# 6.grid_search_br(select_dataset,dataset_original,dataset_original_remedial,dataset_original_mlsmote,dataset_remedial_mlsmote,features_test,labels_test,dataset_features)
#     input : the dataset's name, the 4 datasets, the test features and labels, the number of features
#     output: best scores and metrics
#-----------------------------------------------------------------------------#
#--------------------------IMPORT LIBRARIES-----------------------------------#
import time
import pandas as pd
import numpy as np
from pathlib import Path
import utilities.label_imbalance as imba
from sklearn.preprocessing import MinMaxScaler,RobustScaler,QuantileTransformer,PowerTransformer,StandardScaler
from sklearn.ensemble import RandomForestClassifier
from imblearn.ensemble import BalancedRandomForestClassifier
from skmultilearn.problem_transform import BinaryRelevance
from sklearn.svm import SVC
from sklearn.model_selection import KFold
from imblearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score,f1_score
#from sklearn.pipeline import make_pipeline,Pipeline,FeatureUnion
from skmultilearn.adapt import MLkNN
from sklearn.model_selection import GridSearchCV
#-----------------------------------------------------------------------------#
#-------------------------------FUNCTIONS-------------------------------------#
def dataframe_bytes_to_int(labels):
    xxx=pd.DataFrame()
    for i in range(labels.shape[1]):
        xx=[]
        for x in labels[labels.columns[i]]:
            xx.append(int(x.decode()))
        xxx[labels.columns[i]]=xx
    return(xxx)
    
def transform_dataframe_to_dictionary(labels):
    labels_dictionary={}
    #labels_dictionary['key'] = []
    for i in range(len(labels)):
        #instance='instance_'+str(i)
        labels_dictionary.setdefault('instance', [])
        x_series=labels.iloc[i]
        list_of_ones = list(x_series.loc[x_series != 0].index)
        labels_dictionary['instance'].append(list_of_ones)
        
    labels_dictionary2=labels_dictionary['instance']  
    labels_dictionary = { i : labels_dictionary2[i] for i in range(0, len(labels_dictionary2) ) }    
    return(labels_dictionary)
    
def return_string_from_list(mylist):
    tmp_list=[]
    for j in range(len(mylist)):
        for i in mylist[j]:
            if isinstance(i, str):
                tmp_list.append(i)
    return(tmp_list)

def find_dataset_in_folder(dataset_name):
    if dataset_name=='emotions':
        dataset_instances=593
        dataset_features=72
        dataset_labels=6
    if dataset_name=='yeast':
        dataset_name='yeast'
        dataset_instances=2417
        dataset_features=103
        dataset_labels=13
    if dataset_name=='scene':
        dataset_name='scene'
        dataset_instances=2407
        dataset_features=294
        dataset_labels=6
    if dataset_name=='corel5k':
        dataset_name='corel5k'
        dataset_instances=5000
        dataset_features=499
        dataset_labels=374   
        
    data_folder = Path('datasets/'+dataset_name+'/')
    file_to_open_data = data_folder / str(dataset_name+'.arff')
    file_to_open_train= data_folder / str(dataset_name+'-train.arff')
    file_to_open_test= data_folder / str(dataset_name+'-test.arff')
    if not data_folder.exists():
        print("Oops,the folder: "+str(data_folder)+" ,doesn't exist!")
    else:
        print("Yay,the folder: "+str(data_folder)+" ,exists!")
    if not file_to_open_data.exists():
        print("Oops,the file: "+str(file_to_open_data)+" ,doesn't exist!")
    else:
        print("Yay,the file: "+str(file_to_open_data)+" ,exists!")
    if not file_to_open_train.exists():
        print("Oops,the file: "+str(file_to_open_train)+" ,doesn't exist!")
    else:
        print("Yay,the file: "+str(file_to_open_train)+" ,exists!")
    if not file_to_open_test.exists():
        print("Oops,the file: "+str(file_to_open_test)+" ,doesn't exist!")
    else:
        print("Yay,the file: "+str(file_to_open_test)+" ,exists!")
    print('')
    return(file_to_open_data,file_to_open_train,file_to_open_test,dataset_instances,dataset_features,dataset_labels)


def grid_search_mlknn(select_dataset,dataset_original,dataset_original_remedial,dataset_original_mlsmote,dataset_remedial_mlsmote,features_test,labels_test,dataset_features):
    # datasets
    if select_dataset =="original":
        # find the metrics for the initial data
        df=dataset_original.copy()
    elif select_dataset =="original-remedial":
        # find the metrics for the data after REMEDIAL algorithm
        df=dataset_original_remedial.copy()
    elif select_dataset =="original-mlsmote":    
        # find the metrics for the data after MLSMOTE algorithm
        df=dataset_original_mlsmote.copy()
    elif select_dataset =="remedial-mlsmote":
        # find the metrics for the data after REMEDIAL and MLSMOTE algorithms
        df=dataset_remedial_mlsmote.copy()
    
    x_test=features_test.copy()
    y_test=labels_test.copy()
    
    x_train=df.iloc[:,:dataset_features]
    x_train.columns=x_test.columns
    #x_train.columns=range(x_train.shape[1])
    y_train=df.iloc[:,dataset_features:]
    y_train.columns=y_test.columns
    #y_train.columns=range(y_train.shape[1])
    
    # # transform into numpy array
    x_test_array=np.asarray(x_test)
    y_test_array=np.asarray(y_test)
    x_train_array=np.asarray(x_train)
    y_train_array=np.asarray(y_train)
    y_train_array= np.array(y_train_array, dtype=int)
    
    scalers_to_test =[MinMaxScaler(), RobustScaler(), QuantileTransformer(n_quantiles=len(df)), PowerTransformer(method='yeo-johnson')]
    
    for i in scalers_to_test:
        pipeline_MLKNN = Pipeline([
            ("scaler",i ),
            ("extra_scaler", StandardScaler()),
            ("classifier", MLkNN())              
            ])
        parameters = {}
        parameters['classifier__k'] = [2,3,4,5,6,7,8,9,10]
        parameters['classifier__s'] = [0.1, 0.25, 0.5, 0.75, 1.0]
        parameters['extra_scaler__with_mean']= ["passthrough",True, False]
        parameters['extra_scaler__with_std'] = ["passthrough",True, False]
        
        num_splits=10
        score = 'f1_micro'
        print("Grid search for MLKNN of",len(scalers_to_test)*len(parameters['classifier__k'])*len(parameters['classifier__s'])*len(parameters['extra_scaler__with_mean'])*len(parameters['extra_scaler__with_std'])*num_splits,"fits...")
        clf_MLKNN = GridSearchCV(pipeline_MLKNN, parameters, verbose=1,cv=num_splits, scoring=score).fit(x_train_array, y_train_array)
        print("Grid search for : ", i)
        print("The best model parameters are : ",clf_MLKNN.best_params_)
        print("The best model estimator is   : ",clf_MLKNN.best_estimator_) 
        print("The best model training score is : ",clf_MLKNN.best_score_)
        print('Final test score is: ', clf_MLKNN.score(x_test_array, y_test_array))
        imba.print_metrics(y_test,clf_MLKNN.predict(x_test),clf_MLKNN)
        
def grid_search_br(select_dataset,dataset_original,dataset_original_remedial,dataset_original_mlsmote,dataset_remedial_mlsmote,features_test,labels_test,dataset_features):
    # datasets
    if select_dataset =="original":
        # find the metrics for the initial data
        df=dataset_original.copy()
    elif select_dataset =="original-remedial":
        # find the metrics for the data after REMEDIAL algorithm
        df=dataset_original_remedial.copy()
    elif select_dataset =="original-mlsmote":    
        # find the metrics for the data after MLSMOTE algorithm
        df=dataset_original_mlsmote.copy()
    elif select_dataset =="remedial-mlsmote":
        # find the metrics for the data after REMEDIAL and MLSMOTE algorithms
        df=dataset_remedial_mlsmote.copy()
    
    x_test=features_test.copy()
    y_test=labels_test.copy()
    
    x_train=df.iloc[:,:dataset_features]
    x_train.columns=x_test.columns
    #x_train.columns=range(x_train.shape[1])
    y_train=df.iloc[:,dataset_features:]
    y_train.columns=y_test.columns
    #y_train.columns=range(y_train.shape[1])
    
    # # transform into numpy array
    x_test_array=np.asarray(x_test)
    y_test_array=np.asarray(y_test)
    x_train_array=np.asarray(x_train)
    y_train_array=np.asarray(y_train)
    y_train_array= np.array(y_train_array, dtype=int)
    
    scalers_to_test =[MinMaxScaler(), RobustScaler(), QuantileTransformer(n_quantiles=len(df)), PowerTransformer(method='yeo-johnson')]
    #BalancedRandomForestClassifier
    for i in scalers_to_test:
        pipeline_MLKNN = Pipeline([
            ("scaler",i ),
            ("extra_scaler", StandardScaler()),
            ("clf", BinaryRelevance(classifier=RandomForestClassifier(n_estimators=100,criterion="entropy")))              
            ])
        parameters = {}
        #parameters['clf__classifier__n_estimators'] = [50,150,200]
        #parameters['clf__classifier__criterion'] = ["gini", "entropy"]
        parameters['extra_scaler__with_mean']= ["passthrough",True, False]
        parameters['extra_scaler__with_std'] = ["passthrough",True, False]
        
        num_splits=10
        score = 'f1_micro'
        #print("Grid search for MLKNN of",len(scalers_to_test)*len(parameters['classifier__k'])*len(parameters['classifier__s'])*len(parameters['extra_scaler__with_mean'])*len(parameters['extra_scaler__with_std'])*num_splits,"fits...")
        clf_BRRF = GridSearchCV(pipeline_MLKNN, parameters, verbose=1,cv=num_splits, scoring=score).fit(x_train_array, y_train_array)
        print("Grid search for : ", i)
        print("The best model parameters are : ",clf_BRRF.best_params_)
        print("The best model estimator is   : ",clf_BRRF.best_estimator_) 
        print("The best model training score is : ",clf_BRRF.best_score_)
        print('Final test score is: ', clf_BRRF.score(x_test_array, y_test_array))
        imba.print_metrics(y_test,clf_BRRF.predict(x_test),clf_BRRF)
        
