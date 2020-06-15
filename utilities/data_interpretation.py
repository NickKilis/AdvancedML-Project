#-------------------------------FUNCTIONS-------------------------------------#
#
# 1.norm_probabilities(prob_of_labels)
#     input : probabilities
#     output: normalized probabilities (their sum equal to 1)
#
# 2.hyperparameters_control(select_dataset,select_scaler,select_extra_scaler,select_classifier,classifier_params,
#            dataset_original,dataset_original_remedial,dataset_original_mlsmote,dataset_remedial_mlsmote,features_test,labels_test,dataset_features)
#     input : the dataset's name, the 4 datasets, the test features and labels, the number of features, the classifier's parameters
#     output: x_train,y_train,x_test,y_test,classifier,y_pred
#
# 3.lime_interpreter(dataset_features,x_train,x_test,classifier,model_name,rng=True,instance=None)
#     input : the number of features,x_train,x_test,classifier,model_name, a random instance (rng=True) or a specific instance
#     output: html file with unique name (model name), with the interpretation
#
#-----------------------------------------------------------------------------#
#--------------------------IMPORT LIBRARIES-----------------------------------#
import numpy as np
from sklearn.preprocessing import MinMaxScaler,RobustScaler,QuantileTransformer,PowerTransformer,StandardScaler
from sklearn.ensemble import RandomForestClassifier
#from imblearn.ensemble import BalancedRandomForestClassifier
from skmultilearn.problem_transform import BinaryRelevance
from sklearn.svm import SVC
from sklearn.preprocessing import MinMaxScaler,RobustScaler,QuantileTransformer,PowerTransformer,StandardScaler
from skmultilearn.problem_transform import BinaryRelevance
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression,LogisticRegression
from lime.lime_tabular import LimeTabularExplainer
from scipy.sparse import csr_matrix
from skmultilearn.adapt import MLkNN
#-----------------------------------------------------------------------------#
#-------------------------------FUNCTIONS-------------------------------------#
def norm_probabilities(prob_of_labels):
        def norm(probs):
            prob_factor = 1 / sum(probs)
            return [prob_factor * p for p in probs]
        result= np.empty((0,6), float)
        for row in range(len(prob_of_labels)):
            result=np.vstack([result,norm(prob_of_labels[row])])
        return(result)

def hyperparameters_control(select_dataset,select_scaler,select_extra_scaler,select_classifier,classifier_params,
                            dataset_original,dataset_original_remedial,dataset_original_mlsmote,dataset_remedial_mlsmote,features_test,labels_test,dataset_features):
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
    
    # normalization-scaling
    if select_scaler=="minmax":
        scaler = MinMaxScaler()                             # this does not keep the negative values
    elif select_scaler=="quantile":                                         
        scaler = QuantileTransformer(n_quantiles=len(df))   # this does not keep the negative values
    elif select_scaler=="robust":
        scaler = RobustScaler()                             # this keeps the negative values
    elif select_scaler=="power":
        scaler = PowerTransformer(method='yeo-johnson')     # this keeps the negative values
        
    x_test=features_test.copy()
    y_test=labels_test.copy()
    
    x_train=df.iloc[:,:dataset_features]
    x_train.columns=x_test.columns
    #x_train.columns=range(x_train.shape[1])
    y_train=df.iloc[:,dataset_features:]
    y_train.columns=y_test.columns
    #y_train.columns=range(y_train.shape[1])
    
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)
    
    # # transform into numpy array
    x_test_array=np.asarray(x_test)
    # y_test_array=np.asarray(y_test)
    x_train_array=np.asarray(x_train)
    y_train_array=np.asarray(y_train)
    y_train_array= np.array(y_train_array, dtype=int)
    
    x_train_array = scaler.fit_transform(x_train_array)
    x_test_array = scaler.transform(x_test_array)    
    
    if select_extra_scaler=="standard":
        sc=StandardScaler(with_mean=classifier_params["with_mean"],with_std=classifier_params["with_std"])
        x_train = sc.fit_transform(x_train)
        x_test = sc.transform(x_test)

    print("")
    print("The number of training instances is : ",len(x_train))
    print("The number of testing instances is  : ",len(x_test))

    if select_classifier=="randomforest":
        classifier = BinaryRelevance(classifier=RandomForestClassifier(n_estimators=classifier_params["n_estimators"],criterion=classifier_params["criterion"]))
    elif select_classifier=="mlknn":
        classifier = MLkNN(k=classifier_params["k"],s=classifier_params["s"])
    
    # Load train and test partitions to fit the classifiers
    classifier.fit(x_train_array, y_train_array)
    y_pred = classifier.predict(x_test_array)
    y_pred=y_pred.toarray()
    
    # create the probabilities of the labels
    #prob_of_labels=classifier.predict_proba(x_test).toarray()
    # PLOT
    #prob_of_labels_norm=norm_probabilities(prob_of_labels)
    return(x_train,y_train,x_test,y_test,classifier,y_pred)

def lime_interpreter(dataset_features,x_train,x_test,classifier,model_name,rng=True,instance=None):
    feature_names = ["f" + str(i) for i in range(dataset_features)]  #
    explainer = LimeTabularExplainer(x_train,feature_names = feature_names,discretize_continuous=True)
    
    def wrapped_fn(x_test):
        p = classifier.predict_proba(x_test).toarray()
        p_norm=norm_probabilities(p)
        return p_norm
    if rng:
        idx = np.random.randint(0, x_test.shape[0])
    else:
        idx=instance
    exp = explainer.explain_instance(x_test[idx],predict_fn =wrapped_fn)
    exp.save_to_file(model_name+'.html')
    print("Iterpretation can be found as an HTML file in the currect directory, named :")
    print(model_name)
    print("")