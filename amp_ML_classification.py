import pandas as pd
import numpy as np
import sklearn
#FOR RESULTS
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.model_selection import cross_validate
from sklearn.model_selection import StratifiedKFold
import ML_Models
import normalization_teqniques
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
from collections import Counter
import seaborn as sns
import time
from sklearn import preprocessing


main_dir = "AMP_FICI_Classification"  #
result_counter=1
parent_dir = "C:/Users/Admin/Desktop/MAO_Machine_Learning_Library/" #change with it.
train_data_dic = parent_dir + main_dir + "/dataset/" + "AMP_data_for_training_and_validation.csv"
test_data_dic = parent_dir + main_dir + "/dataset/" + "AMP_data_for_external_test.csv"
results_dic= parent_dir + main_dir +"/results/"


##########TRAIN-TARGET SPLIT##########
train_data = pd.read_csv (train_data_dic)
train_data['Antimic_MW'] = train_data['Antimic_MW'].apply(lambda x: str(x).replace(u'\xa0', u''))
train_data["Antimic_MW"] = pd.to_numeric(train_data["Antimic_MW"])

training_data=train_data.iloc[:,1:26]
data_targets=train_data.iloc[:,27]
training_data_with_Clsfr=pd.concat([training_data, data_targets], axis=1, join='inner')

external_test = pd.read_csv (test_data_dic)
external_test['Antimic_MW'] = external_test['Antimic_MW'].apply(lambda x: str(x).replace(u'\xa0', u''))
external_test["Antimic_MW"] = pd.to_numeric(external_test["Antimic_MW"])

external_test_data=external_test.iloc[:,1:26]
external_test_targets=external_test.iloc[:,27]
external_test_data_with_Clsfr=pd.concat([external_test_data, external_test_targets], axis=1, join='inner')



# Because they are correlated with others: (Comment if you still want to keep them)
training_data=training_data.drop(["Len","DCP","Ratio_H_T","Antimic_Chrg","Hydrophi","Amph_In","AMP_MW"], axis=1)
external_test_data=external_test_data.drop(["Len","DCP","Ratio_H_T","Antimic_Chrg","Hydrophi","Amph_In","AMP_MW"], axis=1)

##########FEATURE NORMALIZATION##########
train_data_and_external_test_data = pd.concat([training_data, external_test_data], axis=0,
                                               join='inner',ignore_index=True)

normalizers,train_data_and_external_test_data_normalized = normalization_teqniques.get_normalize_functions(train_data_and_external_test_data)


# normalizers,training_data_normalized = normalization_teqniques.get_normalize_functions(training_data)
# normalizers,external_test_data_normalized = normalization_teqniques.get_normalize_functions(external_test_data)

##########FEATURE ENCODING##########
d = {'No Interaction': 0, 'Synergism': 1}
data_targets_LE=data_targets.map(d)
external_test_targets=external_test_targets.map(d)


norm_count=2 # best case
training_data_OHE_SMOTE=[]
external_test_data_OHE_normalized=[]
for i in range(len(normalizers)):

    #training_data_OHE = pd.get_dummies(training_data_normalized[i])

    if i > -1: # this because robust normalizer chosen according to acc value.
        #train_data_and_external_test_data = pd.concat([training_data_normalized[i], external_test_data_normalized[i]], axis=0, join='inner')
        #train_data_and_external_test_data =pd.get_dummies(train_data_and_external_test_data)
        train_data_and_external_test_data = pd.get_dummies(train_data_and_external_test_data_normalized[i])
        #limit=len(training_data_normalized[i])
        limit = len(training_data)
        training_data_OHE = train_data_and_external_test_data.iloc[0:limit,:]
        external_test_data_OHE = train_data_and_external_test_data.iloc[limit:,:]

##########SMOTE########## RESAMPLING ROW
    sm = SMOTE(random_state=42)
    training_data_OHE_SMOTE_temp, data_targets_LE_SMOTE = sm.fit_resample(training_data_OHE, data_targets_LE)
    training_data_OHE_SMOTE.append(training_data_OHE_SMOTE_temp)
    external_test_data_OHE_normalized.append(external_test_data_OHE)
# Open comments below if you need to check normalizers
##########TRAINING########## DIFFERENT NORMALIZATION
# # models= ML_Models.get_models()
# models= ML_Models.best_get_models()

# scores_pd=pd.DataFrame()
#
#
# for i in range(len(normalizers)):
#
#     ##########Train-Test Split##########
#     X_train, X_test, Y_train, Y_test = train_test_split(training_data_OHE_SMOTE[i], data_targets_LE_SMOTE,
#                                                         test_size=0.25, random_state=3)
#
#     for name, model in models:
#         kfold = StratifiedKFold(n_splits=5, n_repeats=3, random_state=3, shuffle=True)
#         cv_results = cross_val_score(model, X_train, Y_train, cv=kfold, scoring='accuracy')
#
#         temp_pd = pd.DataFrame({'Normalizer': normalizers[i], 'Scores': cv_results, 'Classfiers': name})
#         scores_pd=scores_pd.append(temp_pd)
#         # results.append(cv_results)
#         # names.append(name+'-'+normalizers[i])
#         print('%s: %f (%f)' % (name+'-'+normalizers[i], cv_results.mean(), cv_results.std()))
#



##########TRAINING########## DIFFERENT CLASSIFIER
# Open comments below if you need to check classfiers
# models= ML_Models.get_models()
# # models= ML_Models.best_get_models()
#
# scores_pd=pd.DataFrame()
# #ROBUST NORMALIZER WINS!
X_train, X_test, Y_train, Y_test = train_test_split(training_data_OHE_SMOTE[norm_count], data_targets_LE_SMOTE,
                                                      test_size=0.25, random_state=1)

#
# classification_scoring = {'acc': 'accuracy',
#             'f1m':'f1_macro', 'rec_macro':'recall_macro','pre_macro':'precision_macro',
#             'jaccard':'jaccard_macro',
#             'auc':'roc_auc_ovr_weighted','f1w':'f1_weighted'}
#
# kfold = StratifiedKFold(n_splits=5, n_repeats=3, random_state=3, shuffle=True)

#
# for name, model in models:
#
#     start = time.time()
#     cv_results = cross_validate(model, X_train, Y_train, cv=kfold, scoring=classification_scoring,error_score="raise", return_train_score=True)
#     end = time.time()
#
#     temp_pd = pd.DataFrame({ 'ACC Scores': cv_results['test_acc'],'Train ACC': cv_results['train_acc'],
#                              'Classfiers': name,'fit time': cv_results['fit_time'],'Score Time': cv_results['score_time'],
#                              'F1Macro': cv_results['test_f1m'],'RecMacro': cv_results['test_rec_macro'],
#                              'PreMacro': cv_results['test_pre_macro'],'Jaccard': cv_results['test_jaccard'],
#                              'TestAuc': cv_results['test_auc'],'TrainAuc': cv_results['train_auc'],'Elapsed Time': end-start} )
#     scores_pd=scores_pd.append(temp_pd)
#     print('%s: %f (%f)' % (name, cv_results['test_acc'].mean(), cv_results['test_acc'].std()))
#


##########TRAINING########## HYPERPARAMETTER TUNING
# from tpot import TPOTClassifier
# tpot_clf = TPOTClassifier(generations=2, population_size=50,
#                           verbosity=2, offspring_size=20, scoring='accuracy', cv=5,n_jobs=8)
# #Training and prediction
#
# tpot_clf.fit(X_train, Y_train)
#
# tpot_pred = tpot_clf.score(X_test, Y_test)
# print("Tpot score: ",tpot_pred)
# tpot_clf.export('tpot_digits_pipeline_amp_v2.py')
# print()

print('debug')
##########TEST DATA PREDICTION##########
from lightgbm import LGBMClassifier
#model=LGBMClassifier(learning_rate=0.1, max_depth=10, max_features=0.3, min_samples_leaf=7, min_samples_split=10,
#                            n_estimators=100, subsample=0.8500000000000001)

model=LGBMClassifier(learning_rate=0.1, max_depth=20, max_features=0.7, min_samples_leaf=7,
                            n_estimators=100, subsample=0.8)

model.fit(X_train, Y_train)
# import pickle
# filename = 'finalized_AMP_FICI_model.sav'
# pickle.dump(model, open(filename, 'wb'))

predictions = model.predict(X_test)
print(accuracy_score(Y_test, predictions))
print(confusion_matrix(Y_test, predictions))
#print(classification_report(Y_test, predictions))
print('debug')

X_test_tmp=X_test
Y_test_tmp=Y_test


########### Spacial Cases ###########
#Gram negative Gram postive case
from sklearn.metrics import roc_auc_score
from sklearn.utils import shuffle
tmp_data=X_test
tmp_data['FICI'] = Y_test.values
spe_case1_data =tmp_data[tmp_data['Gram_Gram-negative'] == True]

X_test=spe_case1_data.iloc[:,0:-1]
Y_test=spe_case1_data.iloc[:,-1]

X_test, Y_test = shuffle(X_test, Y_test, random_state=1)
predictions = model.predict(X_test)
print('Gram negative: ACC: ',accuracy_score(Y_test, predictions))
print('Gram negative: AUC: ',roc_auc_score(Y_test, model.predict_proba(X_test)[:, 1]))
print('debug')
#
X_test=X_test_tmp
Y_test=Y_test_tmp
tmp_data=X_test
tmp_data['FICI'] = Y_test.values
spe_case1_data =tmp_data[tmp_data['Gram_Gram-positive'] == True]

X_test=spe_case1_data.iloc[:,0:-1]
Y_test=spe_case1_data.iloc[:,-1]

X_test, Y_test = shuffle(X_test, Y_test, random_state=1)
predictions = model.predict(X_test)
print('Gram positive: ACC: ',accuracy_score(Y_test, predictions))
print('Gram positive: AUC: ',roc_auc_score(Y_test, model.predict_proba(X_test)[:, 1]))
print('debug')




##########  EXTERNAL TEST DATA PREDICTION##########
X_test, Y_test = shuffle(external_test_data_OHE_normalized[norm_count], external_test_targets, random_state=2)
predictions = model.predict(X_test)
print(accuracy_score(Y_test, predictions))
print(confusion_matrix(Y_test, predictions))
#print(classification_report(external_test_targets, predictions))
print('debug')


########### Spacial Cases ###########
#LOOCV by organism
tmp_data=training_data_OHE_SMOTE[norm_count]
tmp_data['FICI'] = data_targets_LE_SMOTE.values

#print('Unique O: ',training_data.Syn_Spe.unique())
print('Unique Organism Count: ',len(training_data.Syn_Spe.unique()))

# starting from 82 OHE
scores_org=pd.DataFrame()
for organism_index in range(82,82+len(training_data.Syn_Spe.unique())):
 spe_train_data =tmp_data[tmp_data[tmp_data.columns[organism_index]] == False]
 spe_test_data =tmp_data[tmp_data[tmp_data.columns[organism_index]] == True]
 X_train=spe_train_data.iloc[:,0:-1]
 Y_train=spe_train_data.iloc[:,-1]
 X_test=spe_test_data.iloc[:,0:-1]
 Y_test=spe_test_data.iloc[:,-1]
 model.fit(X_train, Y_train)
 predictions = model.predict(X_test)
 if len(Y_test.unique())>1:
    auc_s=roc_auc_score(Y_test, model.predict_proba(X_test)[:, 1])
 else:
    auc_s=1
 #print(tmp_data.columns[organism_index],': ',accuracy_score(Y_test, predictions))
 temp_pd = pd.DataFrame({'Test Set': str(tmp_data.columns[organism_index]), 'ACC': accuracy_score(Y_test, predictions),'AUC': auc_s}, index=[0])
 scores_org = pd.concat([scores_org, temp_pd], ignore_index=True)

print(scores_org)
print("AVG:", scores_org.mean(numeric_only=True))
scores_org.to_csv("organism.csv", encoding='utf-8', header=True, float_format='%.4f')

print('debug')



########### Spacial Cases ###########
#LOOCV by drug class
tmp_data=training_data_OHE_SMOTE[norm_count]
tmp_data['FICI'] = data_targets_LE_SMOTE.values

#print('Unique O: ',training_data.Syn_Spe.unique())
print('Unique Class Count: ',len(training_data.Clss.unique()))

# starting from 49 OHE
scores_clss=pd.DataFrame()
for class_index in range(49,49+len(training_data.Clss.unique())):
 spe_train_data =tmp_data[tmp_data[tmp_data.columns[class_index]] == False]
 spe_test_data =tmp_data[tmp_data[tmp_data.columns[class_index]] == True]
 X_train=spe_train_data.iloc[:,0:-1]
 Y_train=spe_train_data.iloc[:,-1]
 X_test=spe_test_data.iloc[:,0:-1]
 Y_test=spe_test_data.iloc[:,-1]
 model.fit(X_train, Y_train)
 predictions = model.predict(X_test)
 if len(Y_test.unique())>1:
    auc_s=roc_auc_score(Y_test, model.predict_proba(X_test)[:, 1])
 else:
    auc_s=1
 #print(tmp_data.columns[class_index],': ',accuracy_score(Y_test, predictions))
 temp_pd = pd.DataFrame({'Test Set': str(tmp_data.columns[class_index]), 'ACC': accuracy_score(Y_test, predictions),
                         'AUC': auc_s}, index=[0])
 scores_clss = pd.concat([scores_clss, temp_pd], ignore_index=True)

print(scores_clss)
print("AVG:", scores_clss.mean(numeric_only=True))
scores_clss.to_csv("drug_Class.csv", encoding='utf-8', header=True, float_format='%.4f')

print('debug')

