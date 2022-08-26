# %% 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import sys
import os
from pathlib import Path as path

import pickle

import sklearn
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, MinMaxScaler, scale
from sklearn import metrics
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, auc
from sklearn.pipeline import Pipeline
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn import tree 
from sklearn.tree import DecisionTreeClassifier


seed = int(sys.argv[-1])
np.random.seed(seed)

def read_and_clean(location,filename):
    os.chdir(location)

    df = pd.read_csv(filename).drop('Unnamed: 0',axis = 1).set_index('date')

    df['time_to_sell'] = np.where(df['percent_change'] < 0, False,True)
    print(df.groupby('time_to_sell').count()['percent_change'])

    X = df.drop(['percent_change','open','close','time_to_sell','compound_scaled_avg','code','low','high','price_change'],axis = 1)
    y = np.array(df['time_to_sell'])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3)

    return [X, X_train, X_test], [y, y_train, y_test], df  
X, Y, df = read_and_clean('capstone/data/tweet-sentiment-predict-btc-prices/Data','df_minute.csv')

def classification_reporter(Y,predicted_y_test):
    target_names = np.sort(np.unique(Y[0]))
    target_names=target_names.tolist()
    target_names = [str(x) for x in target_names]
    # Print an entire classification report.
    class_report = metrics.classification_report(Y[2], predicted_y_test, target_names = target_names)
    print(class_report)
def roc_auc_plot(y_test,predicted_y_test):
    fpr, tpr, threshold = roc_curve(y_test, predicted_y_test)
    # Store the AUC.
    roc_auc = auc(fpr, tpr)
    plt.title('Receiver Operating Characteristic')
    plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
    plt.legend(loc = 'lower right')
    plt.plot([0, 1], [0, 1],'r--')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.show()
    plt.close()
def confusion_matrix_maker(y_test,predicted_y_test):
    cm = confusion_matrix(y_test, predicted_y_test)
    print(cm)
    print(round(accuracy_score(y_test, predicted_y_test), 4))
    plt.imshow(cm, interpolation = 'nearest', cmap = plt.cm.Wistia)
    classNames = ['Buy', 'Sell']
    plt.title('Confusion Matrix - Test Data')
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    tick_marks = np.arange(len(classNames))
    plt.xticks(tick_marks, classNames, rotation = 45)
    plt.yticks(tick_marks, classNames)
    s = [['True Buy', 'False Sell'], ['False Buy', 'True Sell']]
    for i in range(2):
        for j in range(2):
            plt.text(j,i, str(s[i][j]) + " = " + str(cm[i][j]))
    plt.show()
    plt.close()
    return cm
def optimal_parameter(values,test_results):
    best_test_value = max(test_results)
    best_test_index = test_results.index(best_test_value)
    best_value = values[best_test_index]
    return(best_value)
def create_tree(model,X,name):
    fig = plt.figure(figsize=(40,40))
    # Visualize `clf_fit_small`
    tree.plot_tree(model, feature_names= X[0].columns, filled=True)
    # Save figure
    plt.savefig(name+'.jpeg', dpi = 1600)
    plt.show()
    plt.close()
def classifier(algo,X,Y,df,cv):
    if 'knn' in algo:
        scaler = StandardScaler()
        X[1] = scaler.fit_transform(X[1])
        X[2] = scaler.transform(X[2])
        model = KNeighborsClassifier()
        if 'grid' in algo:
            parameters = {
                'n_neighbors' : [x for x in range(1,101)]
            }
            grid = GridSearchCV(model, parameters, cv = cv, scoring = 'accuracy',verbose = 2)
            best_model = grid.fit(X[1], Y[1])
            print((grid.best_params_))
            model = KNeighborsClassifier(n_neighbors = best_model.best_estimator_.get_params()['n_neighbors'])
    if 'log' in algo:
        scaler = MinMaxScaler()
        X[1] = scaler.fit_transform(X[1])
        X[2] = scaler.transform(X[2])
        model = LogisticRegression()
        if 'grid' in algo:
            parameters = {
                'penalty' : ['l1','l2'], 
                'C'       : np.logspace(0, 10, 10),
                'solver'  : ['saga'],
                'max_iter': [7500], 'verbose' : [0]
            }
            grid = GridSearchCV(model,                    # model
                   param_grid = parameters,   # hyperparameters
                   scoring='accuracy',        # metric for scoring
                   cv=cv)                     # number of folds
            best_model = grid.fit(X[1], Y[1])
            penalty = best_model.best_estimator_.get_params()['penalty']
            constant = best_model.best_estimator_.get_params()['C']
            solver = best_model.best_estimator_.get_params()['solver']
            model = LogisticRegression(penalty = penalty, C = constant, solver = solver)
    if 'tree' in algo:
        model = DecisionTreeClassifier()
        if 'grid' in algo:
            parameters = {
                'max_depth':  2**np.arange(0, 7, 1),#, endpoint = True),
                'max_leaf_nodes': list(range(2, 101)),  #changed from 2,101
                'min_samples_leaf': np.linspace(0.1, 0.5, 5, endpoint = True),
                'min_samples_split': np.linspace(0.1, 1.0, 10, endpoint=True)
            }
            grid = GridSearchCV(model,                    # model
                param_grid = parameters,   # hyperparameters
                scoring='accuracy',        # metric for scoring
                cv=cv, verbose = 2)                     # number of folds
            best_model = grid.fit(X[1], Y[1])
            max_depth = int(best_model.best_estimator_.get_params()['max_depth'])
            max_leaf_nodes = int(best_model.best_estimator_.get_params()['max_leaf_nodes'])
            min_samples_split = (best_model.best_estimator_.get_params()['min_samples_split'])
            min_samples_leaf = (best_model.best_estimator_.get_params()['min_samples_leaf'])
            model = DecisionTreeClassifier(
                max_depth = max_depth, 
                max_leaf_nodes = max_leaf_nodes, 
                min_samples_split = min_samples_split,
                min_samples_leaf = min_samples_leaf
                )
    model.fit(X[1], Y[1])
    if 'tree' in algo:
        create_tree(model, X, algo)
    predicted_y_test = model.predict(X[2])
    classification_reporter(Y,predicted_y_test)
    cm=confusion_matrix_maker(Y[2],predicted_y_test)
    roc_auc_plot(Y[2],predicted_y_test)
    if 'grid' not in algo:
        return accuracy_score(Y[2], predicted_y_test), model, cm
    else:
        return accuracy_score(Y[2], predicted_y_test), best_model, cm

cv = int(sys.argv[2])
val, selected_model, cm = classifier(sys.argv[1],X,Y,df,cv)
tn, fp, fn, tp = cm.ravel()
sell_precision = tp/(tp+fp)
sell_recall = tp/(tp+fn)
model_final_dict = {
    'metric': ["accuracy"],
    'values':[val],
    'model_name':[sys.argv[1]],
    'cross_validation':[sys.argv[2]],
    'seed_number':[seed],
    'sell_precision':[sell_precision],
    'sell_recall': [sell_recall],
    'f1': [2* (sell_precision * sell_recall)/(sell_precision + sell_recall)],
    'true_buy': [tn],
    'true_sell': [tp],
    'false_buy': [fn],
    'false_sell': [fp]
    }
model_final = pd.DataFrame(data = model_final_dict)
if 'grid' in sys.argv[1] and ('log' in sys.argv[1] or 'tree' in sys.argv[1]):
    print(selected_model.best_estimator_.get_params())
#if 'log' in sys.argv[1] and 'grid' in sys.argv[1]:
#    penalty = selected_model.best_estimator_.get_params()['penalty']
#    constant = selected_model.best_estimator_.get_params()['C']
#    solver = selected_model.best_estimator_.get_params()['solver']
#    print('Best penalty: ', penalty)
#    print('Best C: ', constant)
#    print('Best Solver: ', solver)

if 'classification_model_scores.sav' in os.listdir():
    loaded_model = pickle.load(open("classification_model_scores.sav", "rb"))
    model_final = pd.concat([loaded_model,model_final],axis = 0)
    model_final = model_final.drop_duplicates(subset = 'model_name')
    pickle.dump(model_final,open('classification_model_scores.sav','wb'))
else:
    pickle.dump(model_final,open('classification_model_scores.sav','wb'))
print(model_final.sort_values(by = 'f1', ascending = False))
model_final.to_csv('classification_scores.csv', index = False)



    


# %%
