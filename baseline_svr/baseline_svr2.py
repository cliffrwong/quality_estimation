import pickle
import math
import numpy as np
import optunity
import <optunity></optunity>.metrics
import sklearn.svm
import pickle

from sklearn.externals import joblib
from sklearn import svm, datasets
from sklearn.model_selection import GridSearchCV
from score import printScores


finalModel = "final_bayes2.pkl"

data_dir = '/home/cliffrwong/Documents/data/qe/'

train_features_file = 'task1_en-de_training.baseline17.features'
train_target_file = 'train.hter'

dev_features_file = 'task1_en-de_dev.baseline17_dev.features'
dev_target_file = 'dev.hter'

test_features_file = 'task1_en-de_test.baseline17.features'
test_target_file = 'test.hter'

nn_prediction_file = 'test_prediction.txt'


train_features = np.loadtxt(data_dir+train_features_file, dtype=np.float32)
train_target = np.loadtxt(data_dir+train_target_file, dtype=np.float32)
dev_features = np.loadtxt(data_dir+dev_features_file, dtype=np.float32)
dev_target = np.loadtxt(data_dir+dev_target_file, dtype=np.float32)
features = np.vstack((train_features, dev_features))
target = np.hstack((train_target, dev_target))


# score function: twice iterated 10-fold cross-validated accuracy
@optunity.cross_validated(x=features, y=target, num_folds=10)
def svr_mse(x_train, y_train, x_test, y_test, logC, logGamma, logEpsilon):
    model = sklearn.svm.SVR(C=10 ** logC, gamma=10 ** logGamma,
                     epsilon=10 ** logEpsilon).fit(x_train, y_train)
    decision_values = model.predict(x_test)
    return optunity.metrics.mse(y_test, decision_values)


# Model selection/Hyperparameter Optimization
def model_selection():
    pmap8 = optunity.parallel.create_pmap(8)
    hps, _, _ = optunity.minimize(svr_mse, num_evals=400,
                                solver_name='particle swarm', 
                                logC=[5, 7], 
                                logGamma=[-6.5, -5],
                                logEpsilon=[.7, 2],
                                # pmap=optunity.pmap)
                                pmap=pmap8)
    print(hps)
    optimal_model = sklearn.svm.SVR(C=10 ** hps['logC'],
                                gamma=10 ** hps['logGamma'], 
                                epsilon=10 ** hps['logEpsilon']
                                ).fit(features, target)
    joblib.dump(optimal_model, finalModel)
    
# Get scores for test file.
def test(features_file, target_file):
    clf = joblib.load(finalModel)
    features = np.loadtxt(features_file, dtype=np.float32)
    target = np.loadtxt(target_file, dtype=np.float32)
    prediction = clf.predict(features)
    printScores(target, prediction)

# Get scores for data from NN. Target file is a numpy pickle file
def nn_test(target_file):
    prediction = pickle.load(open(data_dir+nn_prediction_file, 'rb'))
    prediction = [x[0][0] for x in prediction]
    target = np.loadtxt(target_file, dtype=np.float32)
    printScores(target, prediction)

    # print(target)

def main():
    # Cross validation to optimize hyperparameters
    model_selection()
    
    # Score model on test set 
    test(data_dir+test_features_file, data_dir+test_target_file)
    # nn_test(data_dir+test_target_file)


if __name__ == "__main__":
    main()