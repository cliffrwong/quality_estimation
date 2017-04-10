import pickle
import math
import numpy as np
import optunity
import optunity.metrics
import sklearn.svm

from sklearn.externals import joblib
from sklearn import svm, datasets
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error
from scipy.stats import pearsonr

optimal_model_file = "optimal.pkl"

data_dir = '/Users/cliff/Documents/data/qe/'

train_features_file = 'task1_en-de_training.baseline17.features'
train_target_file = 'train.hter'

dev_features_file = 'task1_en-de_dev.baseline17_dev.features'
dev_target_file = 'dev.hter'

test_features_file = 'task1_en-de_test.baseline17.features'
test_target_file = 'test.hter'

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
    numparticles = 30
    num_generations = 10
    hpConstraints = {'logC':[-4, 3], 'logGamma':[-6, 0], 'logEpsilon':[-2, 1]}
    solver = optunity.solvers.ParticleSwarm.ParticleSwarm(num_particles, 
                                                    num_generations, 
                                                    max_speed=None, 
                                                    phi1=2.0, 
                                                    phi2=2.0,
                                                    hpConstraints)
    optimal_pars, _, _ = optunity.optimize(solver, svr_mse, 
                                           maximize = False, max_evals=300)
    print(optimal_pars)
    optimal_model = sklearn.svm.SVR(C=10 ** optimal_pars['logC'],
                                    gamma=10 ** optimal_pars['logGamma'], 
                                    epsilon=10 ** optimal_pars['logEpsilon']
                                    ).fit(features, target)
    joblib.dump(optimal_model, optimal_model_file)
    
# Get scores for test file.
# Should be Pearson’s r = 0.3510, MAE = 0.1353, RMSE = 0.1839
def test(features_file, target_file):
    clf = joblib.load(optimal_model_file)
    features = np.loadtxt(features_file, dtype=np.float32)
    target = np.loadtxt(target_file, dtype=np.float32)
    prediction = clf.predict(features)
    print('Pearson\'s r:', pearsonr(target, prediction))
    print('RMSE:', math.sqrt(mean_squared_error(target, prediction)))
    print('MAE:', mean_absolute_error(target, prediction))
    
def main():
    # Cross validation to optimize hyperparameters
    model_selection()
    
    # Score model on test set 
    # test(data_dir+test_features_file, data_dir+test_target_file)


if __name__ == "__main__":
    main()