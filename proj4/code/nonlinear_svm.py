import cv2
import numpy as np
import pickle
from utils import load_image, load_image_gray
import cyvlfeat as vlfeat
import sklearn.metrics.pairwise as sklearn_pairwise
from sklearn.svm import LinearSVC, SVC
from sklearn.model_selection import train_test_split, cross_val_score, ShuffleSplit, GridSearchCV, KFold
from IPython.core.debugger import set_trace
from scipy import stats

def tuning(train_image_feats, train_labels, test_image_feats):

  categories = list(set(train_labels))
  
  svms = SVC(kernel='rbf')

  parameter_grid = {'C': [0.1,0.5,0.8,1],
                    'gamma': [0.01,0.1,1,10,100]}
  
  cross_validation = KFold(n_splits=5, shuffle=True)

  grid_search = GridSearchCV(svms, param_grid=parameter_grid, cv=cross_validation)

  grid_search.fit(train_image_feats, train_labels)
  print('Best score: {}'.format(grid_search.best_score_))
  print('Best parameters: {}'.format(grid_search.best_params_))
  
  return grid_search.best_params_

def svm(train_image_feats, train_labels, test_image_feats):
  """
  This function will train a non-linear SVM for classification, including tuning
  parameters gamma and C, and classify test images data.

  Args:
  -   train_image_feats:  N x d numpy array, where d is the dimensionality of
          the feature representation
  -   train_labels: N element list, where each entry is a string indicating the
          ground truth category for each training image
  -   test_image_feats: M x d numpy array, where d is the dimensionality of the
          feature representation. You can assume N = M, unless you have changed
          the starter code
  Returns:
  -   test_labels: M element list, where each entry is a string indicating the
          predicted category for each testing image
  """

  categories = list(set(train_labels))
  
  svms = {cat: SVC(C=1, gamma=10, kernel='rbf') for cat in categories}
  
  test = []
  for cat in categories:
    labels = np.zeros(len(train_labels)) - 1
    labels[np.array(train_labels) == cat] = 1
    svms[cat].fit(train_image_feats, labels)
    confidence = svms[cat].decision_function(test_image_feats)
    test.append(confidence)
      
  test = np.transpose(np.array(test))
  test_labels = [categories[np.argmax(test[i])] for i in range(len(test))]


  return test_labels