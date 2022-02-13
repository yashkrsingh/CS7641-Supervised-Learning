from sklearn.ensemble import AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from processing import *
from learner import *

dt_parameters = {'max_depth': np.arange(1, 10),
                 'min_samples_leaf': np.arange(1, 20)}

ada_parameters = {'learning_rate': np.linspace(0.01, 0.1, 10),
                  'n_estimators': np.arange(10, 500, 50)}

knn_parameters = {'n_neighbors': [3, 5, 7, 9, 11, 13, 15],
                  'metric': ['minkowski', 'euclidean', 'manhattan']}

svm_parameters = {'C': np.linspace(.1, 50, 10),
                  'kernel': ['linear', 'rbf']}

mlp_parameters = {'hidden_layer_sizes': np.arange(50, 300, 25),
                  'activation': ['identity', 'logistic', 'tanh', 'relu']}
#                   'solver': ['sgd', 'lbfgs', 'adam']

if __name__ == '__main__':
    fetus = load_fetus_data()
    wine = load_wine_data()

    seed = 42
    np.random.seed(seed)

    fetus_train_x, fetus_train_y, fetus_test_x, fetus_test_y = split_data_set(fetus, seed)
    wine_train_x, wine_train_y, wine_test_x, wine_test_y = split_data_set(wine, seed)
    results = pd.DataFrame(columns=['data', 'precision', 'recall', 'f1', 'accuracy'])

    # Decision Tree Classifier
    fetus_train_result = vanilla_fit(DecisionTreeClassifier(), fetus_train_x, fetus_train_y, fetus_test_x, fetus_test_y)
    validation_curve('fetus', DecisionTreeClassifier(), dt_parameters, fetus_train_x, fetus_train_y)
    fetus_test_result = do_grid_search('fetus', DecisionTreeClassifier(), dt_parameters, fetus_train_x, fetus_train_y, fetus_test_x, fetus_test_y)
    # {'max_depth': 9, 'min_samples_leaf': 1}

    wine_train_result = vanilla_fit(DecisionTreeClassifier(), wine_train_x, wine_train_y, wine_test_x, wine_test_y)
    validation_curve('wine', DecisionTreeClassifier(), dt_parameters, wine_train_x, wine_train_y)
    wine_test_result = do_grid_search('wine', DecisionTreeClassifier(), dt_parameters, wine_train_x, wine_train_y, wine_test_x, wine_test_y)
    # {'max_depth': 9, 'min_samples_leaf': 1}

    results.loc[results.shape[0]] = classification_scores('dt-fetus-untuned', fetus_train_result)
    results.loc[results.shape[0]] = classification_scores('dt-fetus-optimal', fetus_test_result)
    results.loc[results.shape[0]] = classification_scores('dt-wine-untuned', wine_train_result)
    results.loc[results.shape[0]] = classification_scores('dt-wine-optimal', wine_test_result)

    # Adaboost Decision Tree Classifier
    fetus_train_result = vanilla_fit(AdaBoostClassifier(DecisionTreeClassifier()), fetus_train_x, fetus_train_y, fetus_test_x, fetus_test_y)
    validation_curve('fetus', AdaBoostClassifier(DecisionTreeClassifier(max_depth=1)), ada_parameters, fetus_train_x, fetus_train_y)
    fetus_test_result = do_grid_search('fetus', AdaBoostClassifier(DecisionTreeClassifier(max_depth=1)), ada_parameters, fetus_train_x, fetus_train_y, fetus_test_x, fetus_test_y)
    # {'learning_rate': 0.6, 'n_estimators': 400}
    # {'learning_rate': 0.09000000000000001, 'n_estimators': 460}

    wine_train_result = vanilla_fit(AdaBoostClassifier(DecisionTreeClassifier()), wine_train_x, wine_train_y, wine_test_x, wine_test_y)
    validation_curve('wine', AdaBoostClassifier(DecisionTreeClassifier(max_depth=2, criterion='entropy')), ada_parameters, wine_train_x, wine_train_y)
    wine_test_result = do_grid_search('wine', AdaBoostClassifier(DecisionTreeClassifier(max_depth=2, criterion='entropy')), ada_parameters, wine_train_x, wine_train_y, wine_test_x, wine_test_y)
    # {'learning_rate': 0.9, 'n_estimators': 450}
    # {'learning_rate': 0.09000000000000001, 'n_estimators': 460}

    results.loc[results.shape[0]] = classification_scores('ada-fetus-untuned', fetus_train_result)
    results.loc[results.shape[0]] = classification_scores('ada-fetus-optimal', fetus_test_result)
    results.loc[results.shape[0]] = classification_scores('ada-wine-untuned', wine_train_result)
    results.loc[results.shape[0]] = classification_scores('ada-wine-optimal', wine_test_result)

    # KNN Classifier
    fetus_train_result = vanilla_fit(KNeighborsClassifier(n_jobs=-1), fetus_train_x, fetus_train_y, fetus_test_x, fetus_test_y)
    validation_curve('fetus', KNeighborsClassifier(n_jobs=-1), knn_parameters, fetus_train_x, fetus_train_y)
    fetus_test_result = do_grid_search('fetus', KNeighborsClassifier(n_jobs=-1), knn_parameters, fetus_train_x, fetus_train_y, fetus_test_x, fetus_test_y)
    # {'n_neighbors': 3, 'weights': 'distance'}
    # {'metric': 'minkowski', 'n_neighbors': 3}

    wine_train_result = vanilla_fit(KNeighborsClassifier(n_jobs=-1), wine_train_x, wine_train_y, wine_test_x, wine_test_y)
    validation_curve('wine', KNeighborsClassifier(n_jobs=-1), knn_parameters, wine_train_x, wine_train_y)
    wine_test_result = do_grid_search('wine', KNeighborsClassifier(n_jobs=-1), knn_parameters, wine_train_x, wine_train_y, wine_test_x, wine_test_y)
    # {'n_neighbors': 7, 'weights': 'distance'}
    # {'metric': 'minkowski', 'n_neighbors': 3}

    results.loc[results.shape[0]] = classification_scores('knn-fetus-untuned', fetus_train_result)
    results.loc[results.shape[0]] = classification_scores('knn-fetus-optimal', fetus_test_result)
    results.loc[results.shape[0]] = classification_scores('knn-wine-untuned', wine_train_result)
    results.loc[results.shape[0]] = classification_scores('knn-wine-optimal', wine_test_result)

    # SVM Classifier
    fetus_train_result = vanilla_fit(SVC(), fetus_train_x, fetus_train_y, fetus_test_x, fetus_test_y)
    validation_curve('fetus', SVC(), svm_parameters, fetus_train_x, fetus_train_y)
    fetus_test_result = do_grid_search('fetus', SVC(), svm_parameters, fetus_train_x, fetus_train_y, fetus_test_x, fetus_test_y)
    #  {'C': 16.733333333333334, 'kernel': 'rbf'}

    wine_train_result = vanilla_fit(SVC(), wine_train_x, wine_train_y, wine_test_x, wine_test_y)
    validation_curve('wine', SVC(), svm_parameters, wine_train_x, wine_train_y)
    wine_test_result = do_grid_search('wine', SVC(), svm_parameters, wine_train_x, wine_train_y, wine_test_x, wine_test_y)
    # {'C': 38.91111111111111, 'kernel': 'rbf'}

    results.loc[results.shape[0]] = classification_scores('svm-fetus-untuned', fetus_train_result)
    results.loc[results.shape[0]] = classification_scores('svm-fetus-optimal', fetus_test_result)
    results.loc[results.shape[0]] = classification_scores('svm-wine-untuned', wine_train_result)
    results.loc[results.shape[0]] = classification_scores('svm-wine-optimal', wine_test_result)

    # NN Classifier
    fetus_train_result = vanilla_fit(MLPClassifier(learning_rate_init=.001, max_iter=10000), fetus_train_x, fetus_train_y, fetus_test_x, fetus_test_y)
    validation_curve('fetus', MLPClassifier(learning_rate_init=.001, max_iter=10000), mlp_parameters, fetus_train_x, fetus_train_y)
    fetus_test_result = do_grid_search('fetus', MLPClassifier(learning_rate_init=.001, max_iter=10000), mlp_parameters, fetus_train_x, fetus_train_y, fetus_test_x,fetus_test_y)
    # {'hidden_layer_sizes': 275, 'solver': 'adam'}
    # {'activation': 'relu', 'hidden_layer_sizes': 125}

    wine_train_result = vanilla_fit(MLPClassifier(learning_rate_init=.01, max_iter=10000), wine_train_x, wine_train_y, wine_test_x, wine_test_y)
    validation_curve('wine', MLPClassifier(learning_rate_init=.01, max_iter=10000), mlp_parameters, wine_train_x, wine_train_y)
    wine_test_result = do_grid_search('wine', MLPClassifier(learning_rate_init=.01, max_iter=10000), mlp_parameters, wine_train_x, wine_train_y, wine_test_x,wine_test_y)
    # {'hidden_layer_sizes': 200, 'solver': 'lbfgs'}
    # {'activation': 'logistic', 'hidden_layer_sizes': 100}

    results.loc[results.shape[0]] = classification_scores('nn-fetus-untuned', fetus_train_result)
    results.loc[results.shape[0]] = classification_scores('nn-fetus-optimal', fetus_test_result)
    results.loc[results.shape[0]] = classification_scores('nn-wine-untuned', wine_train_result)
    results.loc[results.shape[0]] = classification_scores('nn-wine-optimal', wine_test_result)

    results.to_csv('results.csv', sep=',', encoding='utf-8')
