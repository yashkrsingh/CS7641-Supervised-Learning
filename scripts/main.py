from sklearn.ensemble import AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from processing import *
from learner import *

dt_parameters = {'max_depth': np.arange(1, 10),
                 'min_samples_leaf': np.arange(1, 20)}

ada_parameters = {'learning_rate': np.linspace(0.1, 1.0, 10),
                  'n_estimators': np.arange(50, 500, 50)}

knn_parameters = {'n_neighbors': np.arange(3, 10),
                  'weights': ['uniform', 'distance']}

svm_parameters = {'C': np.linspace(.1, 50, 10),
                  'kernel': ['linear', 'rbf']}

mlp_parameters = {'hidden_layer_sizes': [(10, ), (10, 10), (10, 10, 10), (100, ), (100, 100)],
                  'solver': ['sgd', 'lbfgs', 'adam']}

if __name__ == '__main__':
    cancer = load_cancer_data()
    wine = load_wine_data()

    seed = 42
    np.random.seed(seed)

    cancer_train_x, cancer_train_y, cancer_test_x, cancer_test_y = split_data_set(cancer, seed)
    wine_train_x, wine_train_y, wine_test_x, wine_test_y = split_data_set(wine, seed)
    results = pd.DataFrame(columns=['data', 'precision', 'recall', 'f1', 'accuracy'])

    # Decision Tree Classifier
    cancer_train_result = vanilla_fit(DecisionTreeClassifier(), cancer_train_x, cancer_train_y, cancer_test_x, cancer_test_y)
    validation_curve('cancer', DecisionTreeClassifier(), dt_parameters, cancer_train_x, cancer_train_y)
    cancer_test_result = do_grid_search('cancer', DecisionTreeClassifier(), dt_parameters, cancer_train_x, cancer_train_y, cancer_test_x, cancer_test_y)

    wine_train_result = vanilla_fit(DecisionTreeClassifier(), wine_train_x, wine_train_y, wine_test_x, wine_test_y)
    validation_curve('wine', DecisionTreeClassifier(), dt_parameters, wine_train_x, wine_train_y)
    wine_test_result = do_grid_search('wine', DecisionTreeClassifier(), dt_parameters, wine_train_x, wine_train_y, wine_test_x, wine_test_y)

    results.loc[results.shape[0]] = classification_scores('dt-cancer-untuned', cancer_train_result)
    results.loc[results.shape[0]] = classification_scores('dt-cancer-optimal', cancer_test_result)
    results.loc[results.shape[0]] = classification_scores('dt-wine-untuned', wine_train_result)
    results.loc[results.shape[0]] = classification_scores('dt-wine-optimal', wine_test_result)

    # Adaboost Decision Tree Classifier
    cancer_train_result = vanilla_fit(AdaBoostClassifier(DecisionTreeClassifier(max_depth=3)), cancer_train_x, cancer_train_y, cancer_test_x, cancer_test_y)
    validation_curve('cancer', AdaBoostClassifier(DecisionTreeClassifier(max_depth=3)), ada_parameters, cancer_train_x, cancer_train_y)
    cancer_test_result = do_grid_search('cancer', AdaBoostClassifier(DecisionTreeClassifier(max_depth=3)), ada_parameters, cancer_train_x, cancer_train_y, cancer_test_x, cancer_test_y)

    wine_train_result = vanilla_fit(AdaBoostClassifier(DecisionTreeClassifier(max_depth=3)), wine_train_x, wine_train_y, wine_test_x, wine_test_y)
    validation_curve('wine', AdaBoostClassifier(DecisionTreeClassifier(max_depth=3)), ada_parameters, wine_train_x, wine_train_y)
    wine_test_result = do_grid_search('wine', AdaBoostClassifier(DecisionTreeClassifier(max_depth=3)), ada_parameters, wine_train_x, wine_train_y, wine_test_x, wine_test_y)

    results.loc[results.shape[0]] = classification_scores('ada-cancer-untuned', cancer_train_result)
    results.loc[results.shape[0]] = classification_scores('ada-cancer-optimal', cancer_test_result)
    results.loc[results.shape[0]] = classification_scores('ada-wine-untuned', wine_train_result)
    results.loc[results.shape[0]] = classification_scores('ada-wine-optimal', wine_test_result)

    # KNN Classifier
    cancer_train_result = vanilla_fit(KNeighborsClassifier(n_jobs=-1), cancer_train_x, cancer_train_y, cancer_test_x, cancer_test_y)
    validation_curve('cancer', KNeighborsClassifier(n_jobs=-1), knn_parameters, cancer_train_x, cancer_train_y)
    cancer_test_result = do_grid_search('cancer', KNeighborsClassifier(n_jobs=-1), knn_parameters, cancer_train_x, cancer_train_y, cancer_test_x, cancer_test_y)

    wine_train_result = vanilla_fit(KNeighborsClassifier(n_jobs=-1), wine_train_x, wine_train_y, wine_test_x, wine_test_y)
    validation_curve('wine', KNeighborsClassifier(n_jobs=-1), knn_parameters, wine_train_x, wine_train_y)
    wine_test_result = do_grid_search('wine', KNeighborsClassifier(n_jobs=-1), knn_parameters, wine_train_x, wine_train_y, wine_test_x, wine_test_y)

    results.loc[results.shape[0]] = classification_scores('knn-cancer-untuned', cancer_train_result)
    results.loc[results.shape[0]] = classification_scores('knn-cancer-optimal', cancer_test_result)
    results.loc[results.shape[0]] = classification_scores('knn-wine-untuned', wine_train_result)
    results.loc[results.shape[0]] = classification_scores('knn-wine-optimal', wine_test_result)

    # SVM Classifier
    cancer_train_result = vanilla_fit(SVC(), cancer_train_x, cancer_train_y, cancer_test_x, cancer_test_y)
    validation_curve('cancer', SVC(), svm_parameters, cancer_train_x, cancer_train_y)
    cancer_test_result = do_grid_search('cancer', SVC(), svm_parameters, cancer_train_x, cancer_train_y, cancer_test_x, cancer_test_y)

    wine_train_result = vanilla_fit(SVC(), wine_train_x, wine_train_y, wine_test_x, wine_test_y)
    validation_curve('wine', SVC(), svm_parameters, wine_train_x, wine_train_y)
    wine_test_result = do_grid_search('wine', SVC(), svm_parameters, wine_train_x, wine_train_y, wine_test_x, wine_test_y)

    results.loc[results.shape[0]] = classification_scores('svm-cancer-untuned', cancer_train_result)
    results.loc[results.shape[0]] = classification_scores('svm-cancer-optimal', cancer_test_result)
    results.loc[results.shape[0]] = classification_scores('svm-wine-untuned', wine_train_result)
    results.loc[results.shape[0]] = classification_scores('svm-wine-optimal', wine_test_result)

    # NN Classifier
    cancer_train_result = vanilla_fit(MLPClassifier(learning_rate_init=.001, max_iter=10000), cancer_train_x, cancer_train_y, cancer_test_x, cancer_test_y)
    validation_curve('cancer', MLPClassifier(learning_rate_init=.001, max_iter=10000), mlp_parameters, cancer_train_x, cancer_train_y)
    cancer_test_result = do_grid_search('cancer', MLPClassifier(learning_rate_init=.001, max_iter=10000), mlp_parameters, cancer_train_x, cancer_train_y, cancer_test_x,cancer_test_y)

    wine_train_result = vanilla_fit(MLPClassifier(learning_rate_init=.001, max_iter=10000), wine_train_x, wine_train_y, wine_test_x, wine_test_y)
    validation_curve('wine', MLPClassifier(learning_rate_init=.001, max_iter=10000), mlp_parameters, wine_train_x, wine_train_y)
    wine_test_result = do_grid_search('wine', MLPClassifier(learning_rate_init=.001, max_iter=10000), mlp_parameters, wine_train_x, wine_train_y, wine_test_x,wine_test_y)

    results.loc[results.shape[0]] = classification_scores('nn-cancer-untuned', cancer_train_result)
    results.loc[results.shape[0]] = classification_scores('nn-cancer-optimal', cancer_test_result)
    results.loc[results.shape[0]] = classification_scores('nn-wine-untuned', wine_train_result)
    results.loc[results.shape[0]] = classification_scores('nn-wine-optimal', wine_test_result)

    results.to_csv('results.csv', sep=',', encoding='utf-8')
