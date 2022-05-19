import numpy as np
from sklearn.metrics import mean_squared_error
import numpy as np
from sklearn.neural_network import MLPRegressor
import scipy.io as sio
from sklearn.model_selection import KFold
import random
from copy import deepcopy
class SimulatedAnnealing(object):
    """Feature selection with simulated annealing algorithm.
    parameters
    ----------
    initT: int or float, default: 100
        The maximum temperature
    minT: int or float, default: 1
        The minimum temperature
    alpha：float， default:0.98
        Decay coefficient of temperature
    iteration: int, default:50
        Balance times at present temperature
    features: int
        The number of attributes in the original data
    init_features_sel: int
        The index of selected fatures
    estimator: object
        A supervised learning estimator with a `fit` method.
    Attributes
    ----------
    temp_history: array
        record the temperatures
    best_cost_history: array
        record the MSEs
    best_solution_history: array
        record the solutions
    """

    def __init__(self, features, init_features_sel, estimator, initT=50, minT=1, alpha=0.95, iteration=50):

        self.initT = initT
        self.minT = minT
        self.alpha = alpha
        self.iteration = iteration
        self.feature_size = features
        self.init_feature_sel = init_features_sel
        self.estimator = estimator

    def get_initial_solution(self):
        sol = np.arange(self.feature_size)
        np.random.shuffle(sol)
        return sol[:self.init_feature_sel]

    def get_cost(self, solution, x_train, x_test, y_train, y_test):
        """ compute the evaluated results of current solution
        :param solution: array of shape (selected, )
        :param x_train: array of shape (n_samples, n_features)
        :param x_test: array of shape (n_samples, n_features)
        :param y_train: array of shape (n_samples, )
        :param y_test: array of shape (n_samples, n_features)
        :return: mse
        """
        limited_train_data = self.get_data_subset(x_train, solution)
        limited_test_data = self.get_data_subset(x_test, solution)
        estimator = self.estimator.fit(limited_train_data, y_train)
        return estimator.score(limited_test_data, y_test)

    @staticmethod
    def get_data_subset(x_data, soln):
        return x_data[:, soln]

    def get_neighbor(self, current_solution, temperature):
        """
        :param current_solution: array of shape (selected, )
        :param temperature: int or float.
        :return: selected ：the index of selected features, array of shape (selected, ).
        """
        all_features = range(self.feature_size)
        selected = current_solution
        not_selected = np.setdiff1d(all_features, selected)
        # swap one selected feature with one non-selected feature
        num_swaps = int(
            min(np.ceil(np.abs(np.random.normal(0, 0.1 * len(selected) * temperature))), np.ceil(0.1 * len(selected))))
        feature_out = random.sample(range(0, len(selected)), num_swaps)
        selected = np.delete(selected, feature_out)
        feature_in = random.sample(range(0, len(not_selected)), num_swaps)
        selected = np.append(selected, not_selected[feature_in])
        return selected

    @staticmethod
    def get_probability(temperature, delta_cost):
        return np.exp(delta_cost / temperature)
    def k_fold_valid(self,K,x_train,y_train,next_solution):
        kf = KFold(n_splits=K,shuffle=True)
        i = 0
        total_acc = 0.0
        for train_index, test_index in kf.split(x_train):
            i=i+1
            train_x,train_y = x_train[train_index],y_train[train_index]
            test_x,test_y = x_train[test_index],y_train[test_index]
            total_acc += self.get_cost(next_solution, train_x, test_x, train_y, test_y)
        return total_acc/i
    def fit(self, x_train, x_test, y_train, y_test):
        """
        :param x_train: array of shape (n_samples, n_features)
        :param x_test: array of shape (n_samples, n_features)
        :param y_train: array of shape (n_samples, )
        :param y_test: array of shape (n_samples, )
        :return:
        best_solution: the index of final selected attributes, array of shape (selected, )
        best_cost : max acc
        """
        temperature = self.initT
        K = 3
        solution = self.get_initial_solution()
        cost = self.k_fold_valid(K, x_train, y_train, solution)
        temp_history = [temperature]
        best_cost_history = []
        best_solution_history = []
        best_cost = cost
        best_solution = solution
        while temperature > self.minT:
            for k in range(self.iteration):
                next_solution = self.get_neighbor(solution, temperature)
                # next_cost = self.get_cost(next_solution, x_train, x_test, y_train, y_test)
                next_cost = self.k_fold_valid(K, x_train, y_train, next_solution)
                probability = 0
                if next_cost < cost:
                    probability = self.get_probability(temperature, 2000*(next_cost - cost))
                if next_cost > cost or np.random.random() < probability:
                    cost = next_cost
                    solution = next_solution
                if next_cost > best_cost:
                    best_cost = cost
                    best_solution = solution
            temperature *= self.alpha
            temp_history.append(temperature)
            best_cost_history.append(best_cost)
            best_solution_history.append(best_solution)
        limited_train_data = self.get_data_subset(x_train, best_solution)
        estimator = deepcopy(self.estimator)
        estimator = estimator.fit(limited_train_data, y_train)
        limited_test_data = self.get_data_subset(x_test, best_solution)
        acc = estimator.score(limited_test_data, y_test)
        print('finally acc in test dataset is :',acc)
        self.temp_history_ = temp_history
        self.best_cost_history_ = best_cost_history
        self.best_solution_history = best_solution_history
        return best_solution, best_cost
