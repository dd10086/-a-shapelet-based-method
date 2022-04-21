import numpy as np
from sklearn.metrics import mean_squared_error
import numpy as np
from sklearn.neural_network import MLPRegressor
import scipy.io as sio
from sklearn.model_selection import train_test_split
import random
from copy import deepcopy
class SimulatedAnnealing(object):

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

    def get_cost(self, solution, x_train, y_train):

        estimator = deepcopy(self.estimator)
        limited_train_data = self.get_data_subset(x_train, solution)
        estimator = estimator.fit(limited_train_data, y_train)
        return estimator.score(limited_train_data, y_train)


    @staticmethod
    def get_data_subset(x_data, soln):
        return x_data[:, soln]

    def get_neighbor(self, current_solution, temperature):

        all_features = range(self.feature_size)
        selected = current_solution
        not_selected = np.setdiff1d(all_features, selected)

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

    def fit(self, x_train, x_test, y_train, y_test):
        temperature = self.initT
        solution = self.get_initial_solution()
        cost = self.get_cost(solution, x_train, y_train)
        temp_history = [temperature]
        best_cost_history = []
        best_solution_history = []

        best_cost = cost
        best_solution = solution
        while temperature > self.minT:
            for k in range(self.iteration):
                next_solution = self.get_neighbor(solution, temperature)
                next_cost = self.get_cost(next_solution, x_train, y_train)
                probability = 0
                if next_cost < cost:
                    probability = self.get_probability(temperature, 2000*(next_cost - cost))
                if next_cost > cost or np.random.random() < probability:
                    cost = next_cost
                    solution = next_solution
                if next_cost > best_cost:
                    best_cost = cost
                    best_solution = solution
            print("current temperature：", round(temperature, 2))
            print("the best accuracy of train set in current temperature is ：", best_cost)
            temperature *= self.alpha
            temp_history.append(temperature)
            best_cost_history.append(best_cost)
            best_solution_history.append(best_solution)
        estimator = deepcopy(self.estimator)
        limited_train_data = self.get_data_subset(x_train, best_solution)
        limited_test_data = self.get_data_subset(x_test, best_solution)
        estimator = estimator.fit(limited_train_data, y_train)
        best_acc = estimator.score(limited_test_data, y_test)
        print('the best accuracy of test set is: ',best_acc)
        self.temp_history_ = temp_history
        self.best_cost_history_ = best_cost_history
        self.best_solution_history = best_solution_history
        return best_solution, best_cost
