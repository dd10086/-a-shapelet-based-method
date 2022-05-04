import math
import numpy
import torch
import os
import random
import sklearn
import sklearn.linear_model
import joblib
from sklearn.cluster import KMeans
from collections import Counter
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from SimulatedAnneal import SimulatedAnnealing
from sklearn.svm import SVC
import timeit
from losses.loss import cal_adj1
from losses.loss import cal_adj2
import utils
import losses
import networks
import slide


class TimeSeriesEncoderClassifier(sklearn.base.BaseEstimator,
                                  sklearn.base.ClassifierMixin):

    def __init__(self, compared_length,
                 batch_size, epochs, lr,
                 encoder, params, in_channels, cuda=False, gpu=0):
        self.architecture = ''
        self.cuda = cuda
        self.gpu = gpu
        self.batch_size = batch_size
        self.epochs = epochs
        self.lr = lr
        self.encoder = encoder
        self.params = params
        self.in_channels = in_channels
        self.loss = losses.loss.PNTripletLoss(
            compared_length
        )
        self.classifier = sklearn.svm.SVC()
        self.optimizer = torch.optim.Adam(self.encoder.parameters(), lr=lr)

    def save_shapelet(self, prefix_file, shapelet, shapelet_dim):
        '''
        write the shapelet and its dimension to file
        '''
        # save shapelet
        fo_shapelet = open(os.path.join(prefix_file,"shapelet.txt"), "w")
        for j in range(len(shapelet)):
            shapelet_tmp = numpy.asarray(shapelet[j])
            s = shapelet_tmp.reshape(1, -1)
            numpy.savetxt(fo_shapelet, s)
        fo_shapelet.close()
        # save shapelet variable
        fo_shapelet_dim = open(os.path.join(prefix_file,"shapelet_dim.txt"), "w")
        numpy.savetxt(fo_shapelet_dim, shapelet_dim)
        fo_shapelet_dim.close()

    def load_shapelet(self, prefix_file):
        '''
        load the shapelet and its dimension from disk
        '''
        # save shapelet
        fo_shapelet = prefix_file + "shapelet.txt"
        with open(fo_shapelet, "r") as fo_shapelet:
            shapelet = []
            for line in fo_shapelet:
                shapelet.append(line)
        fo_shapelet.close()
        # save shapelet dimension
        fo_shapelet_dim = open(prefix_file + "shapelet_dim.txt", "r")
        shapelet_dim = numpy.loadtxt(fo_shapelet_dim)
        fo_shapelet_dim.close()

        return shapelet, shapelet_dim

    def save_encoder(self, prefix_file):
        """
        Saves the encoder and the SVM classifier.
        """
        torch.save(
            self.encoder.state_dict(),
            prefix_file + '_' + self.architecture + '_encoder.pth'
        )

    def save(self, prefix_file):
        """
        Saves the encoder and the SVM classifier.
        """
        self.save_encoder(prefix_file)
        joblib.dump(
            self.classifier,
            prefix_file + '_' + self.architecture + '_classifier.pkl'
        )

    def load_encoder(self, prefix_file):
        """
        Loads an encoder.
        """
        if self.cuda:
            self.encoder.load_state_dict(torch.load(
                prefix_file + '_' + self.architecture + '_encoder.pth',
                map_location=lambda storage, loc: storage.cuda(self.gpu)
            ))
        else:
            self.encoder.load_state_dict(torch.load(
                prefix_file + '_' + self.architecture + '_encoder.pth',
                map_location=lambda storage, loc: storage
            ))

    def load(self, prefix_file):
        """
        Loads an encoder and an SVM classifier.
        """
        self.load_encoder(prefix_file)
        self.classifier = joblib.load(
            prefix_file + '_' + self.architecture + '_classifier.pkl'
        )

    def fit_svm_linear(self, features, y):
        """
        Trains the classifier using precomputed features. Uses an svm linear
        classifier.
        """
        self.classifier = SVC(kernel='linear', gamma='auto')
        self.classifier.fit(features, y)  # 相当于是一个训练分类器的操作

        return self.classifier

    def fit_encoder(self, X, node_number, y=None, save_memory=False, verbose=False):
        """
        Trains the encoder unsupervisedly using the given training data.
        """
        train = torch.from_numpy(X)
        if self.cuda:
            train = train.cuda(self.gpu)
        train_torch_dataset = utils.Dataset(X)
        train_generator = torch.utils.data.DataLoader(
            train_torch_dataset, batch_size=self.batch_size, shuffle=True
        )
        epoch = 100
        min_loss = 1000
        best_epoch = -1
        total_losses = []
        for i in range(epoch): #
            epoch_start = timeit.default_timer()
            j = 0
            losses = []
            for batch in train_generator:
                j += 1
                batch_start = timeit.default_timer()
                if self.cuda:
                    batch = batch.cuda(self.gpu)
                self.optimizer.zero_grad()
                slide_num = 3  # todo 超参数，  可以修改
                alpha = 0.6
                loss = 0
                for m in range(slide_num):
                    loss += self.loss(alpha, self.params['Adj'], epoch, i,
                                     batch, self.encoder, self.params, node_number, save_memory=save_memory
                                     )
                    alpha = alpha - 0.2
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                losses[0].append(loss.item())
                del loss
            print('epoch{}, train finished'.format(i + 1))
            total_losses.append(numpy.array(losses[0]). sum())
            epoch_end = timeit.default_timer()
            print("epoch {} time: {}".format(i + 1, (epoch_end - epoch_start) / 60))
        return self.encoder


    def fit(self, X, y, test, test_labels, prefix_file, cluster_num, node_number, save_memory=False, verbose=False):
        """
        Trains sequentially the encoder unsupervisedly and then the classifier
        using the given labels over the learned features.
        """
        final_shapelet_num = 50
        # Fitting encoder
        encoder_start = timeit.default_timer()
        self.encoder = self.fit_encoder(
            X, node_number, y=y, save_memory=save_memory, verbose=verbose
        )
        encoder_end = timeit.default_timer()
        print("encode time: ", (encoder_end - encoder_start) / 60)
        # shapelet discovery
        discovery_start = timeit.default_timer()
        shapelet, shapelet_dim,utility= self.shapelet_discovery(node_number, X, y, cluster_num,
                                                                             batch_size=100)
        discovery_end = timeit.default_timer()
        print("discovery time: ", (discovery_end - discovery_start) / 60)
        # shapelet transformation
        transformation_start = timeit.default_timer()
        features = self.shapelet_transformation(X, shapelet, shapelet_dim)
        test_features = self.shapelet_transformation(test, shapelet, shapelet_dim)
        transformation_end = timeit.default_timer()
        print("transformation time: ", (transformation_end - transformation_start) / 60)
        estimator = SVC(kernel='linear', gamma='auto')
        sa = SimulatedAnnealing(initT=50,
                                minT=1,
                                alpha=0.95,
                                iteration=50,
                                features=features.shape[1],
                                init_features_sel=final_shapelet_num,
                                estimator=estimator)
        best_solution, best_acc = sa.fit(features, test_features, y, test_labels)
        save_shapelet = []
        for i in range(len(best_solution)):
            save_shapelet.append(shapelet[best_solution[i]])
        self.save_shapelet(prefix_file, save_shapelet, shapelet_dim[best_solution])
        return self

    def encode(self, X, batch_size=50):
        """
        Outputs the representations associated to the input by the encoder.
        """
        test = utils.Dataset(X)
        test_generator = torch.utils.data.DataLoader(
            test, batch_size=batch_size
        )
        features = numpy.zeros((numpy.shape(X)[0], self.out_channels))
        self.encoder = self.encoder.eval()

        count = 0
        with torch.no_grad():
            for batch in test_generator:
                if self.cuda:
                    batch = batch.cuda(self.gpu)
                features[
                count * batch_size: (count + 1) * batch_size
                ] = self.encoder(batch, 0, 0)
                count += 1

        self.encoder = self.encoder.train()
        return features

    def shapelet_discovery(self, node_number, X, train_labels, cluster_num,
                           batch_size=100):  # todo batch_size may be alter
        '''
        slide raw time series as candidates
        encode candidates
        cluster new representations
        select the one nearest to centroid
        trace back original candidates as shapelet
        '''
        slide_num = 3
        alpha = 0.6
        beta = 6
        count = 0
        X_slide_num = []
        gama = 0.5
        for m in range(slide_num):
            # slide the raw time series and the corresponding class and variate label
            X_slide, candidates_dim, candidates_class_label = slide.slide_MTS_dim_step(X, train_labels, alpha)
            X_slide_num.append(numpy.shape(X_slide)[0])
            beta = beta - 2
            alpha = beta / 10
            test = utils.Dataset(X_slide)
            test_generator = torch.utils.data.DataLoader(test, batch_size=batch_size)
            self.encoder = self.encoder.eval()
            # encode slide TS
            with torch.no_grad():
                for batch in test_generator:
                    if self.cuda:
                        batch = batch.cuda(self.gpu)
                    # 2D to 3D
                    self.params['Adj'].data = cal_adj2(numpy.array(batch), node_number)
                    sample_num = len(batch)
                    temp_representation = torch.zeros(node_number, batch.shape[1],
                                                      dtype=torch.float64)
                    temp_representation[0:sample_num, :] = batch
                    batch = self.encoder(temp_representation.reshape(node_number, 1, -1), 0, 0)[0:sample_num]
                    if count == 0:
                        representation = batch
                    else:
                        representation = numpy.concatenate((representation, batch), axis=0)
                    count += 1
            self.encoder = self.encoder.train()
            count = 0
            # concatenate the new representation from different slides
            if m == 0:
                representation_all = representation
                representation_dim = candidates_dim
                representation_class_label = candidates_class_label
            else:
                representation_all = numpy.concatenate((representation_all, representation), axis=0)
                representation_dim = representation_dim + candidates_dim
                representation_class_label = numpy.concatenate((representation_class_label, candidates_class_label),
                                                               axis=0)
        # cluster all the new representations
        num_cluster = cluster_num
        kmeans = KMeans(n_clusters=num_cluster)
        kmeans.fit(representation_all)
        # init candidate as list
        candidate = []
        candidate_first_representation=[]
        candidate_dim = numpy.zeros(num_cluster)
        candidate_cluster_size = []
        X_slide_disc_total = []
        utility = []
        for k in range(slide_num):
            X_slide_disc_total.append(slide.slide_MTS_dim(X, (0.6 - k * 0.2)))# 0.6,0.4,0.2
        for i in range(num_cluster):
            tmp = representation_all[kmeans.labels_ == i]
            candidate_cluster_size.append(tmp.shape[0])
            dim_in_cluster_i = list()
            class_label_cluster_i = list()
            dist = math.inf
            if tmp.shape[0] > 0:
                match_start = (numpy.where(representation_all == tmp[0]))[0][0]
                match_len = tmp.shape[0]
                dist = numpy.linalg.norm(tmp - kmeans.cluster_centers_[i],axis=1)#todo
                min_index = dist.argmin()
                tmp_candidate_first_representation = tmp[min_index]
                nearest = match_start+min_index
                class_label_cluster_i = representation_class_label[match_start:match_start+match_len]
                dim_in_cluster_i = representation_dim[match_start:match_start+match_len]
                sum_X_slide_num = 0
                for k in range(slide_num):
                    sum_X_slide_num += X_slide_num[k]
                    if (nearest < sum_X_slide_num):
                        index_slide = nearest - sum_X_slide_num + X_slide_num[k]
                        candidate_tmp = X_slide_disc_total[k][index_slide]
                        candidate_dim[i] = index_slide % numpy.shape(X)[1]
                        break
                del tmp
                class_label_top = (Counter(class_label_cluster_i).most_common(1)[0][1] / len(class_label_cluster_i))
                dim_label_top = (Counter(dim_in_cluster_i).most_common(1)[0][1] / len(dim_in_cluster_i))
                if (class_label_top < (1 / numpy.unique(train_labels).shape[0])) or (
                        dim_label_top < (1 / numpy.shape(X)[1])):
                    del candidate_dim[-1]
                    continue
                candidate_first_representation.append(tmp_candidate_first_representation)
                candidate.append(candidate_tmp)
        for i in range(num_cluster):
            ed_dist_sum = 0
            for j in range(len(candidate_first_representation)):
                ed_dist_sum += numpy.linalg.norm(
                    candidate_first_representation[i] - candidate_first_representation[j])
            utility.append(gama * candidate_cluster_size[i] + (1 - gama) * ed_dist_sum)
        del representation_all
        return candidate, candidate_dim,utility

    def shapelet_transformation(self, X, candidate, candidate_dim):
        '''
        transform the original multivariate time series into the new one vector data space
        transformed date label the same with original label
        '''
        # init transformed data with list
        feature = []
        # transform original time series
        for i in range(numpy.shape(X)[0]):
            for j in range(len(candidate)):
                dist = math.inf
                candidate_tmp = numpy.asarray(candidate[j])
                for k in range(numpy.shape(X)[2] - numpy.shape(candidate_tmp)[0] + 1):
                    difference = X[i, int(candidate_dim[j]),
                                 0 + k: int(numpy.shape(candidate_tmp)[0]) + k] - candidate_tmp
                    feature_tmp = numpy.linalg.norm(difference)
                    if feature_tmp < dist:
                        dist = feature_tmp
                feature.append(dist)
        # turn list to array and reshape
        feature = numpy.asarray(feature)
        feature = feature.reshape(numpy.shape(X)[0], len(candidate))
        return feature  # x_num x feature_num

    def predict(self, X, batch_size=50):
        """
        Outputs the class predictions for the given test data.
        """
        features = self.encode(X, batch_size=batch_size)
        return self.classifier.predict(features)

    def score(self, X, y, shapelet, shapelet_dim, utility_sort_index, final_shapelet_num):
        """
        Outputs accuracy of the SVM classifier on the given testing data.
        """
        features = self.shapelet_transformation(X, shapelet, shapelet_dim, utility_sort_index, final_shapelet_num)
        return self.classifier.score(features, y)


class CausalCNNEncoderClassifier(TimeSeriesEncoderClassifier):
    """
    Wraps a causal CNN encoder of time series as a PyTorch module and a
    SVM classifier on top of its computed representations in a scikit-learn
    class.
    """

    def __init__(self, Adj=None, compared_length=50, batch_size=1, epochs=100, lr=0.001,
                 channels=10, depth=1,
                 reduced_size=10, out_channels=10, kernel_size=4,
                 in_channels=1, cuda=False, gpu=0):
        super(CausalCNNEncoderClassifier, self).__init__(
            compared_length, batch_size,
            epochs, lr,
            self.__create_encoder(Adj, in_channels, channels, depth, reduced_size,
                                  out_channels, kernel_size, cuda, gpu),
            self.__encoder_params(Adj, in_channels, channels, depth, reduced_size,
                                  out_channels, kernel_size),
            in_channels, cuda, gpu
        )
        self.Adj = Adj
        self.architecture = 'CausalCNN'
        self.channels = channels
        self.depth = depth
        self.reduced_size = reduced_size
        self.out_channels = out_channels
        self.kernel_size = kernel_size

    def __create_encoder(self, Adj, in_channels, channels, depth, reduced_size,
                         out_channels, kernel_size, cuda, gpu):
        encoder = networks.DC_GNN.DCGNNEncoder(
            Adj, in_channels, channels, depth, reduced_size, out_channels,
            kernel_size
        )
        encoder.double()
        if cuda:
            encoder.cuda(gpu)
        return encoder

    def __encoder_params(self, Adj, in_channels, channels, depth, reduced_size,
                         out_channels, kernel_size):
        return {
            'Adj': Adj,
            'in_channels': in_channels,
            'channels': channels,
            'depth': depth,
            'reduced_size': reduced_size,
            'out_channels': out_channels,
            'kernel_size': kernel_size
        }

    def encode_sequence(self, X, batch_size=50):
        """
        Outputs the representations associated to the input by the encoder,
        from the start of the time series to each time step (i.e., the
        evolution of the representations of the input time series with
        repect to time steps).

        Takes advantage of the causal CNN (before the max pooling), wich
        ensures that its output at time step i only depends on time step i and
        previous time steps.
        """

        test = utils.Dataset(X)
        test_generator = torch.utils.data.DataLoader(
            test, batch_size=batch_size
        )
        length = numpy.shape(X)[2]
        features = numpy.full(
            (numpy.shape(X)[0], self.out_channels, length), numpy.nan
        )
        self.encoder = self.encoder.eval()
        causal_cnn = self.encoder.network[0]
        linear = self.encoder.network[3]

        count = 0
        with torch.no_grad():
            for batch in test_generator:
                if self.cuda:
                    batch = batch.cuda(self.gpu)
                # First applies the causal CNN
                output_causal_cnn = causal_cnn(batch)
                after_pool = torch.empty(
                    output_causal_cnn.size(), dtype=torch.double
                )
                if self.cuda:
                    after_pool = after_pool.cuda(self.gpu)
                after_pool[:, :, 0] = output_causal_cnn[:, :, 0]
                # Then for each time step, computes the output of the max
                # pooling layer
                for i in range(1, length):
                    after_pool[:, :, i] = torch.max(
                        torch.cat([
                            after_pool[:, :, i - 1: i],
                            output_causal_cnn[:, :, i: i + 1]
                        ], dim=2),
                        dim=2
                    )[0]
                features[
                count * batch_size: (count + 1) * batch_size, :, :
                ] = torch.transpose(linear(
                    torch.transpose(after_pool, 1, 2)
                ), 1, 2)
                count += 1

        self.encoder = self.encoder.train()
        return features

    def get_params(self, deep=True):
        return {
            'compared_length': self.loss.compared_length,
            'batch_size': self.batch_size,
            'epochs': self.epochs,
            'lr': self.lr,
            'channels': self.channels,
            'depth': self.depth,
            'reduced_size': self.reduced_size,
            'kernel_size': self.kernel_size,
            'in_channels': self.in_channels,
            'out_channels': self.out_channels,
            'cuda': self.cuda,
            'gpu': self.gpu
        }

    def set_params(self, Adj, compared_length, batch_size, epochs, lr,
                   channels, depth, reduced_size, out_channels, kernel_size,
                   in_channels, cuda, gpu):
        self.__init__(Adj,
                      compared_length, batch_size, epochs, lr, channels, depth,
                      reduced_size, out_channels, kernel_size, in_channels, cuda, gpu
                      )
        return self
