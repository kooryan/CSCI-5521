#import libraries
import numpy as np

class Kmeans:
    def __init__(self,k=8): # k is number of clusters
        self.num_cluster = k
        self.center = None # centers for different clusters
        self.cluster_label = np.zeros([k,]) # class labels for different clusters
        self.error_history = []

    def fit(self, X, y):
        # initialize the centers of clutsers as a set of pre-selected samples
        init_idx = [1, 200, 500, 1000, 1001, 1500, 2000, 2005]
        self.center = [X[i] for i in init_idx] # cluster means using some data point

        num_iter = 0 # number of iterations for convergence

        # initialize the cluster assignment
        prev_cluster_assignment = np.zeros([len(X),]).astype('int')
        cluster_assignment = np.zeros([len(X),]).astype('int')
        is_converged = False

        print(cluster_assignment.shape)

        # iteratively update the centers of clusters till convergence
        while not is_converged:
            # iterate through the samples and compute their cluster assignment (E step)
            for i in range(len(X)):
                # use euclidean distance to measure the distance between sample and cluster centers
                distance = 999999999
                cluster = 0
                for j in range(len(self.center)):
                    dist = np.linalg.norm(X[i] - self.center[j])
                    if dist < distance:
                        distance = dist
                        cluster = j
                # determine the cluster assignment by selecting the cluster whose center is closest to the sample
                cluster_assignment[i] = cluster

            # update the centers based on cluster assignment (M step)
            for i in range(len(self.center)):
                self.center[i] = np.mean(X[cluster_assignment == i], axis=0)

            # compute the reconstruction error for the current iteration
            cur_error = self.compute_error(X, cluster_assignment)
            self.error_history.append(cur_error)

            # reach convergence if the assignment does not change anymore
            is_converged = True if (cluster_assignment==prev_cluster_assignment).sum() == len(X) else False
            prev_cluster_assignment = np.copy(cluster_assignment)
            num_iter += 1

        # compute the class label of each cluster based on majority voting (remember to update the corresponding class attribute)
        for i in range(len(self.center)):
            dict = {0: 0, 8: 0, 9: 0}

            cluster = y[cluster_assignment == i]
            for j in cluster:
                dict[j] += 1

            label = max(dict, key=dict.get)
            self.cluster_label[i] = label

        print(self.cluster_label)

        return num_iter, self.error_history

    def predict(self,X):
        # predicting the labels of test samples based on their clustering results
        prediction = np.ones([len(X),]) # placeholder

        # iterate through the test samples
        for i in range(len(X)):
            # find the cluster of each sample
            min_idx = 0
            min_dist = 9999999
            for j in range(len(self.center)):
                dist = np.linalg.norm(X[i] - self.center[j])
                if dist < min_dist:
                    min_dist = dist
                    min_idx = j

            # use the class label of the selected cluster as the predicted class
            prediction[i] = self.cluster_label[min_idx]

        return prediction

    def compute_error(self,X,cluster_assignment):
        # compute the reconstruction error for given cluster assignment and centers
        error = 0 # placeholder
        for i in range(len(X)):
            error += np.power(np.linalg.norm(X[i] - self.center[cluster_assignment[i]]), 2)

        return error

    def params(self):
        return self.center, self.cluster_label
