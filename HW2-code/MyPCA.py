import numpy as np

class PCA():
    def __init__(self,num_dim=None):
        self.num_dim = num_dim
        self.mean = np.zeros([1,784]) # means of training data
        self.W = None # projection matrix

    def fit(self,X):
        # normalize the data to make it centered at zero (also store the means as class attribute)
        self.mean = np.mean(X, axis=0)
        X_center = X - self.mean

        cov = np.cov(X_center, rowvar=False)
        # finding the projection matrix that maximize the variance (Hint: for eigen computation, use numpy.eigh instead of numpy.eig)
        w, v = np.linalg.eigh(cov)

        w_ = w[::-1] # order from highest to low
        v_ = v[:, ::-1]

        if self.num_dim is None:
            sum = 0
            i = 0
            perc = 0
            print(i)
            while perc < 90:
                sum += w_[i]
                perc = (sum / np.sum(w_)) * 100
                i += 1
            # select the reduced dimension that keep >90% of the variance

            # store the projected dimension
            self.num_dim = i # placeholder

        # determine the projection matrix and store it as class attribute
        self.W = v_[:,:self.num_dim]
        print(self.W.shape)

        # project the high-dimensional data to low-dimensional one
        X_pca = X_center @ self.W

        return X_pca, self.num_dim

    def predict(self,X):
        # normalize the test data based on training statistics
        # self.mean = np.mean(X, axis=0)
        X_center = X - self.mean

        # project the test data
        X_pca = X_center @ self.W # placeholder

        return X_pca

    def params(self):
        return self.W, self.mean, self.num_dim
