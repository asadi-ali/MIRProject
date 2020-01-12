import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

pca = PCA(n_components=2)

def flat(X):
    return pca.fit_transform(X)

def visualize_clustering(X, y, path=None):
    X_2D = flat(X)
    plt.scatter(X_2D[:, 0], X_2D[:, 1], c=y, s=10, cmap='viridis')
    plt.show()
    if path is not None:
        plt.savefig(path)