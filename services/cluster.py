import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

import services.vectorspace as vs
from services.document_manager import process_english_document, documents_cnt, document_base
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from services.visualize import visualize_clustering
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import scipy.cluster.hierarchy as hc

clt = None
clt_type = None
df = None
X = None
y = None

def load_data(path, rep_type, n_comps=0, scale=True):
    global df, X
    df = pd.read_csv(path, encoding='latin1')
    X = []
    cnt = 0
    documents = []
    for id, text in zip(df['ID'], df['Text']):
        documents.append(process_english_document(text))

    if rep_type == 'word2vec':
        corpus = document_base + documents
        bag = []
        for doc in corpus:
            bag.extend(doc)
        dictionary = set(bag)
        vs.train_doc2vec(corpus)

    for document in documents:
        vector = vs.doc2tf_idf(document, documents_cnt()) if rep_type == 'tf-idf' else vs.doc2vec(document, dictionary)
        X.append(vector)
        cnt += 1

    if n_comps:
        pca = PCA(n_components=n_comps, random_state=0)
        X = pca.fit_transform(X)
    if scale:
        mms = MinMaxScaler()
        mms.fit(X)
        X = mms.transform(X)


def train_classifier(type):
    global clt, clt_type, y
    clt_type = type
    K = range(1, 11, 1)
    if clt_type == 'kmeans':
        print("Training kmeans on training data")
        inertia = []
        for k in K:
            print('Set k = %d' % k)
            clt = KMeans(n_clusters=k, random_state=0, init='k-means++').fit(X)
            inertia.append(clt.inertia_)
        print("Training completed")
        plt.plot(K, inertia, 'bx-')
        plt.xlabel('k')
        plt.ylabel('Sum_of_squared_distances')
        plt.show()
        plt.savefig('data/kmeans-elbow')
        print('Input optimal k')
        clt = KMeans(n_clusters=int(input()), random_state=0, init='k-means++').fit(X)

    if clt_type == 'gaussian-mixture':
        print("Training gaussian mixture model on training data")
        score = []
        for k in K:
            print('Set k = %d' % k)
            clt = GaussianMixture(n_components=k, random_state=0).fit(X)
            score.append(clt.score(X))
        print("Training completed")
        plt.plot(K, score, 'bx-')
        plt.xlabel('k')
        plt.ylabel('Score')
        plt.show()
        plt.savefig('data/gmm-elbow')
        print('Input optimal k')
        clt = GaussianMixture(n_components=int(input()), random_state=0).fit(X)

    if clt_type == 'hierarchical':
        hc.dendrogram(hc.linkage(X, method='ward'))
        plt.show()
        plt.savefig('data/dendrogram')
        print('Input optimal k')
        clt = AgglomerativeClustering(n_clusters=int(input()))
        print("Training completed")


def cluster_data(path=None):
    global X, y
    if clt_type == 'hierarchical':
        y = clt.fit_predict(X)
    else:
        y = clt.predict(X)
    if path is not None:
        df['Cluster'] = y
        df.to_csv(path, columns = ['ID', 'Cluster'])

def visualize(path=None):
    visualize_clustering(X, y, path=path)
