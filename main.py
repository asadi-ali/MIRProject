import server

if __name__ == '__main__':
    server.initialize()
    clustering_methods = ['kmeans', 'gaussian-mixture', 'hierarchical']
    vectorization_methods = ['tf-idf', 'word2vec']
    for vec in vectorization_methods:
        server.load_to_cluster(vec)
        for clt in clustering_methods:
            server.cluster(clt, 'data/phase3_%s_%s' % (clt, vec))
    # server.learn('naive_bayes')
    # server.serve()
