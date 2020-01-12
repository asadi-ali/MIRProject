import server

if __name__ == '__main__':
    server.initialize()
    server.cluster('hierarchical', 'tf-idf')
    # server.learn('naive_bayes')
    # server.serve()
