import server

if __name__ == '__main__':
    server.initialize()
    server.learn('naive_bayes')
    server.serve()
