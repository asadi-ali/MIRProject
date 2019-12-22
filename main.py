import server

if __name__ == '__main__':
    server.initialize()
    server.learn('knn')
    server.serve()
