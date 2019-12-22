import server

if __name__ == '__main__':
    server.initialize()
    server.learn('random_forest')
    server.serve()
