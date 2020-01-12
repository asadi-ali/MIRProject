import json
import os
from pprint import pprint


def calculate_page_rank(alpha=0.85):
    iterations = 100
    graph = {}
    page_rank = {}
    for filename in os.listdir('crawler/papers'):
        with open('crawler/papers/%s' % filename) as f:
            data = json.loads(f.read())
            graph[data['id']] = data['references']
            page_rank[data['id']] = 1.0

    for paper_id in page_rank.keys():
        new_neis = []
        for nei in graph[paper_id]:
            if nei in page_rank:
                new_neis.append(nei)
        graph[paper_id] = new_neis

    for i in range(iterations):
        new_page_rank = {}
        for paper_id in page_rank.keys():
            new_value = 1. - alpha
            length = len(graph[paper_id])
            for nei in graph[paper_id]:
                new_value += alpha * page_rank[nei] / length
            new_page_rank[paper_id] = new_value
        page_rank = new_page_rank
        ans = 0
        for x, v in page_rank.items():
            ans += v
    pprint(page_rank)


calculate_page_rank(0.85)
