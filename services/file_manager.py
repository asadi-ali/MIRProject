import json
from services import compress


def save(file_address, inverted_index):
    with open(file_address, "w") as f:
        f.write(json.dumps(inverted_index) + '\n')


def load(file_address):
    with open(file_address) as f:
        return json.loads(f.readline())


def save_uncompress(file_address, inverted_index):
    with open(file_address, "w") as f:
        for word, posting_list in inverted_index.items():
            res = word
            for doc_id, _ in posting_list:
                res += ' ' + bin(doc_id)[2:].zfill(32)
            f.write(res + "\n")


def load_uncompress(file_address):
    inverted_index = {}
    with open(file_address) as f:
        for line in f.readlines():
            word, *posting_list = line.split(' ')
            inverted_index[word] = [int(doc_id, 2) for doc_id in posting_list]

    return inverted_index


def save_gamma(file_address, inverted_index):
    with open(file_address, "w") as f:
        for word, posting_list in inverted_index.items():
            gaps = []
            pre = 0
            for doc_id, _ in posting_list:
                gaps.append(doc_id - pre)
                pre = doc_id

            res = word + ' ' + compress.gamma_compress(gaps)
            f.write(res + "\n")


def load_gamma(file_address):
    inverted_index = {}
    with open(file_address) as f:
        for line in f.readlines():
            word, string = line.split(' ')
            gaps = compress.gamma_uncompress(string)
            inverted_index[word] = []
            pre = 0
            for gap in gaps:
                pre += gap
                inverted_index[word].append(pre)

    return inverted_index


def save_variable(file_address, inverted_index):
    with open(file_address, "w") as f:
        for word, posting_list in inverted_index.items():
            gaps = []
            pre = 0
            for doc_id, _ in posting_list:
                gaps.append(doc_id - pre)
                pre = doc_id

            res = word + ' ' + compress.variable_compress(gaps)
            f.write(res + "\n")


def load_variable(file_address):
    inverted_index = {}
    with open(file_address) as f:
        for line in f.readlines():
            word, string = line.split(' ')
            gaps = compress.variable_uncompress(string)
            inverted_index[word] = []
            pre = 0
            for gap in gaps:
                pre += gap
                inverted_index[word].append(pre)

    return inverted_index
