

def gamma_compress(integer):
    if not integer:
        return ''

    res = bin(integer)[3:]
    return '1' * len(res) + '0' + res


def gamma_uncompress(string):
    pass
