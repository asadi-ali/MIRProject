

def gamma_compress(integers):
    result = ''
    for integer in integers:
        if not integer:
            return ''

        res = bin(integer)[3:]
        result += '1' * len(res) + '0' + res

    return result


def gamma_uncompress(string):
    pass
