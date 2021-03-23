import ssss
import time
# print(ssss.encrypt_str("secret",10,2))

def fetch_shares(secret, intercept, degree):
    shares = ssss.encrypt_str(secret, intercept, degree)
    result = ''
    for share in shares:
        result += share + '\n'

    return result


def fetch_shares_px(pixels_list, intercept, degree):
    shares = ssss.encrypt_int_px(pixels_list, intercept, degree)
    return shares


def fetch_shares_cheating(secret, intercept, degree):
    shares = ssss.encrypt_str_cheating(secret, intercept, degree)
    result = ''
    for share in shares:
        result += share + '\n'

    return result


def fetch_shares_arr(secret, intercept, degree):
    shares = ssss.encrypt_str(secret, intercept, degree)
    return shares


def fetch_secret(shares, degree):
    secret = ssss.decrypt_str(shares, degree)
    return secret


def fetch_secret_cheating(shares, degree):
    shares_f = []
    shares_g = []
    length = len(shares)
    for i in range(int(length/2)):
        shares_f.append(shares[i])
    for i in range(int(length/2), length):
        shares_g.append(shares[i])

    secret = ssss.decrypt_str_cheating(shares_f, shares_g, degree)
    return secret


def fetch_secret_px(x_values, y_values, degree):
    pixels_list = ssss.decrypt_px(x_values, y_values, degree)

    return pixels_list


if __name__ == "__main__":
    pixels_list = [244, 19, 37, 255, 8]
    t1 = time.time()
    shares = fetch_shares_px(pixels_list, 10, 2)
    t2 = time.time()
    print(t2-t1)
    print(shares)

    x_values = [1, 2, 3]
    y_values = [[204, 255, 37, 91, 93], [179, 216, 51, 12, 209], [139, 52, 51, 168, 132]]
    secret = fetch_secret_px(x_values, y_values, 2)
    print(secret)
