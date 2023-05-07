import numpy as np
from PIL import Image
import time
from numba import njit
from line_profiler import LineProfiler

# @njit
# def cal_distance(x, y, x2, y2, A_padding, B, p):
#     patch_a = A_padding[x:x + p, y:y + p, :].astype(np.float32)
#     patch_b = B[x2 - p:x2 + p + 1, y2 - p:y2 + p + 1, :]
#     return np.sum((patch_a - patch_b)**2)


@njit
def cal_distance(x, y, x2, y2, A_padding, B, p):
    sum = 0
    for i in range(p + p + 1):
        for j in range(p + p + 1):
            for k in range(3):
                a = float(A_padding[x + i, y + j, k])
                bb = B[x2 - p + i, y2 - p + j, k]
                sum += (a - bb)**2
    return sum


def reconstruction(f, A, B):
    A_h = np.size(A, 0)
    A_w = np.size(A, 1)
    temp = np.zeros_like(A)
    for i in range(A_h):
        for j in range(A_w):
            temp[i, j, :] = B[f[i, j][0], f[i, j][1], :]
    Image.fromarray(temp).save('tmp.jpg')


@njit
def initialization_loop(A_padding, B, f, dist, A_h, A_w, random_B_r,
                        random_B_c, p):
    for i in range(A_h):
        for j in range(A_w):
            x, y = random_B_r[i, j], random_B_c[i, j]
            f[i, j, 0] = x
            f[i, j, 1] = y
            dist[i, j] = cal_distance(i, j, x, y, A_padding, B, p)


def initialization(A, B, A_h, A_w, B_h, B_w, p_size):
    p = p_size // 2
    random_B_r = np.random.randint(p, B_h - p, [A_h, A_w])
    random_B_c = np.random.randint(p, B_w - p, [A_h, A_w])
    A_padding = np.pad(A, ((p, p), (p, p), (0, 0)), mode='edge')
    f = np.zeros([A_h, A_w, 2], dtype=np.int32)
    dist = np.zeros([A_h, A_w])
    initialization_loop(A_padding, B, f, dist, A_h, A_w, random_B_r,
                        random_B_c, p)
    return f, dist, A_padding


@njit
def propagation(f, x, y, A_h, A_w, dist, A_padding, B, p_size, is_odd):
    p = p_size // 2
    if is_odd:
        d_left = dist[max(x - 1, 0), y]
        d_up = dist[x, max(y - 1, 0)]
        d_current = dist[x, y]
        idx = np.argmin(np.array([d_current, d_left, d_up]))
        if idx == 1:
            f[x, y] = f[max(x - 1, 0), y]
            dist[x, y] = cal_distance(x, y, f[x, y][0], f[x, y][1], A_padding,
                                      B, p)
        if idx == 2:
            f[x, y] = f[x, max(y - 1, 0)]
            dist[x, y] = cal_distance(x, y, f[x, y][0], f[x, y][1], A_padding,
                                      B, p)
    else:
        d_right = dist[min(x + 1, A_h - 1), y]
        d_down = dist[x, min(y + 1, A_w - 1)]
        d_current = dist[x, y]
        idx = np.argmin(np.array([d_current, d_right, d_down]))
        if idx == 1:
            f[x, y] = f[min(x + 1, A_h - 1), y]
            dist[x, y] = cal_distance(x, y, f[x, y][0], f[x, y][1], A_padding,
                                      B, p)
        if idx == 2:
            f[x, y] = f[x, min(y + 1, A_w - 1)]
            dist[x, y] = cal_distance(x, y, f[x, y][0], f[x, y][1], A_padding,
                                      B, p)


@njit
def random_search(f, x, y, B_h, B_w, dist, A_padding, B, p_size, alpha=0.5):
    p = p_size // 2
    i = 4
    search_h = B_h * alpha**i
    search_w = B_w * alpha**i
    b_x = f[x, y][0]
    b_y = f[x, y][1]
    while search_h > 1 and search_w > 1:
        search_min_r = max(b_x - search_h, p)
        search_max_r = min(b_x + search_h, B_h - p)
        random_b_x = np.random.randint(search_min_r, search_max_r)
        search_min_c = max(b_y - search_w, p)
        search_max_c = min(b_y + search_w, B_w - p)
        random_b_y = np.random.randint(search_min_c, search_max_c)
        d = cal_distance(x, y, random_b_x, random_b_y, A_padding, B, p)
        if d < dist[x, y]:
            dist[x, y] = d
            f[x, y, 0] = random_b_x
            f[x, y, 1] = random_b_y
        search_h *= alpha
        search_w *= alpha


def NNS(img, ref, p_size, itr):
    A_h = np.size(img, 0)
    A_w = np.size(img, 1)
    B_h = np.size(ref, 0)
    B_w = np.size(ref, 1)
    f, dist, img_padding = initialization(img, ref, A_h, A_w, B_h, B_w, p_size)
    for itr in range(1, itr + 1):
        if itr % 2 == 0:
            for i in range(A_h - 1, -1, -1):
                for j in range(A_w - 1, -1, -1):
                    propagation(f, i, j, A_h, A_w, dist, img_padding, ref,
                                p_size, False)
                    random_search(f, i, j, B_h, B_w, dist, img_padding, ref,
                                  p_size)
        else:
            for i in range(A_h):
                for j in range(A_w):
                    propagation(f, i, j, A_h, A_w, dist, img_padding, ref,
                                p_size, True)
                    random_search(f, i, j, B_h, B_w, dist, img_padding, ref,
                                  p_size)
        print("iteration: %d" % (itr))
    return f


if __name__ == "__main__":
    img = np.array(Image.open("./cup_a.jpg"))
    ref = np.array(Image.open("./cup_b.jpg"))
    p = 3
    itr = 5

    start = time.time()
    f = NNS(img, ref, p, itr)
    end = time.time()
    print(end - start)

    # Use the following codes profile the program
    # lp = LineProfiler()
    # lp_wrapper = lp(NNS)
    # f = lp_wrapper(img, ref, p, itr)
    # lp.print_stats()

    reconstruction(f, img, ref)
