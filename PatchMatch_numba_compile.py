import numpy as np
from numba import njit
from numba.pycc import CC

cc = CC('patch_match_module')


@njit
@cc.export('cal_distance', 'f4(i4, i4, i4, i4, u1[:, :, :], u1[:, :, :], i4)')
def cal_distance(x, y, x2, y2, A_padding, B, p):
    sum = 0
    for i in range(p + p + 1):
        for j in range(p + p + 1):
            for k in range(3):
                a = float(A_padding[x + i, y + j, k])
                bb = B[x2 - p + i, y2 - p + j, k]
                sum += (a - bb)**2
    return sum


@njit
@cc.export(
    'initialization_loop',
    'void(u1[:, :, :], u1[:, :, :], i4[:, :, :], f4[:, :], i4, i4, i4[:, :], i4[:, :], i4)'
)
def initialization_loop(A_padding, B, f, dist, A_h, A_w, random_B_r,
                        random_B_c, p):
    for i in range(A_h):
        for j in range(A_w):
            x, y = random_B_r[i, j], random_B_c[i, j]
            f[i, j, 0] = x
            f[i, j, 1] = y
            dist[i, j] = cal_distance(i, j, x, y, A_padding, B, p)


@njit
@cc.export(
    'propagation',
    'void(i4[:, :, :], i4, i4, i4, i4, f4[:, :], u1[:, :, :], u1[:, :, :], i4, b1)'
)
def propagation(f, x, y, A_h, A_w, dist, A_padding, B, p_size, is_odd):
    p = p_size // 2
    if is_odd:
        d_left = dist[max(x - 1, 0), y]
        d_up = dist[x, max(y - 1, 0)]
        d_current = dist[x, y]
        # idx = np.argmin(np.array([d_current, d_left, d_up]))

        if d_left < d_current and d_left < d_up:
            f[x, y] = f[max(x - 1, 0), y]
            dist[x, y] = cal_distance(x, y, f[x, y][0], f[x, y][1], A_padding,
                                      B, p)
        elif d_up < d_current and d_up < d_left:
            f[x, y] = f[x, max(y - 1, 0)]
            dist[x, y] = cal_distance(x, y, f[x, y][0], f[x, y][1], A_padding,
                                      B, p)
    else:
        d_right = dist[min(x + 1, A_h - 1), y]
        d_down = dist[x, min(y + 1, A_w - 1)]
        d_current = dist[x, y]
        # idx = np.argmin(np.array([d_current, d_right, d_down]))
        if d_right < d_current and d_right < d_down:
            f[x, y] = f[min(x + 1, A_h - 1), y]
            dist[x, y] = cal_distance(x, y, f[x, y][0], f[x, y][1], A_padding,
                                      B, p)
        elif d_down < d_current and d_down < d_right:
            f[x, y] = f[x, min(y + 1, A_w - 1)]
            dist[x, y] = cal_distance(x, y, f[x, y][0], f[x, y][1], A_padding,
                                      B, p)


@njit
@cc.export(
    'random_search',
    'void(i4[:, :, :], i4, i4, i4, i4, f4[:, :], u1[:, :, :], u1[:, :, :], i4, f4)'
)
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


if __name__ == "__main__":
    cc.compile()
