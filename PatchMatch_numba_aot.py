import numpy as np
from PIL import Image
import time
import patch_match_module


def reconstruction(f, A, B):
    A_h = np.size(A, 0)
    A_w = np.size(A, 1)
    temp = np.zeros_like(A)
    for i in range(A_h):
        for j in range(A_w):
            temp[i, j, :] = B[f[i, j][0], f[i, j][1], :]
    Image.fromarray(temp).save('tmp.jpg')


def initialization(A, B, A_h, A_w, B_h, B_w, p_size):
    p = p_size // 2
    random_B_r = np.random.randint(p, B_h - p, [A_h, A_w])
    random_B_c = np.random.randint(p, B_w - p, [A_h, A_w])
    A_padding = np.pad(A, ((p, p), (p, p), (0, 0)), mode='edge')
    f = np.zeros([A_h, A_w, 2], dtype=np.int32)
    dist = np.zeros([A_h, A_w], dtype=np.float32)
    patch_match_module.initialization_loop(A_padding, B, f, dist, A_h, A_w,
                                           random_B_r, random_B_c, p)
    return f, dist, A_padding


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
                    patch_match_module.propagation(f, i, j, A_h, A_w, dist,
                                                   img_padding, ref, p_size,
                                                   False)
                    patch_match_module.random_search(f, i, j, B_h, B_w, dist,
                                                     img_padding, ref, p_size,
                                                     0.5)
        else:
            for i in range(A_h):
                for j in range(A_w):
                    patch_match_module.propagation(f, i, j, A_h, A_w, dist,
                                                   img_padding, ref, p_size,
                                                   True)
                    patch_match_module.random_search(f, i, j, B_h, B_w, dist,
                                                     img_padding, ref, p_size,
                                                     0.5)
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

    # lp = LineProfiler()
    # lp_wrapper = lp(NNS)
    # f = lp_wrapper(img, ref, p, itr)
    # lp.print_stats()
    reconstruction(f, img, ref)
