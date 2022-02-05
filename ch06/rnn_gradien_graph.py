import numpy as np
import matplotlib.pyplot as plt

if __name__ == "__main__":
    N = 2
    H = 3
    T = 20

    dh = np.ones((N, H))
    np.random.seed(3)
    #Wh = np.random.randn(H, H)
    Wh = np.random.randn(H, H) * 0.5

    norm_list = []
    for t in range(T):
        dh = np.dot(dh, Wh.T)
        norm = np.sqrt(np.sum(dh**2)) / N
        norm_list.append(norm)

    print(norm_list)

    plt.plot(np.arange(len(norm_list)), norm_list)
    plt.xticks([0, 4, 9, 14, 19], [1, 5, 10, 15, 20])
    plt.xlabel('시간 크기(time_step)')
    plt.ylabel('노름 (norm)')
    plt.show()
