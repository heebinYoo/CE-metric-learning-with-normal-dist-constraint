import numpy as np

def next(M):
    n = M.shape[1] + 1
    mid = M * (np.sqrt(n ** 2 - 1) / np.sqrt(n ** 2))
    left = 1 / n * np.ones((n, 1))
    bottom = np.zeros((1, n))
    bottom[0, -1] = -1

    temp = np.concatenate((mid, left), axis=1)
    temp = np.concatenate((temp, bottom), axis=0)
    return temp


M = np.array([[1.0], [-1.0]])

for k in range(3):
    print(M)
    # for i in range(M.shape[0]):
        # print(np.linalg.norm(M[i]))
    for i in range(M.shape[0]):
        for j in range(M.shape[0]):
            if i > j:
                print(np.dot(M[i], M[j])/(np.linalg.norm(M[i])*np.linalg.norm(M[j])))
    M = next(M)








