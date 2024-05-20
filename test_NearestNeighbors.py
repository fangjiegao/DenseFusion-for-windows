from sklearn.neighbors import NearestNeighbors
import numpy as np

X = np.array([[-1, -1],
              [-2, -1],
              [-3, -2],
              [1, 1],
              [2, 1],
              [3, 2]])

Y = np.array([[1, 5],
              [3, 3]])

nbrs = NearestNeighbors(n_neighbors=1, algorithm='ball_tree').fit(X)
distances, indices = nbrs.kneighbors(Y)

print(indices)
print(distances)


# [[5]
#  [5]]
# [[3.60555128]
#  [1.        ]]

# ---------- check ----------

def distEuclid(x, y):
    distance = np.sqrt(np.sum(np.square(x - y)))
    return distance


d = np.zeros((2, 6), dtype=float)
for i in range(len(Y)):
    for j in range(len(X)):
        d[i, j] = distEuclid(X[j], Y[i])
print(d)

# [[6.32455532 6.70820393 8.06225775 4.         4.12310563 3.60555128]
#  [5.65685425 6.40312424 7.81024968 2.82842712 2.23606798 1.        ]]
