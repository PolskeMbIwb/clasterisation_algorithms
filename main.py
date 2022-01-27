import os
import shutil
import numpy as np
import random
import matplotlib.pyplot as plt

np.set_printoptions(formatter={'float': '{: 0.5f}'.format})


def _centers_setup(k):
    prev_centers = np.ones((k, DIM)) * -1
    new_centers = np.random.rand(k, DIM) * HIGH_VALUE
    return new_centers, prev_centers


def _set_cluster_dots(data_r, k, prev_centers):
    clusters = [[] for _ in range(k)]  # списки с номерами точек
    for dot_num in range(len(data_r)):
        lower_dist = float("inf")
        closest_center = -1
        for cluster_number in range(k):
            dist = np.linalg.norm(data_r[dot_num] - prev_centers[cluster_number])
            if dist < lower_dist:
                lower_dist = dist
                closest_center = cluster_number
        clusters[closest_center].append(dot_num)
    return clusters


def _setup_conditions(k, alg_type):
    output_dir = './' + alg_type + '_means'
    if os.path.isdir(output_dir):
        shutil.rmtree(output_dir)
    os.mkdir(output_dir)
    new_centers, prev_centers = _centers_setup(k)
    iteration = 0
    return iteration, new_centers, prev_centers


def draw_before(d):
    if DIM == 2:
        ddata = d.transpose()
        ax.scatter(x=ddata[0], y=ddata[1], s=20, alpha=0.75, color="black")
        ax.set_title("Before")


def draw_centers(iteration, new_centers, ax1):
    a = 200 * (1 - (iteration / MAX_ITERATIONS) * 6)
    a = a if a > 0 else 1
    for i in range(len(new_centers)):
        if DIM == 3:
            ax1.scatter(new_centers[i][0], new_centers[i][1], new_centers[i][2],
                        color=(tuple([a / 255 for _ in range(3)])),
                        alpha=0.75, marker='^')
        else:
            ax1.scatter(x=new_centers[i][0], y=new_centers[i][1], s=20, color=(tuple([a / 255 for _ in range(3)])),
                        marker='^')
            ax1.set_title("After")


def do_k_means(data_r: np.array, k, max_iter):
    """
    Алгоритм к-средних
    :param data_r: Начальные данные
    :param k: Число центров
    :param max_iter: Максимальное число итераций
    :return: Номер последней итерации и вычисленные координаты центров
    """
    iteration, new_centers, prev_centers = _setup_conditions(k, 'k')
    ax1 = plt.subplot(222) if DIM != 3 else fig.add_subplot(projection='3d')
    clusters = []
    while np.all(prev_centers != new_centers) and iteration < max_iter:  # основной цикл алгоритма
        prev_centers = list(new_centers)
        clusters = _set_cluster_dots(data_r, k, prev_centers)  # точки, разбитые по кластерам
        new_centers = np.zeros((k, DIM))
        for i in range(k):  # пересчет позиций центров кластеров
            for dot in clusters[i]:
                new_centers[i] += data_r[dot]
            new_centers[i] /= len(clusters[i]) if len(clusters[i]) != 0 else 1
        iteration += 1
        draw_centers(iteration, new_centers, ax1)
    draw_centers(max_iter, new_centers, ax1)
    draw_k_means(ax1, clusters, data_r)
    return iteration, new_centers


def draw_k_means(ax1, clusters, data_r):
    for cluster in clusters:
        val = np.array([data_r[dot] for dot in cluster]).transpose()
        if DIM == 3:
            ax1.scatter(val[0], val[1], val[2], color=COLORS[clusters.index(cluster)], alpha=0.75)
        else:
            ax1.scatter(x=val[0], y=val[1], s=20, color=COLORS[clusters.index(cluster)], alpha=0.75)
    plt.savefig(r'./k_means/k_plot.png')


def do_fuzzy_c(data_r, k, max_iter, epsilon=0.01, m=2):
    """
    Алгоритм fuzzy c-means
    :param data_r: Начальные данные
    :param k: Число центров
    :param max_iter: Максимальное число итераций
    :param epsilon: Точность подсчета координат центров
    :param m: Коэффициент точности подсчета границ кластеров / коэффициент неопределённости
    :return: Номер последней итерации и вычисленные координаты центров
    """
    iteration, settled_centers, prev_centers = _setup_conditions(k, 'c')
    ax1 = plt.subplot(222) if DIM != 3 else fig.add_subplot(projection='3d')
    membershipMatrix = np.array([[_membership_value(point, settled_centers, i, m) for i in range(CENTERS_NUM)]
                                 for point in data_r])
    while np.all((settled_centers - prev_centers)) > epsilon and \
            iteration < max_iter and \
            np.all(settled_centers != prev_centers):
        # while iteration < max_iter and np.all(settled_centers != prev_centers):
        iteration += 1
        prev_centers = settled_centers
        settled_centers = _update_centroids(membershipMatrix, data_r, m)
        membershipMatrix = np.array([[_membership_value(point, settled_centers, i, m) for i in range(CENTERS_NUM)]
                                     for point in data_r])
        draw_centers(iteration, settled_centers, ax1)
    draw_centers(max_iter, settled_centers, ax1)
    draw_FCM(ax1, data_r, membershipMatrix)
    return iteration, settled_centers


def draw_FCM(ax1, data_r, mus):
    colors = []
    for j in range(len(data_r)):
        color = np.add.reduce([np.array(COLORS[i]) * mus[j][i] for i in range(CENTERS_NUM)]) / CENTERS_NUM
        color *= 3
        colors.append(tuple(color))
    for i in range(len(data_r)):
        if DIM == 2:
            ax1.scatter(x=data_r[i][0], y=data_r[i][1], color=colors[i], s=20)
        else:
            ax1.scatter(data_r[i][0], data_r[i][1], data_r[i][2], color=colors[i], s=20)
    plt.savefig(r'./c_means/c_plot.png')


def _membership_value(point, centroids, i, m):
    proto_membership = [(np.linalg.norm(point - centroids[i]) / np.linalg.norm(point - centroids[k])) ** (2 / (m - 1))
                        for k in range(len(centroids))]
    mu = 1 / sum(proto_membership)
    return mu


def _update_centroids(mus, dots, m):
    mus_t = mus.transpose()
    new_centers = np.zeros((CENTERS_NUM, DIM))
    for i in range(len(new_centers)):
        top = np.add.reduce([dots[j] * (mus_t[i][j] ** m) for j in range(len(dots))])
        bot = np.sum([(mus_t[i][j] ** m) for j in range(len(dots))])
        new_centers[i] = top / bot
    return new_centers


DIM = 2  # draw methods can be used only with two- or three-dimensional datasets
TYPE = "C"  # 'C' for Fuzzy-C-Means, 'K' for K-Means
LENGTH = 100  # amount of data records
CENTERS_NUM = 3
MAX_ITERATIONS = 100
HIGH_VALUE = 100  # top-border of value for randomly generated data
# np.random.seed(1)
COLORS = [tuple(random.randint(0, 255) / 255 for _ in range(3)) for i in range(CENTERS_NUM)]
DATA = np.random.rand(LENGTH, DIM) * HIGH_VALUE

fig = plt.figure(num=TYPE + "-means", figsize=(12, 12))
if DIM == 2:
    ax = plt.subplot(221)
    draw_before(DATA)

iterations_number, centers = do_k_means(DATA, CENTERS_NUM, MAX_ITERATIONS)
print("Total iterations: ", iterations_number)
print("Counted centers: ")
print(centers)

plt.subplot(222).clear()
iterations_number, centers = do_fuzzy_c(DATA, CENTERS_NUM, MAX_ITERATIONS)
print("Total iterations: ", iterations_number)
print("Counted centers: ")
print(centers)
plt.show()
