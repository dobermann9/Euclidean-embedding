import numpy as np
from numba import jit
import matplotlib.pyplot as plt
import scipy

# helper functions for the "Euclidean embedding of wormholes" - jupyter notebooks
# Author: Tristan Baumann

@jit
def jit_cross3(a, b):
    """Helper function for fast_stress: third component of the cross product"""
    return a[0]*b[1]-a[1]*b[0]


@jit
def fast_stress(X, l1, l2, distances, angles, triplets):
    """Graph stress function, optimized by jit"""
    x = np.reshape(X, (int(X.shape[0]/2), 2))
    stress = 0
    for t in triplets:
        i, j, k = t
        stress += l1*(np.dot(x[j]-x[i], x[j]-x[k]) - distances[i, j] *
                      distances[j, k]*np.cos(angles[i, j, k]))**2
        stress += l2*(jit_cross3(x[j]-x[i], x[j]-x[k]) - distances[i, j] *
                      distances[j, k]*np.sin(angles[i, j, k]))**2
    return stress


@jit
def fast_stress_wdegree(X, l1, l2, distances, angles, triplets, degrees):
    """Graph stress function with node degree, optimized by jit"""
    x = np.reshape(X, (int(X.shape[0]/2), 2))
    stress = 0
    for t in triplets:
        i, j, k = t
        stress += (1/degrees[j])*l1*(np.dot(x[j]-x[i], x[j]-x[k]) - distances[i, j] *
                                     distances[j, k]*np.cos(angles[i, j, k]))**2
        stress += (1/degrees[j])*l2*(jit_cross3(x[j]-x[i], x[j]-x[k]) -
                                     distances[i, j]*distances[j, k]*np.sin(angles[i, j, k]))**2
    return stress


def angle(u, v):
    """returns angle of the vector v-u"""
    w = v-u
    return np.arctan2(w[1], w[0])


def angularDifference(x, measurements):
    """Returns how well the true directions in embedding x fit with the measurements"""
    stress = 0
    for m in measurements:
        u1 = (x[m[1]]-x[m[0]])/np.linalg.norm(x[m[1]]-x[m[0]])
        v1 = np.array([np.cos(m[2]), np.sin(m[2])])
        stress += np.linalg.norm(u1-v1)**2

        u2 = (x[m[0]]-x[m[1]])/np.linalg.norm(x[m[0]]-x[m[1]])
        v2 = np.array([np.cos(m[3]), np.sin(m[3])])
        stress += np.linalg.norm(u2-v2)**2
    return stress


def findBestRotation(x, m):
    """Returns the embedding with the least angular difference to measurement m"""
    best_stress = 10000
    best_i = None
    best_em = None

    for i in range(-180, 180):
        a = np.deg2rad(i)
        xr = x@np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
        stress = angularDifference(xr, m)
        if stress < best_stress:
            best_stress = stress
            best_i = i
            best_em = xr
    print("Least stress: {}, rotation by {}°".format(best_stress, best_i))
    return best_em


def drawLineFromMeasurement(x, m, length, color):
    """
    Draws three lines described by the measurement  m=[object1 object2 angle_o1o2, 
    angle_o2o1].
    """
    plt.plot([x[m[0]][0], x[m[1]][0]], [x[m[0]][1], x[m[1]][1]],
             color=color, linestyle='--', linewidth=2)
    if m[2] is not None:
        plt.plot([x[m[0]][0], x[m[0]][0]+length*np.cos(m[2])],
                 [x[m[0]][1], x[m[0]][1]+length*np.sin(m[2])],
                 color=color, linewidth=4)
    if m[3] is not None:
        plt.plot([x[m[1]][0], x[m[1]][0]+length*np.cos(m[3])],
                 [x[m[1]][1], x[m[1]][1]+length*np.sin(m[3])],
                 color=color, linewidth=4)


def drawPointingsWH(emb, data):
    """draws the four "pointing" pairs recorded in the wormhole maze"""
    drawLineFromMeasurement(emb, data[0], 3, 'r')
    drawLineFromMeasurement(emb, data[1], 3, 'b')
    drawLineFromMeasurement(emb, data[2], 3, 'g')
    drawLineFromMeasurement(emb, data[3], 3, 'y')


def vectorAlongPath(p, pos, D, A):
    """
    Sums a path of local angles and distances.
    pos is only relevant for the relative starting direction
    """
    x = np.zeros(2, float)
    a = angle(pos[p[0]], pos[p[1]])
    d = D[p[0], p[1]]
    x += np.array([np.cos(a)*d, np.sin(a)*d])
    for i in range(2, len(p)):
        a += np.pi+A[p[i-2], p[i-1], p[i]]
        d = D[p[i-1], p[i]]
        x += np.array([np.cos(a)*d, np.sin(a)*d])
    return x


def plotLineAndMeasurement(m, pos, color, linewidth):
    """Draws direct line between the node pair and measurement"""
    plt.plot([pos[m[0]][0], pos[m[1]][0]], [pos[m[0]][1], pos[m[1]][1]],
             color=color, linewidth=linewidth)
    plt.plot([pos[m[0]][0], pos[m[0]][0]+5*np.cos(m[2])],
             [pos[m[0]][1], pos[m[0]][1]+5*np.sin(m[2])],
             color=color, linestyle='--', linewidth=linewidth)


def plotVectorAndMeasurement(vec, m, pos, color, linewidth):
    """Draws path-vector-addition and measurement"""
    plt.plot([pos[m[0]][0], pos[m[0]][0]+vec[0]], [pos[m[0]][1], pos[m[0]][1]+vec[1]],
             color=color, linewidth=linewidth)
    plt.plot([pos[m[0]][0], pos[m[0]][0]+5*np.cos(m[2])],
             [pos[m[0]][1], pos[m[0]][1]+5*np.sin(m[2])],
             color=color, linestyle='--', linewidth=linewidth)


def makeLocal(G, gt, emb, data):
    """"Turns the measurements in data into local directions based on the embedding"""
    local_data = []
    for d in data:
        nbr = [i for i in G[d[0]]][0]
        # r1 = fr3.angle(gt[nbr],gt[d[0]])-fr3.angle(emb[nbr],emb[d[0]])
        # local_data.append([d[0], d[1], d[2]-r1])
        r1 = angle(gt[d[0]], gt[nbr])
        r2 = angle(emb[d[0]], emb[nbr])
        local_data.append([d[0], d[1], d[2]+(r2-r1)])
    return local_data


def cohens_d(x1, x2, circular=False):
    """Calculates Cohen's d effect size. Optionally also for circular data."""
    if circular:
        n1 = x1.shape[0]
        n2 = x2.shape[0]
        mean_x1 = scipy.stats.circmean(x1, np.pi, -np.pi)
        mean_x2 = scipy.stats.circmean(x2, np.pi, -np.pi)

        var1 = 1/(n1-1)*np.sum(((x1-mean_x1+np.pi)%(2*np.pi)-np.pi)**2)
        var2 = 1/(n2-1)*np.sum(((x1-mean_x1+np.pi)%(2*np.pi)-np.pi)**2)

        s = np.sqrt(((n1-1)*var1+(n2-1)*var2)/(n1+n2-2))
        d = ((mean_x1-mean_x2+np.pi)%(2*np.pi)-np.pi)/s
        return d
    else:
        d = (np.mean(x1) - np.mean(x2)) / (np.sqrt((np.var(x1,ddof=1) + 
            np.var(x2,ddof=1)) / 2))
        return d