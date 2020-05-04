"""

"""
import math
import sys
import numpy as np
import pandas as pd
from numba import jit
import matplotlib.pyplot as plt

sys.setrecursionlimit(10000000)
N = 40
X = pd.read_csv("data/X.csv", index_col=0, dtype=np.float32)

subV = [[0 for _ in range(N)] for _ in range(N)]
for i in range(N):
    for j in range(N):
        if X.iat[i, j] > 0:
            subV[i][j] = [0 for _ in range(np.int(X.iat[i, j]) - 1)]
subV = pd.DataFrame(subV, index=np.arange(1, 41), columns=np.arange(1, 41))
subV[subV == 0] = None

subC = np.array(subV)
subC = subC.tolist()
subV = np.array(subV).tolist()

C = np.zeros(N).T
V = np.zeros(N).T

pre_V = V
pre_sub_V = subV  # 用来清零

minS = 0.01
min_cost = 1e10
price_0 = 100
price_1 = 220
SA = 0
SB = 0
kase = 0


def safeCheck():
    for i in range(N):
        if V[i] < minS:
            return False
        for j in range(N):
            if subV[i][j] is not None:
                for k in range(len(subV[i[j]])):
                    if subV[i][j][k] < minS:
                        return False
    return True


def getSafeValue(distance):
    return 1 / (1 + math.exp(distance))


def sub_addSafeValue(x, y):
    global V
    global subV
    if subC[x][y] is not None:
        length = len(subC[x][y])
        for z in range(length):
            if subC[x][y][z] > 0:
                V[x] += getSafeValue(z + 1)
                V[y] += getSafeValue(length - z)
                for k in range(length):
                    subV[x][y][k] += getSafeValue(abs(k - z))


def addSafeValue(x):
    global V
    global subV
    if C[x] > 0:
        V[x] += 1
        for y in range(N):
            sub_addSafeValue(x, y)

            if subV[x][y] is not None:
                d = len(subV[x][y])
                V[y] += getSafeValue(d + 1)
                for z in range(d):
                    subV[x][y][z] += getSafeValue(z + 1)  # 子节点增加安全值。

            if subV[x][y] is not None:
                d = len(subV[x][y])
                for z in range(d):
                    subV[y][x][z] += getSafeValue(d - z)


def ensureSafeValue():
    for i in range(N):
        addSafeValue(i)


def cancleSafeValue():
    global V
    global subV
    V = pre_V
    subV = pre_sub_V


# def sub_sub_dfs(i, j, k):
#     global SA
#     length = len(subC[i][j])
#     if k == length:
#         sub_dfs(i, j + 1)
#     else:
#         sub_sub_dfs(i, j, k + 1)  # 子节点摄像头不打开
#         # print(k)
#         subC[i][j][k] = 1
#         SA += 1
#         sub_sub_dfs(i, j, k + 1)  # 子节点摄像头打开
#         subC[i][j][k] = 0  # 复原
#         SA -= 1
#
#
# def sub_dfs(i, j):
#     if i == N and j == N:
#         dfs(0)
#     else:
#         if j == N:
#             sub_dfs(i + 1, 0)
#         else:
#             sub_dfs(i, j + 1)
#             if subC[i][j] is not None:
#                 sub_sub_dfs(i, j, 0)


def dfs(i):
    global SB, C, V, subC, subV, min_cost, kase
    if i == N:
        ensureSafeValue()
        if safeCheck():
            cost = price_0 * SA + price_1 * SB
            print("now cases: %d" % kase)
            print("now cost : %.2f" % cost)
            if cost < min_cost:
                min_cost = cost
                # 文件保存
                data0 = pd.DataFrame(C)
                data1 = pd.DataFrame(subC)

                data0.to_csv("data/data0.csv")
                data1.to_csv("data/data1.csv")
        cancleSafeValue()
    else:
        dfs(i + 1)       # 不安装摄像头

        C[i] = 1
        SB += 1
        dfs(i + 1)       # 安装摄像头
        C[i] = 0
        SB -= 1


def subsubdfs(i, j, k):
    global SA, kase
    if i == N:
        dfs(0)
    else:
        if j == N:
            subsubdfs(i + 1, 0, 0)
        else:
            if subC[i][j] is None:
                subsubdfs(i, j + 1, 0)
            else:
                distance = len(subC[i][j])
                if k == distance:
                    subsubdfs(i, j + 1, 0)
                else:
                    subsubdfs(i, j, k + 1)

                    subC[i][j][k] = 1
                    SA += 1
                    subsubdfs(i, j, k + 1)
                    SA -= 1
                    subC[i][j][k] = 0


if __name__ == '__main__':
    subsubdfs(0, 0, 0)
    # print(subV[39][36])

