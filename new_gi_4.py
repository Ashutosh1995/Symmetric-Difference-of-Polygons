import networkx as nx
import numpy as np
import numpy.linalg as la
import random
from networkx import simple_cycles
import math
from string import ascii_lowercase
import matplotlib.pyplot as plt
import collections
from networkx import find_cycle
from numpy import array
from math import fabs
import  pdb
import sympy
from sympy import *
import matplotlib.patches as mpatches

class Node:
    def __init__(self, t, mark=False):
        self.x = t[0]
        self.y = t[1]
        self.mark = mark

    def point(self):
        return [self.x, self.y]

    def __repr__(self):
        return "[{},{}]".format(self.x, self.y)

    def __getitem__(self,item):
        return (self.x, self.y)[item]


class Graph:

    def __init__(self, fromPolygon):
        P = fromPolygon
        self.nodes = [Node(P[0])]
        self.edges = []
        self.positions = {self.nodes[0]: P[0]}

        for i in range(1,len(P)):
            n = Node(P[i])
            self.positions[n] = P[i]
            self.nodes.append(n)
            self.edges.append([self.nodes[i-1],self.nodes[i]])

        self.edges.append([self.nodes[-1],self.nodes[0]])

    def draw(self):
        self.G = nx.DiGraph()
        self.G.add_nodes_from(self.nodes)
        self.G.add_edges_from(self.edges)
        nx.draw(self.G, pos=self.positions)

def edgeIntersect(e1, e2):
    p,q = e1
    r,s = e2
    xp = p.x
    yp = p.y
    xq, yq = q.x, q.y
    xr, yr = r.x, r.y
    xs, ys = s.x, s.y
    A = np.array([
        [(xq-xp), (xr-xs)],
        [(yq-yp), (yr-ys)]
    ], dtype=np.float32)
    B = np.array([
        xr - xp,
        yr - yp
    ], dtype=np.float32)
    bool = True
    try:
        t, u = np.linalg.solve(A,B)
        if not(0 <= t and t <= 1.0  and 0 <= u and u <= 1.0):
            bool = False

    except np.linalg.LinAlgError as e:
        return (0,0,xp,yp,xq,yq,xr,yr,xs,ys, False)
    return (t,u,xp,yp,xq,yq,xr,yr,xs,ys,bool)

def shuffleListonAbsoluteDist(gamma,l):
    temp_dict = {}
    loos = []
    for temp in l:
        dist = abs(temp[0] - gamma[0]) + abs(temp[1] - gamma[1])
        temp_dict[dist] = temp
    keylist = temp_dict.keys()
    keylist.sort()
    for key in keylist:
        loos.append(temp_dict[key])
    return loos

def computeIntersect(P,Q):
    graphVertex = []
    graphVertexPos = {}
    Matrix = [[None for x in range(len(P.edges))] for y in range(len(Q.edges))]
    is_broken = [[None for x in range(len(P.edges))] for y in range(len(Q.edges))]
    for ind1, var1 in enumerate(P.edges):
        for ind2, var2 in enumerate(Q.edges):
            a,b, xm,ym,xn,yn,xo,yo,xr,yr, bool = edgeIntersect(var1,var2)
            if bool == False:
                Matrix[ind1][ind2] = None
                is_broken[ind1][ind2] = 0
                continue
            else:
                a = int(a) if np.asscalar(a).is_integer() else a
                b = int(b) if np.asscalar(b).is_integer() else b
                alpha = var1[0].point()[0] + a*(var1[1].point()[0] - var1[0].point()[0])
                beta = var1[0].point()[1] + a*(var1[1].point()[1] - var1[0].point()[1])
                new_node = Node((alpha,beta), mark=True)
                Matrix[ind1][ind2] = new_node
                is_broken[ind1][ind2] = 1
                graphVertex.append(new_node)
                graphVertexPos[new_node] = new_node.point()
    return Matrix, is_broken, graphVertex, graphVertexPos

def BreakEdge(varTemp, MatTemp):
    new_edges = []
    varTemp0 = varTemp[0]
    varTemp1 = varTemp[1]
    MatTemp = [x for x in MatTemp if x is not None]
    matTemp = shuffleListonAbsoluteDist(varTemp0, MatTemp)
    for tom,jerry in enumerate(matTemp):
        #print("TOm and jerry is", tom, jerry)
        if len(MatTemp) == 1:
            #print([varTemp0,jerry])
            new_edges.append([varTemp0,jerry])
            new_edges.append([jerry,varTemp1])
        elif len(MatTemp) > 1:
            if tom == 0:
                new_edges.append([varTemp0,jerry])
            elif tom == len(matTemp) -1:
                k = matTemp[len(matTemp) -1]
                k1 = matTemp[len(matTemp) -2]
                new_edges.append([k1,k])
                new_edges.append([k,varTemp1])
            else:
                loo = tom - 1
                new_edges.append([matTemp[loo], jerry])
    return new_edges

def drawGraph(P,Q,graphVertex,graphVertexPos, EdgeP, EdgeQ, EdgeNone1, EdgeNone2, epoch):
    G = nx.DiGraph()
    pos = {}
    G.add_nodes_from(P.nodes)
    pos.update(P.positions)
    G.add_nodes_from(Q.nodes)
    pos.update(Q.positions)
    G.add_nodes_from(graphVertex)
    pos.update(graphVertexPos)
    G.add_edges_from(EdgeNone1)
    G.add_edges_from(EdgeNone2)
    G.add_edges_from(EdgeP)
    G.add_edges_from(EdgeQ)

    labels = {}
    for idx, node in enumerate(G.nodes()):
        labels[node] = ascii_lowercase[idx]

    nx.draw_networkx_nodes(G, pos)
    nx.draw_networkx_edges(G, pos)
    nx.draw_networkx_edges(G,pos, edgelist=EdgeNone1+EdgeP, width=8,alpha=0.5,edge_color='r')
    nx.draw_networkx_edges(G,pos, edgelist=EdgeNone2+EdgeQ,width=8,alpha=0.5,edge_color='b')
    nx.draw_networkx_labels(G, pos, labels, font_size=16)
    red_patch = mpatches.Patch(color='red', label='Prediction')
    blue_patch = mpatches.Patch(color='blue', label='Ground Truth')
    plt.legend(handles=[red_patch, blue_patch], loc='upper center',)
    plt.suptitle("Epoch: " + str(epoch), fontsize=35)
    plt.savefig(str(epoch) + '.png', format="PNG")
    plt.show()
    #plt.savefig(str(epoch) + '.png')
    simpleCycles = list(nx.simple_cycles(G))
    simpleCycles.sort(key=len)
    del simpleCycles[len(simpleCycles)-2:len(simpleCycles)]
    return simpleCycles, G

def getDictFunc(PM, L):
    DC = {}
    for i in range(PM.shape[0]):
        for j in range(PM.shape[1]):
            DC[PM[i,j]] = L.nodes[j][i]
    return DC

def isCollinear(a, b, c):
    ax = a[0]
    ay = a[1]
    bx = b[0]
    by = b[1]
    cx = c[0]
    cy = c[1]
    M = ((ay - by) * (ax - cx) - (ay - cy) * (ax - bx))
    if fabs(M) <= 1e-9:
        return True
    else:
        return False

def nearestPredictedCoordinates(T, grV, V, X, PSympyMat):
    for f in T:
        for ind, var in enumerate(f):
            if var in grV:
                for i,j in enumerate(V.nodes):
                    if i == len(V.nodes) - 1:
                        k = 0
                    else:
                        k = i + 1
                    m = V.nodes[k]
                    Var = isCollinear(var,m,j)
                    if Var == True:
                        dist = np.linalg.norm(np.array(var.point()) - np.array(m.point()))
                        dist_ = np.linalg.norm(np.array(j.point()) - np.array(m.point()))
                        alpha = dist/dist_
                        exp1 = alpha*PSympyMat[0,i] + (1-alpha)*PSympyMat[0,k]
                        exp2 = alpha*PSympyMat[1,i] + (1-alpha)*PSympyMat[1,k]
                        f[ind] = [exp1,exp2]
            else:
                continue

    return T

def AreaCalculation(UPSC, DC_, P_Mat):
    GradMat = np.zeros((P_Mat.shape[0],P_Mat.shape[1]))
    Area = 0
    for u in UPSC:
        for ind, var in enumerate(u):
            if ind == len(u) -1:
                h = 0
            else:
                h = ind+1
            temp1 = u[h]
            AR1 = var[0]*temp1[1]
            AR1 = AR1/2
            AR2 = var[1]*temp1[0]
            AR2 = AR2/2
            Area += AR1 - AR2
    Ar_ = expand(Area)
    for i in range(P_Mat.shape[0]):
        for j in range(P_Mat.shape[1]):
            temp = Ar_.diff(P_Mat[i,j])
            GradMat[i,j] = temp.evalf(subs=DC_)
            temp = 0

    GM = zip(*GradMat)
    return GM


if __name__=="__main__":




    #First graph
    #PG = [(-2,0),(0,1),(2,0)]
    #QG = [(-2,0.5),(0,-1),(2,0.5)]
    #Q = Graph(fromPolygon=QG)

    #Second graph
    QG = [(1.5,0.5), (1.5,1.5), (0.5,1.5), (0.5,-0.1)]
    Q = Graph(fromPolygon=QG)
    PG = [(0,0), (0,1), (2,2), (1,0)]

    for epoch in range(20):
        if epoch < 5:
            learning_rate = 0.35
        else:
            learning_rate = 0.15
        P = Graph(fromPolygon=PG)
        E1 = []
        E2 = []
        EP = []
        EQ = []

        MatPQ, is_brokenPQ, grVertex, grVertexPos = computeIntersect(P,Q)
        MatQP = map(list, zip(*MatPQ))
        is_brokenQP = map(list, zip(*is_brokenPQ))

        for ind,var in enumerate(P.edges):
            if any(v is not 0 for v in is_brokenPQ[ind][:]) == True:
                EP.append(BreakEdge(var,MatPQ[ind][:]))
            elif all(v is 0 for v in is_brokenPQ[ind][:]) == True:
                E1.append(var)

        for ind1,var1 in enumerate(Q.edges):
            if any(v1 is not 0 for v1 in is_brokenQP[ind1][:]) == True:
                EQ.append(BreakEdge(var1,MatQP[ind1][:]))
            elif all(v1 is 0 for v1 in is_brokenQP[ind1][:]) == True:
                E2.append(var1)

        EPP = [val for sublist in EP for val in sublist]
        EQQ = [val for sublist in EQ for val in sublist]


        SC, G = drawGraph(P,Q, grVertex, grVertexPos, EPP, EQQ, E1, E2, epoch)
        PMat = MatrixSymbol('P',2,len(P.nodes))
        DICT = getDictFunc(PMat, P)
        UpdatedSC = nearestPredictedCoordinates(SC, grVertex, P, Q, PMat)
        GM = AreaCalculation(UpdatedSC, DICT, PMat)

        DiffGM = [tuple(learning_rate*x for x in s) for s in GM]

        PG = [(a[0] - b[0], a[1] - b[1]) for a, b in zip(PG, DiffGM)]
        # P.P = UPPG
        #print(PG)
