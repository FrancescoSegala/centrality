import networkx as nx
from random import randint
import itertools
from math import sqrt
import numpy.linalg
import csv

"""
notation:
    Graph: G = ( V , E )
"""


G = nx.read_gml("../Datasets/karate.gml",label="id")


def read_csv_Graph(path) :
    L = nx.Graph()
    with open(path, "rb") as csvfile:
        reader = csv.reader(csvfile, delimiter='\n', quotechar='#')
        for row in reader:
            stw = row[0].split(",")[:3]
            L.add_edge( int(stw[0]) , int(stw[1]) )
            L[int(stw[0])][int(stw[1])]["weight"] = int(stw[2])
    return L


################################################################################
################ Degree centrality for a Query set Q ###########################

"""
Degree centrality:
which is defined as the number of links incident upon a node (i.e., the number of ties that a node has).
The degree can be interpreted in terms of the immediate risk of a node for catching whatever is flowing through
the network (such as a virus,or some information).In the case of a directed network (where ties have direction),
we usually define two separate measures of degree centrality, namely indegree and outdegree.
Accordingly, indegree is a count of the number of ties directed to the node and outdegree is the number of ties that
the node directs to others. When ties are associated to some positive aspects such as friendship or collaboration,
indegree is often interpreted as a form of popularity, and outdegree as gregariousness.

The centrality wrt a query set may be the risk of a node to be infected or touched if a subset of node Q is compromised, the more nodes
in Q you are connected with, the more your risk to be affected by the condition in Q

inDegree and outDegree are supposed to represent two opposite values:
    inDegree  is the proportion on nodes that point to V and belongs to Q
    outDegree represents the opposite i.e. the nodes in Q a node V points

so they may represent for the same example above the :
    inDegree is the risk to be victim of an attack by a compromised set of nodes Q with incoming connections
    outDegree on the other hand is the risk to connect to a compromised node in the set Q

cost O( |E| )
"""



def degree_centrality_in_query_set( G , Q = list(G.nodes) ):
    V = list(G.nodes)
    norm = 1.0 / (len(V) - 1.0 )
    D = dict.fromkeys( V , 0.0 )
    for e in G.edges() :
        u,v = e
        if u in Q or v in Q :
            D[u] += norm
            D[v] += norm
    return D


def degree_centrality_in_query_set_Direct(G , Q = list(G.nodes) ):
    V = list(G.nodes)
    norm = 1.0 / (len(V) - 1.0 )
    inD = dict.fromkeys( V , 0.0 )
    outD = dict.fromkeys( V , 0.0 )
    for e in G.edges() :
        u,v = e
        if u in Q :
            inD[v] += norm
        if v in Q :
            outD[u] += norm
    return ( inD , outD )



################################################################################
################ Closeness Centrality for a Query set Q ########################

"""
for a node u in V is defined as the inverse of the sum of all the shortest paths between u and any node in Q.

cost of the algorithm : O ( |V| * |Q| * shortest_path_length ) that for nx.shortest_path_length is computed by dijkstra
so his cost is O (|V^2|+|E|) with the assumption that in general |E| = O(|V^2|) then the overall cost is O( V^3 * Q )

This centrality measure cane be an indcator of how a node U is close to the query set Q that may be a Query set of webpages
indicating then how it is pertinent the page wrt Q, also it may be a measure of proximity in threat analysis for a set of vulnerable
machine Q etc.
"""
#TODO is correct this ? questions for Q as normalization factor?

def closeness_in_query_set ( G , Q = list( G.nodes ) ):
    V = list(G.nodes)
    norm = len( Q ) - 1.0
    C = dict.fromkeys(V , 0.0)
    for v in V :
        sumSPL = 0.0 # sum of all the shortest paths len
        for u in Q :
            sumSPL += nx.shortest_path_length(G , source=v , target=u )
        C[v] = norm/sumSPL
    return C



################################################################################
############ betweennes centrality for a Query set Q ###########################

"""
two implementation, for unweighted and weighted graph.

"""

def betweenness_in_query_set_brandes_01( G , Q =list(G.nodes), endpoints = True ):
    V = list(G.nodes)
    B = dict.fromkeys(V, 0.0)
    for s in V :
        S , P , sigma = BFS_source( G , s )
        delta = dict.fromkeys(V, 0.0)
        #S returns vertices in order of non-increasing distance from s
        if endpoints:
            B[s] = len(S)-1
        while S :
            w = S.pop()
            for v in P[w] :
                delta[v] += (sigma[v]/sigma[w]) * (1+delta[w])
            if w != s :
                if endpoints:
                    B[w] += delta[w]+1
                else :
                    B[w] += delta[w]
    normalize = (len(V)-1) * (len(V)-2)
    for b in B.keys():
        B[b] *= 1.0 /normalize
    return B



def betweenness_in_query_set( G , Q = list(G.nodes), endpoints = True, normalization = True):
    #this method computes the betweenness of all the nodes wrt the query set Q
    #endpoints are counted in the shortest path count according to the notes taken during the lectures
    V = list(G.nodes)
    #SPS is a dict with all pairs of shortest paths , key = (s,t) , value = [[node,,..,node],..., [node,...,node]]
    SPS = all_pairs_shortest_paths(G , V)
    # B is the dict of betweennes key = node , value = betweenness
    B = dict.fromkeys( V , 0.0 )
    for v in V :
        #st is a tuple (s,t)
        for st in SPS.keys() :
            if not endpoints :
                condition = v!=st[0] and v!=st[1]
            else :
                condition = True
            aux = 0.0
            #sp is a shortest path from s to t
            for sp in SPS[st]  :
                if v in sp and condition and v in Q:
                    aux+=1
            B[v] += aux / len(SPS[st])
        if normalization :
            B[v] = ( 2*B[v]) / ((len(V)-1)*(len(V)-2))
    return B


def betweenness_in_query_set_weight( G , Q = list(G.nodes), endpoints = True, normalization = True, directed = False):
    #this method computes the betweenness of all the nodes wrt the query set Q
    #endpoints are counted in the shortest path count according to the notes taken during the lectures
    R , acc , P = compute_weight_matrix( G , Q , directed=directed )
    V = list(G.nodes)
    #SPS is a dict with all pairs of shortest paths , key = (s,t) , value = [[node,,..,node],..., [node,...,node]]
    SPS = all_pairs_shortest_paths(G , V)
    # B is the dict of betweennes key = node , value = betweenness
    B = dict.fromkeys( V , 0.0 )
    for v in V :
        #st is a tuple (s,t)
        for st in SPS.keys() :
            if not endpoints :
                condition = v!=st[0] and v!=st[1]
            else :
                condition = True
            aux = 0.0
            #sp is a shortest path from s to t
            for sp in SPS[st]  :
                if v in sp and condition and v in Q:
                    aux+=1
            B[v] += (aux / len(SPS[st])) * R[sp]
        if normalization :
            B[v] = ( 2*B[v]) / ((len(V)-1)*(len(V)-2))
    return B



################################################################################
########### eigenvector centrality for a Query set Q ###########################



def eigenvector_in_query_set( G , Q=list(G.nodes) ):
    #part of this code is from https://networkx.github.io/documentation/stable/_modules/networkx/algorithms/centrality/eigenvector.html
    #reshaped to fit this purposes with Q query set
    max_iter = 100
    treshold = 1.0e-6
    #x is the starting vector
    x = dict([(n,1.0/len(G)) for n in G])
    # normalize starting vector
    s = 1.0/sum(x.values())
    for k in x:
        x[k] *= s
    nnodes = len(G.nodes)
    for i in range(max_iter):
        xlast = x
        x = dict.fromkeys(xlast, 0)
        # do the multiplication y^T = x^T AQ
        for n in x: #n is each node
            for nbr in G[n]: # nbr iterate over the neighbors of n
                Aij = 0
                if nbr in Q or n in Q :
                    Aij = G[n][nbr].get("weight", 1)
                x[nbr] += xlast[n] * Aij
        # normalize vector
        try:
            s = 1.0/sqrt(sum(v**2 for v in x.values()))
        # this should never be zero?
        except ZeroDivisionError:
            s = 1.0
        for n in x:
            x[n] *= s
        # check convergence
        err = sum([abs(x[n]-xlast[n]) for n in x])
        if err < nnodes*treshold:
            return x
    return nx.PowerIterationFailedConvergence(max_iter)



################################################################################
################## Kats centrality for a Query set Q ###########################


def kats_in_query_set( G , Q = list(G.nodes) , alpha = None ):
    #here the same idea of eigenvector_centrality is used adjusted with the requirements for kats centrality
    max_iter = 100
    treshold = 1.0e-6
    nnodes = len(G.nodes)
    if alpha is None :
        #alpha is set as the inverse of the largest eigenvalue of A^T
        L = nx.normalized_laplacian_matrix(G)
        e = numpy.linalg.eigvals(L.A)
        alpha = 1.0 /  max(e)
    x = dict([(n,1.0) for n in G])
    for i in range(max_iter):
        xlast = x
        x = dict.fromkeys(xlast, 0)
        # do the multiplication x^T = (alpha x^T AQ )+1
        for n in x: #n is each node
            for nbr in G[n]: # nbr iterate over the neighbors of n
                Aij = 0
                if nbr in Q or n in Q :
                    Aij = G[n][nbr].get("weight", 1)
                x[nbr] += alpha * xlast[n] * Aij + 1
        # normalize vector
        try:
            s = 1.0/sqrt(sum(v**2 for v in x.values()))
        # this should never be zero?
        except ZeroDivisionError:
            s = 1.0
        for n in x:
            x[n] *= s
        # check convergence
        err = sum([abs(x[n]-xlast[n]) for n in x])
        if err < nnodes*treshold:
            return x
    return nx.PowerIterationFailedConvergence(max_iter)

################################################################################
################### AUX METHODS ################################################

#aux method from:
#https://networkx.github.io/documentation/latest/_modules/networkx/algorithms/centrality/betweenness.html

def BFS_source( G , s ):
    #this is an aux method for the betweenness centrality, it returns sigma that is a dict where each entry represent the number of
    #shortest paths between s and the node , S that is the stack of all the nodes encoutered from s , and finally P that is a dict where for
    #each node there is a list of predecessors following the definition : P_s (v) : {  u in V : (u,v) in E , d_G(s,v)= d_G(s,w)+w(u,v) }
    #S returns vertices in order of non-increasing distance from s
    S = []
    P = {}
    for v in G:
        P[v] = []
    sigma = dict.fromkeys(G, 0.0)    # sigma[v]=0 for v in G
    D = {}
    sigma[s] = 1.0
    D[s] = 0
    Q = [s]
    while Q:   # use BFS to find shortest paths
        v = Q.pop(0)
        S.append(v)
        Dv = D[v]
        sigmav = sigma[v]
        for w in G[v]:
            if w not in D:
                Q.append(w)
                D[w] = Dv + 1
            if D[w] == Dv + 1:   # this is a shortest path, count paths
                sigma[w] += sigmav
                P[w].append(v)  # predecessors
    return S, P, sigma



def all_pairs_shortest_paths( G , Q ):
    #computes all the shortest paths between any s and any t in Q
    #potentially O(|Q|^2 * (|E|+|V|))
    # Q bounded is this a good solution?

    #return value is a dict where {key : value} are respetctively (s,t) : all_shortest_paths(s,t)
    all_pairs = [x for x in itertools.combinations( Q , 2 ) ]
    SPS = {}
    for (s,t) in all_pairs :
        SPS[ (s,t) ] = [ x for x in nx.all_shortest_paths( G , source=s , target=t ) ]
    return SPS



def select_query_nodes( G , seed , q=None ):
    #cost O (seed * V )
    # as in the paper it creates a set of query nodes Q , predetermined params are radius = 2 , q = 10 for smal datasets and q = 20 for big datasets
    # the assumption is that q << size(all balls with radius 2)
    # seed is the size of the initial set S of seeds nodes
    if q is None :
        q = 10
        # assumption that 200 nodes is the boundary for big dataset
        if len(G.nodes) > 200 :
            q = 20
    S = [0] * seed
    index=0
    V = list(G.nodes)
    for i in range(seed):
        S[index] = V[randint(0,len(V))-1]
        index+=1
    distance_one = []
    for s in S :
        distance_one+=[x for x in G.neighbors(s)]
    distance_two = [] + distance_one
    for one in distance_one:
        distance_two+=[x for x in G.neighbors(one)]
        #distance_two = list(set(distance_two))
        #keep all the occurrences because if a node appears many times is more likely an important one and thus shoul have
        #more probability to be in Q , however in big networks is unlikely to have lots of repetitions
    Q = []
    for j in range(q*seed) :
        aux = randint(0,len(distance_two)-1)
        Q += [ distance_two[aux] ]
        distance_two=list( filter(lambda x : x != distance_two[aux] , distance_two) ) #removing all the occurrences of distance_two[aux]
    return Q



def compute_weight_matrix(G , Q = list(G.nodes) , directed = False ):
    #this method should return a matrix of weights normalized
    # NOTE nodes that are not in Q have weight 0 , in fact for a node that has P equal to zero his weight would be zero
        #thus only nodes that are both in Q will have a weight > 0. according to Rij = Pi*Pj / sum_ij( Pi*Pj )
    V = list(G.nodes)
    P = dict.fromkeys( V, 0 )
    R = {}
    for v in V :
        #P[v] contains the sum of all the weights of the incidents edges
        if v not in Q :
            P[v] = 0
        else :
            P[v] += sum([d["weight"] for (u,v,d) in G.edges(v, data=True)])
    accum = 0
    for i in V :
        for j in V :
            if i == j :
                R[(i,j)] = 0
            else:
                R[(i,j)] = P[i]*P[j]
            accum += R[(i,j)]
    #normalization
    for r in R.keys():
        R[r] = float(R[r])/accum
    return R , accum , P




################################################################################
############################ Test METHODS ######################################




def test_betweenness( G ):
    print "----------------------------------------------------------"
    Q = select_query_nodes(G , 3 )
    print "--------------------Q-------------------------------------"
    print Q
    print "--------------------Betweennes Q---------------------------"
    B = betweenness_in_query_set( G, Q , normalization = True)
    print B
    print "--------------------Betweennes all-------------------------"
    print nx.betweenness_centrality(G , endpoints = True )


def test_eigenvector( G ):
    print "----------------------------------------------------------"
    Q = select_query_nodes(G , 3 )
    print "--------------------Q-------------------------------------"
    print Q
    print "--------------------Eigenvector Q-------------------------"
    X = eigenvector_in_query_set(G , Q)
    print X
    print "--------------------Eigenvector all-----------------------"
    print nx.eigenvector_centrality( G )



def test_kats( G ):
    print "----------------------------------------------------------"
    Q = select_query_nodes(G , 3 )
    print "--------------------Q-------------------------------------"
    print Q
    print "--------------------Kats centrality Q---------------------"
    X = kats_in_query_set(G , Q)
    print X



def test_degree(G):
    print "----------------------------------------------------------"
    Q = select_query_nodes(G , 3 )
    print "--------------------Q-------------------------------------"
    print Q
    print "--------------------degree centrality Q---------------------"
    D = degree_centrality_in_query_set(G, Q)
    print D
    print "---------------------degree centrality -----------------------"
    print nx.degree_centrality(G)
    print "\n\nNOTE: for each node in Q the degree centrality should be the same"




def test_closeness(G):
        print "----------------------------------------------------------"
        Q = select_query_nodes(G , 1 )
        print "--------------------Q-------------------------------------"
        print Q
        print "--------------------Closeness centrality Q---------------------"
        D = closeness_in_query_set(G, Q)
        print D
        print "---------------------degree centrality -----------------------"
        print nx.closeness_centrality(G)


def test(G):
    print betweenness_in_query_set_brandes_01(G)
    print"-------------------------------------"
    print nx.betweenness_centrality(G , endpoints = True)


############## MAIN ############################################################

if __name__ == "__main__":
    #G = read_csv_Graph("../Datasets/soc-sign-bitcoinalpha.csv")
    #test_betweenness(G)
    G = nx.read_gml("../Datasets/karate.gml",label="id")
    test( G )

##############fine##############################################################
