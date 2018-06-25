import networkx as nx
import csv
from random import randint
import matplotlib.pyplot as plt
from centrality import degree_centrality_in_query_set
from centrality import degree_centrality_in_query_set_Direct
from centrality import read_csv_Graph
from centrality import select_query_nodes




################################################################################
###################### testing degree_centrality_in_query_set###################
################################################################################

def test_1(G,Q, epsilon):
    print "Q: ",
    print Q ,
    print " size: "+str(len(Q)) ,
    print "size of G :"+str(len(G))
    D_Q = degree_centrality_in_query_set(G , Q )
    D= nx.degree_centrality( G )
    cond = dict.fromkeys(G,False)
    error = dict.fromkeys(G,0.0)
    for elem in E_Q.keys():
        cond[elem] = abs(D_Q[elem]-D[elem])<epsilon
        error[elem] = abs(D_Q[elem]-D[elem])
        if cond:
            print "node : " +str(elem)+ ";Deg_Q : " + str(D_Q[elem]) +" ;Deg : "+ str(D[elem]) + ";tresh :" + str( cond[elem] )+";error : "+str(error[elem])
    inDeg,OutDeg = degree_centrality_in_query_set_Direct( G , Q )
    for i in inDeg.keys() :
        print "node i: "+str(i)+";Indeg :"+str(inDeg[i])
    for j in OutDeg.keys() :
        print "node j: "+str(i)+";Outdeg :"+str(OutDeg[j])


# saved in a consistent way, possible retrieving trough parsing
################################################################################
def main():
    G = nx.read_gml("../Datasets/netscience.gml",label="id")
    Q = select_query_nodes(G , 3)
    test_1(G , Q ,0.0001)






if __name__ == "__main__":
    main()
