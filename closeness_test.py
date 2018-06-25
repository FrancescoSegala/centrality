import networkx as nx
import csv
from random import randint
import matplotlib.pyplot as plt
from centrality import closeness_in_query_set
from centrality import read_csv_Graph
from centrality import select_query_nodes




################################################################################
###################### testing closeness_in_query_set ########################
################################################################################

def test_1(G,Q, epsilon):
    print "Q: ",
    print Q ,
    print " size: "+str(len(Q)) ,
    print "size of G :"+str(len(G))
    C_Q = closeness_in_query_set(G , Q )
    C= nx.closeness_centrality( G )
    cond = dict.fromkeys(G,False)
    error = dict.fromkeys(G,0.0)
    for elem in E_Q.keys():
        cond[elem] = abs(C_Q[elem]-C[elem])<epsilon
        error[elem] = abs(C_Q[elem]-C[elem])
        if cond:
            print "node : " +str(elem)+ ";Clos_Q : " + str(C_Q[elem]) +" ;Clos : "+ str(C[elem]) + ";tresh :" + str( cond[elem] )+";error : "+str(error[elem])

# saved in a consistent way, possible retrieving trough parsing
################################################################################
def main():
    G = nx.read_gml("../Datasets/netscience.gml",label="id")
    Q = select_query_nodes(G , 3)
    test_1(G, Q,0.0001 )






if __name__ == "__main__":
    main()
