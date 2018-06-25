import networkx as nx
import csv
from random import randint
import matplotlib.pyplot as plt
from centrality import kats_in_query_set
from centrality import read_csv_Graph
from centrality import select_query_nodes




################################################################################
###################### testing kats_in_query_set ###############################
################################################################################

def test_1(G,Q,epsilon):
    print "Q: ",
    print Q ,
    print " size: "+str(len(Q)) ,
    print "size of G :"+str(len(G))
    K= nx.katz_centrality(G)
    try:
        K_Q = kats_in_query_set(G , Q )
    except:
        print "Kats_in_query_set do not converges"
        print K
        return
    cond = dict.fromkeys(G,False)
    error = dict.fromkeys(G,0.0)
    for elem in K.keys():
        cond[elem] = abs(K_Q[elem]-K[elem])<epsilon
        error[elem] = abs(K_Q[elem]-K[elem])
        if cond:
            print "node : " +str(elem)+ ";Kats_Q : " + str(K_Q[elem]) +" ;Kats : "+ str(K[elem]) + ";tresh :" + str( cond[elem] )+";error : "+str(error[elem])

# saved in a consistent way, possible retrieving trough parsing

################################################################################
def main():
    G = nx.read_gml("../Datasets/netscience.gml",label="id")
    Q = select_query_nodes(G , 3)
    test_1(G , Q,0.0001 )






if __name__ == "__main__":
    main()
