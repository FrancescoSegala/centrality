import networkx as nx
import csv
import matplotlib.pyplot as plt
from random import randint
from centrality import *





################################################################################
###################### testing eigenvector_in_query_set ########################
################################################################################

def test_1(G,Q, epsilon):
    print "Q: ",
    print Q ,
    print " size: "+str(len(Q)) ,
    print "size of G :"+str(len(G))
    E= nx.eigenvector_centrality( G )
    try:
        E_Q = eigenvector_in_query_set(G , Q )
    except:
        print "Eigenvector_in_query_set do not converge"
        print Q
        return
    cond = dict.fromkeys(G,False)
    error = dict.fromkeys(G,0.0)
    if E_Q == None :
        print "Eigenvector_in_query_set do not converges"
        return
    for elem in E_Q.keys():
        cond[elem] = abs(E_Q[elem]-E[elem])<epsilon
        error[elem] = abs(E_Q[elem]-E[elem])
        if cond:
            print "node : " +str(elem)+ ";Eigen_Q : " + str(E_Q[elem]) +" ;Eigen : "+ str(E[elem]) + ";tresh :" + str( cond[elem] )+";error : "+str(error[elem])

# saved in a consistent way, possible retrieving trough parsing
################################################################################
def main():
    G = nx.read_gml("../Datasets/netscience.gml",label="id")
    Q = select_query_nodes(G , 3)
    test_1( G , Q ,0.0001)






if __name__ == "__main__":
    main()
