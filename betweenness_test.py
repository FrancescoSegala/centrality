import networkx as nx
from random import randint
import csv
import matplotlib.pyplot as plt
from centrality import betweenness_in_query_set_brandes_01
from centrality import betweenness_in_query_set
from centrality import read_csv_Graph
from centrality import select_query_nodes

################################################################################
###################### testing betweenness_in_query_set_brandes_01 #############
################################################################################
G = nx.read_gml("../Datasets/karate.gml",label="id")


V = list(G.nodes)

Q = [33,34,1,3,5,7,2,18,16,15,19,23]
def test_1(G,V,Q):
    B11 = betweenness_in_query_set(G , V )
    B02 =  betweenness_in_query_set_brandes_01(G , V)
    B01= nx.betweenness_centrality(G , endpoints = True)
    for elem in B01.keys():
        cond = abs(B01[elem]-B02[elem])<0.001
        if cond:
            print "node " +str(elem)+ " brandes : " + str(B02[elem]) +" , betweennes: "+ str(B01[elem]) + " tresh:" + str( cond )

################################## draw the graph ##############################

#pos = nx.spring_layout(G)

#labels = dict([(x,x) for x in G.nodes])

#nx.draw_networkx_nodes(G, pos, node_size=700)

#nx.draw_networkx_edges(G, pos, edgelist=G.edges,width=6)

#nx.draw_networkx_labels(G, pos, labels=labels,font_size=20, font_family='sans-serif')

#plt.axis('off')
#plt.show()

################################################################################
################################################################################
###################### testing betweenness_in_query_set ########################
################################################################################

def test_2 (G,V,Q,epsilon):
    B_Q = betweenness_in_query_set( G , Q ) #this is regard Q
    B_V = betweenness_in_query_set( G )   #here Q = V
    B = nx.betweenness_centrality(G, endpoints=True)
    cond = dict.fromkeys(G,False)
    error = dict.fromkeys(G,0.0)
    for node in B_Q.keys():
        cond[node] = abs(B_V[node] - B[node] ) < epsilon
        error[node] = abs(B_V[node] - B[node] )
    print "-------test--------Betweennes vs betweenness_in_query_set(Q=V)-------------"
    print "B_V[node] - B[node]"
    total_error = 0
    for node in B_Q.keys():
        print "B_V["+str(node)+"] :"+ str(B_V[node]) +";B["+str(node)+"] :"+str(B[node] )+";cond :"+cond[elem]+";error: "+str(error[elem])
        total_error += error[node]
    print "\n\ntotal_error :" ,
    print total_error
    print "----------------------------------------------------------------------------"
    print "--------------test betweenness_in_query_set with Q--------------------------"
    for node in B_Q.keys():
        print "B_Q["+node+"] :"+ str(B_Q[node])




def main():
    G = nx.read_gml("../Datasets/netscience.gml",label="id")
    Q = select_query_nodes(G , 3)
    test_2( G , list(G.nodes) , Q , 0.001 )






if __name__ == "__main__":
    main()















#
