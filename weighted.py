#generate a weighted graph for test
import matplotlib.pyplot as plt
import networkx as nx
import csv


G = nx.Graph()
L = nx.Graph()

G.add_edge('a', 'b', weight=1)
G.add_edge('a', 'c', weight=3)
G.add_edge('c', 'd', weight=1)
#G.add_edge('c', 'e', weight=0.7)
#G.add_edge('c', 'f', weight=0.9)
G.add_edge('a', 'd', weight=2)

elarge = [(u, v) for (u, v, d) in G.edges(data=True) if d['weight'] > 0.5]
esmall = [(u, v) for (u, v, d) in G.edges(data=True) if d['weight'] <= 0.5]


def compute_weight_matrix(G , Q = None , directed = False ):
    #this method should return a matrix of weights normalized
    V = list(G.nodes)
    P = dict.fromkeys( V, 0 )
    R = {}
    for v in V :
        if v not in Q :
            P[v] = 0
        else :
            P[v] += sum([d["weight"] for (u,v,d) in G.edges(v, data=True)])
    #
    accum = 0
    for i in V :
        for j in V :
            if i == j :
                R[(i,j)] = 0
            else:
                R[(i,j)] = P[i]*P[j]
                accum += P[i]*P[j]
    #normalization
    for r in R.keys():
        R[r] = float(R[r])/accum
    return R , accum , P


r,a,p = compute_weight_matrix(G,G.nodes)
for v in p.keys() :
    p[v] = v+" "+str(p[v])
#print r
#print a

for nbr in G["a"]:
    print nbr

    a = "../Datasets/prova.csv"

with open(a, "rb") as csvfile:
    reader = csv.reader(csvfile, delimiter='\n', quotechar='#')
    for row in reader:
        stw = row[0].split(",")[:3]
        L.add_edge( int(stw[0]) , int(stw[1]) )
        L[int(stw[0])][int(stw[1])]["weight"] = int(stw[2])
# SOURCE, TARGET, RATING, TIME #


print L.nodes

pos = nx.spring_layout(G)  # positions for all nodes

edge_labels = dict([((u,v,),d['weight']) for u,v,d in G.edges(data=True)])
# nodes
nx.draw_networkx_nodes(G, pos, node_size=700)

# edges
nx.draw_networkx_edges(G, pos, edgelist=elarge,width=6)

nx.draw_networkx_edges(G, pos, edgelist=esmall, width=6, alpha=0.5, edge_color='b', style='dashed')

# labels


nx.draw_networkx_labels(G, pos, labels=p ,font_size=20, font_family='sans-serif')
nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)
plt.axis('off')
plt.show()
