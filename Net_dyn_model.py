import networkx as nx
import numpy as np
import sys, os, time
import random
random.seed()
from random import random
import bisect
import matplotlib.pyplot as plt
import multiprocessing

class Dyn_model(object):
    def __init__(self):
        print(">>>>> Simulating preferential deletion model in dynamic networks")

    def _init_graph(self):
        """
        Initialize the graph model
        """
        self.G = nx.Graph()
        self.G.add_nodes_from([1,2,3,4,5])
        self.G.add_edges_from([(1,2),(2,3),(2,4)\
                        ,(2,5),(3,4),(4,5)])

    def _birth_proc(self):
        s_b = []
        d_t = self.G.degree()
        n_m = self.G.number_of_edges()
        n_t = self.G.number_of_nodes()
        g_n = self.G.nodes()
        for k in g_n:
            if len(s_b)==0:
                s_b.append(d_t[k])
            else:
                s_b.append(s_b[-1] + d_t[k])

        prf_attch_node_idx = bisect.bisect_left(s_b, random()*s_b[-1])
        new_node = max(g_n)+1
        self.G.add_node(new_node)
        self.G.add_edge(g_n.keys()[prf_attch_node_idx-1], new_node)

    def _death_proc(self):
        s_d = []
        d_t = self.G.degree()
        n_m = self.G.number_of_edges()
        n_t = self.G.number_of_nodes()
        g_n = self.G.nodes()
        for k in g_n:
            if len(s_d) == 0:
                s_d.append( n_t - d_t[k])
            else:
                s_d.append(s_d[-1] + n_t - d_t[k])
        delete_node_idx = bisect.bisect_left(s_d, random()*s_d[-1])
        self.G.remove_node(g_n.keys()[delete_node_idx-1])

    def _evolve_time(self, prob):
        p = prob[0]
        n_a = prob[1]
        Nn = 0
        En = 0
        print('>>>>> Graph model evolving over time with p = ' + str(p))
        for j in xrange(n_a):
            self._init_graph()
            N = []
            E = []
            for i in xrange(2*5000):
                N.append(self.G.number_of_nodes())
                E.append(self.G.number_of_edges())
                if random() <= p:
                    self._birth_proc()
                else:
                    self._death_proc()
            Nn += np.array(N)
            En += np.array(E)
        return (Nn/float(n_a), En/float(n_a))

    def _gen_analytical(self, p):
        En = []
        Ee = []
        q = 1-p
        for t in xrange(5):
            En.append(((p-q)*(t+1)*10000)+ (2*q) + 5)
            Ee.append(p*(p-q)*(t+1)*10000 + 6)

        return (En,Ee)


    def _gen_figures(self, p):
        d_t = self.G.degree()
        n_m = self.G.number_of_edges()
        n_t = self.G.number_of_nodes()
        g_n = self.G.nodes()
        s_b = []
        for k in g_n:
            if d_t[k] != 0:
                s_b.append(d_t[k])
        S_b = sorted(s_b)
        k = [i for i in xrange(2,S_b[-1])]
        a_k = [i**(-1-(float(2*p)/float((2*p)-1))) for i in k]

        S_b_dict = {i:S_b.count(i) for i in S_b}
        x = S_b_dict.values()

        y = [x[-i] for i in xrange(1,len(x)+1)]
        A_k = [a_k[-i] for i in xrange(1,len(a_k)+1)]

        x = np.flip(np.cumsum(y), 0)
        A_k = np.flip(np.cumsum(A_k), 0)

        prob_dist = (np.array(x))/float(sum(S_b_dict.values()))
        prob_dist_anl = (np.array(A_k))/float(sum(a_k))
        plt.figure()
        xx = list(S_b_dict.keys())

        plt.loglog(xx, prob_dist, 'bs')
        plt.loglog(k, prob_dist_anl)
        plt.savefig("figure5.jpg", dpi=150)
        plt.show()

def run():
    t0 = time.time()
    dyn_model = Dyn_model()
    n_nodes = []
    n_edges = []
    for prob in [[0.6, 5], [0.75,2], [0.9, 1], [0.8, 1]]:
        try:
            result = dyn_model._evolve_time(prob)
        except Exception as e:
            os.system('clear')
            print('Graph death occured! Reinitializong the simulator ...')
            os.system('python Net_dyn_model.py')
        n_nodes.append(5*result[0])
        n_edges.append(5*result[1])
    dyn_model._gen_figures(0.8)
    sys.exit()
    print('\nDone! ... Plotting the figures ...')
    print(float(time.time()-t0)/60)
    EN = []
    EE = []
    for prob in [0.6,0.75,0.9]:
        analytical = dyn_model._gen_analytical(prob)
        EN.append(analytical[0])
        EE.append(analytical[1])

    plt.figure()
    x = [5*i for i in xrange(2000, 10000)]
    t = [10000*(i+1) for i in xrange(5)]
    plt.plot(x, n_nodes[0][2000:])
    plt.plot(t, EN[0], 'r^', label='0.6')
    plt.plot(x, n_nodes[1][2000:])
    plt.plot(t, EN[1], 'rs', label='0.75')
    plt.plot(x, n_nodes[2][2000:])
    plt.plot(t, EN[2], 'ro', label='0.9')
    plt.plot()
    plt.legend()
    plt.savefig("figure2.jpg", dpi=150)

    plt.figure()
    plt.plot(x, n_edges[0][2000:])
    plt.plot(t, EE[0], 'r^', label='0.6')
    plt.plot(x, n_edges[1][2000:])
    plt.plot(t, EE[1], 'rs', label='0.75')
    plt.plot(x, n_edges[2][2000:])
    plt.plot(t, EE[2], 'ro', label='0.9')
    plt.plot()
    plt.legend()
    plt.savefig("figure3.jpg", dpi=150)

    dyn_model._gen_figures(0.8)

def main():
    run()


if __name__=='__main__':
    main()
