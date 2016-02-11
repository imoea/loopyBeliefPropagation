#!/usr/bin/env python3

"""
An implementation of Loopy Belief Propagation
"""

__author__ = "Joshua Wong"
__version__ = "1.0"

from os.path import abspath
import argparse
import numpy as np
import time

class Timer:    
    """ Timer """
    def __enter__(self):
        self.start = time.clock() # start
        return self

    def __exit__(self, *args):
        self.end = time.clock() # end
        self.i = self.end - self.start # time taken

def getData(filename):
    """ Return node and inter-node potentials for belief propagation
        Note: all potentials should be >= 1 (i.e. in the log-domain)
        Input:  [filename] path to file containing data of the following format:

            label1 label2 ...
            node_count[int]
            node1[str] node1_potential1[float] node1_potential2[float] ...
            node2[str] node2_potential1[float] node2_potential2[float] ...
            ...
            node1[str] node2[str] node1_2_potential[float]
            ...
        
        Output: [labels] list of strings
                [nodes] list of strings
                [PHI] numpy array of node potentials
                [PSI] numpy array of inter-node potentials """

    with open(filename, 'r') as fi:
        labels = fi.readline().strip().split() # there should be at least 2 labels
        node_count = int(fi.readline().strip())
        potentials = [ line.split() for line in fi.read().strip().split('\n') ]

    nodes, lines, PHI = [], [], [] # initialise

    for i in range(node_count):
        nodes.append(potentials[i][0])
        PHI.append([ phi for phi in map(float, potentials[i][1:]) ]) # single node potentials

    for i in range(node_count, len(potentials)):
        lines.append(potentials[i][:2] + [float(potentials[i][2])]) # inter-node potentials
    
    index = { nodes[i]: i for i in range(node_count) }
    PSI = np.zeros((node_count, node_count)) # initialise

    for line in lines:
        PSI[index[line[0]], index[line[1]]] = line[2] # populate adjacency matrix

    return labels, nodes, np.array(PHI), PSI + PSI.T

def getMessage(i, PHIi, PSIi, label_j, labels, neighbours, messages):
    """ Return the message update from node i to all other nodes
        Input:  [i] integer index of node i
                [PHIi] numpy array of node i potentials
                [PSIi] numpy array of potentials between node i and its neighbours
                [label_j] label of node j
                [labels] list of strings
                [neighbours] list of indices of the neighbours of i
                [messages] dict of numpy arrays of messages between all nodes
        Output: [messages] numpy array of messages from node i to all other nodes """

    m = { label: np.zeros(messages[label].shape[0]) for label in labels } # initialise
    zeros = np.zeros(len(PSIi))

    for neighbour in neighbours:
        S = set(neighbours) - {neighbour}
        for label in labels: # compute contribution to each neighbour from all other neighbours of i
            m[label][neighbour] = np.prod([ messages[label][k,i] for k in S ]) + .00000001

    return sum( np.multiply(PHIi[i] * (PSIi if label_i == label_j else zeros), m[label_i]) for i, label_i in enumerate(labels) )
    
def getScores(i, PHIi, labels, neighbours, messages):
    """ Return the scores of each label of a node
        Input:  [i] integer index of node i
                [PHIi] numpy array of node i potentials
                [labels] list of strings
                [neighbours] list of indices of the neighbours of i
                [messages] dict of numpy arrays of messages between all nodes
        Output: [scores] dict of floats """

    scores = { label: PHIi[label_i] * np.prod([ messages[label][j,i] for j in neighbours ]) for label_i, label in enumerate(labels) }

    # normalise scores
    alpha = sum( scores[label] for label in labels )
    if alpha:
        for label in labels:
            scores[label] *= 1 / alpha

    return scores

def getLabels(scores):
    """ Return the predicted label of a node
        Input:  [scores] list of dicts of floats
        Output: [results] list of strings """

    return [ max(score, key=score.get) for score in scores ]

def LBP(labels, nodes, PHI, PSI):
    """ Return the polarity of a list of nodes computed using loopy belief propagation
        Input:  [labels] list of strings
                [nodes] list of strings
                [PHI] numpy array of node potentials
                [PSI] numpy array of inter-node potentials
        Output: [results] list of strings
                [scores] list of dicts of floats """

    node_count = len(nodes)

    with Timer() as t: # do some extra initialisation
        messages = { label: (PSI > 0).astype(float) for label in labels } # initialise propagation matrices
        neighbours = [ list(np.nonzero(PSI[i])[0]) for i in range(node_count) ] # neighbours of all nodes

    print("[{:.3f}s] Graph built. Density: {}".format(t.i, 2. * np.sum(PSI > 0) / (node_count * node_count - node_count)))

    print("\nStarting propagation on a {} matrix...".format(PSI.shape))

    loops = 0
    while True:
        with Timer() as t: # compute messages
            loops += 1
            old_messages = { label: messages[label].copy() for label in labels } # archive messages

            for i in range(node_count): # for each node
                for label in labels: # compute message from node i to its neighbours
                    messages[label][i,:] = getMessage(i, PHI[i,:].flatten(), PSI[i,:].flatten(), label, labels, neighbours[i], old_messages)

                alpha = [ alpha for alpha in map(lambda x : 1/x if x else 1, np.sum([ messages[label][i,:] for label in labels ], axis=0)) ] # compute normaliser
                for label in labels:
                    messages[label][i,:] *= alpha # normalise messages

        print("[{:.3f}s] Loop {} completed.\tMessage change: {}".format(t.i, loops, np.sum([ abs(old_messages[label] - messages[label]) for label in labels ])))

        if np.product([ np.allclose(old_messages[label], messages[label]) for label in labels ]): # halt if messages stop changing
            break

    print("Propagation complete.\n")

    with Timer() as t: # compute final labels
        scores = [ getScores(i, PHI[i,:], labels, neighbours[i], messages) for i in range(node_count) ]
        results = getLabels(scores)

    print("[{:.3f}s] Final labels computed. Objective value: {}".format(t.i, sum( max(score.values()) for score in scores )))

    return results, scores

def printResults(filename, labels, nodes, results, scores):
    """ Print results to file
        Input:  [filename] path to data file
                [labels] list of strings
                [nodes] list of strings
                [results] list of strings
                [scores] list of dicts of floats """

    filename += '.results'
    with open(filename, 'w') as fo:
        fo.write("ID\tNode\tLabel\t{}\n".format('\t'.join(labels)))
        for i in range(len(nodes)):
            fo.write("{}\t{}\t{}\t{}\n".format(i, nodes[i], results[i], '\t'.join(str(scores[i][label]) for label in labels)))
    
    print("Results written to '{}'.".format(abspath(filename)))

def main():
    """ Run if called from command-line """

    parser = argparse.ArgumentParser()
    parser.add_argument('filename', help="path to file")
    args = parser.parse_args()

    labels, nodes, PHI, PSI = getData(args.filename) # read file
    results, scores = LBP(labels, nodes, PHI, PSI) # run propagation
    printResults(args.filename, labels, nodes, results, scores) # print results

if __name__ == '__main__':
    main()
