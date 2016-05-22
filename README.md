An implementation of [Loopy Belief Propagation](https://en.wikipedia.org/wiki/Belief_propagation) in Python 3

### Usage

Example input file:
    
    yes no maybe
    5
    a 1 5 3
    b 1 2 4
    c 5 1 2
    d 3 3 1
    e 1 6 2
    a b 1.7
    a c 1.3
    a e 2.3
    b c 2.4
    c d 2
    d e 1.2

To run LBP:
    
	python3 LBP.py [filename]
