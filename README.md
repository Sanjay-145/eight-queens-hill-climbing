# Hill Climbing Algorithm for Eight Queens Problem
## AIM

To develop a code to solve eight queens problem using the hill-climbing algorithm.

## THEORY
At each iteration, the hill-climbing search algorithm moves to the best successor of the current
node according to an objective function. The algorithm does not maintain a search tree, so the data structure for the current node need only
record the state and the value of the objective function.

## DESIGN STEPS

### STEP 1:
Import required python packages.


### STEP 2:
Define the Initial State and get the number of random conflicts at the initial, then using the objective function calculates.


### STEP 3:
Make a decision whether to change if a state with a better objective function value, or stay in the current state.
### STEP 4:
Repeat the process until the total number of conflicts, or the Objective function, becomes zero.
### STEP 5:
By calculating the time taken by the function to reduce the conflict for varying number of iterations.
### Step 6:
Plot a graph between time taken and iterations.


# PROGRAM
## Developed By: P.Sanjay
## Register Number: 212220230042
~~~
%matplotlib inline
import matplotlib.pyplot as plt
import random
import math
import sys
from collections import defaultdict, deque, Counter
from itertools import combinations
from IPython.display import display
from notebook import plot_NQueens

class Problem(object):
    """The abstract class for a formal problem. A new domain subclasses this,
    overriding `actions` and `results`, and perhaps other methods.
    The default heuristic is 0 and the default action cost is 1 for all states.
    When yiou create an instance of a subclass, specify `initial`, and `goal` states 
    (or give an `is_goal` method) and perhaps other keyword args for the subclass."""

    def __init__(self, initial=None, goal=None, **kwds): 
        self.__dict__.update(initial=initial, goal=goal, **kwds) 
        
    def actions(self, state):        
        raise NotImplementedError
    def result(self, state, action): 
        raise NotImplementedError
    def is_goal(self, state):        
        return state == self.goal
    def action_cost(self, s, a, s1): 
        return 1
    
    def __str__(self):
        return '{0}({1}, {2})'.format(
            type(self).__name__, self.initial, self.goal)

class Node:
    "A Node in a search tree."
    def __init__(self, state, parent=None, action=None, path_cost=0):
        self.__dict__.update(state=state, parent=parent, action=action, path_cost=path_cost)

    def __str__(self): 
        return '<{0}>'.format(self.state)
    def __len__(self): 
        return 0 if self.parent is None else (1 + len(self.parent))
    def __lt__(self, other): 
        return self.path_cost < other.path_cost


failure = Node('failure', path_cost=math.inf) # Indicates an algorithm couldn't find a solution.
cutoff  = Node('cutoff',  path_cost=math.inf) # Indicates iterative deepening search was cut off.

def expand(problem, state):
    return problem.actions(state)

class NQueensProblem(Problem):

    def __init__(self, N):
        super().__init__(initial=tuple(random.randint(0,N-1) for _ in tuple(range(N))))
        self.N = N

    def actions(self, state):
        """ finds the nearest neighbors"""
        neighbors = []
        for i in range(self.N):
            for j in range(self.N):
                if j == state[i]:
                    continue
                s1 = list(state)
                s1[i]=j
                new_state = tuple(s1)
                yield Node(state=new_state)

    def result(self, state, row):
        """Place the next queen at the given row."""
        col = state.index(-1)
        new = list(state[:])
        new[col] = row
        return tuple(new)

    def conflicted(self, state, row, col):
        """Would placing a queen at (row, col) conflict with anything?"""
        return any(self.conflict(row, col, state[c], c)
                   for c in range(col))

    def conflict(self, row1, col1, row2, col2):
        """Would putting two queens in (row1, col1) and (row2, col2) conflict?"""
        return (row1 == row2 or  # same row
                col1 == col2 or  # same column
                row1 - col1 == row2 - col2 or  # same \ diagonal
                row1 + col1 == row2 + col2)  # same / diagonal

    def goal_test(self, state):
        return not any(self.conflicted(state, state[col], col)
                       for col in range(len(state)))

    def h(self, node):
        """Return number of conflicting queens for a given node"""
        num_conflicts = 0
        # Write your code here
        for (r1,c1) in enumerate(node.state):
            for (r2,c2) in enumerate(node.state):
                if (r1,c1)!=(r2,c2):
                    num_conflicts += self.conflict(r1,c1,r2,c2) 
        return num_conflicts

def shuffled(iterable):
    """Randomly shuffle a copy of iterable."""
    items = list(iterable)
    random.shuffle(items)
    return items

def argmin_random_tie(seq, key):
    """Return an element with highest fn(seq[i]) score; break ties at random."""
    return min(shuffled(seq), key=key)

def hill_climbing(problem,iterations = 10000):
    # as this is a stochastic algorithm, we will set a cap on the number of iterations        
    current = Node(problem.initial)
    i=1
    while i < iterations:
        neighbors = expand(problem,current.state)
        if not neighbors:
            break
        neighbour = argmin_random_tie(neighbors,key=lambda node:problem.h(node))
        if problem.h(neighbour)<=problem.h(current):
            current.state= neighbour.state
            if problem.goal_test(current.state)==True:
                print('The Goal state is reached at {0}'.format(i))
                return current 
                
        i += 1        
    return current    

nq1=NQueensProblem(8)
plot_NQueens(nq1.initial)
n1 = Node(state=nq1.initial)
num_conflicts = nq1.h(n1)
print("Initial Conflicts = {0}".format(num_conflicts))
sol1=hill_climbing(nq1,iterations=20000)
sol1.state
num_conflicts = nq1.h(sol1)
print("Final Conflicts = {0}".format(num_conflicts))
plot_NQueens(list(sol1.state))


import time
start=time.time()
end=time.time()
print("The total time required for 20000 iterations is {0:.4f} seconds".format(end-start))

iterations=[10,20,30,40,50,1000,2000,3000,4000,5000,10000]
time_taken=[]
num=1
for each_i in iterations:
    print("Type {0}:\tIterations:{1}".format(num,each_i))
    n1 = Node(state=nq1.initial)
    num_conflicts = nq1.h(n1)
    print("Initial Conflicts = {0}".format(num_conflicts))
    start=time.time()
    sol1=hill_climbing(nq1,iterations=each_i)
    end=time.time()
    print(sol1.state)
    num_conflicts = nq1.h(sol1)
    print("Final Conflicts = {0}".format(num_conflicts))
    print("The total time required for 20000 iterations is {0:.4f} seconds\n\n".format(end-start))
    time_taken.append(end-start)
    num+=1
    
plt.title("Number of Iterations VS Time taken")
plt.xlabel("Iteration")
plt.ylabel("Time taken")
plt.plot(iterations,time_taken)
plt.show()    

~~~


## OUTPUT:
### Results.
![2](https://user-images.githubusercontent.com/75235426/169993013-947bcefe-1362-415a-a5c0-ef24695d1a4a.jpg)
### Shows the timetaken for each iterations.
![4](https://user-images.githubusercontent.com/75235426/169993200-819f01eb-7a30-4fc6-a5f0-55a913f6749e.jpg)
![5](https://user-images.githubusercontent.com/75235426/169993174-f14ed62b-2da8-4546-bcc2-ec7879b5c0f2.jpg)
![6](https://user-images.githubusercontent.com/75235426/169993187-17070dc8-aa12-4bec-a055-f73045896039.jpg)
when the iterations increase, to complete the search time increases too.
### 8-Queen (initial conflicts).
![1](https://user-images.githubusercontent.com/75235426/169993304-27432bb2-e177-48a2-9adb-bb4d10c61d64.jpg)
#### (final solution)
![3](https://user-images.githubusercontent.com/75235426/169993356-055436d0-bbcc-4fc9-b723-eb8d866be65d.jpg)
## Time Complexity Plot
#### Plot a graph for the various value of N and time(seconds)
![7](https://user-images.githubusercontent.com/75235426/169993382-5dab7ca0-7ee8-451b-bc54-6fde034aae7e.jpg)


## RESULT:
Hence, this code solves the eight queens problem using the hill-climbing algorithm that has been implemented.
