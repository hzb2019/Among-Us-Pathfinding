# Among Us Pathfinding

This was the final project for our Fall 2021 CSE 331 Algorithms and Data Structures course.
It was to give us practice with Graph data structures and the implementation of their common functions.
This probably does not have much actual use for the game Among Us.

All project code can be found in Graph.py, tests can be found in tests.py, all tests pass as of Python 3.9.

BFS implementation uses an array copy call to keep track of searched nodes that ruined its time complexity. It can instead be implemented using a dictionary like the A* search implementation to maintain O(V+E) time complexity.

Final Score: 97/100
