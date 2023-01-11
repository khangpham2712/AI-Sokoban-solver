import sys
import collections
import numpy as np
import heapq
import time
# from memory_profiler import memory_usage
# import argparse

class PriorityQueue:
    """Define a PriorityQueue data structure that will be used"""

    #Constructor
    def __init__(self):
        self.Heap = []
        self.Count = 0

    # Push to Priority Queue with item and its priority
    def push(self, item, priority):
        entry = (priority, self.Count, item)
        heapq.heappush(self.Heap, entry)
        self.Count += 1
    # Pop the smallest item from Queue
    def pop(self):
        (_, _, item) = heapq.heappop(self.Heap)
        return item
    # check queue empty
    def isEmpty(self):
        return len(self.Heap) == 0



"""Load puzzles and define the rules of sokoban"""
def transferToGameState(layout):
    # Transfer the layout of initial puzzle
    # Parameter is initilization of problem
    """
        Layout e.g.:
            ######
            #.  .#
            #    #
            # $$ #
            #@   #
            ######
    """
    layout = [x.replace("\n", "") for x in layout]
    layout = [",".join(layout[i]) for i in range(len(layout))]
    layout = [x.split(",") for x in layout]
    maxColsNum = max([len(x) for x in layout])
    for irow in range(len(layout)):
        for icol in range(len(layout[irow])):
            if layout[irow][icol] == " ":
                layout[irow][icol] = 0  # free space
            elif layout[irow][icol] == "#":
                layout[irow][icol] = 1  # wall
            elif layout[irow][icol] == "@":
                layout[irow][icol] = 2  # player
            elif layout[irow][icol] == "$":
                layout[irow][icol] = 3  # box
            elif layout[irow][icol] == ".":
                layout[irow][icol] = 4  # goal
            elif layout[irow][icol] == "*":
                layout[irow][icol] = 5  # box on goal
        colsNum = len(layout[irow])
        if colsNum < maxColsNum:
            layout[irow].extend([1 for _ in range(maxColsNum - colsNum)])
    # return the game state as matrix
    """
        Return value e.g.:
        1 1 1 1 1 1
        1 4 0 0 4 1
        1 0 0 0 0 1
        1 0 3 3 0 1
        1 2 0 0 0 1
        1 1 1 1 1 1
    """ 
    return np.array(layout) 


def PosOfPlayer(gameState):
    """Return the position of agent"""
    return tuple(np.argwhere(gameState == 2)[0])  # e.g. (2, 2)


def PosOfBoxes(gameState):
    """Return the positions of boxes"""
    return tuple(
        tuple(x) for x in np.argwhere((gameState == 3) | (gameState == 5))
    )  # e.g. ((2, 3), (3, 4), (4, 4), (6, 1), (6, 4), (6, 5))


def PosOfWalls(gameState):
    """Return the positions of walls"""
    return tuple(tuple(x) for x in np.argwhere(gameState == 1))  # e.g. like those above


def PosOfGoals(gameState):
    """Return the positions of goals"""
    return tuple(
        tuple(x) for x in np.argwhere((gameState == 4) | (gameState == 5))
    )  # e.g. like those above


def isEndState(posBox):
    """Check if all boxes are on the goals (i.e. pass the game)"""
    return sorted(posBox) == sorted(posGoals)


def isLegalAction(action, posPlayer, posBox):
    """Check if the given action is legal"""
    # action is an action of allAction (e.g. [[-1, 0, "u"]])

    xPlayer, yPlayer = posPlayer
    # the move was a push (Push is upper char)
    if action[-1].isupper():
        x1, y1 = xPlayer + 2 * action[0], yPlayer + 2 * action[1]
    else:
        x1, y1 = xPlayer + action[0], yPlayer + action[1]
    return (x1, y1) not in posBox + posWalls


def legalActions(posPlayer, posBox):
    """Return all legal actions for the agent in the current game state"""

    # Upper char means moving box
    allActions = [
        [-1, 0, "u", "U"],
        [1, 0, "d", "D"],
        [0, -1, "l", "L"],
        [0, 1, "r", "R"],
    ]

    xPlayer, yPlayer = posPlayer
    legalActions = []
    for action in allActions:
        x1, y1 = xPlayer + action[0], yPlayer + action[1]
        if (x1, y1) in posBox:  # the move was a push
            action.pop(2)  # drop the little letter
        else:
            action.pop(3)  # drop the upper letter

        if isLegalAction(action, posPlayer, posBox): #check if action is illegal action
            legalActions.append(action)
        else:
            continue
    return tuple(tuple(x) for x in legalActions)  # e.g. ((0, -1, 'l'), (0, 1, 'R'))


def updateState(posPlayer, posBox, action):
    """Return updated game state after an action is taken"""
    xPlayer, yPlayer = posPlayer  # the previous position of player
    newPosPlayer = [
        xPlayer + action[0],
        yPlayer + action[1],
    ]  # the current position of player
    posBox = [list(x) for x in posBox]
    if action[-1].isupper():  # if pushing, update the position of box
        posBox.remove(newPosPlayer) # remove previous box's position
        posBox.append([xPlayer + 2 * action[0], yPlayer + 2 * action[1]]) # add current box's position
    
    
    posBox = tuple(tuple(x) for x in posBox) # Save new box position as a new tuple
    newPosPlayer = tuple(newPosPlayer) # Save new player position as a new tuple
    return newPosPlayer, posBox # return new value

#This is a function used to check if the box is stuck among the walls or other boxes leading to failure or not
def isFailed(posBox):
    """This function used to observe if the state is potentially failed, then prune the search"""
    #rotate deadlock state
    rotatePattern = [
        [0, 1, 2, 3, 4, 5, 6, 7, 8],
        [2, 5, 8, 1, 4, 7, 0, 3, 6],
        [0, 1, 2, 3, 4, 5, 6, 7, 8][::-1],
        [2, 5, 8, 1, 4, 7, 0, 3, 6][::-1],
    ]
    #flip deadlock state
    flipPattern = [
        [2, 1, 0, 5, 4, 3, 8, 7, 6],
        [0, 3, 6, 1, 4, 7, 2, 5, 8],
        [2, 1, 0, 5, 4, 3, 8, 7, 6][::-1],
        [0, 3, 6, 1, 4, 7, 2, 5, 8][::-1],
    ]
    # merge list to check all situation
    allPattern = rotatePattern + flipPattern

    for box in posBox:
        if box not in posGoals:
            #these are the surrounding positions
            board = [
                (box[0] - 1, box[1] - 1),     #top left
                (box[0] - 1, box[1]),         #top
                (box[0] - 1, box[1] + 1),     #top right
                (box[0], box[1] - 1),         #left
                (box[0], box[1]),             #center
                (box[0], box[1] + 1),         #right
                (box[0] + 1, box[1] - 1),     #bottom left
                (box[0] + 1, box[1]),         #bottom
                (box[0] + 1, box[1] + 1),     #bottom right
            ]

            #check Box in corner
            for pattern in allPattern:
                newBoard = [board[i] for i in pattern]
#   W       or      B W     or        W      or    W B
#   B W             W               W B              W
                if newBoard[1] in posWalls and newBoard[5] in posWalls:
                    return True
#   B W     or      B B     or      W B      or     W W
#   B W             W W             W B             B B
                elif (
                    newBoard[1] in posBox
                    and newBoard[2] in posWalls
                    and newBoard[5] in posWalls
                ):
                    return True
#   B W     or      B B     or      B B      or     W B
#   B B             B W             W B             B B
                elif (
                    newBoard[1] in posBox
                    and newBoard[2] in posWalls
                    and newBoard[5] in posBox
                ):
                    return True
#   B B
#   B B
                elif (
                    newBoard[1] in posBox
                    and newBoard[2] in posBox
                    and newBoard[5] in posBox
                ):
                    return True
#     B W       or        W           or                    or      W   
#   W B                   B B                     B W               B B
#                           W                   W B                   W 
                elif (
                    newBoard[1] in posBox
                    and newBoard[2] in posWalls
                    and newBoard[3] in posWalls     
                ):
                    return True
    return False


"""Implement all approcahes"""


def breadthFirstSearch():
    """Implement breadthFirstSearch approach"""
    # Get the original position of the boxes and the player
    beginBox = PosOfBoxes(gameState)
    beginPlayer = PosOfPlayer(gameState)

    # Initalize the game's state space
    startState = (
        beginPlayer,
        beginBox,
    )  # e.g. testcase1. ((4, 1), ((3, 2), (3, 3)))

    # store states, the deque object acts as a queue- FIFO. 
    # A container for nodes that have been found in the search tree.
    # first, we will save the game start state to the frontier
    frontier = collections.deque([[startState]])  
    # store actions
    actions = collections.deque([[0]]) 
    # store visted nodes 
    exploredSet = set()
    # while frontier not empty:
    while frontier:
        # get the node from the left side of the frontier
        node = frontier.popleft()
        # get the action from the left side of the actions
        node_action = actions.popleft()
        if isEndState(node[-1][-1]): # if the game end (matched with goal state):
            # print the solution
            print("SOLUTION : ")
            printAllSolve(node, node_action)
            print("LIST OF STEPS TO TAKE TO SOLUTION:")
            
            print(",".join(node_action[1:]).translate(str.maketrans({',': ' - ', 'U': 'UP','u': 'UP','l': 'LEFT','L': 'LEFT','d': 'DOWN','D': 'DOWN','r': 'RIGHT','R': 'RIGHT',})))

            break
        # if the current node is unvisited:
        if node[-1] not in exploredSet: 
            # add this node to the exploredSet
            exploredSet.add(node[-1])
            # consider all next moves with the current node : UP - DOWN - LEFT -RIGHT
            for action in legalActions(node[-1][0], node[-1][1]):
                # new game state when doing action
                newPosPlayer, newPosBox = updateState(node[-1][0], node[-1][1], action)
                # check the deadlock
                if isFailed(newPosBox):
                    continue # skip if deadlock
                # Add new state to the right side of the deque.
                frontier.append(node + [(newPosPlayer, newPosBox)])
                # Add this action to the right side of the deque.
                actions.append(node_action + [action[-1]])


def depthFirstSearch():
    """Implement depthFirstSearch approach"""
    # Get the original position of the boxes and the player
    beginBox = PosOfBoxes(gameState)
    beginPlayer = PosOfPlayer(gameState)

    # Initalize the game's state space
    startState = (beginPlayer, beginBox) # e.g. testcase1. ((4, 1), ((3, 2), (3, 3)))
    # store states, the deque object acts as a stack- FILO. 
    # A container for nodes that have been found in the search tree.
    # first, we will save the game start state to the frontier.
    frontier = collections.deque([[startState]])
    # store actions
    actions = [[0]]
    # store visted nodes 
    exploredSet = set()
    # while frontier not empty:
    while frontier:
        # get the node from the right side of the frontier
        node = frontier.pop()
        # get the action from the right side of the actions
        node_action = actions.pop()
        if isEndState(node[-1][-1]): # if the game end (matched with goal state):
            # print the solution
            print("SOLUTION : ")
            printAllSolve(node, node_action)
            print("LIST OF STEPS TO TAKE TO SOLUTION:")
            #
            print(",".join(node_action[1:]).translate(str.maketrans({',': ' - ', 'U': 'UP','u': 'UP','l': 'LEFT','L': 'LEFT','d': 'DOWN','D': 'DOWN','r': 'RIGHT','R': 'RIGHT',})))
            break
        # if the current node is unvisited:
        if node[-1] not in exploredSet:
            # add this node to the exploredSet
            exploredSet.add(node[-1])
            # consider all next moves with the current node : UP - DOWN - LEFT -RIGHT
            for action in legalActions(node[-1][0], node[-1][1]):
                # new game state when doing action
                newPosPlayer, newPosBox = updateState(node[-1][0], node[-1][1], action)
                # check the deadlock
                if isFailed(newPosBox):
                    continue # skip if deadlock
                # Add new state to the right side of the deque.
                frontier.append(node + [(newPosPlayer, newPosBox)])
                # Add this action to the right side of the deque.
                actions.append(node_action + [action[-1]])


def heuristic(posPlayer, posBox):
    """A heuristic function to calculate the overall distance between the else boxes and the else goals"""
    distance = 0
    # Listing Box that node in goal and sort them
    completes = set(posGoals) & set(posBox)
    sortposBox = list(set(posBox).difference(completes))
    sortposGoals = list(set(posGoals).difference(completes))
    # Calculate distance amongs box and goal
    for i in range(len(sortposBox)):
        distance += (abs(sortposBox[i][0] - sortposGoals[i][0])) + (
            abs(sortposBox[i][1] - sortposGoals[i][1])
        )
    return distance


def cost(actions):
    """A cost function"""
    return len([x for x in actions if x.islower()])


def aStarSearch():
    """Implement aStarSearch approach"""
    # Get the original position of the boxes and the player
    beginBox = PosOfBoxes(gameState)
    beginPlayer = PosOfPlayer(gameState)

    # Initalize the game's state space
    start_state = (beginPlayer, beginBox)  # e.g. testcase1. ((4, 1), ((3, 2), (3, 3)))
    # Using priority Queue to store state 
    # Push starting information (beginPlayer, beginBox) to priority  queue
    frontier = PriorityQueue()
    frontier.push([start_state], heuristic(beginPlayer, beginBox))

    # Create a set to save explored node
    exploredSet = set()
    # Get action from state, use heuristc function generate next action and push to Queue
    actions = PriorityQueue()
    actions.push([0], heuristic(beginPlayer, start_state[1]))
    while frontier:
        # get the node from the frontier of state queue
        node = frontier.pop()
        # get action from action queue
        node_action = actions.pop()

        #check the solution have found yet, if so, return and print solution
        if isEndState(node[-1][-1]):
            print("SOLUTION : ")
            printAllSolve(node, node_action)
            print("LIST OF STEPS TO TAKE TO SOLUTION:")
            #
            print(",".join(node_action[1:]).translate(str.maketrans({',': ' - ', 'U': 'UP','u': 'UP','l': 'LEFT','L': 'LEFT','d': 'DOWN','D': 'DOWN','r': 'RIGHT','R': 'RIGHT',})))

            break
        # if the current node is unvisited: 
        if node[-1] not in exploredSet:
            # add this node to the exploredSet
            exploredSet.add(node[-1])
            # Calcutlate the code of the action
            Cost = cost(node_action[1:])
            # 
            for action in legalActions(node[-1][0], node[-1][1]):
                newPosPlayer, newPosBox = updateState(node[-1][0], node[-1][1], action)
                # check the deadlock
                if isFailed(newPosBox):
                    continue # skip if deadlock
                # using Heuristic function to calculate cost of new player's position and new box's position
                Heuristic = heuristic(newPosPlayer, newPosBox)
                # enqueue new node to priority queue
                frontier.push(node + [(newPosPlayer, newPosBox)], Heuristic + Cost)
                # enqueue new action to priority queue 
                actions.push(node_action + [action[-1]], Heuristic + Cost)
    


"""Read command"""
def readCommand(argv):

    from optparse import OptionParser
    parser = OptionParser()
    parser.add_option(
        "-l",
        "--level",
        dest="sokobanLevels",
        help="level of game to play",
        default="miniTest1.txt",
    )
    parser.add_option(
        "-m", "--method", dest="agentMethod", help="research method", default="bfs"
    )
    
    args = dict()
    options, _ = parser.parse_args(argv)
    with open("sokobanLevels/" + options.sokobanLevels, "r") as f:
        layout = f.readlines()
    args["layout"] = layout
    args["method"] = options.agentMethod
    return args


#This is actually a function used for updating map instead of printing like its name says
def printCurrentMap(pos):
    # pos = ((4, 2), ((3, 2), (3, 3)))
    new_map = layout
    new_map = [x.replace("\n", "") for x in new_map]
    new_map = [",".join(layout[i]) for i in range(len(new_map))]
    new_map = [x.split(",") for x in new_map]
    maxColsNum = max([len(x) for x in new_map])
    for irow in range(len(new_map)):
        for icol in range(len(new_map[irow])):
            if new_map[irow][icol] == "@":  new_map[irow][icol] = " "  # player
            elif (new_map[irow][icol] == "$"):  new_map[irow][icol] = " "  # box
            elif (new_map[irow][icol] == "*"):  new_map[irow][icol] = "."  # box on goal

    pos_player = pos[0]
    pos_box = pos[1]
    num_box = len(pos_box)
    new_map[pos_player[0]][pos_player[1]] = "@"

    for i in range(num_box):
        if new_map[pos_box[i][0]][pos_box[i][1]] == ".": new_map[pos_box[i][0]][pos_box[i][1]] = "*"
        else : new_map[pos_box[i][0]][pos_box[i][1]] = "$"
    
         
    return new_map

#print indicators for each step
def printAllSolve(node, node_action): 
    for i in range(len(node)):
        if(node_action[i]==0):
            print('\nSTART')
        elif(node_action[i]=='u' or node_action[i]=='U'): #up
            print('\n; '+str(i)+'. NEXT STEP: UP')
        elif(node_action[i]=='d' or node_action[i]=='D'): #down
            print('\n; '+str(i)+'. NEXT STEP: DOWN')
        elif(node_action[i]=='l' or node_action[i]=='L'): #left
            print('\n; '+str(i)+'. NEXT STEP: LEFT')
        elif(node_action[i]=='r' or node_action[i]=='R'): #right
            print('\n; '+str(i)+'. NEXT STEP: RIGHT')
        
        #update map
        new_map = printCurrentMap(node[i])
        re = ''
        for irow in range(len(new_map)):
            for icol in range(len(new_map[irow])):
                re += new_map[irow][icol]
        #print map
        print(re)

if __name__ == "__main__":
    # Start time for calculate time used for each method
    time_start = time.time()

    # Load layout, method option from user input via terminal
        # layout is the initialization of the problem
    layout, method = readCommand(sys.argv[1:]).values()

    # transfer Layout loaded to Game state
    gameState = transferToGameState(layout)

    # get position of walls and goals base on game state
    posWalls = PosOfWalls(gameState)
    posGoals = PosOfGoals(gameState)

    # Run method base on user's choice
    # Calculate method using for each method
    if method == "astar":
        print('SEARCH WITH STRATEGY ASTAR')
        aStarSearch()
    elif method == "dfs":
        print('SEARCH WITH STRATEGY DFS')
        depthFirstSearch()
    elif method == "bfs":
        print('SEARCH WITH STRATEGY BrFS')
        breadthFirstSearch()
    else:
        raise ValueError("Invalid method.")
    
    # End time for calculate time used for each method
    time_end = time.time()

    # Print time used
    print("Runtime of %s: %.2f second." % (method, time_end - time_start))


