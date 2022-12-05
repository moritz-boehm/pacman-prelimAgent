# myTeam.py
# ---------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).
import heapq
import queue
import random

import game
import util
from captureAgents import CaptureAgent


#################
# Team creation #
#################

def create_team(first_index, second_index, is_red,
                first='AStarAgent', second='AStarAgent', num_training=0):
    """
    This function should return a list of two agents that will form the
    team, initialized using firstIndex and secondIndex as their agent
    index numbers.  isRed is True if the red team is being created, and
    will be False if the blue team is being created.

    As a potentially helpful development aid, this function can take
    additional string-valued keyword arguments ("first" and "second" are
    such arguments in the case of this function), which will come from
    the --redOpts and --blueOpts command-line arguments to capture.py.
    For the nightly contest, however, your team will be created without
    any extra arguments, so you should make sure that the default
    behavior is what you want for the nightly contest.
    """

    # The following line is an example only; feel free to change it.
    return [eval(first)(first_index), eval(second)(second_index)]


class AgentSuperclass(CaptureAgent):
    """
    we put the functionality into here so that the attacker and defender can both inherit from it to get the
    functionality and we would not have duplicated code (at this point we only have an attacker, but we plan
    to implement a defender as well)
    """

    def registerInitialState(self, gameState):
        self.start = gameState.get_agent_position(self.index)
        CaptureAgent.register_initial_state(self, gameState)
        self.midWidth = int(gameState.data.layout.width / 2)
        self.height = int(gameState.data.layout.height)

    def entryPoints(self, gameState):
        # this is a list of Positions where the pacman can enter his "homeside"
        entries = []
        if self.red:
            i = self.midWidth - 1
        else:
            i = self.midWidth + 1

        for j in range(self.height):
            entries.append((i, j))

        validPositions = []
        for i in entries:
            if not gameState.has_wall(i[0], i[1]):
                validPositions.append(i)
        return validPositions

    def homeDistance(self, gameState):
        # this function depends upon the entry points of boundaryPosition and returns the minimum out of the
        # mazeDistances between the current position and the reentry points
        myState = gameState.get_agent_state(self.index)
        validReentry = self.entryPoints(gameState)
        dist = 9999
        for validPosition in validReentry:
            tempDist = self.get_maze_distance(validPosition, myState.get_position())
            if tempDist < dist:
                dist = tempDist
        return dist

    def closestGhostDistance(self, gameState):
        # this function is used by the heuristic to find out the distance between ourselves and the closest ghost
        myPosition = gameState.get_agent_state(self.index).get_position()
        enemies = []
        for i in self.get_opponents(gameState):
            enemies.append(gameState.get_agent_state(i))

        ghosts = []
        for a in enemies:
            if not a.is_pacman and a.get_position() is not None:
                ghosts.append(a)

        if len(ghosts) > 0:
            distances = []

            for g in ghosts:
                distances.append(self.get_maze_distance(myPosition, g.get_position()))
            return min(distances)
        else:
            return None

    def heuristic1(self, state, gameState):
        # the heuristic makes the agent dodge the ghosts so he does not run into them
        # the problem we still have is that food is too interesting so he can dodge the ghosts, but he will
        # go for food and die if the ghost follows him
        heuristic = 0
        if self.closestGhostDistance(gameState) is not None:
            enemies = [gameState.get_agent_state(i) for i in self.get_opponents(gameState)]
            ghosts = [a for a in enemies if not a.is_pacman and a.scared_timer < 2 and a.get_position() is not None]
            if ghosts is not None and len(ghosts) > 0:
                ghostPositions = [ghost.get_position() for ghost in ghosts]
                ghostDists = [self.get_maze_distance(state, ghostPosition) for ghostPosition in ghostPositions]
                ghostDist = min(ghostDists)
                if ghostDist < 2:
                    heuristic = pow((5 - ghostDist), 4)
        return heuristic

    def aStarSearch(self, problem, gameState, heuristic):
        # we are using the method we have written in the first project we did but return only one action that
        # the chooseAction() method can use
        # the only difficulty is that we need to define a problem that has a goalState to know when to stop
        # so our idea was to define problems like finding food, running away from ghosts, chasing pacmans, getting back
        # home or escaping from a chasing ghost
        # (up until now we are only using problems for finding food and returning home)

        from util import PriorityQueue
        frontier = PriorityQueue()
        start_state = problem.getStartState()
        visitedNodes = []
        path = []
        if problem.isGoalState(start_state):
            return path

        h = heuristic(start_state, gameState)
        frontier.push((start_state, [], 0), h)

        while True:
            if frontier.isEmpty():
                return

            node = frontier.pop()

            state = node[0]
            path = node[1]
            cost = node[2]

            visitedNodes.append(state)

            if problem.isGoalState(state):
                print(path[0])
                return path

            successors = problem.getSuccessors(state)

            for successor in successors:
                newPath = path + [successor[1]]

                newCost = successor[2] + cost

                priority = newCost + heuristic(successor[0], gameState)

                if successor[0] in visitedNodes:
                    continue

                flag = True
                for index, (p, c, m) in enumerate(frontier.heap):
                    if m[0] == successor[0]:
                        # print("found equal position")
                        flag = False
                        # own update
                        if (priority < p):
                            del frontier.heap[index]
                            frontier.push((successor[0], newPath, newCost), priority)
                            heapq.heapify(frontier.heap)
                        break

                if flag:
                    frontier.push((successor[0], newPath, newCost), priority)


class AStarAgent(AgentSuperclass):
    """
  A Dummy agent to serve as an example of the necessary agent structure.
  You should look at baselineTeam.py for more details about how to
  create an agent as this is the bare minimum.
  """

    def __init__(self, index, time_for_computing=.1):
        super().__init__(index, time_for_computing)
        self.start = None

    def register_initial_state(self, game_state):
        """
        This method handles the initial setup of the
        agent to populate useful fields (such as what team
        we're on).

        A distanceCalculator instance caches the maze distances
        between each pair of positions, so your agents can use:
        self.distancer.getDistance(p1, p2)

        IMPORTANT: This method may run for at most 15 seconds.
        """

        '''
        Make sure you do not delete the following line. If you would like to
        use Manhattan distances instead of maze distances in order to save
        on initialization time, please take a look at
        CaptureAgent.register_initial_state in captureAgents.py.
        '''
        CaptureAgent.register_initial_state(self, game_state)

        '''
        Your initialization code goes here, if you need any.
        '''
        self.start = game_state.get_agent_position(self.index)
        self.midWidth = int(game_state.data.layout.width / 2)
        self.height = int(game_state.data.layout.height)

    def getGhostDistance(self, gameState, index):
        myPosition = gameState.getAgentState(self.index).getPosition()
        ghost = gameState.getAgentState(index)
        dist = self.get_maze_distance(myPosition, ghost.getPosition())
        return dist

    def choose_action(self, game_state):
        if game_state.data.timeleft < self.homeDistance(game_state) + 30 or game_state.get_agent_state(
                self.index).num_carrying > 5:
            problem = ReturnHome(game_state, self, self.index)
            if len(self.aStarSearch(problem, game_state, self.heuristic1)) == 0:
                return 'Stop'
            else:
                return self.aStarSearch(problem, game_state, self.heuristic1)[0]
        problem = EatFood(game_state, self, self.index)
        return self.aStarSearch(problem, game_state, self.heuristic1)[0]

    def get_successor(self, game_state, action):
        """
        Finds the next successor which is a grid position (location tuple).
        """
        successor = game_state.generate_successor(self.index, action)
        pos = successor.get_agent_state(self.index).get_position()
        if pos != util.nearestPoint(pos):
            # Only half a grid position was covered
            return successor.generate_successor(self.index, action)
        else:
            return successor


class DefensiveAStarAgent(AgentSuperclass):
    """this is a defensive Agent. Whenever there are invaders on our side. It tries to defend. Otherwise it will move to
    the nearest accesspoint of an attacker. In the future it might be interesting to experiment with a dynamic change of
    behaviour considering the current score, opponent behaviour etc."""

    def __init__(self, index, time_for_computing=.1):
        super().__init__(index, time_for_computing)
        self.start = None

    def register_initial_state(self, game_state):
        """
        This method handles the initial setup of the
        agent to populate useful fields (such as what team
        we're on).

        A distanceCalculator instance caches the maze distances
        between each pair of positions, so your agents can use:
        self.distancer.getDistance(p1, p2)

        IMPORTANT: This method may run for at most 15 seconds.
        """

        '''
        Make sure you do not delete the following line. If you would like to
        use Manhattan distances instead of maze distances in order to save
        on initialization time, please take a look at
        CaptureAgent.register_initial_state in captureAgents.py.
        '''
        CaptureAgent.register_initial_state(self, game_state)

        '''
        Your initialization code goes here, if you need any.
        '''
        self.start = game_state.get_agent_position(self.index)
        self.midWidth = int(game_state.data.layout.width / 2)
        self.height = int(game_state.data.layout.height)

    def choose_action(self, game_state):
        opposition = [game_state.get_agent_state(agentIndex) for agentIndex in self.get_opponents(game_state)]
        visibleOpponents = [p for p in opposition if p.is_pacman and p.get_position is not None]
        #if there are no visible opponents on the homeside the agent should go to the closest entry point
        if len(visibleOpponents) == 0:
            problem = ReturnHome(game_state, self, self.index)
            nextMoves = self.aStarSearch(problem, game_state, self.heuristic1)
            if len(nextMoves) == 0:
                return "Stop"
            else:
                return nextMoves[0]
        #if there are visible invaders, the agent should try and catch them
        else:
            problem = CatchAttackers(game_state, self, self.index)
            nextMoves = self.aStarSearch(problem, game_state, self.heuristic1)[0]
            if len(nextMoves) == 0:
                return "Stop"
            else:
                return nextMoves[0]





class PositionSearchProblem:
    """
    We use Searchproblems in order to use a general A-Star Algorithm
    This is the ancestor class for all problems, which are applicable in specific circumstances
    """

    def __init__(self, gameState, agent, agentIndex=0, costFn=lambda x: 1):
        self.walls = gameState.getWalls()
        self.costFn = costFn
        self.startState = gameState.getAgentState(agentIndex).getPosition()

    def getStartState(self):
        return self.startState

    def isGoalState(self, state):

        util.raiseNotDefined()

    def getSuccessors(self, state):
        successors = []
        for action in [game.Directions.NORTH, game.Directions.SOUTH, game.Directions.EAST, game.Directions.WEST]:
            x, y = state
            dx, dy = game.Actions.direction_to_vector(action)
            nextx, nexty = int(x + dx), int(y + dy)
            if not self.walls[nextx][nexty]:
                nextState = (nextx, nexty)
                cost = self.costFn(nextState)
                successors.append((nextState, action, cost))

        return successors

    def getCostOfActions(self, actions):
        if actions is None:
            return 999999
        x, y = self.getStartState()
        cost = 0
        for action in actions:
            # Check figure out the next state and see whether its' legal
            dx, dy = game.Actions.direction_to_vector(action)
            x, y = int(x + dx), int(y + dy)
            if self.walls[x][y]:
                return 999999
            cost += self.costFn((x, y))
        return cost


class EatFood(PositionSearchProblem):
    """
   in search Food the goal is to move to the next fooditem
  """

    def __init__(self, gameState, agent, agentIndex=0):
        self.food = agent.get_food(gameState)
        self.capsule = agent.get_capsules(gameState)
        self.startState = gameState.get_agent_state(agentIndex).get_position()
        self.walls = gameState.get_walls()
        self.costFn = lambda x: 1
        self.carry = gameState.get_agent_state(agentIndex).num_carrying

    def isGoalState(self, state):
        # the goal state is the position of food or capsule
        x, y = state
        return self.food[int(x)][int(y)]


class ReturnHome(PositionSearchProblem):
    """
  Used to go back home
  """

    def __init__(self, gameState, agent, agentIndex=0):
        self.costFn = lambda x: 1
        self.homeBoundary = agent.entryPoints(gameState)
        self.startState = gameState.get_agent_state(agentIndex).get_position()
        self.walls = gameState.get_walls()

    def getStartState(self):
        return self.startState

    def isGoalState(self, state):
        return state in self.homeBoundary


class CatchAttackers(PositionSearchProblem):
    """This problem is used to declare the problem, where the homeside has to be defended"""

    def __init__(self, gameState, agent, agentIndex=0):
        self.food = agent.get_food(gameState)
        self.startState = gameState.get_agent_state(agentIndex).get_position()
        self.walls = gameState.get_walls()
        self.costFn = lambda x: 1
        self.opposition = [gameState.get_agent_state(agentIndex) for agentIndex in agent.get_opponents(gameState)]
        self.pacmen = [p for p in self.opposition if p.is_pacman and p.get_position is not None]
        if self.pacmen[0].get_position() is not None:
            self.pacmenPosition = [p.get_position() for p in self.pacmen]
            print(self.pacmenPosition)
        else:
            self.pacmenPosition = None

    def isGoalState(self, state):
        return state in self.pacmenPosition
