# PriorityQueue implementation for BEST FIRST SEARCH
import itertools,math
from heapq import heappush, heappop, heapify

class PriorityQueue:

  def __init__(self):
    # list of entries states arranged based on priority score in a heap
    self.priorityqueue = []
    # keeping a trace of states query terms already explored and map it to state object
    self.entry_finder = {}
    # unique sequence count
    self.counter = itertools.count()

  def add_state(self,state, priority=0):
    'Add a new state or update the priority of an existing state'
    #before inserting the data sort the query string of state
    count = next(self.counter)
    entry = [-1*priority, count, state] # multiply -1 to make it a max heap
    self.entry_finder[state] = entry
    heappush(self.priorityqueue, entry)

  def pop_state(self):
    'Remove and return the lowest priority state.'
    # only unique states are considered
    priority, count, state = heappop(self.priorityqueue)
    # not deleting this entry as we do  not want to execute the same state
    return (state,-1*priority) # multiply -1 to change to actual value
  
  def check_state_exists(self,state):
    'Check if a state is already in the priority queue'
    if state in self.entry_finder:
      #if exists do not add it again, already explored
      return False
    else:
      #if does not exists add it again, explore
      return True

  def __len__(self):
    return len(self.priorityqueue)
