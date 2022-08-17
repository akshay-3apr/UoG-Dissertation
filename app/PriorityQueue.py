# PriorityQueue implementation for BEST FIRST SEARCH
import itertools,math,json
from heapq import heappush, heappop, heapify

class PriorityQueue:

  def __init__(self):
    # list of entries states arranged based on priority score in a heap
    self.priorityqueue = []
    # keeping a trace of states query terms already explored and map it to state object
    self.cur_entry_finder = {}
    # self.load_existing_states()
    # unique sequence count
    self.counter = itertools.count()

  def add_state(self,state, priority=0):
    'Add a new state or update the priority of an existing state'
    #before inserting the data sort the query string of state
    count = next(self.counter)
    entry = [-1*priority, count, state] # multiply -1 to make it a max heap
    self.cur_entry_finder[state] = entry
    # if state not in self.existing_entry_finder:
      #if state is not in the existing states, add unique state
      # self.existing_entry_finder[state] = entry
    heappush(self.priorityqueue, entry)

  def pop_state(self):
    'Remove and return the lowest priority state.'
    # only unique states are considered
    priority, count, state = heappop(self.priorityqueue)
    # not deleting this entry as we do  not want to execute the same state
    return (state,-1*priority) # multiply -1 to change to actual value
  
  def check_state_exists(self,state):
    'Check if a state is already in the priority queue'
    if state in self.cur_entry_finder:
      #if exists do not add it again, already explored
      return False
    else:
      #if does not exists add it again, explore
      return True

  def load_existing_states(self):
    'Load the priority of existing states'
    try:
      with open('data/existing_unique_bfs_states.txt','r') as f:
        self.existing_entry_finder = json.load(f)
    except FileNotFoundError:
      self.existing_entry_finder = {}
  
  def save_existing_states(self):
    'Save the unique existing states'
    with open('data/existing_unique_bfs_states.txt','w') as f:
      f.write(json.dumps(self.existing_entry_finder))
  
  def already_explored(self,state):
    'Check if a state is already explored'
    return state in self.existing_entry_finder

  def get_state_score(self,state):
    'Get the priority of a state'
    return -1*self.existing_entry_finder[state][0]

  def __len__(self):
    return len(self.priorityqueue)
