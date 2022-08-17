from nltk.stem.porter import PorterStemmer

class Node:

  def __init__(self,token,score,parentNode):
    self.token=token
    self.score=score
    self.parentNode=parentNode
    if self.parentNode != None:
      self.parentNode.addChildNode(self)
    self.childNodes=[]
    self.stemmer = PorterStemmer()
  
  def setToken(self,token):
    self.token=token
  
  def getToken(self):
    return self.token

  def addChildNode(self,node):
    self.childNodes.append(node)
    return self
  
  def addChildNodes(self,nodes: list):
    self.childNodes = nodes
    return self

  def getChildNodes(self):
    return self.childNodes
  
  def getStemmedQueryTerms(self):

    tokens = self.token.split(" ") if len(self.token.split(" ")) > 0 else [self.token]
    tokens = [self.stemmer.stem(token) for token in tokens]

    if self.parentNode == None:
      return tokens

    return self.parentNode.getStemmedQueryTerms() + tokens
  
  def getQueryTerms(self):

    tokens = self.token.split(" ") if len(self.token.split(" ")) > 0 else [self.token]

    if self.parentNode == None:
      return tokens

    return self.parentNode.getQueryTerms() + tokens
  
  def getParent(self):
    
    if self.parentNode == None:
      return [self]

    return [self] + self.parentNode.getParent()

  def treeHeight(self):
    if self.parentNode == None:
        return 0

    return 1 + self.parentNode.treeHeight()
  
  
  def treeDepth(self):
    # No children means depth zero below.
    if len(self.childNodes) == 0:
        return 0

    # Otherwise get deepest child recursively.
    deepestChild = 0
    for child in self.childNodes:
        childDepth = child.treeDepth()
        if childDepth > deepestChild:
          deepestChild = childDepth

    # Depth of this node is one plus the deepest child.
    return 1 + deepestChild

  
  
  def __str__(self):
    return f"{self.token} :: {self.score} :: {len(self.childNodes)} :: {self.parentNode.token if self.parentNode else 'None'}"
  
  def __repr__(self):
    return f"{self.token} :: {self.score} :: {len(self.childNodes)} :: {self.parentNode.token if self.parentNode else 'None'}"