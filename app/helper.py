#import libraries
from nltk.stem.porter import PorterStemmer
import nltk
#import evaluation matrices: kendall's tau and ranked-bias overlap
from rbo import RankingSimilarity
from scipy.stats import kendalltau
from scipy import spatial
from rake_nltk import Rake
import pandas as pd
from tqdm import tqdm
import math
from PriorityQueue import PriorityQueue
import gensim

## Get document posting list
def getDocuments(index,row):
  '''
  - fetch document terms from posting list
  - parameters:
    - 1. index: pyTerrier index to retrieve the terms from posting
    - 2. row: row of dataframe to fetch the document terms
  '''
  docid=row["docid"]
  terms = []
  pointer = index.getDocumentIndex().getDocumentEntry(docid)
  for p in index.getDirectIndex().getPostings(pointer):
    termid = p.getId()
    term = index.getLexicon()[termid].getKey()
    # print(f"\t{term}::{p.getFrequency()}")
    terms.append({term:p.getFrequency()})
  row['documentTerms']=terms
  return row

## Fetch raw documents
def getDocumentsCorpus(corpus,row):
  '''
  - fetch document terms from posting list
  - parameters:
    - 1. corpus: corpus of documents
    - 2. row: row of dataframe to fetch the document terms
  '''
  row['doctext']=corpus[corpus.docno==int(row.docno)].text.values[0]
  return row


## Filter records from subgroup top-k
def filterTopKRankRecord(documents,index,topK=10):
  '''
    - filter out top documents
    - parameters:
      - 1. documents: sorted dataframe over rank to filter out rows
      - 2. index: dataset index to fetch the posting list
      - 3. topk: value filter out top ranked documents from dataframe
  '''
  documents = documents[documents['rank']<=topK]
  for idx,document in documents.iterrows():
    terms = []
    pointer = index.getDocumentIndex().getDocumentEntry(int(document.docno))
    if pointer is None:
      docid = index.getMetaIndex().getDocument("docno",str(document.docno))
      pointer = index.getDocumentIndex().getDocumentEntry(docid)
    for p in index.getDirectIndex().getPostings(pointer):
      termid = p.getId()
      terms.append(index.getLexicon()[termid].getKey())
    documents.at[idx,'doctext_parsed'] = " ".join(terms)
  return documents


def generateVocabulary(documents,index):
  '''
  - building vocabulary 
  - parameters:
    - 1. documents: dataframe of documents
    - 2. index: pyTerrier index to retrieve the terms from posting
  '''
  queryCorpus = {}
  for name,group in documents.groupby("qid",as_index=False):
    vocabulary = []
    for idx,doc in group.iterrows():
      pointer = index.getDocumentIndex().getDocumentEntry(int(doc.docno))
      if pointer is None:
        docid = index.getMetaIndex().getDocument("docno",str(doc.docno))
        pointer = index.getDocumentIndex().getDocumentEntry(docid)
      for p in index.getDirectIndex().getPostings(pointer):
        termid = p.getId()
        vocabulary.append(index.getLexicon()[termid].getKey())
    ## added to preserve the order of terms generated from documents
    queryCorpus[str(name)] = sorted(set(vocabulary), key=vocabulary.index)
  
  return queryCorpus

def generateLimitedVocabulary(documents,index):
  '''
  - building vocabulary and restrict it with textrank
  - parameters:
    - 1. documents: dataframe of documents
    - 2. index: pyTerrier index to retrieve the terms from posting
  '''
  nltk.download('punkt')
  queryCorpus = {}
  stemmer = PorterStemmer()
  r = Rake()
  for name,group in documents.groupby("qid",as_index=False):
    vocabulary = []
    for idx,doc in group.iterrows():
      r.extract_keywords_from_text(doc.text)
      rankedList = r.get_ranked_phrases_with_scores()
      imp_words = []
      for tokens in rankedList:
        words = tokens[1].translate(str.maketrans('', '', string.punctuation))
        for token in words.split(" "):
          if token not in imp_words and tokens[0]>3.0:
            imp_words.append(stemmer.stem(token))
      # print(imp_words)
      vocabulary.extend(imp_words)
      # print(vocabulary)
    ## added to preserve the order of terms generated from documents
    queryCorpus[str(name)] = sorted(set(vocabulary), key=vocabulary.index)
  
  return queryCorpus

def generateRLMVocabulary(rlmscores,noOfTerms):
  '''
  - building vocabulary and restrict the number terms in vocabulary with RLM score
  - parameters:
    - 1. rlmscores: dataframe for every qid with respective terms and their scores
    - 2. noOfTerms: number of terms to be included in the vocabulary
  '''
  queryCorpus = {}
  for qid in rlmscores.qid.unique():
    subset = rlmscores[rlmscores.qid==qid]
    termRLMScores = {term.split("^")[0]:term.split("^")[1] for term in subset['query'].values[0].split(" ")[1:]}
    termRLMScores = list(dict(sorted(termRLMScores.items(),key=lambda val:val[1],reverse=True)).keys())
    queryCorpus[str(qid)] = termRLMScores[0:noOfTerms]

  return queryCorpus

def generateWord2vecVocabulary(topicQueries,maxnumstates,documents,index):
  '''
  - building vocabulary and restrict the number terms in vocabulary with word2vec score
  - parameters:
    - 1. documents: dataframe of documents
    - 2. index: pyTerrier index to retrieve the terms from posting
  '''
  wv = gensim.models.KeyedVectors.load_word2vec_format('data/Google Word2Vec Model/GoogleNews-vectors-negative300.bin',binary=True)

  queryCorpus = generateVocabulary(documents,index)
  queryw2vvocabulary = {}
  wordnotinvocab=[]
  gensimvocab = wv.index_to_key
  for row in tqdm(topicQueries.to_dict(orient="records")):
    qid, query = row['qid'], row['query']
    #convert dict to vector
    stemmedword2vec={}
    for word in queryCorpus[qid]:
      similarwords={}
      if word.lower() in gensimvocab:
        for msword,score in wv.most_similar(word,topn=5):
          if PorterStemmer().stem(msword)==word and msword.lower() not in similarwords:
            similarwords[msword.lower()]=wv[msword]
        if len(similarwords)>0:
          stemmedword2vec[word] = sum(similarwords.values())/len(similarwords)
        else:
          stemmedword2vec[word] = wv[word]
      else:
        wordnotinvocab.append(word)

    #get query terms -> convert them to vector -> get cosine similarity -> sort -> get top 10 for each word
    queryw2vvocabulary[qid] = []
    vocabulary = {}
    for qword in query.split(" "):
      if qword in gensimvocab:
        qwordvec = wv[qword]
        similarwords = list(zip(stemmedword2vec.keys(),wv.cosine_similarities(qwordvec,list(stemmedword2vec.values()))))
        for word,score in similarwords: #[:10]
          if word in vocabulary:
            if score>vocabulary[word]:
              vocabulary[word] = score
          else:
            vocabulary[word] = score

    vocabulary = sorted(vocabulary.items(),key=lambda x:x[1],reverse=True)
    queryw2vvocabulary[qid] = [word for word,score in vocabulary[:maxnumstates]]
    queryw2vvocabulary[qid].extend(wordnotinvocab)

  # print(queryw2vvocabulary)
  return queryw2vvocabulary


# function to calculate similarity measure
def similarity(bm25,colbert,simmilaritymatrix="jaccard"):
  '''
  - calculate similarity measure
  - parameters:
    - 1. bm25: dataframe of bm25 scores
    - 2. colbert: dataframe of colbert scores
    - 3. simmilaritymatrix: similarity matrix to be used
  '''
  score = None
  if simmilaritymatrix.lower() == "jaccard":
    qid=bm25.qid.unique()[0]
    bm25querysubset = set(bm25[bm25.qid==qid]['docno'])
    colbertquerysubset = set(colbert[colbert.qid==qid]['docno'])
    score = float(len(bm25querysubset.intersection(colbertquerysubset))) / len(bm25querysubset.union(colbertquerysubset))
  elif simmilaritymatrix.lower() == "kendall":
    qid=bm25.qid.unique()[0]
    bm25 = bm25[bm25.qid==qid]
    colbert = colbert[colbert.qid==qid]
    merged = colbert.merge(bm25,left_on='docno',right_on="docno")
    # print(merged.rank_x, merged.rank_y)
    score, _ = kendalltau(merged.rank_x, merged.rank_y)
    score = 0.0 if math.isnan(score) else score
  elif simmilaritymatrix.lower()=="rbo":
    qid=bm25.qid.unique()[0]
    bm25_list = list(bm25[bm25.qid==qid].docno.values)
    bm25 = sorted(set(bm25[bm25.qid==qid].docno.values),key=bm25_list.index)
    # print("bm25: ",bm25)
    colbert = colbert[colbert.qid==qid].docno.values
    # print("colbert: ",colbert)
    score = RankingSimilarity(bm25, colbert).rbo(p=0.9)
  else:
    print(f"{simmilaritymatrix} similarity yet to be implemented.")
  return score



## feature selection using best first algorithm
def bestfirstsearch(args,vocabulary,qid,originalQuery,basemodel,comparisonDocSet,simmilaritymatrix):
  '''
    -This function is used to implement best first search algorithm
    -Parameters:
      - 1. args: arguments from command line
      - 2. vocabulary: list of all the words in the top50 documents order by RLM score
      - 3. qid: qid of the query
      - 4. originalQuery: original query
      - 5. basemodel: basemodel used to retrieve the documents
      - 6. comparisonDocSet: res file of the deep learning model
      - 7. simmilaritymatrix: similarity matrix to be used for similarity measure
  '''

  addtermsonly=args.addtermsonly
  maxnumterms=args.maxnumterms
  maxnumstates=args.maxnumstates
  stemmer = PorterStemmer()
  priorityQueue = PriorityQueue()
  bestperformingnode=None
  originalQueryTerms = sorted(set(map(stemmer.stem,originalQuery.split(" ")))) #sort aplhabetically
  originalTermCount=len(originalQueryTerms)
  if addtermsonly:
    cleanedquery = " ".join(originalQueryTerms)
    # if priorityQueue.already_explored(cleanedquery):
    #   basemodelscore = priorityQueue.get_state_score(cleanedquery)
    # else:
    retrieved_docs = basemodel.search(cleanedquery)
    retrieved_docs.qid=qid
    basemodelscore = float(similarity(retrieved_docs,comparisonDocSet,simmilaritymatrix=simmilaritymatrix))
    bestperformingnode = (cleanedquery,basemodelscore)
    priorityQueue.add_state(cleanedquery,basemodelscore)
  else:
    originalTermCount = 1
    for term in originalQueryTerms:
      # if priorityQueue.already_explored(term):
      #   basemodelscore = priorityQueue.get_state_score(term)
      # else:
        retrieved_docs = basemodel.search(term)
        retrieved_docs.qid=qid
        if not retrieved_docs.empty:
          basemodelscore = float(similarity(retrieved_docs,comparisonDocSet,simmilaritymatrix=simmilaritymatrix))
        # else:
        #   basemodelscore = None
      # if basemodelscore is not None:
          bestperformingnode = (term,basemodelscore)
          priorityQueue.add_state(term,basemodelscore)
          if term in vocabulary:
            vocabulary.remove(term)
            vocabulary = [term] + vocabulary
          else:
            vocabulary = [term] + vocabulary

  # print("Vocabulary: ",vocabulary)
  # print("Base model simmilarity score:",bestperformingnode)

  while len(priorityQueue)>0:
    # print("Length priorityQueue:",len(priorityQueue))
    currentState = priorityQueue.pop_state()
    # print("currentState: ",currentState)

    if currentState[1] > bestperformingnode[1]:
      bestperformingnode=currentState
    
    if bestperformingnode[1] == 1.0:
      break

    # print("Best performing node: ",bestperformingnode)

    for token in vocabulary:
      newquery = currentState[0] + " " +token
      expandedquery = " ".join(sorted(newquery.split(" "))) #sort aplhabetically
      #if state not already explored and token is not in previous query
      if priorityQueue.check_state_exists(expandedquery) and token not in currentState[0]:
        # if priorityQueue.already_explored(expandedquery):
        #   score = priorityQueue.get_state_score(expandedquery)
        # else:
          # print(expandedquery)
          retrieved_docs = basemodel.search(expandedquery)
          # print(retrieved_docs.info())
          retrieved_docs.qid=qid
          score = float(similarity(retrieved_docs,comparisonDocSet,simmilaritymatrix=simmilaritymatrix))
          # print(expandedquery,score)
          newaddedtermcount=len(expandedquery.split(" ")) - originalTermCount
          if newaddedtermcount <= maxnumterms and score != currentState[1]:
            priorityQueue.add_state(expandedquery,score)
      
    if len(priorityQueue) >= maxnumstates:
      break
    
  if len(priorityQueue) >= maxnumstates:
    state,score = priorityQueue.pop_state()
    if bestperformingnode[1] < score:
      bestperformingnode = (state,score)
  
  # priorityQueue.save_existing_states()
  # print(priorityQueue.cur_entry_finder)
  return bestperformingnode

def calTermWeights(queryVocabularies,basemodel,dllm_docs,queries,similarity_type,addtermsonly=False):
  '''
  -This function is used to calculate the term weights for each terms in vocabulary
  -Parameters:
    - 1. queryVocabularies: list of all the words in the top50 documents order by RLM score
    - 2. basemodel: basemodel used to retrieve the documents
    - 3. dllm_docs: res file of the deep learning model
    - 4. queries: list of all the queries
    - 5. similarity_type: similarity measure to be used
    - 6. addtermsonly: boolean value to indicate whether to just add terms to query or to remove terms from query as well
  '''
  # changing dtype of colbert_docs docno column for comparison
  stemmer = PorterStemmer()
  dllm_docs = dllm_docs.astype({'docno':'str'})
  jaccsimilarity = []
  for row in tqdm(queries.to_dict(orient="records")):
      qid,query = row['qid'],row['query']
      queryVocab = queryVocabularies[str(qid)]

      if addtermsonly:
        #get basemodel score
        cleanquery = " ".join(list(map(stemmer.stem,query.split(" "))))
        retrieved_doc = basemodel.search(cleanquery)
        retrieved_doc.qid = qid
        basesimscore = similarity(retrieved_doc,dllm_docs,simmilaritymatrix=similarity_type)
        ##retrieve expaneded query score
        words = [word for word in queryVocab if not word in cleanquery.split(" ")]
        expanded_query = [(count,cleanquery+" "+word) for count,word in enumerate(words,start=1)]
        retrieved_doc = basemodel.transform(pd.DataFrame(expanded_query,columns=["qid","query"]))
        retrieved_doc.qid=qid
        # print("\n",retrieved_doc.info())
        for idx,group in retrieved_doc.groupby("query",as_index=False):
          word = group['query'].unique()[0].split(" ")[-1]
          jscore = similarity(group,dllm_docs,simmilaritymatrix=similarity_type)
          diff = checkIncrement(basesimscore,jscore)
          jaccsimilarity.append((qid,query,group['query'].unique()[0],word,jscore,diff))
      else:
        ##Improvement###
        querywords = list(map(stemmer.stem,query.split(" ")))
        #remove duplicate terms from original query to reduce loop count
        querywords = sorted(set(querywords),key=querywords.index)
        for queryword in querywords:
            #get basemodel score
            retrieved_doc = basemodel.search(queryword)
            retrieved_doc.qid = qid
            if not retrieved_doc.empty:
              #check similarity between dllm and basemodel for each original query term
              basesimscore = similarity(retrieved_doc,dllm_docs,simmilaritymatrix=similarity_type)
              # fetch similarity score adding vocabulary term to cal incremental score
              vocaboqwords=list(filter(lambda qw: qw != queryword ,querywords))
              terms = [term for term in queryVocab if term not in querywords]
              terms.extend(vocaboqwords)
              expanded_query = [(count,queryword+" "+word) for count,word in enumerate(terms,start=1)]
              expanded_query_df = pd.DataFrame(expanded_query,columns=["qid","query"])
              retrieved_doc = basemodel.transform(expanded_query_df)
              retrieved_doc.qid=qid
              # print("\n",retrieved_doc.info())
              for _,group in retrieved_doc.groupby("query",as_index=False):
                word = group['query'].unique()[0].split(" ")[-1]
                jscore = similarity(group,dllm_docs,simmilaritymatrix=similarity_type)
                diff = checkIncrement(basesimscore,jscore)
                jaccsimilarity.append((qid,query,group['query'].unique()[0],word,jscore,diff))

  return jaccsimilarity

#function to calculate improvement
def checkIncrement(baseScore,currentScore):
  '''
  -This function is used to calculate the incremental improvement of the expanded query with original query
  -Parameters:
    - 1. baseScore: score of the original query
    - 2. currentScore: score of the expanded query
  '''
  #check relative increment in the jaccard similarity score for each term and  handle division by zero
  if currentScore == 0.0 and baseScore == 0.0:
    return 0.0
  return float(((currentScore-baseScore)+0.1)/(baseScore+0.1))

## fetch features based on jaccard similarity
def fetchFeature(group,addtermsonly=True,rowcount=3):
  '''
  -This function is used to fetch the features based on jaccard similarity score
  -Parameters:
    - 1. group: group of documents from where to fetch the selected documents
    - 2. addtermsonly: boolean value to indicate whether to split the query term or not
    - 3. rowcount: number of rows to be fetched from the group
  '''
  stemmer = PorterStemmer()
  if addtermsonly:
    group = group.sort_values("improvement",ascending=False).head(rowcount)
    query = group["original_query"].unique()[0]
    cleanquery = " ".join(list(map(stemmer.stem,query.split(" "))))
    group["expanded_terms"] = group['word'].str.cat(sep=' ')
    group["final_query"] = cleanquery + " " + group['expanded_terms']
  else:
    group = group.sort_values("improvement",ascending=False).head(50)
    expanded_query =  group['expanded_query'].str.cat(sep=" ").split(" ")
    group["expanded_terms"] = " ".join(sorted(set(expanded_query), key=expanded_query.index)[0:rowcount])
    group["final_query"] = group['expanded_terms']
  return group