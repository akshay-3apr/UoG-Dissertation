from argparse import ArgumentParser
from corpus import Dataset
from tqdm import tqdm
import pyterrier as pt
import pandas as pd
import random
from distutils.util import strtobool
from nltk.stem.porter import PorterStemmer
from helper import filterTopKRankRecord,calTermWeights,generateVocabulary,similarity,checkIncrement,fetchFeature

def fetchtermweights(args,dllm_res,dataset,topicQueries):

    ## get topK of ColBERT reranked output
    dllm = dllm_res.groupby("qid",as_index=False).head(10)

    ##filter out test_topics
    # topicQueries = test_topics[test_topics.qid.isin(colbert_res.qid)]

    ## get baseline model
    wmodel=args.wmodel
    basemodel = pt.BatchRetrieve(dataset.getIndex(),num_results=10,wmodel=wmodel,properties={"termpipelines" : "Stopwords"})
    retrieved_docs = basemodel.transform(topicQueries)
    vocabdocs = retrieved_docs.groupby("qid",as_index=False).apply(lambda subgroup:filterTopKRankRecord(subgroup,dataset.getIndex(),topK=10))
    #set flag to split query terms
    addtermsonly= args.addtermsonly

    #initiate the process
    queryVocabulary = generateVocabulary(vocabdocs,index=dataset.getIndex())
    result = calTermWeights(queryVocabulary,basemodel,dllm,topicQueries,similarity_type='jaccard',addtermsonly=addtermsonly)
    termWeights = pd.DataFrame(result, columns =['qid','original_query','expanded_query','word','jscore','improvement'])
    termWeights.to_csv(f"data/ColBERT_{wmodel}_termweights_trec_dl_top_10_addtermsonly_{addtermsonly}.csv",\
        header=['qid','original_query','expanded_query','word','jscore','improvement'],index=False)

    return termWeights

def greedysearch(args,pt):
    '''
    - Initiate the greedy search algorithm
    - Parameters:
        - args: command line arguments
        - pt: pyterrier object
    '''
    random.seed(9678)

    #top k documents to retrieve
    topK=args.topK

    ## get termWeights
    termWeightModel=args.wmodel

    ##Number of terms to consider
    noOfTerms = args.maxnumterms

    #split query term
    addtermsonly= args.addtermsonly

    # get topicQueries and filter out test_topics
    assert args.topics is not None, "Please provide the topic tsv file without header"
    topicQueries = args.topics
    topicQueries = pd.read_csv(topicQueries,sep="\t",names=["qid", "query"], index_col=False, dtype=str)

    assert args.index_path is not None, "PyTerrier Index path is not specified"
    dataset = Dataset(args.index_path)

    #get deep learning reranked output csv file
    dltopK = args.dltop1000

    dllm_db = pd.read_csv(dltopK,sep="\t",header=None,names=["qid","number","docno","rank","score","msg"],index_col=False)
    dllm_db.sort_values(["qid","rank"],inplace=True)
    dllm_db = dllm_db.astype({'qid':'str'})
    # colbert_db.info()

    dllm_res = dllm_db[dllm_db.qid.isin(topicQueries.qid.unique())]

    if args.relevant:
        dllm = dllm_res.groupby('qid', as_index=False).apply(lambda g: g[g['rank'] <= topK])
    else:
        dllm = dllm_res.groupby('qid', as_index=False).head(topK)
    dllm = dllm.astype({'docno':'str'})

    #cal term weights
    if not args.optimise:
        assert args.termweights is not None, "Please provide precomputed term weights to calculate similarity score"
        termWeights = pd.read_csv(args.termweights,header=0,names=["qid","original_query","expanded_query","word","jscore","improvement"])
    else:
        print("Calculating term weights to optimise")
        termWeights = fetchtermweights(args,dllm_res,dataset,topicQueries)

    termWeights = termWeights[~termWeights.word.isin(termWeights.original_query.unique()[0].split(' '))]
    queries = termWeights.groupby('qid',as_index=False).apply(lambda group:fetchFeature(group,addtermsonly,rowcount=noOfTerms)).reset_index()
    queries.drop(['level_0','level_1'],axis=1,inplace=True)
    queries = queries.loc[:,['qid','original_query','expanded_terms','final_query']].groupby('qid',as_index=False).first()
    queries.qid=queries.qid.astype('str')

    ## similarity matrix
    simmilaritymatrix=args.evalmatrix

    ## get baseline performance
    basemodel = pt.BatchRetrieve(dataset.getIndex(),num_results=topK,wmodel=termWeightModel,properties={"termpipelines" : "Stopwords"})
    jaccsimilarity = []

    ## to clean original query
    stemmer = PorterStemmer()
    
    print("Calculating similarity score for each optimal query")
    for idx,data in tqdm(queries.iterrows()):
        cleanquery = " ".join(list(map(stemmer.stem,data.final_query.split(" "))))
        retrieved_doc = basemodel.search(cleanquery)
        retrieved_doc.qid = data.qid
        base = similarity(retrieved_doc,dllm,simmilaritymatrix=simmilaritymatrix)
        retrieved_doc = basemodel.search(data.final_query)
        retrieved_doc.qid = data.qid
        score = similarity(retrieved_doc,dllm,simmilaritymatrix=simmilaritymatrix)
        diff = checkIncrement(base,score)
        jaccsimilarity.append((data.qid,data.original_query,data.final_query,data.expanded_terms,score,diff))

    print(*jaccsimilarity,sep="\n")
    df = pd.DataFrame(jaccsimilarity, columns =['qid','original_query','final_query','word','score','improvement'])
    if args.optimise:
        df.to_csv(f"data/{termWeightModel}_greedySearch_top{topK}_{noOfTerms}_addtermsonly_{addtermsonly}_{simmilaritymatrix}.csv",\
        header=['qid','original_query','final_query','word','score','improvement'],index=False)
    print(f"\n{termWeightModel} average {simmilaritymatrix} considering {noOfTerms} expanded query terms: ",round(df.score.mean(),4),"\n")


if __name__=="__main__":
    parser = ArgumentParser(description='Process cmd arguments for Greedy Search')
    parser.add_argument('--topK', dest='topK', default=10,type=int, help='topK of dl model output to compare with LM model')
    parser.add_argument('--index_path', dest='index_path',default=None, help='path to trec 2019 index')
    parser.add_argument('--dltop1000', dest='dltop1000', default=None, help='path to dltop1000 of DL model csv file')
    parser.add_argument('--topics', dest='topics', default=None, help='path to topics csv file')
    parser.add_argument('--qrels', dest='qrels', default=None, help='path to qrels csv file')
    parser.add_argument('--termweights', dest='termweights', default=None, help='path to termweights csv file')
    parser.add_argument('--wmodel', dest='wmodel', default="BM25", help='weight model to be used to fetch records')
    parser.add_argument('--addtermsonly', dest='addtermsonly', type=lambda x:bool(strtobool(x)) , default=True, help='boolean value to add terms to the query and remove when false')
    parser.add_argument('--maxnumterms', dest='maxnumterms', default=3,type=int, help='maximum number of terms to consider for optimal query')
    parser.add_argument('--evalmatrix', dest='evalmatrix', default="jaccard", help='evaluation matrix to get the scores')
    parser.add_argument('--optimise', dest='optimise', type=lambda x: bool(strtobool(x)),default=False, help='Please set if you want to optimise on the res file')
    parser.add_argument('--relevant', dest='relevant', type=lambda x: bool(strtobool(x)),default=False, help='Please set if you want to fetech result for relevant set ')
    args=parser.parse_args()
    # print(args)
    # checks if pyterrier init method is called
    if not pt.started():
        pt.init(boot_packages=["com.github.terrierteam:terrier-prf:-SNAPSHOT"],logging='ERROR')

    greedysearch(args,pt)