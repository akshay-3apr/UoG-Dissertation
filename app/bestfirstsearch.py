from argparse import ArgumentParser
from corpus import Dataset
import random
from tqdm import tqdm
import pyterrier as pt
import pandas as pd
from distutils.util import strtobool
from nltk.stem.porter import PorterStemmer
from helper import filterTopKRankRecord, generateVocabulary, similarity, generateRLMVocabulary, generateLimitedVocabulary,bestfirstsearch,generateWord2vecVocabulary
import json


def buildVocabulary(args, dataset, dllm, topicQueries):
    '''
    - Build vocabulary for each topic query
    - Parameters:
        - args: command line arguments
        - dataset: pyterrier dataset
        - dllm: deep learning re-ranked output
        - topicQueries: topic queries
    '''
    # topK = 50 # args.topK
    # wmodel = args.wmodel
    # basemodel = pt.BatchRetrieve(dataset.getIndex(), num_results=topK, wmodel=wmodel)

    queryVocabulary = None
    if args.termselection.lower() == "rm3":
        assert args.rm3vocab is not None, "Please provide the rm3 vocabulary tsv file"
        # qid,query_0,query
        rlmvocab = pd.read_csv(args.rm3vocab, header=None, names=["qid", "query_0", "query"], index_col=False, dtype=str)
        queryVocabulary = generateRLMVocabulary(rlmvocab,args.maxbranching)
        # print(queryVocabulary)
    elif args.termselection.lower() == "textrank":
        # initiate restricting vocabulary with TextRank
        queryVocabulary = generateLimitedVocabulary(dllm, index=dataset.getIndex())
    elif args.termselection.lower() == "word2vec":
        # initiate restricting vocabulary with TextRank
        if args.word2vecvocab is not None:
            print("Building vocabulary from file...")
            with open(args.word2vecvocab) as f:
                queryVocabulary = json.load(f)
        else:
            queryVocabulary = generateWord2vecVocabulary(topicQueries,args.maxbranching,dllm,index=dataset.getIndex())
            with open("data/word2vecvocabulary.txt", 'w') as f:
                f.write(json.dumps(queryVocabulary))
    else:
        queryVocabulary = generateVocabulary(dllm, index=dataset.getIndex())

    return queryVocabulary


def main(args):
    '''
        - Main function to initiate the best first search algorithm
    '''
    # top k documents to retrieve
    topK = args.topK

    # get weight model
    weightModel = args.wmodel

    # add and removal of terms to expanded query
    addtermsonly = args.addtermsonly
    
    # maximum number of vocabulary terms to consider
    maxbranching = args.maxbranching

    # get topicQueries and filter out test_topics
    assert args.topics is not None, "Please provide the topic tsv file without header"
    topicQueries = args.topics
    topicQueries = pd.read_csv(topicQueries,sep="\t",names=["qid", "query"], index_col=False, dtype=str)

    assert args.index_path is not None, "PyTerrier Index path is not specified"
    dataset = Dataset(args.index_path)

    # get topK of deep learning result -> reranked output res file
    dltopK = args.dltop1000
    dllm_db = pd.read_csv(dltopK, sep="\t", header=None, names=["qid", "number", "docno", "rank", "score", "msg"], index_col=False)
    dllm_db.sort_values(["qid", "rank"], inplace=True)
    dllm_db = dllm_db.astype({'qid':'str'})

    dllm_res = dllm_db[dllm_db.qid.isin(topicQueries.qid.unique())]
    dllm = dllm_res.groupby('qid', as_index=False).head(topK)
    dllm = dllm.astype({'docno':'str'})

    basemodel = pt.BatchRetrieve(dataset.getIndex(), num_results=topK, wmodel=weightModel, properties={"termpipelines" : "Stopwords"})

    similaritymatrix = args.evalmatrix
    results = []

    if similaritymatrix.lower()=='rbo':
        assert args.optimalquery is not None, "Please provide the optimal query file"
        optimalBFS_df = pd.read_csv(args.optimalquery, names=["qid","expandedquery","evaluationmetric","score","originalquery"],header=0, index_col=False, dtype=str)
        # print(optimalBFS_df.head(2))

        for row in tqdm(optimalBFS_df.to_dict(orient="records")):
            qid,query = row['qid'],row['expandedquery']
            if qid != "qid":
                retrieved_docs = basemodel.search(query)
                retrieved_docs.qid=qid
                simscore = float(similarity(retrieved_docs,dllm,simmilaritymatrix="rbo"))
                results.append((qid,query,similaritymatrix,simscore))
    else:
        
        # build vocabulary
        queryVocabulary = buildVocabulary(args, dataset, dllm, topicQueries)
        
        print("Running best first search")
        for row in tqdm(topicQueries.to_dict(orient="records")):
            qid, query = row['qid'], row['query']
            if qid != "qid":
                bpn = bestfirstsearch(args,queryVocabulary[qid], qid, query, basemodel, dllm, similaritymatrix)
                results.append((qid, bpn[0], similaritymatrix, bpn[1]))

    print(*results,sep='\n')
    df = pd.DataFrame(results, columns=["qid","expandedquery","evaluationmetric","score"])
    df = df.merge(topicQueries,left_on="qid",right_on="qid")
    print(f"Average {similaritymatrix} score of {weightModel} using BFS: ", df.score.mean())
    df.to_csv(f"data/{weightModel}_BESTFIRST_{similaritymatrix}_trec_dl_top{topK}_terms_{maxbranching}_md{args.maxnumterms}_ms{args.maxnumstates}_addtermsonly_{addtermsonly}.csv",header=["qid","expandedquery","evaluationmetric","score","originalquery"], index=False)

if __name__ == "__main__":
    parser = ArgumentParser(description='Process cmd arguments for Greedy Search')
    parser.add_argument('--index_path', dest='index_path',default=None, help='path to trec 2019 index')
    parser.add_argument('--dltop1000', dest='dltop1000', default=None,help='path to dltop1000 of DL model csv file')
    parser.add_argument('--topics', dest='topics',default=None, help='path to topics csv file')
    parser.add_argument('--qrels', dest='qrels',default=None, help='path to qrels csv file')
    parser.add_argument('--rm3vocab', dest='rm3vocab',default=None, help='path to rm3 vocabulary file')
    parser.add_argument('--wmodel', dest='wmodel', default="BM25",help='weight model to be used to fetch records')
    parser.add_argument('--addtermsonly', dest='addtermsonly', type=lambda x: bool(strtobool(x)),default=True, help='boolean value to add terms to the query and remove when false')
    parser.add_argument('--evalmatrix', dest='evalmatrix',default="jaccard", help='evaluation matrix to get the scores')
    parser.add_argument('--termselection', dest='termselection',default="RM3", help='way to select terms for optimal query')
    parser.add_argument('--topK', dest='topK', default=10, type=int,help='topK of dl model output to compare with LM model')
    parser.add_argument('--maxbranching', dest='maxbranching',default=30, type=int, help='maximum number of terms to consider in vocabulary')
    parser.add_argument('--maxnumstates', dest='maxnumstates',default=50, type=int, help='maximum number of states to be considered')
    parser.add_argument('--maxnumterms', dest='maxnumterms', default=5, type=int, help='maximum number of terms to consider for optimal query')
    parser.add_argument('--optimalquery', dest='optimalquery', default=None, help='Please provide the optimal query file for best first search')
    parser.add_argument('--word2vecvocab', dest='word2vecvocab', default=None, help='Please provide the path to word2vec vocabulary file')
    args = parser.parse_args()
    # checks if pyterrier init method is called
    try:
        if not pt.started():
            pt.init(boot_packages=["com.github.terrierteam:terrier-prf:-SNAPSHOT"], logging='ERROR')
    except Exception as e:
        # print(e)
        print("Not able to load pyTerrier. Try running the python file again")
        exit(1)
    
    # code to fetch terms from corpus and check the jaccard simiarity between the LMs and ColBERT
    main(args)
