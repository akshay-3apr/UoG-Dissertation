from argparse import ArgumentParser
from corpus import Dataset
import random
from tqdm import tqdm
import pyterrier as pt
import pandas as pd
from distutils.util import strtobool
from nltk.stem.porter import PorterStemmer
from helper import similarity
import json

def caloptimalrlmscore(args,dataset,termcount,docCount):
    # Calculate optimal rlmscore for a given number of terms and documents
     # top k documents to retrieve
    topK = args.topK

    # get weight model
    weightModel = args.wmodel

    # similarity matrix
    similaritymatrix = args.evalmatrix

    # get topicQueries and filter out test_topics
    assert args.topics is not None, "Please provide the topic tsv file without header"
    topicQueries = args.topics
    topicQueries = pd.read_csv(topicQueries,sep="\t",names=["qid", "query"], index_col=False, dtype=str)

    # get topK of deep learning result -> reranked output res file
    dltopK = args.dltop1000
    dllm_db = pd.read_csv(dltopK, sep="\t", header=None, names=["qid", "number", "docno", "rank", "score", "msg"], index_col=False)
    dllm_db.sort_values(["qid", "rank"], inplace=True)
    dllm_db = dllm_db.astype({'qid':'str'})

    dllm_res = dllm_db[dllm_db.qid.isin(topicQueries.qid.unique())]
    if args.relevant:
        dllm = dllm_res.groupby('qid', as_index=False).apply(lambda g: g[g['rank'] <= topK])
    else:
        dllm = dllm_res.groupby('qid', as_index=False).head(topK)
    dllm = dllm.astype({'docno':'str'})
    
    basemodel = pt.BatchRetrieve(dataset.getIndex(),num_results=topK,wmodel=weightModel)
    rm3_pipe = basemodel >> pt.rewrite.RM3(dataset.getIndex(),fb_terms=termcount,fb_docs=docCount) >> basemodel
    retrieved_docs = rm3_pipe.transform(topicQueries)
    
    #cal basemodel scores
    basemodel_performance=[]
    for idx,subgroup in retrieved_docs.groupby("qid",as_index=False):
        basemodel_performance.append((subgroup.qid.unique()[0],subgroup["query"].unique()[0],weightModel,\
            similaritymatrix,float(similarity(subgroup,dllm,simmilaritymatrix=similaritymatrix))))
    
    # df = pd.DataFrame(basemodel_performance,columns=["qid","query","wmodel","similaritymatrix","score"])
    score = sum(list(zip(*basemodel_performance))[-1])/len(list(zip(*basemodel_performance))[-1])
    print(f"Average {similaritymatrix} score of {weightModel} using BFS: ",score)
    return score


if __name__ == "__main__":
    parser = ArgumentParser(description='Process cmd arguments for RM3 calculation')
    parser.add_argument('--index_path', dest='index_path',default=None, help='path to trec 2019 index')
    parser.add_argument('--dltop1000', dest='dltop1000', default=None,help='path to neural model topDocument csv file')
    parser.add_argument('--topics', dest='topics',default=None, help='path to topics csv file')
    parser.add_argument('--qrels', dest='qrels',default=None, help='path to qrels csv file')
    parser.add_argument('--wmodel', dest='wmodel', default="BM25",help='weight model to be used to fetch records')
    parser.add_argument('--evalmatrix', dest='evalmatrix',default="jaccard", help='evaluation matrix to get the scores')
    parser.add_argument('--topK', dest='topK', default=10, type=int,help='topK of dl model output to compare with LM model')
    parser.add_argument('--gridsearchlist', dest='gridsearchlist', default=None, help='Please provide the grid search values for rlm score calculation (",") separated')
    parser.add_argument('--optimise', dest='optimise', type=lambda x: bool(strtobool(x)),default=False, help='Please set if you want to optimise on the res file')
    parser.add_argument('--relevant', dest='relevant', type=lambda x: bool(strtobool(x)),default=False, help='Please set if you want to fetech result for relevant set ')
    args = parser.parse_args()

    # checks if pyterrier init method is called
    try:
        if not pt.started():
            pt.init(boot_packages=["com.github.terrierteam:terrier-prf:-SNAPSHOT"], logging='ERROR')
    except Exception as e:
        print("Not able to load pyTerrier. Try running the python file again")
        print(e)
        exit(1)

    # code to fetch terms from corpus and check the jaccard simiarity between the LMs and ColBERT
    bestscore = 0.0
    bestterms = 0
    bestdoccs = 0
    gridsearchparam = list(map(int,args.gridsearchlist.split(",")))
    assert args.index_path is not None, "PyTerrier Index path is not specified"
    dataset = Dataset(args.index_path)
    if args.optimise:
        for fb_docs in gridsearchparam:
            for fb_terms in gridsearchparam:
                print(fb_terms,fb_docs)
                score = caloptimalrlmscore(args,dataset,fb_terms,fb_docs)
                if score > bestscore:
                    bestscore = score
                    bestterms = fb_terms
                    bestdoccs = fb_docs
    else:
        bestterms = int(input("Enter the maximum number of terms to be used for the RLM score calculation: "))
        bestdoccs = int(input("Enter the maximum number of documents to be used for the RLM score calculation: "))
        bestscore = caloptimalrlmscore(args,dataset,bestterms,bestdoccs)
   
    print("Best score",bestscore,"Best terms",bestterms,"Best doccs",bestdoccs)