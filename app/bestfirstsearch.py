from argparse import ArgumentParser
from corpus import Dataset
import random
from tqdm import tqdm
import pyterrier as pt
import pandas as pd
from distutils.util import strtobool
from nltk.stem.porter import PorterStemmer
from helper import filterTopKRankRecord, generateVocabulary, similarity, generateRLMVocabulary, generateLimitedVocabulary,bestfirstsearch


def buildVocabulary(args, dataset, dllm, topicQueries):
    '''
    - Build vocabulary for each topic query
    - Parameters:
        - args: command line arguments
        - dataset: pyterrier dataset
        - dllm: deep learning re-ranked output
        - topicQueries: topic queries
    '''
    topK = 50
    wmodel = args.wmodel

    basemodel = pt.BatchRetrieve(dataset.getIndex(), num_results=topK, wmodel=wmodel)

    queryVocabulary = None
    if args.termselection.lower() == "rm3":
        rm3_pipe = basemodel >> pt.rewrite.RM3(dataset.getIndex(), fb_terms=args.maxbranching, fb_docs=topK)
        rlmscore = rm3_pipe.transform(topicQueries)
        rlmscore.to_csv("data/top50vocabularyRLMScore.csv",index=False,header=["qid","query_0","query"])
        queryVocabulary = generateRLMVocabulary(rlmscore, dllm, index=dataset.getIndex())
    elif args.termselection.lower() == "textrank":
        # initiate restricting vocabulary with TextRank
        queryVocabulary = generateLimitedVocabulary(dllm, index=dataset.getIndex())
    else:
        queryVocabulary = generateVocabulary(dllm, index=dataset.getIndex())

    return queryVocabulary


def main(args):
    '''
        - Main function to initiate the best first search algorithm
    '''
    # top k documents to retrieve
    topK = args.topK

    # get termWeights
    weightModel = args.wmodel

    # Number of terms to consider
    noOfTerms = args.maxnumterms

    # add and removal of terms to expanded query
    addtermsonly = args.addtermsonly
    
    # maximum number of vocabulary terms to consider
    maxbranching = args.maxbranching

    # get deep learning reranked output csv file
    dltopK = args.dltop1000

    dllm_db = pd.read_csv(dltopK, sep=" ", header=None, names=["qid", "number", "docno", "rank", "score", "msg"], index_col=False)
    dllm_db.sort_values(["qid", "rank"], inplace=True)
    dllm_db.qid = dllm_db.qid.astype('str')
    # colbert_db.info()

    dataset = Dataset()
    print("Indexing Deep Learning Language model reranked documents")
    collection = dataset.setCollection(args.collection).getCollection()
    dllm_res = dllm_db.merge(collection, left_on="docno", right_on="docno")
    indexDataset = dllm_res.loc[:, ['text', 'docno']]
    indexDataset.docno = indexDataset.docno.astype('str')
    dataset.buildCustomIndex(indexDataset)

    # get topK of deep learning result -> reranked output res file
    dllm = dllm_res.groupby("qid", as_index=False).apply(lambda subgroup: filterTopKRankRecord(subgroup, dataset.getIndex(), topK=topK))
    dllm.docno = dllm.docno.astype('str')

    basemodel = pt.BatchRetrieve(dataset.getIndex(), num_results=topK, wmodel=weightModel, properties={"termpipelines" : "Stopwords"})

    similaritymatrix = args.evalmatrix
    results = []
    if similaritymatrix.lower()=='rbo':
        assert args.optimalquery is not None, "Please provide the optimal query file"
        optimalBFS_df = pd.read_csv(args.optimalquery, names=["Qid","Expanded Query","Evaluation Metric","Score","Original Query"], index_col=False, dtype=str)

        for row in tqdm(optimalBFS_df.to_dict(orient="records")):
            qid,query = str(row['Qid']),row['Expanded Query']
            retrieved_docs = basemodel.search(query)
            retrieved_docs.qid=qid
            simscore = float(similarity(retrieved_docs,dllm,simmilaritymatrix="rbo"))
            results.append((qid,query,similaritymatrix,simscore))

    else:

        # get topicQueries and filter out test_topics
        assert args.topics is not None, "Please provide the topic file"
        topicQueries = args.topics
        topicQueries = pd.read_csv(topicQueries, names=["qid", "query"], index_col=False, dtype=str)
        
        # build vocabulary
        queryVocabulary = buildVocabulary(args, dataset, dllm, topicQueries)
        
        print("Running best first search")
        for row in tqdm(topicQueries.to_dict(orient="records")):
            qid, query = row['qid'], row['query']
            bpn = bestfirstsearch(args,queryVocabulary[qid], qid, query, basemodel, dllm, similaritymatrix)
            # expandedquery = " ".join([node.token for node in bpn.getParent()[::-1]])
            # results.append((qid, expandedquery, similaritymatrix, bpn.score))
            results.append((qid, bpn[0], similaritymatrix, bpn[1]))

    print(*results,sep='\n')
    df = pd.DataFrame(results, columns=["Qid", "Expanded Query", "Evaluation Metric", "Score"])
    df = df.merge(topicQueries,left_on="Qid",right_on="qid").drop("qid",axis=1)
    print(f"Average {similaritymatrix} score of {weightModel} using BFS: ", df.Score.mean())
    df.to_csv(f"data/{weightModel}_BESTFIRST_{similaritymatrix}_trec_dl_top{topK}_terms_{maxbranching}_md{args.maxnumterms}_ms{args.maxnumstates}_addtermsonly_{addtermsonly}.csv",header=["Qid", "Expanded Query", "Evaluation Metric", "Score", "Original Query"], index=False)

if __name__ == "__main__":
    parser = ArgumentParser(description='Process cmd arguments for Greedy Search')
    parser.add_argument('--collection', dest='collection',default=None, help='path to trec collection file to use')
    parser.add_argument('--dltop1000', dest='dltop1000', default=None,help='path to dltop1000 of DL model csv file')
    parser.add_argument('--topics', dest='topics',default=None, help='path to topics csv file')
    parser.add_argument('--qrels', dest='qrels',default=None, help='path to qrels csv file')
    parser.add_argument('--termweights', dest='termweights',default=None, help='path to termweights csv file')
    parser.add_argument('--topK', dest='topK', default=10, type=int,help='topK of dl model output to compare with LM model')
    parser.add_argument('--wmodel', dest='wmodel', default="BM25",help='weight model to be used to fetch records')
    parser.add_argument('--addtermsonly', dest='addtermsonly', type=lambda x: bool(strtobool(x)),default=True, help='boolean value to add terms to the query and remove when false')
    parser.add_argument('--evalmatrix', dest='evalmatrix',default="jaccard", help='evaluation matrix to get the scores')
    parser.add_argument('--termselection', dest='termselection',default="RM3", help='way to select terms for optimal query')
    parser.add_argument('--maxbranching', dest='maxbranching',default=30, type=int, help='maximum number of terms to consider in vocabulary')
    parser.add_argument('--maxnumstates', dest='maxnumstates',default=50, type=int, help='maximum number of states to be considered')
    parser.add_argument('--maxnumterms', dest='maxnumterms', default=3, type=int, help='maximum number of terms to consider for optimal query')
    parser.add_argument('--optimalquery', dest='optimalquery', default=None, help='Please provide the optimal query file for best first search')
    args = parser.parse_args()
    # checks if pyterrier init method is called
    if not pt.started():
        pt.init(boot_packages=["com.github.terrierteam:terrier-prf:-SNAPSHOT"], logging='ERROR')
    
    # code to fetch terms from corpus and check the jaccard simiarity between the LMs and ColBERT
    main(args)
