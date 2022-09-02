from argparse import ArgumentParser
from corpus import Dataset
from tqdm import tqdm
import pyterrier as pt
import pandas as pd
from pyterrier.measures import MAP,nDCG

def run_experiment(args):
    assert args.index_path is not None, "PyTerrier Index path is not specified"
    dataset = Dataset(args.index_path)

    queries = pd.read_csv(args.topics,sep="\t",names=["qid", "query"], index_col=False, dtype=str)
    test_qrels = pd.read_csv(args.qrels,sep="\t",names=['qid', 'docno', 'label'], index_col=False, dtype=str)
    test_qrels = test_qrels.astype({'label':'int'})
    basemodel = pt.BatchRetrieve(dataset.getIndex(),num_results=10,wmodel="BM25")

    if args.method.lower() == 'rm3':
        bestterms = int(input("Enter the maximum number of terms to be used for the IR Evaluation calculation: "))
        bestdoccs = int(input("Enter the maximum number of documents to be used for the IR Evaluation calculation: "))
        model = basemodel >> pt.rewrite.RM3(dataset.getIndex(),fb_terms=bestterms,fb_docs=bestdoccs)
        expandedquery = model.transform(queries)
        avgwordcount = expandedquery['query'].apply(lambda x:len(x.split(' '))).mean()
        print("Average word Count: ",avgwordcount)
        model = model >> basemodel
    elif args.method.lower()=="greedy" or args.method.lower()=="bfs" :
        print("Average word Count: ",queries['query'].apply(lambda x:len(x.split(' '))).mean())
        model = basemodel
    else:
        res_file_path = str(input("Enter path to res file: "))
        model = pd.read_csv(res_file_path,sep="\t",index_col=False,header=None,dtype=str)
        model = model.loc[:,[0,2,4]]
        model.columns = ['query_id', 'doc_id', 'score']

    eval_df = pt.Experiment(
        retr_systems=[model],
        topics = queries,
        qrels = test_qrels,
        eval_metrics=[MAP@10, nDCG@10],
        round=4,
        dataframe=True,
        names=["bm25"],
        verbose=True
    )
    print(eval_df.to_string())
    eval_df.to_csv("data/eval_results.csv",index=False)

if __name__=='__main__':
    parser = ArgumentParser(description='Process to calculate MAP@10 and nDCG@10')
    parser.add_argument('--index_path', dest='index_path',default=None, help='path to trec 2019 index')
    parser.add_argument('--topics', dest='topics',default=None, help='path to topics csv file')
    parser.add_argument('--qrels', dest='qrels',default=None, help='path to qrels csv file')
    parser.add_argument('--wmodel', dest='wmodel', default="BM25",help='weight model to be used to fetch records')
    parser.add_argument('--method', dest='method', default="None",help='weight model to be used to fetch records')
    args = parser.parse_args()
    # checks if pyterrier init method is called
    try:
        if not pt.started():
            pt.init(boot_packages=["com.github.terrierteam:terrier-prf:-SNAPSHOT"], logging='ERROR')
    except Exception as e:
        print("Not able to load pyTerrier. Try running the python file again")
        print(e)
        exit(1)

    run_experiment(args)

