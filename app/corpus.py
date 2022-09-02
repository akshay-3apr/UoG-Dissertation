## if importing data using pyterrier
## Dataset document: http://data.terrier.org/msmarco_passage.dataset.html
## MS MARCO document: https://microsoft.github.io/msmarco/TREC-Deep-Learning.html
## Get index from PyTerrier dataset

import pandas as pd
import pyterrier as pt

class Dataset:

    def __init__(self,index_path):
        '''
        path to TREC-19 pyterrier index
        '''
        if index_path == "empty":
            self.dataset = pt.get_dataset('trec-deep-learning-passages')
            indexref = self.dataset.get_index(variant='terrier_stemmed')
            self.index = pt.IndexFactory.of(indexref)
        else:
            self.index = pt.IndexFactory.of(index_path)

        self.dataset = None
        self.collection = None
        self.test_topics = None
        self.test_qrels = None

    def importdata(self):
        selecteddataset = int(input("Which dataset to load \n 1. TREC \n 2. MSMARCO-PASSAGE \n Please type the respective number: "))
        if selecteddataset == 1:
            ### TREC DL PASSAGES
            self.dataset = pt.get_dataset('trec-deep-learning-passages')
            indexref = self.dataset.get_index(variant='terrier_stemmed')
            self.index = pt.IndexFactory.of(indexref)
            test_topics_19 = self.dataset.get_topics('test-2019')
            test_topics_20 = self.dataset.get_topics('test-2020')
            test_qrels_19 = self.dataset.get_qrels('test-2019')
            test_qrels_20 = self.dataset.get_qrels('test-2020')
            self.test_topics = pd.concat([test_topics_19,test_topics_20]).drop_duplicates()
            self.test_qrels = pd.concat([test_qrels_19,test_qrels_20]).drop_duplicates()
            self.test_topics.qid=self.test_topics.qid.astype("str")
            print("Topics unique count after removing duplicates: ",self.test_topics.qid.unique().size)
            print("Qrels unique count after removing duplicates: ",self.test_qrels.qid.unique().size)

        elif selecteddataset == 2:
            ### MAMARCO PASSAGE
            self.dataset = pt.get_dataset('msmarco_passage')
            indexref = self.dataset.get_index(variant='terrier_stemmed')
            self.index = pt.IndexFactory.of(indexref)
            test_topics_19 = self.dataset.get_topics('test-2019')
            test_topics_20 = self.dataset.get_topics('test-2020')
            test_qrels_19 = self.dataset.get_qrels('test-2019')
            test_qrels_20 = self.dataset.get_qrels('test-2020')
            self.test_topics = pd.concat([test_topics_19,test_topics_20]).drop_duplicates()
            self.test_qrels = pd.concat([test_qrels_19,test_qrels_20]).drop_duplicates()
            self.test_topics.qid=self.test_topics.qid.astype("str")
            print("Topics unique count after removing duplicates: ",self.test_topics.qid.unique().size)
            print("Qrels unique count after removing duplicates: ",self.test_qrels.qid.unique().size)

        return self

    def buildCustomIndex(self,dataframe):
        ### CUSTOM INDEX from DATAFRAME
        pd_indexer = pt.DFIndexer("./index/custom",overwrite=True)
        indexref = pd_indexer.index(dataframe["text"],dataframe["docno"])
        self.index = pt.IndexFactory.of(indexref)
        return self

    def getIndex(self):
        return self.index

    def getDataset(self):
        return self.dataset

    def setCollection(self,path):
        assert path is not None, "Path to read collection from is empty"
        ##get collection from path
        self.collection = pd.read_csv(path,sep="\t",header=None,names=["docno","text"])
        return self

    def getCollection(self):
        assert self.collection is not None, "Collection not set. Set Collection first using 'setCollection()'"
        return self.collection

    def setTopics(self,path):
        assert path is not None, "Path to read topics from is empty"
        ##get topics from path
        self.topics = pd.read_csv(path,sep="\t",header=None,names=["qid","query"])
        return self

    def getTopics(self):
        return self.test_topics

    def setQrels(self,path):
        assert path is not None, "Path to read qrels from is empty"
        ##get qrels from path
        self.qrels = pd.read_csv(path,sep="\t",header=None,names=["qid","docno","rel"])
        return self

    def getQrels(self):
        return self.test_qrels

    def printCollectionStatistics(self):
        print(self.index.getCollectionStatistics().toString())