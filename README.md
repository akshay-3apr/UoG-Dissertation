# Investigating the Feasibility of Approximating a Neural Retrieval Model’s Search Results with Statistical Model
## UoG-Dissertation
## This project is implemented as the final dissertatino project for the Master degree in Data Science

To get the code running please follow below steps:

Create a directory named project and start a virtual environment.
```
$ mkdir project
$ python3 -m venv project
$ source project/bin/activate
```

Please use `git clone` command to clone this project to your local system where you create the virtual environment.
```
$ cd project
$ git clone https://github.com/akshay-3apr/UoG-Dissertation.git
```

Once you have ran the above command, the project will get extracted to your local system. The following folders will get created
```
.
UoG-Dissertation
├── app
│   ├── Node.py
│   ├── PriorityQueue.py
│   ├── __init__.py
│   ├── bestfirstsearch.py
│   ├── bin
│   │   |--
│   ├── commands.txt
│   ├── corpus.py
│   ├── data
│   │   |--
│   ├── greedysearch.py
│   ├── helper.py
│   ├── include
│   ├── index
│   ├── irevaluation.py
│   ├── requirements.txt
│   ├── rlmscorecal.py
│   └── version.py
└── README.md
```

Before we start with setup of the code. This project is heavily tested on MacOS `Monterey`, `Python 3.8.12` with pip version `pip 21.2.4`. Please make sure your python and pip versions are updated.
Change the directory to start the setup installation:

```
$ cd UoG-Dissertation/app
$ pip install -r requirements.txt
```

The files are installed and now we can run the below commands to collect the results using Discrete Search Space Optimisation Algorithms.

To collect results for best first search algorithm run the below command:
```
$ python -m bestfirstsearch \
--index_path "path/to/pyterrier/index/data.properties" \
--dltop1000 "path/to/neuralmodel/topdocuments/filename.tsv" \
--topics "path/to/topics.tsv" \
--optimalquery "project/app/data/BM25_BESTFIRST_jaccard_trec_dl_top10_terms_30_md15_ms500_addtermsonly_False.csv" \
--rm3vocab "path/to/vocabulary/vocabularyRLMScore_top10docs_terms100.csv" \
--topK 10 \
--wmodel BM25 \
--addtermsonly False \
--evalmatrix jaccard \
--termselection RM3 \
--maxbranching 30 \
--maxnumstates 500 \
--maxnumterms 15 \
--optimise True \
--relevant False
```


> Parameters like `optimise` shdould be turned `False` when not optimising
>
> `relevant` parameter is used when we are evaluating using relevant documents.
>
> `evalmatrix` should be `jaccard` when optimising and `rbo` when evaluating the generated query.
>
> `maxbranching` is vocabulary size.
>
> For rest parameter like `maxnumterms`,`addtermsonly`,`termselection` and `maxnumstates` please look at the submitted dissertation report Table 5.1 in Chapter-5.

To collect results for greedy algorithm run the below command:
```
$ python -m greedysearch \
--index_path "path/to/pyterrier/index/data.properties" \ \
--dltop1000 "path/to/neuralmodel/topdocuments/filename.tsv" \
--termweights "path/to/project/app/data/Greedy_termweights.csv" \
--topics "path/to/topics.tsv" \
--topK 10 \
--addtermsonly False \
--evalmatrix jaccard \
--maxnumterms 15 \
--optimise True \
--relevant False
```

> Parameters like `optimise` shdould be turned `False` when not optimising. `relevant` parameter is used when we are evaluating using relevant documents.
>
> `evalmatrix` should be `jaccard` when optimising and `rbo` when evaluating the generated query.
>
> `maxbranching` is vocabulary size.
>
> For rest parameter like `maxnumterms` and `addtermsonly` please look at the submitted dissertation report Table 5.1 in Chapter-5.

To collect results for RM3 grid Search run the below command:
```
$ python -m rlmscorecal \
--index_path "/Users/akshayprakash/Documents/UoG/Semester 3/dissertation/UoG-Dissertation-git/app/index/custom/data.properties" \
--dltop1000 "/Users/akshayprakash/Documents/UoG/Semester 3/dissertation/Debasis Data/ANCE.2019.res" \
--topics "/Users/akshayprakash/Documents/UoG/Semester 3/dissertation/Trec-DL-data/topics.tsv" \
--topK 10 \
--evalmatrix jaccard \
--gridsearchlist "5,10,20,30,40,50" \
--optimise False \
--relevant False
```

> Parameters like `optimise` shdould be turned `False` when not optimising. `relevant` parameter is used when we are evaluating using relevant documents.
>
> `evalmatrix` should be `jaccard` when optimising and `rbo` when evaluating the generated query.


To collect IR evaluation score of MAP@10 and nDCG@10, run below command
```
$ python -m irevaluation \
--index_path "/Users/akshayprakash/Documents/UoG/Semester 3/dissertation/UoG-Dissertation-git/app/index/custom/data.properties" \
--topics "path/to/final_expanded_query/test_topics.tsv" \
--qrels "/Users/akshayprakash/Documents/UoG/Semester 3/dissertation/Trec-DL-data/eval/test_qrels.tsv"
```

A result and outputs of this experiment are uploaded in below onedrive:
> https://gla-my.sharepoint.com/:f:/g/personal/2684995a_student_gla_ac_uk/ElZho_xKjLBBsSWsUS9sY-MB9ZUxunxQ5XwtlmQRCR4jEw?e=w8LENp