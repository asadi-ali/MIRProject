# MIRProject

The project is an information retrieval system that provides many services using NLP models and machine learning methods. These services include searching a sentence in a database of documents to find the relative documents, classifying new documents in a few pre-defined classes using the training set of labeled documents, and classifying the documents in an unsupervised manner into some clusters. The project consists of three phases that follow:

## Phase 1
* Preprocess and standardize the documents and sentences in four steps: normalization, tokenization, stemming and removing stop words.
* Build two indexing systems on the texts: positional indexer, and bigram indexer.
* Compress the indexing systems with variable byte and gamma code techniques.
* Correct words in a sentence and replace with words that most occur with neighbor words.
* Find the most relevant documents to a document with searching in the tf-idf vector space,
or proximity search.

## Phase 2
* Classify the news documents into four classes (World / Sports / Business / Sci/Tech) with four different methods: Naive Bayes, k-nearest-neighbor, SVM, and Random Forest.
* Report accuracy, precision, and recall of each classifier

## Phase 3
* Crawl documents from Semantic Scholar
* Cluster documents in tf-idf and word2vec vector spaces with three methods: k-means,
gaussian mixture model, hierarchical clustering
* Run the PageRank algorithm on the crawled documents

# Code Description

* main.py and server.py: using the services that are implemented in the files of the services folder that follow:
* document\_manager.py: implementing the base methods for processing documents
* index.py: implementing the indexer classes
* classify.py: running classifiers implemented in the classifiers folder (knn.py, naive\_bayes.py, random\_forest.py, and svm.py)
* cluster.py: implementing clustering methods on the documents
* search.py: implementing the search through documents methods
* vectorspace.py: providing the basic methods for treating documents as vectors
* compress.py: implementing compressing methods for indexers
* file\_manager: implementing methods for working with compressed objects
* page\_rank.py: implementing the page rank algorithm on documents
* spell\_correction.py: implementing correction methods on sentences
* visualize.py: implementing the basic method for visualization
