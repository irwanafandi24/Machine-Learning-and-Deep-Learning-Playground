# Basic NLP (Natural Language Processing)
<p align="justify">NLP is a part of machine learning where we will try to make machine able to understand, analyze, manipulate, and potentially generate human language.  Its goal is to build systems that can make sense of text and perform tasks like <i>machine translation, grammar checking, text summarization, speech recognitiong, topic classification,</i> etc. In this project I will show you the steps <b><i>how to create a simple text classification</i></b> starting from load the dataset until get the predicted class of the text.</p>

**Project Overview**

### Let's Get Started
<p align="justify">The main problem of text data is the data itself. Usually, it comes from the crawling process which is definitely very messy, lots of words are not normal, lots of punctuation and soon. <b><i>The first step, we need to Preprocessing on The Text Data</i></b> and here is the process that commonly used in this step:</p>
<ul>
  <li>Delete multiple word patterns (using regex)</li>
  <li>Remove punctuation insede the words</li>
  <li>Case folding (change all letters to lowercase)</li>
  <li>Tokenization (cut sentences into words form like: unigram (1), bigram(2), trigram(3), n-grams(N-words))</li>
  <li>Stopword removal (removing unimportant words from the sentence)</li>
  <li>Lemmatization (reduces the inflected words properly ensuring that the root word belongs to the language)</li>
  <li>Stemming (reducing inflection(cut the prefix or sufix) in words to their root forms)</li>
</ul>
<p align="justify">Output from this process is array of clean sentences or array of sentence tokens

```sh
#array of clean sentences
["i love reading a book", "could i borrow your pen"] 
#array of sentence tokens
[["i", "love", "reading", "a", "book"], ["could", "i", "borrow", "your", "pen"]]
```
<b><i>The second step is Feature Extraction (Weighting)</i></b>, this process will transform the text data into vector (number) format to make the system is able to process it. There are 2 feature extractions that have been used in this project : <i>bag of words (BOW) and term frequency inverse document frequency (TF-IDF).</i> How they work?<br><br>
**Bag of Words (BoW)**
<p align="justify">BoW model is the simplest form of text representation in numbers, because we just need to count the number of vocab (unique word set of all documents) that occurrence in a document. For example we have 3 document:</p>
<ol>
  <li>saya zakaria suka makan nasi nasi dan lauk</li>
  <li>saya pesan nasi dan tahu</li>
  <li>tahu buat makan</li>
</ol>
<p align="justify">First we have to build a vocabulary from all the unique words in the documents above, and we get this words (sorted):</p>

```sh
["buat", "dan", "lauk", "makan", "nasi", "pesan", "saya", "suka", "tahu", "zakaria"]
```
<p align="justify">Then, we just need to count the number of each words from vocabulary that occurrence in a document. From this method, we will get this result.</p>

| x | buat | dan | lauk | makan | nasi | pesan | saya | suka | tahu | zakaria |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `doc 1` | 0 | 1 | 1 | 1 | 2 | 0 | 1 | 1 | 0 | 1 |
| `doc 2` | 0 | 1 | 0 | 0 | 1 | 1 | 1 | 0 | 1 | 0 |
| `doc 3` | 1 | 0 | 0 | 1 | 0 | 0 | 0 | 0 | 1 | 0 |

**Term Frequency Inverse Document Frequency (TFIDF)**
<p align="justify">TFIDF is a statistical measure that evaluates how relevant a word is to a document in a collection of documents. This is done by multiplying two metrics: how many times a word appears in a document (TF), and the inverse document frequency of the word across a set of documents (IDF).</p>
<p><i>TF formula and its result</i></p>

```sh
TF(t) = (Number of times term t appears in a document) / (Total number of terms in the document).
```
| x | buat | dan | lauk | makan | nasi | pesan | saya | suka | tahu | zakaria |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `doc 1` | 0 | 0.125 | 0.125 | 0.125 | 0.250 | 0 | 0.125 | 0.125 | 0 | 0.125 |
| `doc 2` | 0 | 0.200 | 0 | 0 | 0.200 | 0.200 | 0.200 | 0 | 0.200 | 0 |
| `doc 3` | 0.333 | 0 | 0 | 0.333 | 0 | 0 | 0 | 0 | 0.333 | 0 |
<p><i>IDF formula and its result</i></p>

```sh
IDF(t) = log_e(Total number of documents / Number of documents with term t in it). 
```
| Term | Value |
| :---: | :---: |
| buat | 1.386 |
| dan | 0.916 |
| lauk | 1.386 |
| makan | 0.916 |
| nasi | 0.916 |
| pesan | 1.386 |
| saya | 0.916 |
| suka | 1.386 |
| tahu | 0.916 |
| zakaria | 1.386 |
<p><i>TF x IDF result</i></p>

| x | buat | dan | lauk | makan | nasi | pesan | saya | suka | tahu | zakaria |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `doc 1` | 0 | 0.286| 0.377 | 0.286 | 0.573 | 0 | 0.286 | 0.377 | 0 | 0.377 |
| `doc 2` | 0 | 0.418 | 0 | 0 | 0.418 | 0.549 | 0.418 | 0 | 0.418 | 0 |
| `doc 3` | 0.681 | 0 | 0 | 0.518 | 0 | 0 | 0 | 0 | 0.518 | 0 |

<p align="justify"> After we get the documents vector value, now we can <b><i>Start to Build The Text Model</i></b>. In this step, I used several types of machine learning model with hyperparameter tuning to obtain higher accuracy. This is the accuracy result as well as the type of feature extraction used.</p>
<p align="center"><img src="asset/instagram_flow.PNG" width=100%></p>
<p align="justify">Besides the accuracy of the model, we can also visualize the data like: the user who is most mentioned, most frequently hashtags and cloudword. You can see the results below.
<p align="center"><img src="asset/instagram_flow.PNG" width=100%></p>  

🎉 Congrats... Now, you can start to build your own text classification. Maybe you can use word2vect or doct2vect :)
