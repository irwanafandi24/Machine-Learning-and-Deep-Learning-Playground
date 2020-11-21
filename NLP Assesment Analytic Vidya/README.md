# Basic NLP (Natural Language Processing)
<p align="justify">NLP is a part of machine learning where we will try to make machine able to understand, analyze, manipulate, and potentially generate human language.  Its goal is to build systems that can make sense of text and perform tasks like <i>machine translation, grammar checking, text summarization, speech recognitiong, topic classification,</i> etc. In this project I will show you the steps <b><i>how to create a simple text classification</i></b> starting from load the dataset until get the predicted class of the text.</p>

**Project Overview**

### Let's Get Started
<p align="justify">The main problem of text data is the data itself. Usually, it comes from the crawling process which is definitely very messy, lots of words are not normal, lots of punctuation and soon. In the first step, we need to <b><i>preprocessing on the text data</i></b> and here is the process that commonly used in this step:</p>
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
The second step is <b><i>feature extraction (weighting)</i></b>, this process will transform the text data into vector (number) format to make the system is able to process it. There are 2 feature extractions that have been used in this project : <i>bag of words (BOW) and term frequency inverse document frequency (TF-IDF).</i> How they work?<br><br>
**Bag of Words (BOW)**

**Term Frequency Inverse Document Frequency (TFIDF)**
