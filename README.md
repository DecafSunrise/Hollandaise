# Hollandaise
*"Deep Linguistic Analysis" of free text*  

The intent with this repo is to build out the "special sauce" in NLP tasks, identifying structures in token dependencies and other grammatical structure to extrat structured data from free text. 

## NLP Database formatting
![Document Decomposition](https://user-images.githubusercontent.com/36832027/198940666-af5497f4-8fbc-42e2-8e41-59cc996c4f05.png)
I'm working on a standardized format to decompose documents into constituent components; Paragraph and Sentence tokens. Treating each of these as a separate table both front-loads common NLP tasks (tokenization, cleaning, etc) to save time at runtime, and gives you a flexible schema you can extend with other processes (NER, Sentiment analysis, reading ease scores, etc).

![Hierarchial ID structure](https://user-images.githubusercontent.com/36832027/198940681-66e81baa-c503-4b55-a4a4-09913ddf269c.png)
By assigning a hierarchial GUID structure, you can investigate the data lineage of documents. Auto-generation ensures you can regenerate this structure on-the-fly, if your cleaning or tokenization methods change. Hopefully the enrichment can be automated with database triggers, or python watcher scripts to create implied queues of work to do. 

## Semantic Search
```sentence-transformers``` has some excelent examples of how to fine-tune transformers on your domain, as well as "neural search" for document similarity etc, overcoming the vocabulary mismatch problems common in lexical search and rule-based NLP.
