# NLP
This repository now includes following parts: 

Neural Machine Translation and chatbot are all built on the famous Seq2Seq model with attention. In Neural Machine Translation section there are two kinds of Seq2Seq model. The difference between them lises in the different positions of attention unit. 

The part of Sentiment Analysis is using GRUs to help to analyze whether the sentiment is positive or negtive from the imdb datasets. 

The project of recommendation system uses GRU and MovieLens data set to complete the task of movie recommendation. 

The Speech Recognition project realizes a simple speech recognition using GRU and vedio files recording numbers from zero to nine.

The project of chatbot retrieval also realizes a chatbot, however, it focuses on some special industries such as banking and insurance industry where answering for querying is relatively fixed to some extent. Therefore the project applies the so called 'dual encoder LSTM model' based on retrieval to build the above system.

The project of PosTagger realizes POS Tagging based on deep learning using bidirectional lstm network and the data set we uses is the annotated corpus in People's Daily in January 1998.

NLI project completes a simple natural language inference target based on decomposable attention model using snli data set released by Standford University and pretrained Glove word vectors.
