# Sequence Models

This is the fifth and final course of the deep learning specialization at [Coursera](https://www.coursera.org/specializations/deep-learning) which is moderated by [deeplearning.ai](http://deeplearning.ai/). The course is taught by Andrew Ng.

**<u>This page is just a draft right now.</u>**

## Table of contents
[TOC]

## Course summary
Here are the course summary as its given on the course [link](https://www.coursera.org/learn/nlp-sequence-models):

> This course will teach you how to build models for natural language, audio, and other sequence data. Thanks to deep learning, sequence algorithms are working far better than just two years ago, and this is enabling numerous exciting applications in speech recognition, music synthesis, chatbots, machine translation, natural language understanding, and many others. 
>
> You will:
> - Understand how to build and train Recurrent Neural Networks (RNNs), and commonly-used variants such as GRUs and LSTMs.
> - Be able to apply sequence models to natural language problems, including text synthesis. 
> - Be able to apply sequence models to audio applications, including speech recognition and music synthesis.
>
> This is the fifth and final course of the Deep Learning Specialization.



## Recurrent Neural Networks

> Learn about recurrent neural networks. This type of model has been proven to perform extremely well on temporal data. It has several variants including LSTMs, GRUs and Bidirectional RNNs, which you are going to learn about in this section.

### Why sequence models

- Sequence Models like RNN and LSTMs have greatly transformed learning on sequences in the past few years.
- Examples of sequence data in applications:
  - Speech recognition (**Sequence to sequence**):
    - X:           Wave sequence
    - Y:           Text sequence
  - Music generation (**one to sequence**):
    - X:           (Can be nothing or an integer)
    - Y:           Wave sequence
  - Sentiment classification (**sequence to one**):
    - X:          Text sequence
    - Y:           Integer rating from one to five
  - DNA sequence analysis (**sequence to sequence**):
    - X:           DNA sequence
    - Y:            DNA Labels
  - Machine translation (**sequence to sequence**):
    - X:            Text sequence (In a language)
    - Y:            Text sequence (In other language)
  - Video activity recognition (**Sequence to one**):
    - X:            Video frames
    - Y:             Label (Activity)
  - Name entity recognition  (**Sequence to sequence**):
    - X:            Text sequence
    - Y:             Label sequence
    - Can be used by seach engines to index different type of words inside a text.
- As you can see there are different data with different input and outputs - sequence or one - that can be learned by supervised learning models.
- There are different ways and models to tackle different sequence problem.

### Notation

- In this section we will discuss the notations that we will use through the course.
- **Motivating example**:
  - In the content of name entity recognition application let:
    - X: "Harry Potter and Hermoine Granger invented a new spell."
    - Y:   1   1   0   1   1   0   0   0   0
    - Both elements has a shape of 9. 1 means its a name, while 0 means its not a name.
  - We will index the first word by X<sup><1></sup>
  - ​

### Recurrent Neural Network Model
- ​
### Backpropagation through time
- ​
### Different types of RNNs
- ​
### Language model and sequence generation
- ​
### Sampling novel sequences
- ​
### Vanishing gradients with RNNs
- ​
### Gated Recurrent Unit (GRU)
- ​
### Long Short Term Memory (LSTM)
- ​
### Bidirectional RNN
- ​
### Deep RNNs
- 


## Natural Language Processing & Word Embeddings

> Natural language processing with deep learning is an important combination. Using word vector representations and embedding layers you can train recurrent neural networks with outstanding performances in a wide variety of industries. Examples of applications are sentiment analysis, named entity recognition and machine translation.

### Introduction to Word Embeddings

#### Word Representation
- ​

#### Using word embeddings
- ​

#### Properties of word embeddings
- ​

#### Embedding matrix
- ​

### Learning Word Embeddings: Word2vec & GloVe

#### Learning word embeddings
- ​

#### Word2Vec
- ​

#### Negative Sampling
- ​

#### GloVe word vectors
- ​

### Applications using Word Embeddings

#### Sentiment Classification
- ​

#### Debiasing word embeddings
- 


## Sequence models & Attention mechanism

> Sequence models can be augmented using an attention mechanism. This algorithm will help your model understand where it should focus its attention given a sequence of inputs. This week, you will also learn about speech recognition and how to deal with audio data.

### Various sequence to sequence architectures

#### Basic Models
- ​

#### Picking the most likely sentence
- ​

#### Beam Search
- ​

#### Refinements to Beam Search
- ​

#### Error analysis in beam search
- ​

#### Bleu Score (optional)
- ​

#### Attention Model Intuition
- ​

#### Attention Model
- ​

### Speech recognition - Audio data

#### Speech recognition
- ​

#### Trigger Word Detection
- 








<br><br>
<br><br>
These Notes was made by [Mahmoud Badry](mailto:mma18@fayoum.edu.eg) @2018