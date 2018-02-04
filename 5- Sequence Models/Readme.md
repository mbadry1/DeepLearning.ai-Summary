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
- We will index the first element of X by X<sup><1></sup>, the second X<sup><2></sup> and so on.
  - X<sup><1></sup> = Harry
  - X<sup><2></sup> = Potter
- Similarly, we will index the first element of Y by Y<sup><1></sup>, the second Y<sup><2></sup> and so on.
  - Y<sup><1></sup> = 1
  - Y<sup><2></sup> = 1
- X<sup>\<t></sup> gets an element by index t.
- T<sub>x</sub> is the size of the input sequence and T<sub>y</sub> is the size of the output sequence.
  - T<sub>x</sub> = T<sub>y</sub> = 9 in the last example although they can be different in other problems than name entity one.
- X<sub>(i)\<t></sub><sup>\<t></sup> is the element t of the sequence i in the training. Similarly for Y
- T<sub>x</sub> <sup>(i)</sup> is the size of the input sequence i.  It can be different across the sets. Similarly for Y
- **Representing words**:
  - We will now work in this course with **NLP** which stands for nature language processing. One of the challenges of NLP is how can we represent a word?
  - <u>The first thing</u> we need a **vocabulary** list that contains all the words in our target sets.
    - Example:
      - [a ... And   ... Harry ... Potter ... Zulu ]
      - Each word will have a unique index that it can be represented with.
      - The sorting here is by alphabetic order.
  - Vocabulary sizes in modern applications are from 30,000 to 50,000. 100,000 is not uncommon. Some of the bigger companies uses a million.
  - To build vocabulary list, you can read all the text you have and get m words with the most occurrence, or search online for m most occurrence words.
  - <u>The next step</u> is to create a one **hot encoding sequence** for each word in your dataset given the vocabulary you have created.
  - While converting, what if you meet a word thats not in your dictionary?
    - Well you can add a token in the vocabulary `<UNK>` which stands for unknown text and use its index in filling your one hot vector.
  - Full example can be found here:
    - ![](Images/01.png)

### Recurrent Neural Network Model
- Why not a standard network for sequence problems? There are two problems:
  - Inputs, outputs can be different lengths in different examples!
    - This can be solved in normal NNs by paddings with the maximum lengths but its not a good solution.
  - Doesn't share features learned across different positions of text/sequence.
    - Using a feature sharing like in CNNs can significantly reduce the number of parameters in your model. Thats what we will do in RNNs.
- Recurrent neural networks doesn't have the two mentioned problems.
- Lets build a RNN that solves **name entity recognition** task:
  - ![](Images/02.png)
  - In this problem T<sub>x</sub> = T<sub>y</sub>. In other problems where they aren't equal, the RNN architecture may be different.
  - a<sup><0></sub> is usually initialized with zeros, but some others may initialize it randomly in some cases.
  - There are three weight matrices here: W<sub>ax</sub>, W<sub>aa</sub>, and W<sub>ya</sub> with shapes:
    - W<sub>ax</sub>: (NoOfHiddenNeurons, n<sub>x</sub>)
    - W<sub>aa</sub>: (NoOfHiddenNeurons, NoOfHiddenNeurons)
    - W<sub>ax</sub>: (n<sub>y</sub>, NoOfHiddenNeurons)
- The weight matrix W<sub>aa</sub> is the memory the RNN is trying to maintain from the previous layers.
- A lot of papers and books write the same architecture this way:
  - ![](Images/03.png)
  - Its harder to interpreter. Its easier to roll this drawings to the unrolled version we have descried.
- In the discussed RNN architecture,  the current output y<sup>^t</sup> depends on the previous inputs and activations.
- Lets have this example ' He Said, "Teddy Roosevelt was a great president"  '. In this example Teddy is a person name but we know that from the word **President** that came after Teddy not from **He** and **said** that is before it.
- So limitation of the discussed architecture is that it learns from behind. To address this problem we will later discuss **Bidirectional** RNNs  (BRNNs).
- Now lets discuss the forward propagation equations on the discussed architecture:
  - ![](Images/04.png)
  - The activation function of a is usually tanh or RELU and for y depends on your task choosing some activation functions like sigmoid and softmax. In name entity recognition task we are solving, we will use Sigmoid because we only have two classes.
- In order to help us develop complex RNN architectures, the last equations needs to be simplified a bit.
- **Simplified RNN notation**:
  - ![](Images/05.png)
  - w<sub>a</sub> shape: (NoOfHiddenNeurons, NoOfHiddenNeurons + n<sub>x</sub>)
  - [a<sup><t-1></sup>, x<sup>t</sup>] shape: (NoOfHiddenNeurons + n<sub>x</sub>, 1)
### Backpropagation through time
- Lets see how backpropagation works with the RNN architecture we have developed.
- Often, Deep learning frameworks do backpropagation automatically for you. But its useful to know how it works especially in RNNs.
- Here is the graph:
  - ![](Images/06.png)
  - Where w<sup>a</sup>, b<sup>a</sup>, w<sup>y</sup>, and b<sup>y</sup> are shared across each element in a sequence.
- We will use the cross entropy loss function:
  - ![](Images/07.png)
  - Where the first equation is the loss for one element and the loss for the whole sequence is given by the summation over all the calculated values.
- Graph with losses:
  - ![](Images/08.png)
- The backpropagation here is called **backpropagation through time** because of the passed activation `a` from one sequence element to another.
### Different types of RNNs
- So far we have seen only one RNN architecture in which T<sub>x</sub> equals T<sub>Y</sub> always. In some other problems, they may not equal so we need different architectures.
- The ideas in this section was inspired by Andrej Karpathy [blog](http://karpathy.github.io/2015/05/21/rnn-effectiveness/). Mainly this image has all types:
  - ![](Images/09.jpg)
- The architecture we have descried before is called **Many to Many**.
- In sentiment analysis problem, X is a text while Y is an integer that rangers from 1 to 5. The RNN architecture for that is **Many to One** as in Andrej Karpathy image.
  - ![](Images/10.png)
- A **One to Many** architecture application would be music generation.
  - ![](Images/11.png)
  - Note that starting the second layer we are feeding the generated output back to the network.
- There are another interesting architecture in **Many To Many**. Applications like machine translation inputs and outputs sequences have different lengths in most of the cases. So an alternative Many To Many architecture that fits the translation would be as follows:
  - ![](Images/12.png)
  - There are an encoder and a decoder in the architecture. The encoder encodes the input sequence into one matrix and feed it to the decoder to generate the outputs. Encoder and decoder have different weight matrices.
- There are another architecture which is the **attention** architecture which we will talk about in chapter 3.
### Language model and sequence generation
- RNNs do very well in language model problems. In this section we will build a language model using RNNs.
- **What is a language model**
  - Lets say we are solving a speech recognition problem and some one says a sentence that can be interpreted into to two sentences:
    - The apple and **pair** salad
    - The apple and **pear** salad
  - **Pair** and **pear** sounds exactly the same, so how would a speech recognition application choose from the two.
  - Thats where the language models comes. It gives a probability for the two sentences and the application decides the best based on this probability.
- The job of a language model is given any sentence give a probability of that sentence. Also what is the next sentence probability given a sentence.
- **How to build language modeling with RNNs?**
  - The first thing it to get a **training set**: Large corpus of target language text.
  - Then tokenize this training set by getting the vocabulary and then one-hot each word.
  - Put an end of sentence token `<EOS>` with the vocabulary and include it with each converted sentence. Also like we have mentioned before use the token `<UNK>` for the unknown words.
- Given the sentence "Cats average 15 hours of sleep a day. `<EOS>`"
  - In training time we will use this:
    - ![](Images/13.png)
    - We pass to 0 vector - One hot -  to the first layer.
  - The loss function is defined by cross entropy loss:
    - ![](Images/14.png)
    - `i`  is for all elements in the training set.
- To use this model:
  1.  For predicting the chance of **next word**, we feed the sentence to the RNN and then get the final y<sup>^\<t></sup> hot vector and sort it by maximum probability.
  2.  For taking the **probability of a sentence**, we compute this:
     - p(y<sup><1></sup>, y<sup><2></sup>, y<sup><3></sup>) = p(y<sup><1></sup>) * p(y<sup><2></sup> | y<sup><1></sup>) * p(y<sup><3></sup> | y<sup><1></sup>, y<sup><2></sup>)
     - This is simply feeding the sentence to the RNN and multiply the probability for the given word in all the output hot encoded.
### Sampling novel sequences
- After a sequence model is trained on a language model, to check what the model has learned you can apply it on a sample novel sequence.
- Lets see the steps of how we can sample a novel sequence from a trained sequence language model:
  1. Given this model:
     - ![](Images/15.png)
  2. We first pass a<sup><0></sup> = zeros vector, and x<sup><1></sup> = zeros vector.
  3. Then we choose a prediction from y<sup>^<1></sup> using random distribution. For example it could be "The".
     - In Numpy this can be made using: `numpy.random.choice`
     - This is the line where you get a random sentences each time you run a novel sequence.
  4. We pass the last predicted word with the calculated  a<sup><1></sup>
  5. We keep doing 3 & 4 steps for a fixed length or until we get the `<EOS>` token.
  6. You can reject any `<UNK>` token if you mind finding it in you output.
- So far we have build a word level language model. A **character** level language model also can be made.
- In the character level language model the vocabulary will contain `[a-zA-Z0-9]`, space, and some special characters.
- Cons of character level language model compared to the word level language model:
  - There will be no `<UNK>` token.
  - It can deal with any word.
- But main disadvantage you will have a larger sequences! and also more computationally expensive and harder to train.
### Vanishing gradients with RNNs
- One of the problems with naive RNNs that it runs into **vanishing gradient** problems.

- An RNN that process a sequence data with the size of 10,000 time sets, has 10,000 deep layer! which is so hard to optimize.

- To address the problem lets take an example. Suppose we are working with language modeling problem and there are two sequences that it tries to learn:

  - "The **cat**, which already ate ........................, **was** full"
  - "The **cats**, which already ate ........................, **were** full"
  - The dots represents many words

- What we need to learn here that "was" came with "cat" and that "were" came with "cats". The naive RNN will find it hard to get this relation.

- As we have discussed in Deep neural networks, deeper networks are luckily to get into the vanishing gradient problem. In deep nets to compute the weights of the earlier layers you have to compute all the weights after these weights which causes the gradient to vanish. That also happens with RNNs with a long sequence size.

  - ![](Images/16.png)
  - For computing the word "Was", we need to compute the gradient for everything behind. Multiply by  fractions tends to vanish the gradient, while multiplication of large number tends to explode it.
  - There for half of your weights may not be updated properly!

- In the problem we descried it means that its hard for the network to memorize "was" word all over back to "cat". So in this case, the network won't be identify the singular/plural words so that it give it the right grammar word.

- The conclusion is that RNNs aren't good in **long term dependencies**.

- > In theory, RNNs are absolutely capable of handling such “long-term dependencies.” A human could carefully pick parameters for them to solve toy problems of this form. Sadly, in practice, RNNs don’t seem to be able to learn them. http://colah.github.io/posts/2015-08-Understanding-LSTMs/

- Vanishing gradients is tends to be the bigger problems with RNNs than the Exploding gradients problem. We will discuss how to solve it in the next sections.

- Exploding gradients can be easily seen when your weight values become `NAN` . So one of the ways solve exploding gradient is to apply **gradient clipping** means if your gradient is more than a threshold deal with the gradient as a fixed value. 

- **Extra**:

  - Solution for the Exploding gradient problem:
    - Truncated back propagation.
      - Not to update all the weights in the way back.
      - Not optimal. You won't update all the weights.
    - Gradient clipping.
  - Solution for the Vanishing gradient problem:
    - Weight initialization.
      - Like he initialization.
    - Echo state networks.
    - Use LSTM/GRU networks.
      - Most popular.
      - We will discuss it next.
### Gated Recurrent Unit (GRU)
- GRU is an RNN type that can help solve the vanishing gradient problem and can remember the long term dependencies.

- The basic RNN unit can be visualized to be like this:

  - ![](Images/17.png)

- We will represent the GRU with a similar drawings.

- Each layer in **GRUs**  has a new variable `C` which is the memory cell. It can tell to wether memorize a something or not.

- In GRUs, C<sup>\<t></sup> = a<sup>\<t></sup>

- Equations of the GRUs:

  - ![](Images/18.png)
  - The update gate is between 0 and 1
    - To understand GRUs imagine that the update gate is either 0 or 1 most of the time.
  - So we update the memory cell based on the update cell and the previous cell.

- Lets take the cat sentence example and apply it to understand this equations:

  - Sentence: "The **cat**, which already ate ........................, **was** full"

  - We will suppose that U is 0 or 1 and is a bit that tells us if a singular word needs to be memorized.

  - Splitting the words and get values of C and U at each place:

    - | Word    | Update gate(U)             | Cell memory (C) |
      | ------- | -------------------------- | --------------- |
      | The     | 0                          | val             |
      | cat     | 1                          | newVal          |
      | which   | 0                          | newVal          |
      | already | 0                          | newVal          |
      | ...     | 0                          | newVal          |
      | was     | 1 (I dont need it anymore) | newerVal        |
      | full    | ..                         | ..              |
- Drawing for the GRUs
  - ![](Images/19.png)
  - Drawings like in http://colah.github.io/posts/2015-08-Understanding-LSTMs/ is so popular and makes it easier to understand GRUs and LSTMs. But Andrew Ng finds its better to look at the equations.
- Because the update gate U is usually a small number like 0.00001, GRUs doesn't suffer the vanishing gradient problem.
  - In the equation this makes C<sup>\<t></sup> = C<sup>\<t-1></sup> in a lot of cases.
- Shapes:
  - a<sup>\<t></sup> shape is (NoOfHiddenNeurons, 1)
  - c<sup>\<t></sup> is the same as a<sup>\<t></sup>
  - c<sup>~\<t></sup> is the same as a<sup>\<t></sup>
  - u<sup>\<t></sup> is also the same dimensions of a<sup>\<t></sup>
- The multiplication in the equations are element wise multiplication.
- What has been descried so var is the Simplified GRU unit. Lets now describe the full one:
  - The full GRU contains a new gate that is used with to calculate the candidate C. The gate tells you how relevance is C<sup>\<t-1></sup> to C<sup>\<t></sup>
  - Equations:
    - ![](Images/20.png)
  - Shapes are the same
- So why we use these architectures, why don't we change them, how we know they will work, why not add another gate, why not use the simpler GRU instead of the full GRU; well researchers has experimented over years all the various types of these architectures with many many different versions and also addressing the vanishing gradient problem. They have found that full GRUs are one of the best RNN architectures  to be used for many different problems. You can make your design but put in mind that GRUs and LSTMs are standards.

### Long Short Term Memory (LSTM)

- Other unit that can make you do long term dependences is an LSTM unit and its more powerful and general than GRUs.
- In LSTMs , C<sup>\<t></sup> != a<sup>\<t></sup>
- Here are the equations of an LSTM unit:
  - ![](Images/21.png)
- In GRUs we have an update gate `U`, a relevance gate `r`, and a candidate cell variables C<sup>~\<t></sup> while in LSTMs we have an update gate `U` - Sometimes its called Input gate I -, a forget gate `F`, an output gate `O`, and a candidate cell variables C<sup>~\<t></sup>
- Drawings - which is inspired from http://colah.github.io/posts/2015-08-Understanding-LSTMs/ -: 
  - ![](Images/22.png)
- Some Variants on LSTMs includes:
  - LSTM with **peephole connections**.
    - The normal LSTM with C<sup>\<t-1></sup> included with every gate.
- There isn't a universal superior in choosing between LSTMs and its variants. Some win and some problems and some don't. One of the advantages of GRUs is that its simpler and can be used to build much bigger network but the LSTM is more powerful and general as we said.
### Bidirectional RNN
- There are still some ideas to make a powerful sequence model. One of them is bidirectional RNNs and another is Deep RNNs.
- As we saw before, here are an example on Name entity recognition:
  - ![](Images/23.png)
- The name Teddy cannot be learned from **He** and **said**, but can be learned from **bears**.
- BRNNs fixes this issue.
- Here are BRNNs architecture:
  - ![](Images/24.png)
- Hint that BRNNs are **acyclic graph**!
- The forward propagation goes from left to right, and from right to left. It learns from both sides.
- To make predictions we use y<sup>^\<t><sup> by using the two activations that comes from left and right.
- The blocks here can be any RNN block including the basic RNNs, LSTMs, or GRUs.
- For a lot of NLP or text processing problems, a BRNN with LSTM appears to be commonly used.
- The disadvantage of BRNNs that you need the entire sequence before you can process it. For example in live speech recognition if you used BRNNs you will need to wait for the person who speaks to stop to take the entire sequence and then make your predictions.
### Deep RNNs

- In a lot of cases the standard one layer RNNs will solve your problem. But in some problems its useful to stack some RNN layers to make a deeper network.
- For example, a deep RNN with 3 layers would look like this:
  - ![](Images/25.png)
- In feed forward deep nets we can have till 100 or 200 layer and this would be a so big network. In deep RNNs stacking 3 layers is considered deep and already expensive.
- In some cases you might see a deep network that are connected after recurrent cell.


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