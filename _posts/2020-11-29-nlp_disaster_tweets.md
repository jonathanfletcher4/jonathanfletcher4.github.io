---
layout: post
title: NLP Predicting Disaster Tweets with LSTM and GloVe
---

In this post we will go through the basics of building an NLP neural network using GloVe embeddings to classify tweets about natural disasters. It uses Keras and assumes basic knowledge of neural network architecture (namely just the different types of layers)

First we will preprocess textual data then some basic EDA then compare 1-directional and bidirectional LSTM performance. The goal of this post is to provide an introduction to building NLP models and the models we'll compile will reflect that.

# Data Source

In 2017 Figure-Eight released a dataset of 10,876 tweets (now available through [kaggle](https://www.kaggle.com/jannesklaas/disasters-on-social-media)). Some of the tweets were genuinely about natural disasters for example: *"I can see a fire in the woods"* .Some tweets were not associated to natural disasters but contain terminology related to natural disasters such as *"This mixtape is fire"* . The tweets are hand labelled as relevant or irrelevant

See below for snapshot of some of the tweets and their respective labels.


|      | keyword          | text                                                                                                                                      |   relevant |
|-----:|:-----------------|:------------------------------------------------------------------------------------------------------------------------------------------|-----------:|
| 7785 | police           | -=-0!!!!. Photo: LASTMA officials challenge police for driving against traffic in Lagos http://t.co/8VzsfTR1bG                            |          1 |
| 7956 | rainstorm        | @SavanahResnik @CBS12 I would hide out at the Coldstone at monterrey and us 1. Great place to wait out a rainstorm.                       |          1 |
| 1884 | burning          | WoW Legion ÛÒ Slouching Towards The Broken Isles: Warlords of Draenor wasnÛªt close enough to The Burning Crusad... http://t.co/RKpmoMQMUi                                                                                                                                           |          1 |
| 7294 | nuclear disaster | Chernobyl disaster - Wikipedia the free encyclopedia don't you just love the nuclear technology it's so glorious  https://t.co/GHucazjSxB |          1 |
| 1699 | bridge collapse  | Two giant cranes holding a bridge collapse into nearby homes http://t.co/jBJRg3eP1Q                                                       |          1 |


# Preprocessing

Given these are tweets pulled directly from Twitter significant amounts of preprocessing were requried:

- Punctuation, symbols and numbers were removed using Regex patterns  
- Credit to kaggle.com/gunesevitan (see his work [here](https://www.kaggle.com/gunesevitan/nlp-with-disaster-tweets-eda-full-cleaning#4.-Embeddings-&-Text-Cleaning)) for carrying out a huge amount of manual preprocessing of typos, acronyms and abbreviations on this dataset which we can leverage 

Below is an example of a tweet before and after preprocessing. We see the link and symbols have been removed and set to lower case


    **Original Tweet**: 
     http://t.co/GKYe6gjTk5 Had a #personalinjury accident this summer? Read our advice &amp; see how a #solicitor can help #OtleyHour 
    
    **Preprocessed Tweet**: 
      had a personalinjury accident this summer read our advice  see how a solicitor can help otleyhour
    

## Embeddings
In order for the data to be used in a machine learning model words need to be converted into numbers (tokens) which have meaning and from which predictions can be made. For this project the *glove.twitter.27B.25d* embeddings dataset was used to tokenize the tweets.

There are multiple different GloVe embeddings datasets which are trained on different corpuses of data. This specific embedding is trained on 2 billion tweets (27 billion tokens) and represented in a 25-dimensional vector. Here training means the model learns context around words. Words that appear together or in the same sentence more often are more "similar".

GloVe creates a word-word matrix and uses K-Nearest Neighbours to identify words which are similar. Below is an example of a word-word matrix for the sentences *"I can see a fire in the woods"* and *"This mixtape is fire"*,  greater values mean the words are more linguistically similar and lower values mean the words are less similar

First we load the glove embeddings (you can download this or others [here](https://nlp.stanford.edu/projects/glove/))


```python
## GLOVE ##

glove_embeddings = {}
with open("~/glove.twitter.27B.25d.txt", 'r', encoding="utf8") as f:
    for line in f:
        values = line.split()
        word = values[0]
        vector = np.asarray(values[1:], "float32")
        glove_embeddings[word] = vector
```

- Then we create a dictionary with each unique word as keys and their counts as the values


```python
vocab = {}

for tweet in df['text2']:
    for word in tweet.split():
        try:
            vocab[word] += 1
        except KeyError:
            vocab[word] = 1
```

- Now we check how many of our words exist in the gloVe embeddings.
- In our case **82% of the words in the tweets appear in the GloVe embeddings**. Those that don't are given a deafult token. This percentage can be increased further with additional manual effort to fix typos, abbreviations, acronyms etc.


```python
covered = {}
oov = {}    
n_covered = 0
n_oov = 0

for word in vocab:
    try:
        covered[word] = glove_embeddings[word]
        n_covered += 1
    except:
        oov[word] = vocab[word]
        n_oov += 1
```

With 27 billion words this becomes a huge matrix and extremely computationally expensive to run any analysis on. GloVe uses dimensionality reduction to create a $n \times 25$ representation of the similarities making it easier to use for use cases such as this project.

Below is a 2-dimensional ISOMAP representation of a random sample of 50 embeddings from the GloVe dataset. We can see at the top the words are closer together *evacuation, wildfires, fleeing* which makes logical sense as these are commonly associated with each other. Near the center we have *cities, shelter, residents* close together which also makes logical sense. This mapping is used for all words in our tweet vocabulary which also appear in the GloVe embeddings.



    
![png](../../images/nlp_disaster_tweets/output_16_0.png)
    


# Tweet Inputs
Now we have our embeddings we can prepare the data for training. We'll follow a standard text preprocessing format:

- Create a corpus of unique words in text



```python
from nltk.tokenize import word_tokenize
import nltk

# nltk.download('punkt')

# Creates a list of lists. each list is a tweet, each word is an element in the list
def create_corpus(df, col):
    corpus=[]
    for tweet in df[col]:
        words=[word.lower() for word in word_tokenize(tweet) if((word.isalpha()==1) & (word not in STOPWORDS))]
        corpus.append(words)
    return corpus
```


```python
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# create corpus
corpus = create_corpus(df, 'text')
```

- Tokenize words (assign word to a number) and convert sentences of words to sequences of tokens
 


```python
# max sequence length
MAX_LEN = 50
tokenizer_obj = Tokenizer()

# Assigns each unique word to a numbered tokens
tokenizer_obj.fit_on_texts(corpus)

# Converts tweets from words to numbered tokens
sequences = tokenizer_obj.texts_to_sequences(corpus)
```

- Pad sequences - we want all sequences to have the same length (dimensions) so we choose a max length of sequences and pad shorter sequences with trailing 0's to ensure they are all the same length. This is will be the **input layer** of our neural network


```python
# Pads vectorized tweets with 0's so they are all the same length
tweet_pad = pad_sequences(sequences, maxlen=MAX_LEN, truncating='post', padding='post')
```

- Create gloVe embeddings vectors by retrieving gloVe embedding of each word. This will be the **embeddings layer**  which maps our sequences of tokens to their corresponding gloVe vector representations .


```python
# Dictionary with words as keys and count as values
word_index=tokenizer_obj.word_index
print('Unique words:',len(word_index))

# Number of unique words
num_words=len(word_index)+1

# Matrix for embeddings vectors. Each row is a unique word. Columns are glove embedding vector (in this case 25 cols)
embedding_matrix=np.zeros((num_words,25))

for word,i in word_index.items():
    if i > num_words:
        continue
    
    emb_vec=glove_embeddings.get(word)
    if emb_vec is not None:
        embedding_matrix[i]=emb_vec
```

    

# Keyword Inputs
Since we are also using the Keyword column as a feature we run the same process again. Instead this time we set our max sequence length to 3 instead of 50. Heavily padding unnecessarily is computationally expensive. 

Since the code is largely the same it is excluded from the post but is available in the notebook at the bottom of the post.


# Exploratory Data Analysis

Before modelling we should carry out some EDA to better understand the data. 

The dataset consists of **6,230 Relevant tweets and 4,673 Not Relevant tweets**. Although not quite balanced it is balanced enough such that we don't need to carry out any resampling.

## Tweet and Word Length Distribution
Next we take a look at the average tweet length by class. The distribution of tweet lengths are similar between the 2 classes except a small spike in Relevant tweets of ~115 characters

Similarly with word lengths, the distributions between classes are similar with just a few spikes in shorter words in Not Relevant tweets.

These distributions suggest the two classes are quite similar in structure which may add the complexity of distinguishing  



    
![png](../../images/nlp_disaster_tweets/output_29_0.png)
    


## Common words and Bigrams
Now we take a look at the most common words and bigrams in the tweets. Looking for the differences and similarities in language. (Stopwords such as *a, the, how* etc. have been removed )

We can see the difference between classes when comparing the top 20 most common words. In the Relevant tweets we see some keywords such as *suicide, disaster, police, hiroshima* which are mostly commonly used when discussing disasters/incidents. However in the Not Relevant tweets the most common words are more standard words such as *love, video, body, people* which we commonly see in all types of conversations.



    
![png](../../images/nlp_disaster_tweets/output_32_0.png)
    


With the top 20 bigrams the Relevant tweets again contains pairs of words commonly related to disasters such as *(suicide, bomber), (oil, spill), (california, wildfire)* . The Not Relevant tweets also have some bigrams commonly associated with disasters such as *(body, bag), (burning, buildings)* but mostly area againg general common bigrams

The differences we see in the common words in each class suggest there are at least some patterns a model can be trained on to distinguish between classes.


    
![png](../../images/nlp_disaster_tweets/output_34_0.png)
    


# Model

The models used are both variations of **LSTM (Long-Short Term Memory)** neural network. LSTMs are like common Recurrent Neural Networks but they have the added element of memory. They are very good at "forgetting" unuseful information and retaining useful information from previous states to learn words in meaningful sequences i.e. context. 

For example, if we trained an LSTM using a Harry Potter novel and asked it predict the next words in the sentence *It must have been Lord _*  , an LSTM more likely to "forget" the words *It must have been* since this sequence of words likely appears in many different contexts. It is likely retain the word *Lord* and predict the next word as *Voldermort* because when we see the word *Lord* in Harry Potter novels it is almost if not always followed by the word *Voldermort*

LSTMs ability to retain relevant contextual information makes it a good choice for our use case, since we are trying to identify a meaningful phrases in the tweets. Both a **Left-to-Right LSTM** (as described above) which tries to learn by evaluating words from left-to-right as we do in English,  and a **Bidirectional LSTM** which tries to learn by evaluating from left-to-right and right-to-left were compared. 


## Metric
The evaluation metric used is f1-score since it is equally as important to capture as many disaster tweets as possible (recall) as flagging only as many true positives as possible (precision) given the flagged tweets are then reviewed manually.

A baseline on 0.5 is set as the dataset is roughly balanced this approximately equal to classifying tweets at random




# Build Model

The models we'll compare are an LSTM one-directional network and an LSTM bidirectional network. Each model has 3 identicial hidden layers. There are endless options for the number of hidden layers and their architecture, in this post we aren't going to extensivley test and optimise as this post is primarliy about building a basic an NLP model (also due to computational resource limitations).

### Train and Test sets
- First we create our train and test sets from our padded sequences arrays, both for tweets and keywords


```python
train_index = int(len(df) * 0.8)

x_train = tweet_pad[:train_index]
x_test = tweet_pad[train_index:]

x_train_kw = keyword_pad[:train_index]
x_test_kw = keyword_pad[train_index:]

y_train = df['relevant'].values[:train_index]
y_test = df['relevant'].values[train_index:]
```

### Metric Callback to record performance after each epoch

- *Credit to [this](https://stackoverflow.com/a/56485026) great Stackoverflow answer*. Next we create a callback that allows us to get an **f1-score (or any other metric) for our model after each epoch**. Natively Keras only records loss and accuarcy after each epoch. 

- It runs model.predict() method after each epoch, evaluates performance with sklearns classification_report and outputs into a list. We access the scores using the model.get() method

- We can also pass 2 *Metrics* objects in the same model to score both training set and validaiton set results separately and concurrently


```python
from tensorflow.keras.callbacks import Callback
from sklearn.metrics import recall_score, classification_report

# 1-directional

class Metrics(Callback):
    def __init__(self, x, y):
        self.x = x
        self.y = y if (y.ndim == 1 or y.shape[1] == 1) else np.argmax(y, axis=1)
        self.reports = []

    def on_epoch_end(self, epoch, logs={}):
        y_hat = np.asarray(self.model.predict(self.x))
        y_hat = np.where(y_hat > 0.5, 1, 0) if (y_hat.ndim == 1 or y_hat.shape[1] == 1)  else np.argmax(y_hat, axis=1)
        report = classification_report(self.y,y_hat,output_dict=True)
        self.reports.append(report)
        return
   
    # Utility method
    def get(self, metrics, of_class):
        return [report[str(of_class)][metrics] for report in self.reports]
 

```

## LSTM 1-directional model

Now we can build our LSTM models. We'll start with the 1-directional (left-to-right) model. We build separate models for Tweets and Keywords data, then we use Keras' concatenate function to combine them before compiling

- We start by creating our input layer. The dimensions should match our padded sequence (50 for Tweets data, 3 for Keywords)



```python
from tensorflow.keras.layers import concatenate
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Embedding,LSTM,Dense,SpatialDropout1D, Input, Bidirectional, Dropout
from tensorflow.keras.initializers import Constant
from sklearn.model_selection import train_test_split
from tensorflow.keras.optimizers import Adam

## Tweets input layer ##
input1 = Input(shape=(MAX_LEN,))

## KEYWORD MODEL ##
input2 = Input(shape=(MAX_LEN_KW,))

```

- Next we add our embeddings layer, to map our padded sequences to their gloVe vector representations. 
    - Note that the output_dim here should match the dimensions of your gloVe embeddings, in our case it's 25
    - trainable is set to false since the gloVe is a pretrained dataset


```python
# Tweets Embedding Layer
embedding_layer = Embedding(input_dim=num_words, output_dim=25, embeddings_initializer=Constant(embedding_matrix),
                   input_length=MAX_LEN, trainable=False)(input1)

# Keyword Embeddings layer
embedding_layer_kw = Embedding(input_dim=num_words_kw, output_dim=25, embeddings_initializer=Constant(embedding_matrix_kw),
                   input_length=MAX_LEN2, trainable=False)(input_kw)
```

- Then we add the hidden layers, as mentioned before there are limitless options for these, the 3 layers used are just one example.


```python
# Tweet Hidden layers 
tweet_1 = LSTM(64, dropout=0.2, recurrent_dropout=0.2,  return_sequences = True)(embedding_layer)
tweet_2 = LSTM(64, dropout=0.2, recurrent_dropout=0.2, return_sequences = True)(tweet_1)
tweet_3 = LSTM(64, dropout=0.2, recurrent_dropout=0.2, return_sequences=False)(tweet_2)

# Keyword Hidden layers
kw_1 = Dropout(0.2)(embedding_layer_kw)
kw_2 = LSTM(64, dropout=0.2, recurrent_dropout=0.2,  return_sequences = True)(kw_1)
kw_3 = LSTM(64, dropout=0.2, recurrent_dropout=0.2)(kw_2)
```

- Next we concatenate the models so we can run them concurrently and combine predictions


```python
## CONCATENATE ##
combo = concatenate([tweet_3, kw_3])
```

- Finally we add our Dense layer to ensure outputs are returned as 1-dimension (either 1/0 or a probability)
- And create our Model object with the expected input and output objects


```python
# Dense output layer
output = Dense(1, activation='sigmoid')(combo)

optimzer=Adam(learning_rate=1e-5)

model2 = Model(inputs=[input1, input2], outputs=output)
model2.summary()

model2.compile(loss='binary_crossentropy', optimizer=optimzer)
```

    Model: "model_1"
    __________________________________________________________________________________________________
    Layer (type)                    Output Shape         Param #     Connected to                     
    ==================================================================================================
    input_7 (InputLayer)            [(None, 50)]         0                                            
    __________________________________________________________________________________________________
    input_8 (InputLayer)            [(None, 3)]          0                                            
    __________________________________________________________________________________________________
    embedding_9 (Embedding)         (None, 50, 25)       453175      input_7[0][0]                    
    __________________________________________________________________________________________________
    embedding_10 (Embedding)        (None, 3, 25)        5775        input_8[0][0]                    
    __________________________________________________________________________________________________
    lstm_26 (LSTM)                  (None, 50, 64)       23040       embedding_9[0][0]                
    __________________________________________________________________________________________________
    dropout_1 (Dropout)             (None, 3, 25)        0           embedding_10[0][0]               
    __________________________________________________________________________________________________
    lstm_27 (LSTM)                  (None, 50, 64)       33024       lstm_26[0][0]                    
    __________________________________________________________________________________________________
    lstm_29 (LSTM)                  (None, 3, 64)        23040       dropout_1[0][0]                  
    __________________________________________________________________________________________________
    lstm_28 (LSTM)                  (None, 64)           33024       lstm_27[0][0]                    
    __________________________________________________________________________________________________
    lstm_30 (LSTM)                  (None, 64)           33024       lstm_29[0][0]                    
    __________________________________________________________________________________________________
    concatenate_1 (Concatenate)     (None, 128)          0           lstm_28[0][0]                    
                                                                     lstm_30[0][0]                    
    __________________________________________________________________________________________________
    dense_5 (Dense)                 (None, 1)            129         concatenate_1[0][0]              
    ==================================================================================================
    Total params: 604,231
    Trainable params: 145,281
    Non-trainable params: 458,950
    __________________________________________________________________________________________________
    

## Train Models

Now we can train the model we've just compiled. The epochs and bactch size are set according computational resource available, they can be tweaked as necessary.

Note that we intialise 2 **separate** Metrics objects. The first is to score the training set, the second for the validation set. 


```python
%%time

np.random.seed(175)

# Initialise Metrics callbacks
metrics = Metrics([x_train, x_train_kw], y_train)
metrics2 = Metrics([x_test, x_test_kw], y_test)

# Fit model
model2_fit = model2.fit([x_train, x_train_kw], 
                        y_train, 
                        batch_size=100,
                        epochs=15, 
                        validation_data=([x_test, x_test_kw], y_test), 
                        verbose=2,
                        callbacks=[metrics, metrics2])

print()
```

    Epoch 1/15
    87/87 - 75s - loss: 0.6924 - val_loss: 0.6935
    Epoch 2/15
    87/87 - 53s - loss: 0.6896 - val_loss: 0.6927
    Epoch 3/15
    87/87 - 54s - loss: 0.6855 - val_loss: 0.6910
    Epoch 4/15
    87/87 - 57s - loss: 0.6769 - val_loss: 0.6847
    Epoch 5/15
    87/87 - 46s - loss: 0.6541 - val_loss: 0.6504
    Epoch 6/15
    87/87 - 44s - loss: 0.6046 - val_loss: 0.5931
    Epoch 7/15
    87/87 - 49s - loss: 0.5657 - val_loss: 0.5635
    Epoch 8/15
    87/87 - 37s - loss: 0.5490 - val_loss: 0.5482
    Epoch 9/15
    87/87 - 41s - loss: 0.5386 - val_loss: 0.5336
    Epoch 10/15
    87/87 - 42s - loss: 0.5313 - val_loss: 0.5246
    Epoch 11/15
    87/87 - 56s - loss: 0.5248 - val_loss: 0.5140
    Epoch 12/15
    87/87 - 59s - loss: 0.5227 - val_loss: 0.5064
    Epoch 13/15
    87/87 - 51s - loss: 0.5147 - val_loss: 0.5042
    Epoch 14/15
    87/87 - 56s - loss: 0.5110 - val_loss: 0.4942
    Epoch 15/15
    87/87 - 67s - loss: 0.5080 - val_loss: 0.4931
    
    Wall time: 18min 33s
    

# Bidrectional LSTM model

The process for training the Bidirectional is the same as the 1-directional LSTM just with Bidirectional hidden layers instead. We'll just focus on the results here but the code is available in the notebook link at the end of the page.

    

## Performance

First we look at the loss function over each epoch. The one-directional model improves after each epoch before starting to plateau after 14 epochs. The bidirectional model has a worse starting position but evenutally the loss function improves more than the one-directional model. It also doesn't start to plataeu so if we were to increase the number of epochs we could see better  even performance



We can see that both models score similarly, the Bidirectional model ends slightly better but has far better performance over the first few epochs but plateaus quickly. The 1-directional model conversely starts worse but improves significantly, eventually reaching similar levels to the Bidirectional model

Looking at f1-score both models perform well, far better than the baseline indicating that both the models do a good job at both reducing False Positives (precision) and increasing recall. 



    
![png](../../images/nlp_disaster_tweets/output_57_0.png)
    


## Performance by Keyword
For different types of events it may be easier for the model to identify Relevant tweets given the langauge used for some types of disaster are more likely to appear in general conversation than others. For example *fire* can appear in many common non-disaster conversations whereas *suicide bomber* is less likely to.

Below we can see this reflected in the models score for tweets which contain certain keywords. The model performs better for tweets containing keywords such as *suicide bomber, wreckage, wounded, survived* which are less likely to appear in normal conversation. Whereas for keywords such as *threat, trauma, trouble* the model performs worse, this is likely because there are more contexts which these words appear in general conversation.

To improve performance the model can either fine-tune with different hidden layer combinations or collect more data.


    
![png](../../images/nlp_disaster_tweets/output_60_0.png)
    


# Conclusion
In this post we've explored a dataset of tweets and successfully trained an LSTM neural network to predict whether a tweet which contains language associated with disasters are actually about natural disasters or not

We've seen that it's more difficult to identify disaster tweets when the relevant keyword is more commonly appears in normal conversation. To further imrpove this model we can test different combinations of hidden layers such Convolutional Neural Networks, Recurrent Neural Networks or collect more data to fine tune the model for specific keywords.

Thanks for reading!
