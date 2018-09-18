

import os
import zipfile
import numpy as np
import tensorflow as tf
import spacy
import collections
import string
import math
import random
import re
from six.moves import xrange 

data_index = 0
vocabulary_size = 27000
def get_mean_context_embeds(embeddings, train_inputs):
    """
    :param embeddings (tf.Variable(shape=(vocabulary_size, embedding_size))
    :param train_inputs (tf.placeholder(shape=(batch_size, 2*skip_window))
    returns:
        `mean_context_embeds`: the mean of the embeddings for all context words
        for each entry in the batch, should have shape (batch_size,
        embedding_size)
    """
    # cpu is recommended to avoid out of memory errors, if you don't
    # have a high capacity GPU
    with tf.device('/cpu:0'):
        embed = tf.nn.embedding_lookup(embeddings, train_inputs)
        mean_context_embeds = tf.reduce_mean(embed, 1)
    return mean_context_embeds
def generate_batch(data, batch_size, num_samples, skip_window):
    """
    Generates a mini-batch of training data for the training CBOW
    embedding model.
    :param data (numpy.ndarray(dtype=int, shape=(corpus_size,)): holds the
        training corpus, with words encoded as an integer
    :param batch_size (int): size of the batch to generate
    :param skip_window (int): number of words to both left and right that form
        the context window for the target word.
    Batch is a vector of shape (batch_size, 2*skip_window), with each entry for the batch containing all the context words, with the corresponding label being the word in the middle of the context
    """
    global data_index   
    assert batch_size % num_samples == 0
    assert num_samples <= 2 * skip_window
    
    batch = np.ndarray(shape=(batch_size), dtype=np.int32)
    labels = np.ndarray(shape=(batch_size, 1), dtype=np.int32)
    span = 2 * skip_window + 1  # span is the width of the sliding window
    buffer = collections.deque(maxlen=span)
    if data_index + span > len(data):
        data_index = 0
    buffer.extend(data[data_index:data_index + span]) # initial buffer content = first sliding window
    
    #print('data_index = {}, buffer = {}'.format(data_index, [reverse_dictionary[w] for w in buffer]))

    data_index += span
    for i in range(batch_size // num_samples):
        context_words = [w for w in range(span) if w != skip_window]
        random.shuffle(context_words)
        words_to_use = collections.deque(context_words) # now we obtain a random list of context words
        for j in range(num_samples): # generate the training pairs
            batch[i * num_samples + j] = buffer[skip_window]
            context_word = words_to_use.pop()
            labels[i * num_samples + j, 0] = buffer[context_word] # buffer[context_word] is a random context word
        
        # slide the window to the next position    
        if data_index == len(data):
            buffer = data[:span]
            data_index = span
        else: 
            buffer.append(data[data_index]) # note that due to the size limit, the left most word is automatically removed from the buffer.
            data_index += 1
        
        #print('data_index = {}, buffer = {}'.format(data_index, [reverse_dictionary[w] for w in buffer]))
        
    # end-of-for
    data_index = (data_index + len(data) - span) % len(data) # move data_index back by `span`
    return batch, labels
def adjective_embeddings(data_file, embeddings_file_name, num_steps, embedding_dim):
    pass # Remove this pass line, you need to implement your code for Adjective Embeddings here...
    global vocabulary_size
    words = data_file
    count =[['UNK',-1]]
    count.extend(collections.Counter(words).most_common(vocabulary_size - 1))
    dictionary = {}
    for word,_ in count:
        dictionary[word] = len(dictionary)
    data = list()
    unk_count = 0
    for word in words:
        index = dictionary.get(word, 0)
        if index == 0:  # i.e., one of the 'UNK' words
            unk_count += 1
        data.append(index)
    count[0][1] = unk_count
    reverse_dictionary = dict(zip(dictionary.values(), dictionary.keys()))
    print(len(reverse_dictionary))
    check_skip_window = 1     # How many words to consider left and right.
    batch, labels = generate_batch(data, batch_size = 8,num_samples=2, skip_window=check_skip_window)
#    for i in range(8):  
#        print(batch[i, :], [reverse_dictionary[batch[i, j]] for j in range(check_skip_window*2)],
#          '->', labels[i, 0], reverse_dictionary[labels[i, 0]])

    batch_size = 128
    embedding_size = embedding_dim  # Dimension of the embedding vector.
    skip_window = 1       # How many words to consider left and right.
    
    # We pick a random validation set to sample nearest neighbors. Here we limit the
    # validation samples to the words that have a low numeric ID, which by
    # construction are also the most frequent.
    valid_size = 16     # Random set of words to evaluate similarity on.
    valid_window = 100  # Only pick dev samples in the head of the distribution.
    valid_examples = np.random.choice(valid_window, valid_size, replace=False)
    num_samples = 2
    num_sampled = 64    # Number of negative examples to sample.
    
    graph = tf.Graph()
    
    with graph.as_default():
    
        # Input data.
        train_inputs = tf.placeholder(tf.int32, shape=[batch_size, 2*skip_window])
        train_labels = tf.placeholder(tf.int32, shape=[batch_size, 1])
        valid_dataset = tf.constant(valid_examples, dtype=tf.int32)
    
        # Ops and variables pinned to the CPU because of missing GPU implementation
        with tf.device('/cpu:0'):
            # Look up embeddings for inputs.
            embeddings = tf.Variable(
                tf.random_uniform([vocabulary_size, embedding_size], -1.0, 1.0))
    
            # train_inputs is of shape (batch_size, 2*skip_window)
            mean_context_embeds =\
                get_mean_context_embeds(embeddings, train_inputs)
    
            # Construct the variables for the NCE loss
            weights = tf.Variable(
                tf.truncated_normal([vocabulary_size, embedding_size],
                                    stddev=1.0 / math.sqrt(embedding_size)))
            biases = tf.Variable(tf.zeros([vocabulary_size]))
    
            # Compute the average NCE loss for the batch.
            # tf.nce_loss automatically draws a new sample of the negative labels each
            # time we evaluate the loss.
            loss = tf.reduce_mean(
                tf.nn.sampled_softmax_loss(weights=weights,
                               biases=biases,
                               labels=train_labels,
                               inputs=mean_context_embeds,
                               num_sampled=num_sampled,
                               num_classes=vocabulary_size))
    
            # Construct the SGD optimizer using a learning rate of 1.0.
            optimizer = tf.train.AdamOptimizer(0.001).minimize(loss)
    
            # Compute the cosine similarity between minibatch examples and all embeddings.
            norm = tf.sqrt(tf.reduce_sum(tf.square(embeddings), 1, keep_dims=True))
            normalized_embeddings = embeddings / norm
            valid_embeddings = tf.nn.embedding_lookup(
                normalized_embeddings, valid_dataset)
            similarity = tf.matmul(
                valid_embeddings, normalized_embeddings, transpose_b=True)
    
        # Add variable initializer.
        init = tf.global_variables_initializer()

    
    
    with tf.Session(graph=graph) as session:
        # We must initialize all variables before we use them.
        init.run()
        print('Initialized')
    
        average_loss = 0
        for step in xrange(num_steps):
            batch_inputs, batch_labels = generate_batch(data, batch_size,num_samples, skip_window)
            feed_dict = {train_inputs: batch_inputs, train_labels: batch_labels}
    
            # We perform one update step by evaluating the optimizer op (including it
            # in the list of returned values for session.run()
            _, loss_val = session.run([optimizer, loss], feed_dict=feed_dict)
            average_loss += loss_val
    
            if step % 2000 == 0:
                if step > 0:
                    average_loss /= 2000
                # The average loss is an estimate of the loss over the last 2000 batches.
                print('Average loss at step ', step, ': ', average_loss)
                average_loss = 0
    
            # Note that this is expensive (~20% slowdown if computed every 500 steps)
            if step % 10000 == 0:
                sim = similarity.eval()
                for i in range(valid_size):
                    valid_word = reverse_dictionary[valid_examples[i]]
                    top_k = 8  # number of nearest neighbors
                    nearest = (-sim[i, :]).argsort()[1:top_k + 1]
                    log_str = 'Nearest to %s:' % valid_word
                    for k in range(top_k):
                        close_word = reverse_dictionary[nearest[k]]
                        log_str = '%s %s,' % (log_str, close_word)
                    print(log_str)
                np.save("CBOW_Embeddings", normalized_embeddings.eval())
                #saver.save(session, 'w2vEmbedding', global_step=step)
    
        final_embeddings = normalized_embeddings.eval()
        np.save("CBOW_Embeddings", final_embeddings)

        
    

def process_data(input_data):
    
    words = []
    translator = str.maketrans('\n', ' ', string.punctuation)
    nlp = spacy.load('en')
    with zipfile.ZipFile(input_data) as f:
        #print(f.read(f.namelist()))
        for file in f.namelist():
            words.extend(tf.compat.as_str(f.read(file)).translate(translator).split())

            #print(doc)
            #words.extend(doc)
        print(words[-1])
    return words
    #print(reverse_dictionary)
        


def Compute_topk(model_file, input_adjective, top_k):
    pass # Remove this pass line, you need to implement your code to compute top_k words similar to input_adjective
