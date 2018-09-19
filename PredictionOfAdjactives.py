#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 14 15:08:50 2017

@author: yy
"""
import codecs
import gensim
import zipfile
import numpy as np
import tensorflow as tf
import spacy
import collections
import math
import random

data_index = 0
nlp = spacy.load('en')
vocabulary_size = 10000
entity = {'Person':('Entity','NOUN'),'Norp':('Entity','NOUN'),'Facility':('Entity','NOUN'),'org':('Entity','NOUN'),'gpe':('Entity','NOUN'),'loc':('Entity','NOUN'),'product':('Entity','NOUN'),'event':('Entity','NOUN'),'-PRON-':('-PRON-','PRON'),'law':('Entity','NOUN'),'Language':('entity','NOUN'),'date':('Date', 'NOUN'),'time':('Time','NOUN'),'percent':('Percent','NOUN'),'money':('Money','NOUN'),'quantity':('Quantity','NOUN'),'ordinal':('Ordinal','NOUN'),'cardinal':('Cardinal','NOUN'),'work_of_art':('Entity','NOUN'),'m': ('Money','NOUN')}
def generate_batch(data, batch_size,num_samples, skip_window):
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
    with open(data_file, 'r',encoding = 'utf8') as f:
        data_string=tf.compat.as_str(f.read())
    data = data_string.split()
    global vocabulary_size
    #vocabulary_size = len(collections.Counter(data))
    #Modified reverse_dictionary contains ('word','pos_')
    words = []
    for token in nlp(data_string):
        if not token.text.isalpha():
            continue
        if token.text in entity:
            words.append(entity[token.text])
    #Replace numer and symbol and punt.
        elif token.pos_ == 'NUM':
            words.append(('Num','NUM'))
        elif token.pos_ == 'SYM':
            words.append(('SYM','SYM'))
        elif(token.pos_ != 'PUNCT'):
            words.append((token.text,token.pos_))
    count = [['UNK', -1]]
    count.extend(collections.Counter(words).most_common(vocabulary_size - 1))
    dictionary = dict()
    for word, _ in count:
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
#print(reverse_dictionary)
    check_skip_window = 2      # How many words to consider left and right.
    batch, labels = generate_batch(data, batch_size=8, num_samples = 4,skip_window=check_skip_window)
    batch_size = 128      # Size of mini-batch for skip-gram model.
    embedding_size = embedding_dim  # Dimension of the embedding vector.
    skip_window = 2       # How many words to consider left and right of the target word.
    num_samples = 4         # How many times to reuse an input to generate a label.
    num_sampled = 40      # Sample size for negative examples.
    logs_path = './log/'

# Specification of test Sample:
    sample_size = 20       # Random sample of words to evaluate similarity.
    sample_window = 100    # Only pick samples in the head of the distribution.
    sample_examples = np.random.choice(sample_window, sample_size, replace=False) # Randomly pick a sample of size 16
    
## Constructing the graph...
    graph = tf.Graph()
    with graph.as_default():
        with tf.device('/cpu:0'):
            # Placeholders to read input data.
            with tf.name_scope('Inputs'):
                train_inputs = tf.placeholder(tf.int32, shape=[batch_size])
                train_labels = tf.placeholder(tf.int32, shape=[batch_size, 1])
                
            # Look up embeddings for inputs.
            with tf.name_scope('Embeddings'):            
                sample_dataset = tf.constant(sample_examples, dtype=tf.int32)
                embeddings = tf.Variable(tf.random_uniform([vocabulary_size, embedding_size], -1.0, 1.0))
                embed = tf.nn.embedding_lookup(embeddings, train_inputs)
                
                # Construct the variables for the NCE loss
                nce_weights = tf.Variable(tf.truncated_normal([vocabulary_size, embedding_size],
                                                          stddev=1.0 / math.sqrt(embedding_size)))
                nce_biases = tf.Variable(tf.zeros([vocabulary_size]))
            
            # Compute the average NCE loss for the batch.
            # tf.nce_loss automatically draws a new sample of the negative labels each
            # time we evaluate the loss.
            with tf.name_scope('Loss'):
                loss = tf.reduce_mean(tf.nn.sampled_softmax_loss(weights=nce_weights, biases=nce_biases,
                                                 labels=train_labels, inputs=embed, 
                                                 num_sampled=num_sampled, num_classes=vocabulary_size))
            
            # Construct the Gradient Descent optimizer using a learning rate of 0.01.
            with tf.name_scope('Adam'):
                optimizer = tf.train.AdamOptimizer(0.003).minimize(loss)
    
            # Normalize the embeddings to avoid overfitting.
            with tf.name_scope('Normalization'):
                norm = tf.sqrt(tf.reduce_sum(tf.square(embeddings), 1, keep_dims=True))
                normalized_embeddings = embeddings / norm
                
            sample_embeddings = tf.nn.embedding_lookup(normalized_embeddings, sample_dataset)
            similarity = tf.matmul(sample_embeddings, normalized_embeddings, transpose_b=True)
            
            # Add variable initializer.
            init = tf.global_variables_initializer()
            
            
            # Create a summary to monitor cost tensor
            tf.summary.scalar("cost", loss)
            # Merge all summary variables.
            merged_summary_op = tf.summary.merge_all()

    with tf.Session(graph=graph) as session:
        # We must initialize all variables before we use them.
        session.run(init)
        summary_writer = tf.summary.FileWriter(logs_path, graph=tf.get_default_graph())
        
        print('Initializing the model')
        
        average_loss = 0
        for step in range(num_steps):
            batch_inputs, batch_labels = generate_batch(data,batch_size, num_samples, skip_window)
            feed_dict = {train_inputs: batch_inputs, train_labels: batch_labels}
            
            # We perform one update step by evaluating the optimizer op using session.run()
            _, loss_val, summary = session.run([optimizer, loss, merged_summary_op], feed_dict=feed_dict)
            
            summary_writer.add_summary(summary, step )
            average_loss += loss_val
            
            if step % 2000 == 0:
                if step > 0:
                    average_loss /= 2000
                    # The average loss is an estimate of the loss over the last 5000 batches.
                    print('Average loss at step ', step, ': ', average_loss)
                    average_loss = 0
    
            # Evaluate similarity after every 10000 iterations.
            if step % 10000 == 0:
                sim = similarity.eval() #
                for i in range(sample_size):
                    sample_word = reverse_dictionary[sample_examples[i]][0]
                    top_k = 10  # Look for top-10 neighbours for words in sample set.
                    nearest = (-sim[i, :]).argsort()[1:top_k + 1]
                    log_str = 'Nearest to %s:' % sample_word
                    for k in range(top_k):
                        close_word = reverse_dictionary[nearest[k]][0]
                        log_str = '%s %s,' % (log_str, close_word)
                    print(log_str)
                print()

        final_embeddings = normalized_embeddings.eval()
        index = []
        output = ''
        size_line = ''
        for i in range(len(reverse_dictionary)):
            if reverse_dictionary[i][1] == 'ADJ' and reverse_dictionary[i][0] not in entity:
                index.append(i)
        size_line = '{} {}\n'.format(int(len(index)- 1), int(embedding_dim))
        for i in range(1,len(index)):
            output = output + reverse_dictionary[index[i]][0] + ' '
            for j ,vector in  enumerate(final_embeddings[index[i]]):
                if(j != embedding_dim - 1):
                    output = output + '{} '.format(vector)
                else:
                    output = output + '{}\n'.format(vector)
        output = size_line + output  
        with codecs.open(embeddings_file_name, 'w',encoding='utf8') as f:
            f.write(output)
        f.close()
#Not sure about which writing method is better cause of float accuracy.

#        with codecs.open(embeddings_file_name, 'w',encoding='utf8') as f:
#            print(int(len(index)- 1),file =f ,end = ' ')
#            print(int(embedding_dim),file= f,end = '\n')
#            for i in range(1,len(index)):
#                print(reverse_dictionary[index[i]][0], file =f ,end = ' ')
#                for j ,vector in  enumerate(final_embeddings[index[i]]):
#                    if (j != embedding_dim - 1):
#                        print(vector, file = f,end = ' ')
#                    else:
#                        print(vector, file = f,end = '\n')
#        f.close()
        
def process_data(input_data): 
    words = []
    with zipfile.ZipFile(input_data) as f:
        for file in f.namelist():
            doc = nlp(tf.compat.as_str(f.read(file)))
            tokens = []
            if(doc):
                sentence = ""
                #Replace token with their ent_type_ if it has one.
                for n,token in enumerate(doc):
                    if n!=0:
                        sentence += " "
                    if (token.ent_type_):
                        sentence += token.ent_type_
                    else:
                        sentence += token.text
                doc1 = nlp(sentence)
                #Lemmalization
                for n,token in enumerate(doc1):
                    if '\n' not in token.lemma_:
                        tokens.append(token.lemma_)
                words.extend(tokens)
    #Write file as required
    output_file = input_data[:-4]+".txt"
    output = open(output_file, 'w')
    for token in words:
        output.write(token + " ")
    return output_file
def Compute_topk(model_file, input_adjective, top_k):
    model = gensim.models.KeyedVectors.load_word2vec_format(model_file, binary=False)
    return [a for a, b in model.most_similar(positive=[input_adjective], topn=top_k)]
