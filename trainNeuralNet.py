#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Dec  9 18:04:07 2016

@author: nubby
"""



import tensorflow as tf
from six.moves import cPickle as pickle
import numpy as np
import os
import glob

cwd = os.getcwd()
pickleRoot = cwd + '/pickles/'
mixedPickleRoot = cwd + '/pickles_mixed/'
checkpointFile = cwd + '/checkpoints/model.ckpt'
indexedCheckpointFile = cwd + '/checkpoints/model.ckpt.index'
csvFile = cwd + '/trainResults/trainResults.csv'

def loadPickle(pickleFile, dataset, labels):

    try:
        with open(pickleFile, 'rb') as f:
            
          print "Loading from ", pickleFile
          saved = pickle.load(f)
          datasetNew = saved['dataset'].astype('float32')
          labelsNew = saved['labels'].astype('float32')
    
          del saved  # hint to help gc free up memory
    

          print "Dataset shape: ", datasetNew.shape
          print "Labels shape: ", labelsNew.shape
          
          if (len(dataset) == 0):
              dataset = datasetNew
              labels = labelsNew
          else:
              dataset = np.concatenate((dataset, datasetNew))
              labels = np.concatenate((labels, labelsNew))
              print "Total so far - Dataset shape: ", dataset.shape
              print "Total so far - Labels shape: ", labels.shape
              
          return dataset, labels

    except Exception as e:
      print('Unable to load data from', pickleFile, ':', e)
      return dataset, labels
      
labels =[]
dataset = []

pickleFiles = glob.glob(pickleRoot + "*.pickle")

for pickleFile in pickleFiles:
    dataset, labels = loadPickle(pickleFile, dataset, labels)
    
mixedPickleFiles = glob.glob(mixedPickleRoot + "*.pickle")

for mixedPickleFile in mixedPickleFiles:
    dataset, labels = loadPickle(mixedPickleFile, dataset, labels)
    
print "Total dataset shape: ",  dataset.shape
print "Total labels shape: ", labels.shape

def randomize(dataset, labels):
  permutation = np.random.permutation(len(dataset)) 
  try:
      del shuffled_dataset
  except:
      pass
  shuffled_dataset = dataset[permutation]
  shuffled_labels = labels[permutation]
  return shuffled_dataset, shuffled_labels

randomize(dataset,labels)

data_size = len(dataset)
test_size = 10000
train_size = data_size - test_size

train_dataset = dataset[:train_size -1]
train_labels = labels[:train_size -1]
test_dataset = dataset[train_size:]
test_labels = labels[train_size: ]

print('Training:', train_dataset.shape, train_labels.shape)
print('Testing:', test_dataset.shape, test_labels.shape)  

"""
count = 0
print "Cases where RESIGN is true in train_labels"
for i in range(len(train_labels)):
    if (train_labels[i][82] == 1):
        count+= 1
print count  
"""
  
num_nodes = 1024
batch_size = 1000 

graph = tf.Graph()
with graph.as_default():

  # Input data. For the training data, we use a placeholder that will be fed
  # at run time with a training minibatch.
  tf_train_dataset = tf.placeholder(tf.float32,shape=(batch_size, 83))
  tf_train_labels = tf.placeholder(tf.float32, shape=(batch_size, 83))
  tf_test_dataset = tf.constant(test_dataset, dtype=tf.float32)

  # Variables.
  weights_1 = tf.Variable(tf.truncated_normal([83, num_nodes]), name='weights_1')
  biases_1 = tf.Variable(tf.zeros([num_nodes]), name='biases_1')
  weights_2 = tf.Variable(tf.truncated_normal([num_nodes, 83]), name='weights_2')
  biases_2 = tf.Variable(tf.zeros([83]), name='biases_2')


  
  # Training computation.
  relu_layer=tf.nn.relu(tf.matmul(tf_train_dataset, weights_1) + biases_1)
  logits = tf.matmul(relu_layer, weights_2) + biases_2
  loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits, tf_train_labels))
  
  # Optimizer.
  optimizer = tf.train.GradientDescentOptimizer(0.04).minimize(loss)
  
  # Predictions for the training and test data.
  train_prediction = tf.nn.softmax(logits)
  test_prediction =  tf.nn.softmax(
   tf.matmul(tf.nn.relu(tf.matmul(tf_test_dataset, weights_1) + biases_1), 
                                  weights_2) + biases_2)

num_steps = 16000001
data_size = len(train_dataset)

csv = open(csvFile, 'w')
trainingHeaders = "step,loss,batch_accuracy,test_accuracy\n"
csv.write(trainingHeaders)
      
def accuracy(predictions, labels):
  return (100.0 * np.sum(np.argmax(predictions,1) == np.argmax(labels,1)) / predictions.shape[0])

with tf.Session(graph=graph) as session:
    
    
  checkPointExists = os.path.isfile(checkpointFile) 
  indexedCheckPointExists = os.path.isfile(indexedCheckpointFile)
  if (checkPointExists == True or indexedCheckPointExists == True):
      saver = tf.train.Saver()
        
      print  "Loading in weights and biases."
      saver.restore(session, checkpointFile)
      print "TensorFlow weights and biases restored."
  else:
      print("Checkpoint file not found. So initializing new one.")        
      saver = tf.train.Saver([weights_1, biases_1, weights_2, biases_2]) 
      tf.global_variables_initializer().run()
        
  for step in range(num_steps):
      
    # Pick an offset within the training data, which has been randomized.
    offset = (step * batch_size) % (data_size - batch_size)
    if (offset < batch_size): #re-randomize the data after every pass through whole set
        train_dataset, train_labels = randomize(train_dataset, train_labels)
        print "re-randomizing...."
        
    # Generate a minibatch.
    batch_data = train_dataset[offset:(offset + batch_size),:]
    batch_labels = train_labels[offset:(offset + batch_size),:]
    # Prepare a dictionary telling the session where to feed the minibatch.
    # The key of the dictionary is the placeholder node of the graph to be fed,
    # and the value is the numpy array to feed to it.
    feed_dict = {tf_train_dataset : batch_data, tf_train_labels : batch_labels}
    _, l, predictions = session.run(
      [optimizer, loss, train_prediction], feed_dict=feed_dict)
    if (step % 500 == 0):
      print("Minibatch loss at step %d: %f" % (step, l))
      train_accuracy = accuracy(predictions, batch_labels)
      print("Minibatch accuracy: %.1f%%" % train_accuracy)
      test_accuracy = accuracy(test_prediction.eval(), test_labels)
      print("Test accuracy: %.1f%%" % test_accuracy)
      output = str(step) +  ',' +  str(l) + ',' + str(train_accuracy) + ',' + str(test_accuracy) + '\n'
      csv.write(output)

  print("Final test accuracy: %.1f%%" % accuracy(test_prediction.eval(), test_labels))
  testp = test_prediction.eval()
  
  # Save the variables to disk.
  save_path = saver.save(session, checkpointFile)
  print("Model saved in file: %s" % save_path)
  
csv.close()
print("Training stats saved in file: %s" % csv.name)

  
#Check on resigning condition: saved in 82nd column
count = 0
print "Cases where RESIGN prediction is greater than 0.1 in test predictions:"
for i in range(len(testp)):
    if (testp[i][82] > 0.1):
        count+= 1
print count

#Check on passing condition: saved in 81st column
count = 0
print "Cases where PASS prediction is greater than 0.5 in test predictions:"
for i in range(len(testp)):
    if (testp[i][81] > 0.5):
        count+= 1
print count


count = 0
print "Cases where RESIGN is true in test_labels"
for i in range(len(test_labels)):
    if (test_labels[i][82] == 1):
        count+= 1
print count

count = 0
print "Cases where PASS is true in test_labels"
for i in range(len(test_labels)):
    if (test_labels[i][81] == 1):
        count+= 1
print count