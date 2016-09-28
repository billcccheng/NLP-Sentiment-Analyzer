import sys
import getopt
import os
import math
import operator
import random
from collections import defaultdict
import copy
import numpy as np
import scipy as sp
import scipy.stats

class Perceptron:
  class TrainSplit:
    """Represents a set of training/testing data. self.train is a list of Examples, as is self.test. 
    """
    def __init__(self):
      self.train = []
      self.test = []

  class Example:
    """Represents a document with a label. klass is 'pos' or 'neg' by convention.
       words is a list of strings.
    """
    def __init__(self):
      self.klass = ''
      self.words = []


  def __init__(self):
    """Perceptron initialization"""
    #in case you found removing stop words helps.
    self.stopList = set(self.readFile('../data/english.stop'))
    self.numFolds = 10
    # self.map = defaultdict(int)
    self.weights = defaultdict(int)
    self.input = []
    self.output = []
    self.threshold = 0
    self.lrate = 0.1

  #############################################################################
  # TODO TODO TODO TODO TODO 
  # Implement the Perceptron classifier with
  # the best set of features you found through your experiments with Naive Bayes.

  def classify(self, words):
    """ TODO
      'words' is a list of words to classify. Return 'pos' or 'neg' classification.
    """
    # Write code here
    classify_map = defaultdict(int)
    for word in words:
      classify_map[word]+=1
    self.normalize(classify_map)
    return "pos" if self.output_res(classify_map) >= self.threshold else "neg"
  

  def addExample(self, klass, words):
    """
     * TODO
     * Train your model on an example document with label klass ('pos' or 'neg') and
     * words, a list of strings.
     * You should store whatever data structures you use for your classifier 
     * in the Perceptron class.
     * Returns nothing
    """

    # Write code here
    words = self.filterStopWords(words)
    word_map = defaultdict(int) 
    for word in words:
      word_map[word] += 1
      if word not in self.weights:
        self.weights[word] = random.random()
    self.normalize(word_map)
    self.input.append(copy.deepcopy(word_map))
    self.output.append(1 if klass == "pos" else -1) 

    pass
  
  def train(self, split, iterations):
    """
    * TODO 
    * iterates through data examples
    * TODO 
    * use weight averages instead of final iteration weights
    """
    for example in split.train:
      words = example.words
      self.addExample(example.klass, words)

    # for word_map in self.input:
    #   for key in word_map:
    #     if key not in self.weights:
    #       self.weights[key] = random.random()
    for i in range(iterations):
      totalError = 0
      for j in range(len(self.output)):
        out_res = self.output_res(self.input[j])
        error  = self.output[j] - out_res

        totalError+=error
        for word in self.input[j]:
          # print word
          # sys.exit("Break")
          delta = self.lrate * self.input[j][word] * error
          self.weights[word] += delta
      if not totalError:
        break

  def output_res(self, _input):
    _sum = 0.0
    # print _input
    # sys.exit()    
    for word in _input:
      _sum+=self.weights[word]*_input[word]
    # print _sum
    # sys.exit()
    return 1 if _sum > self.threshold else -1


  def normalize(self, words):
    factor=1.0/sum(words.itervalues())
    for k in words:
        words[k] = words[k]*factor

  # END TODO (Modify code beyond here with caution)
  #############################################################################
  
  
  def readFile(self, fileName):
    """
     * Code for reading a file.  you probably don't want to modify anything here, 
     * unless you don't like the way we segment files.
    """
    contents = []
    f = open(fileName)
    for line in f:
      contents.append(line)
    f.close()
    result = self.segmentWords('\n'.join(contents)) 
    return result

  
  def segmentWords(self, s):
    """
     * Splits lines on whitespace for file reading
    """
    return s.split()

  
  def trainSplit(self, trainDir):
    """Takes in a trainDir, returns one TrainSplit with train set."""
    split = self.TrainSplit()
    posTrainFileNames = os.listdir('%s/pos/' % trainDir)
    negTrainFileNames = os.listdir('%s/neg/' % trainDir)
    for fileName in posTrainFileNames:
      example = self.Example()
      example.words = self.readFile('%s/pos/%s' % (trainDir, fileName))
      example.klass = 'pos'
      split.train.append(example)
    for fileName in negTrainFileNames:
      example = self.Example()
      example.words = self.readFile('%s/neg/%s' % (trainDir, fileName))
      example.klass = 'neg'
      split.train.append(example)
    return split


  def crossValidationSplits(self, trainDir):
    """Returns a lsit of TrainSplits corresponding to the cross validation splits."""
    splits = [] 
    posTrainFileNames = os.listdir('%s/pos/' % trainDir)
    negTrainFileNames = os.listdir('%s/neg/' % trainDir)
    #for fileName in trainFileNames:
    for fold in range(0, self.numFolds):
      split = self.TrainSplit()
      for fileName in posTrainFileNames:
        example = self.Example()
        example.words = self.readFile('%s/pos/%s' % (trainDir, fileName))
        example.klass = 'pos'
        if fileName[2] == str(fold):
          split.test.append(example)
        else:
          split.train.append(example)
      for fileName in negTrainFileNames:
        example = self.Example()
        example.words = self.readFile('%s/neg/%s' % (trainDir, fileName))
        example.klass = 'neg'
        if fileName[2] == str(fold):
          split.test.append(example)
        else:
          split.train.append(example)
      splits.append(split)
    return splits
  
  
  def filterStopWords(self, words):
    """Filters stop words."""
    filtered = []
    for word in words:
      if not word in self.stopList and word.strip() != '':
        filtered.append(word)
    return filtered

def test10Fold(args, accuracy_res):
  pt = Perceptron()
  
  iterations = int(args[1])
  splits = pt.crossValidationSplits(args[0])
  avgAccuracy = 0.0
  fold = 0
  for split in splits:
    classifier = Perceptron()
    accuracy = 0.0
    classifier.train(split,iterations)
  
    for example in split.test:
      words = example.words
      guess = classifier.classify(words)
      if example.klass == guess:
        accuracy += 1.0

    accuracy = accuracy / len(split.test)
    avgAccuracy += accuracy
    print '[INFO]\tFold %d Accuracy: %f' % (fold, accuracy) 
    fold += 1
    accuracy_res.append(accuracy_res)
  avgAccuracy = avgAccuracy / fold
  print '[INFO]\tAccuracy: %f' % avgAccuracy
    
    
def classifyDir(trainDir, testDir,iter):
  classifier = Perceptron()
  trainSplit = classifier.trainSplit(trainDir)
  iterations = int(iter)
  classifier.train(trainSplit,iterations)
  testSplit = classifier.trainSplit(testDir)
  #testFile = classifier.readFile(testFilePath)
  accuracy = 0.0
  for example in testSplit.train:
    words = example.words
    guess = classifier.classify(words)
    if example.klass == guess:
      accuracy += 1.0
  accuracy = accuracy / len(testSplit.train)
  print '[INFO]\tAccuracy: %f' % accuracy

def mean_confidence_interval(data, confidence=0.95):
    a = 1.0*np.array(data)
    n = len(a)
    m, se = np.mean(a), scipy.stats.sem(a)
    h = se * sp.stats.t._ppf((1+confidence)/2., n-1)
    return m-h, m, m+h, h

    
def main():
  (options, args) = getopt.getopt(sys.argv[1:], '')
  accuracy_res = []
  if len(args) == 3:
    classifyDir(args[0], args[1], args[2])
  elif len(args) == 2:
    test10Fold(args, accuracy_res)
  print mean_confidence_interval(accuracy_res)
    
if __name__ == "__main__":
    main()
