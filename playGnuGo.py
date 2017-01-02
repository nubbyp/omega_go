#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 21 08:57:19 2016

@author: nubby

This file allows you to play multiple games against GnuGo at once. I have it
set to play 50 games with one run of this file - ten at a time in ten
separate threads.This can be adjusted of course.

It is not possible to convert this to run in 10 separate processes due 
to the fact that tensorFlow checkpoint files are not serializable. In any case this is not
necessary. The processing power needed to run each thread is small.

At the top of this file you can set OmegaGo's color and GnuGo's color. You can also set a handicap.
Komi is adjusted accordingly.

This file assumes your GnuGo application is in /usr/local/bin/gnugo. Adjust the code if it is
saved elsewhere on your machine. 

It outputs stats on games won and lost. 
"""

import sys
from subprocess import PIPE, Popen
from threading  import Thread
import tensorflow as tf
import numpy as np
import re
import os
import copy
from random import randint
import threading
import queue

cwd = os.getcwd()
modelCheckpoint = cwd + '/checkpoints/model.ckpt'

try:
    from Queue import Queue, Empty
except ImportError:
    from queue import Queue, Empty  # python 3.x

threadCount = 10
totalGamesToPlay = 50
    
BLACK = 1
WHITE = -1

omegaColor = BLACK
gnuColor = WHITE
handicap = 0

if (omegaColor == WHITE):
    print "GnuGo is black. OmegaGo is white"
else:
    print "OmegaGo is black. GnuGo is white"

print "Handicap is: ", str(handicap)

print str(totalGamesToPlay) + " games will be played."
    
allowPass = True
limitPass = False
passThreshold = 0.5

allowResign = True
limitResign = True
resignThreshold = 0.4

ON_POSIX = 'posix' in sys.builtin_module_names

letterDict = {'A':0, 'B':1,'C':2, 'D':3, 'E':4, 'F':5,'G':6,'H':7,'J':8}
numberDict = {0:'A', 1:'B', 2:'C', 3:'D', 4:'E', 5:'F', 6:'G', 7:'H', 8:'J'}

def GTPToArray(gnuMoveGTP): 
    gnuMoveArray = list(gnuMoveGTP)
    xLetter = gnuMoveArray[0]
    yNumber = int(gnuMoveArray[1])
    x = letterDict[xLetter]
    y = yNumber - 1
    return x+(y*9) 
    
def arrayToGTP(idx):
    x = idx % 9
    y = idx / 9
    xLetter = numberDict[x]
    yNumber = y + 1
    result = xLetter + str(yNumber)
    return result


def enqueue_output(out, queue):
    for line in iter(out.readline, b''):
        queue.put(line)
    out.close()


    
def playOmegaMove(omegaMoveGTP, justReadMove,p,q):
    
    if (omegaColor == BLACK):
        color = "black"
    else:
        color = "white"
    
    if (justReadMove == False):
        p.stdin.write('play ' + color + ' ' + omegaMoveGTP + '\n')
        
    # read line without blocking
    try:  
        while True:
            
            #line = q.get_nowait() 
            resp = q.get(timeout=0.3)
           # print "omegaMove resp: ", resp
            if ("=" in resp):
            #    print "Omega's move accepted."
                return "true"
            elif ("illegal move" in resp):
             #   print "Illegal move: ", omegaMoveGTP
                return "false"
            else:
                print "Unknown response to omega move: ", resp
                return "unknown"
    except Empty:
        print "In Empty in playOmegaMove"
        return "unknown"
      

def setBoardSize(p,q):
    
    p.stdin.write('boardsize 9\n')
    # read line without blocking
    try:  
        while True:
            resp = q.get(timeout=0.2)
            if (len(resp.strip()) > 0):
          #       print "board size set to 9x9"
              pass
    except Empty:
        pass           
         
        
def setKomi(komi,p,q):
    p.stdin.write('komi ' + str(komi) + '\n')
    # read line without blocking
    try:  
        while True:
            resp = q.get(timeout=0.2)
            if (len(resp.strip()) > 0):
               if (resp.strip() != "="):
                   print resp
    except Empty:
        pass       
    
def setHandicap(handicap,p,q):
  #  print "Handicap set to " + str(handicap) + " stones"
    p.stdin.write('fixed_handicap ' + str(handicap) + '\n')
    # read line without blocking
    try:  
        while True:
            resp = q.get(timeout=0.2)
            if (len(resp.strip()) > 0):
                # print "handicap points are : ", resp
                pass
    except Empty:
        pass        
    
    
def omegaPass(p,q):
    
        
    if (omegaColor == BLACK):
        color = "black"
    else:
        color = "white"
    
        
    p.stdin.write('play ' + color + ' pass\n')
    # read line without blocking
    try:  
        while True:
            resp = q.get(timeout=0.2)
            if (len(resp.strip()) > 0):
                # print "handicap points are : ", resp
                pass
    except Empty:
        pass       

def getGNUMove(justReadMove,p,q):
    
        
    if (gnuColor == BLACK):
        color = "black"
    else:
        color = "white"
    
    
    if (justReadMove == False):
        p.stdin.write('genmove ' + color + '\n')
        
    # read line without blocking
    try:  
        while True:
            
            #line = q.get_nowait() 
            resp = q.get(timeout=0.5)
         #   print "GNU Move: " , resp
            try:
                m = re.search(r"= ([A-HJ])([1-9])", resp)
                
                xLetter = m.group(1)
                yNumber = m.group(2)
                gtpMove = xLetter + yNumber
                return gtpMove
            except:
                if ("PASS" in resp or "pass" in resp):
                    return "pass"
                elif ("RESIGN" in resp or "resign" in resp):
                    m = re.search(r"= resign", resp)
                    return "resign"
                else:
                    print "Unknown format getting gnu move: ", resp
                    return "error"
    except Empty:
        print "Empty gnu move: "
        return "empty"   
        

        
def getAccurateBoard(p,q):
    
    accurateBoard = np.zeros((1,83)).astype('float32') 
    
    for idx in range(len(accurateBoard[0]) - 2):

        vertex = arrayToGTP(idx)
        
        p.stdin.write('color ' + vertex + '\n')
        # read line without blocking
        try:  
            while True:
                resp = q.get(timeout=0.05)
                if ("black" in resp):
                    position = BLACK
                    accurateBoard[0][idx] = position
                elif ("white" in resp):
                    position = WHITE
                    accurateBoard[0][idx] = position
                elif ("empty" in resp):
                    position = 0 
                    accurateBoard[0][idx] = position
                else:
                    pass

        except Empty:
            pass

            
    return accurateBoard   
    


    
def getScore(p,q):
    
    whoWon = ''
    
    p.stdin.write('final_score\n')
    # read line without blocking
    try:  
        while True:
            resp = q.get(timeout=0.5)
            if (len(resp.strip()) > 0):
                r = re.search(r".*([BW]).*", resp)
                whoWon = r.group(1)
    except Empty:
        pass
    
    return whoWon
        
 


def playOneGame(threadName, workerQueue):
    
    global whiteScore
    global blackScore
    
    while not exitFlag:
        
        
        queueLock.acquire()
        if not workQueue.empty():
            
            queueName = workerQueue.get()
            queueLock.release()
            threadLock.acquire(1)
            print "Starting ", queueName, " in thread: ", threadName
            threadLock.release()
            
            orientation = randint(0,7)
            # You can set orientation from 0 to 7 for some variety in GnuGo's play. Level can be set from 0 to 20.
            p = Popen(['/usr/local/bin/gnugo', '--mode', 'gtp', '--level', '0', '--orientation', str(orientation)], stdout=PIPE, 
                         stdin=PIPE, bufsize=1, close_fds=ON_POSIX)    
            q = Queue()
            t = Thread(target=enqueue_output, args=(p.stdout, q))
            t.daemon = True # thread dies with the program
            t.start()
            
            
            setBoardSize(p,q)
            
            gnuToPlay = False
            omegaToPlay = False
                
            if (gnuColor == BLACK):
                if (handicap > 1):
                    omegaToPlay = True
                else:
                    gnuToPlay = True
            else:
                if (handicap > 1):
                    gnuToPlay = True
                else:
                    omegaToPlay = True
              
            
            if (handicap > 1):
                setHandicap(handicap,p,q)
                
            if (handicap > 0):
                setKomi(0.5,p,q)
            else:
                setKomi(6.5,p,q)
        
            
            gnuPassed = False
            omegaPassed = False
            
            board_array = np.zeros((1,83)).astype('float32')  
            prediction_array = np.zeros((1,83)).astype('float32')  
                
            while True:
                
                choiceCount = 1
                justReadMove = False
                
                while omegaToPlay == True:
                    
                    board_array = getAccurateBoard(p,q)
                    # Need to add pass flag back in after getting board position from Gnu
                    if (gnuPassed == True):
                        board_array[0][81] = gnuColor
            
                    #Board must be fed to tensorflow with black next to move regardless of
                    #what color black is playing
                    tensor_board = copy.deepcopy(board_array) 
                    if (omegaColor == WHITE):
                        tensor_board *= -1
                        
                    #Get the prediction for next move
                    threadLock.acquire(1)
                    test_prediction =  tf.nn.softmax(tf.matmul(tf.nn.relu(tf.matmul(tensor_board, weights_1) + biases_1), 
                                          weights_2) + biases_2)
                       
                    del tensor_board
                    
                    prediction_array = test_prediction.eval(session=sess) 
                 #   print "choiceCount: ", choiceCount
                    threadLock.release()
            
                    
                    omegaMove = np.argsort(prediction_array)[0][-choiceCount]
            
            
                    if (choiceCount > 82):
                    #    print "No more moves possible. "
                        omegaToPlay = False
                        gnuToPlay = False
                    
                    # Pass
                    elif (omegaMove == 81):
            
                        #Always allow omega to pass second
                        if (allowPass == False and gnuPassed == False):
                         #   print "OmegaGo would have passed but not allowing it."
                            omegaToPlay = True
                            omegaPassed = False
                            choiceCount += 1
                        elif (gnuPassed == True):
                         #   print "OmegaGo passed"
                            omegaToPlay = False
                            omegaPass(p,q)
                         #   print "Game over"
                            gnuToPlay = False
                        elif (allowPass == True and limitPass == False):
                         #   print "OmegaGo passed"
                            omegaPass(p,q)
                            omegaPassed = True
                            omegaToPlay = False
                            board_array[0][81] = omegaColor
                            gnuToPlay = True
                        elif (allowPass == True and limitPass == True):
                            if (prediction_array[0][81] > 0.7):
                             #   print "OmegaGo allowed to pass because prediction > ", passThreshold, " ", prediction_array[0][81]
                             #   print "OmegaGo passed"
                                omegaPass(p,q)
                                omegaPassed = True
                                omegaToPlay = False
                                board_array[0][81] = omegaColor
                                gnuToPlay = True
                            else:
                              #  print "OmegaGo would have passed but prediction < ", passThreshold, " so not allowed:", prediction_array[0][81]
                                omegaToPlay = True
                                omegaPassed = False
                                choiceCount += 1
                        else:
                          #  print "Unexpected else in OmegaGo pass code"
                            omegaToPlay = True
                            omegaPassed = False
                            choiceCount += 1
                                
                    elif (omegaMove == 82):
                        if (allowResign == False):
                          #  print "OmegaGo would have resigned but not allowing it."
                            omegaToPlay == True
                            choiceCount += 1
                            omegaPassed = False
                        elif (allowResign == True and limitResign == False):
                         #   print "OmegaGo resigned"
                         #   print "Game over"
                            #There is no way to tell GnuGo you resign
                            gnuToPlay = False
                            omegaToPlay = False
                        elif (allowResign == True and limitResign == True):
                            if (prediction_array[0][82] > resignThreshold):
                             #   print "OmegaGo allowed to resign because prediction >  ",resignThreshold, " ",  prediction_array[0][82]
                             #   print "OmegaGo resigned"
                             #   print "Game over"
                                #There is no way to tell GnuGo you resign
                                gnuToPlay = False
                                omegaToPlay = False
                            else:
                             #   print "OmegaGo would have resigned but prediction < ", resignThreshold, " so not allowed:", prediction_array[0][82]
                                omegaToPlay = True
                                omegaPassed = False
                                choiceCount += 1
                        else:
                         #   print "Unexpected else in OmegaGo resign code"
                            omegaToPlay = True
                            omegaPassed = False
                            choiceCount += 1
                    else:         
                        omegaMoveGTP = arrayToGTP(omegaMove)
                        #    print "omegaMove: ", omegaMoveGTP
                        legalMove = playOmegaMove(omegaMoveGTP, justReadMove,p,q)
            
                        if (legalMove == "false"):
                            omegaMoveGTP = arrayToGTP(omegaMove)
                            omegaToPlay = True
                            omegaPassed = False
                            choiceCount += 1
                            justReadMove = False
                          #  print "choiceCount = ", choiceCount
                        elif (legalMove == "true"):
                            omegaToPlay = False
                            omegaPassed = False
                            board_array = getAccurateBoard(p,q)
                         #   drawHeatMapBoard(board_array[0], prediction_array[0])  
                         #   showCaptures(p,q)                  
                         #   estimateScore(p,q)
                            gnuToPlay = True
                        else: # "unknown"
                            justReadMove = True
                            omegaToPlay = True
                            omegaPassed = False
            
            
                justReadMove = False
                while gnuToPlay == True:
                    
                    gnuMoveGTP = getGNUMove(justReadMove, p,q)
                    if (gnuMoveGTP == "error"):
                        gnuToPlay = True
                    elif (gnuMoveGTP == "empty"):
                        #Try read again
                        justReadMove = True
                        gnuToPlay = True
                    elif (gnuMoveGTP == "pass"):
                      #  print "gnu passes"
                        if (omegaPassed == True):
                          #  print "Game over"
                            gnuToPlay = False
                            omegaToPlay = False
                        else:
                            board_array[0][81] = gnuColor
                            gnuPassed = True
                            gnuToPlay = False
                            omegaToPlay = True
                    elif (gnuMoveGTP == "resign"):
                    #    print "gnu resigns"
                    #    print "game over"
                        gnuToPlay = False
                        omegaToPlay = False
                        
                    else:
                        gnuPassed = False
                        gnuToPlay = False
                        omegaToPlay = True
                        board_array = getAccurateBoard(p,q)
                     #   drawHeatMapBoard(board_array[0], prediction_array[0])  
                     #   showCaptures(p,q)                  
                     #   estimateScore(p,q)
            
                #End of while gnuToPlay
                
                
                if (gnuToPlay == False and omegaToPlay == False):
                    
                    winner = getScore(p,q)
                    
                    # Get lock to synchronize threads
                    threadLock.acquire(1)
                    if (winner == 'W'):
                        whiteScore += 1
                    elif (winner == 'B'):
                        blackScore += 1
                    print "Winner of game ", queueName, " was: ", winner
                    print "White Games Won So Far: ", whiteScore
                    print "Black Games Won So Far: ", blackScore  
                    threadLock.release()
                    
                    p.kill()
                    
                    workQueue.task_done()
                    break
            # end of while True
        else: # workQueue is empty
            queueLock.release() 
    #End of while not exitFlag
#End of playOneGame()  

exitFlag = 0
whiteScore = 0
blackScore = 0

class myThread (threading.Thread):
    def __init__(self, threadID, name, q):
        threading.Thread.__init__(self)
        self.threadID = threadID
        self.name = name
        self.q = q
    def run(self):
        playOneGame(self.name, self.q)


    
num_nodes = 1024

weights_1 = tf.Variable(tf.zeros([83, num_nodes]), name='weights_1')
biases_1 = tf.Variable(tf.zeros([num_nodes]), name='biases_1')
weights_2 = tf.Variable(tf.zeros([num_nodes, 83]), name='weights_2')
biases_2 = tf.Variable(tf.zeros([83]), name='biases_2')

saver = tf.train.Saver()
sess = tf.Session()
    
print "Loading in weights and biases..."
saver.restore(sess, modelCheckpoint)
print "Tensorflow weights and biases restored"

threadLock = threading.Lock()
threads = []
queueLock = threading.Lock()
workQueue = queue.Queue(100)


# Create new threads
for i in range(threadCount):
    thread = myThread(i+1, "Thread-" + str(i+1), workQueue)
    thread.start()
    threads.append(thread)
    

# Fill the queue
queueLock.acquire()
for g in range(totalGamesToPlay):
    workQueue.put("Game-" + str(g+1))
queueLock.release()


# Wait for queue to empty
while not workQueue.empty():
    pass


# Notify threads it's time to exit
exitFlag = 1


# Wait for all threads to complete
for t in threads:
    t.join()

print "White Total: ", whiteScore
print "Black Total: ", blackScore 