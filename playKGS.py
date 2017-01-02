#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 22 08:33:38 2016

@author: nubby

This is the engine that will run on KGS using the kgsGTP java gateway application. See README for 
download details. This engine listens on standard in and writes to standard out in GTP (Go Text Protocol)

NB: Run with python -u when running from KGS gateway

A sample config file is here below:

name=xxxx
password=xxxx
room=Computer Go
mode=both
automatch.rank=20k
rules=japanese
rules.boardSize=9
rules.time=0
verbose=t
gameNotes=Neural net bot about 15 kyu.
engine=python -u /Users/nubby/Desktop/kgsGtp/playKGS.py

This file using GnuGo to establish the final board status of life and death and reports this to KGS to
remove dead stones from the board (final_status_list). Using GnuGo to establish dead stones at the end of a game is standard
practice and is allowed in ranked game and tournament play. Note that we never call genmove in GnuGo here. 
GnuGo does not generate any moves for the bot. We call the play command for both black and white stones so
that GnuGo has a record of the game as it progresses. 

(kgs-genmove_cleanup is not working correctly. This is supposed to be used to dispute dead stones at the end
of a game with the opponent. )

"""
import re
import sys
import tensorflow as tf
import numpy as np
import copy
import os
from subprocess import PIPE, Popen
from threading  import Thread

cwd = os.getcwd()
modelCheckpoint = cwd + '/checkpoints/model.ckpt'

BLACK = 1
WHITE = -1


letterDict = {'a':0, 'b':1,'c':2, 'd':3, 'e':4, 'f':5,'g':6,'h':7,'j':8,
              'A':0, 'B':1, 'C':2, 'D':3, 'E':4, 'F':5, 'G':6, 'H':7, 'J':8}
numberDict = {0:'a', 1:'b', 2:'c', 3:'d', 4:'e', 5:'f', 6:'g', 7:'h', 8:'j'}


prediction_array = np.zeros((1,83)).astype('float32') 
board_array = np.zeros((1,83)).astype('float32')   
omegaLastBoardPosition = np.zeros((1,83)).astype('float32')  
   

num_nodes = 1024

weights_1 = tf.Variable(tf.zeros([83, num_nodes]), name='weights_1')
biases_1 = tf.Variable(tf.zeros([num_nodes]), name='biases_1')
weights_2 = tf.Variable(tf.zeros([num_nodes, 83]), name='weights_2')
biases_2 = tf.Variable(tf.zeros([83]), name='biases_2')

saver = tf.train.Saver()
sess = tf.Session()
    
sys.stderr.write( "Loading in weights and biases.")
saver.restore(sess, modelCheckpoint)
sys.stderr.write("TensorFlow weights and biases restored.")

def enqueue_output(out, queue):
    for line in iter(out.readline, b''):
        queue.put(line)
    out.close()



try:
    from Queue import Queue, Empty
except ImportError:
    from queue import Queue, Empty  # python 3.x

ON_POSIX = 'posix' in sys.builtin_module_names

# You can set orientation from 0 to 7 for some variety in GnuGo's play. Level can be set from 0 to 20.
p = Popen(['/usr/local/bin/gnugo', '--mode', 'gtp', '--level', '0', '--orientation', '0'], stdout=PIPE, 
             stdin=PIPE, bufsize=1, close_fds=ON_POSIX)

q = Queue()
t = Thread(target=enqueue_output, args=(p.stdout, q))
t.daemon = True # thread dies with the program
t.start()



    
def checkMoveLegal(board_array, move, player):

    global omegaColor
    global opponentColor

    board_copy = copy.deepcopy(board_array)

    
    if (board_copy[move] == BLACK or board_copy[move] == WHITE):
        return False
    
    alreadyChecked = []
    group = []
    hasLiberties = False
    checkSide = 0
    
    if (player == "omega"):
        board_copy[move] = omegaColor
        checkSide = omegaColor
    elif (player == "opponent"):
        checkSide = opponentColor
        board_copy[move] = opponentColor


    board_copy = removeDeadStones(board_copy, player)

    board_copy = board_copy[:-2].astype('int')
    board_copy = board_copy.reshape(9,9)
    
    column = move % 9
    row = move / 9
  
   
    hasLiberties = checkGroup(checkSide, (row, column), alreadyChecked, group, hasLiberties, board_copy)

    del board_copy
    
    if (hasLiberties == False):
        return False
    else:
        return True



def removeDeadStones(board_array, player):

    global omegaColor
    global opponentColor
    
    board_copy = copy.deepcopy(board_array)
    
    board_copy = board_copy[:-2].astype('int')
    board_copy = board_copy.reshape(9,9)

    if (player == "omega"):
        removeSide = opponentColor
    elif (player == "opponent"):
        removeSide = omegaColor
    else:
        sys.stderr.write("Unknown player")
        return board_copy
        
    alreadyChecked = []    

    lastPointReached = False
    while lastPointReached == False:
        
        groupHead = 0
        group = []
        hasLiberties = False
    
        #Get the next groupHead
        for row in range(9):
            groupHeadFound = False
            for column in range(9):
                if (row == 8 and column == 8):
                    lastPointReached = True
                if (board_copy[row][column] == removeSide):
                    if ((row,column) in alreadyChecked):
                        continue
                    else:
                        groupHead = (row,column)
                        groupHeadFound = True
                        break
            if groupHeadFound == True:
                break

        if (groupHeadFound == True):
            hasLiberties = checkGroup(removeSide, groupHead, alreadyChecked, group, hasLiberties, board_copy)
            if (hasLiberties == False):
                for tupe in group:
                    row = tupe[0]
                    column = tupe[1]
                    board_copy[row][column] = 0
            #    print "Found group to be removed: ", group

        #End of groupHeadFound == True

    #End of while loop until lastPointReached
    
    board_copy = board_copy.reshape(81)
    board_copy = np.append(board_copy, [0,0])
    return board_copy

#End of removeDeadStones()

def checkGroup(checkSide, tupe, alreadyChecked, group, hasLiberties, board_copy):
    
    alreadyChecked.append(tupe)
    group.append(tupe)
    
    
    row = tupe[0]
    column = tupe[1]
    
    if (row < 8 and ((row+1,column) not in alreadyChecked)):
        if (board_copy[row+1][column] == checkSide): 
            hasLiberties = checkGroup(checkSide, (row+1,column), alreadyChecked, group, hasLiberties, board_copy)
        elif (board_copy[row+1][column] == 0):
            hasLiberties = True

    if (row > 0 and ((row-1,column) not in alreadyChecked)):
        if (board_copy[row-1][column] == checkSide):
            hasLiberties = checkGroup(checkSide, (row-1,column), alreadyChecked, group, hasLiberties, board_copy)
        elif (board_copy[row-1][column] == 0):
            hasLiberties = True
            
    if (column < 8 and ((row,column+1) not in alreadyChecked)):
        if (board_copy[row][column+1] == checkSide):
            hasLiberties = checkGroup(checkSide, (row,column+1), alreadyChecked, group, hasLiberties, board_copy)
        elif (board_copy[row][column+1] == 0):
            hasLiberties = True            

    if (column > 0 and ((row,column-1) not in alreadyChecked)):
        if (board_copy[row][column-1] == checkSide):
            hasLiberties = checkGroup(checkSide, (row,column-1), alreadyChecked, group, hasLiberties, board_copy)
        elif (board_copy[row][column-1] == 0):
            hasLiberties = True  
            
    return hasLiberties


def GTPToArray(moveGTP): 
    moveArray = list(moveGTP)
    xLetter = moveArray[0]
    yNumber = int(moveArray[1])
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

    
def getOmegaMove():
    
    
    global omegaColor
    global opponentColor
    
    global omegaLastBoardPosition
    choiceCount = 1
    
    #Board must be fed to tensorflow with black next to move regardless of
    #what color black is playing
    tensor_board = copy.deepcopy(board_array)
    if (omegaColor == WHITE):
        tensor_board *= -1
    
    #Get the prediction for next move
    test_prediction =  tf.nn.softmax(tf.matmul(tf.nn.relu(tf.matmul(tensor_board, weights_1) + biases_1), 
                          weights_2) + biases_2)
       
    del tensor_board
    
    prediction_array = test_prediction.eval(session=sess) 
    
    while True:

        if (choiceCount > 82):
            print "No more moves possible. "
            break
        
        omegaMove = np.argsort(prediction_array)[0][-choiceCount]
            
        if (omegaMove == 81):
            
            omegaLastBoardPosition = copy.deepcopy(board_array)
    
            return "pass"
        
        elif (omegaMove == 82):
            
            return "resign"

            
        omegaMoveGTP = arrayToGTP(omegaMove)

        
        if (board_array[0][omegaMove] != 0):

            sys.stderr.write( "Illegal omega move: point already taken: " + omegaMoveGTP)
            choiceCount += 1
            continue
        
        elif (checkMoveLegal(board_array[0], omegaMove, "omega") == False):
          
            sys.stderr.write("Illegal omega move: suicide rule violated: " + omegaMoveGTP)
            choiceCount += 1
            continue
            
        proposedPosition = copy.deepcopy(board_array)
        proposedPosition[0][omegaMove] = omegaColor
        proposedPosition[0] = removeDeadStones(proposedPosition[0], "omega")
    
        if (np.array_equal(proposedPosition[0],omegaLastBoardPosition[0])):
    
            sys.stderr.write("Illegal omega move: KO rule violation: " + omegaMoveGTP)
            choiceCount += 1
            continue
        
        else:         
            
            
            print >> sys.stderr, board_array[0]

            board_array[0][omegaMove] = omegaColor

            print >> sys.stderr, board_array[0]

            board_array[0] = removeDeadStones(board_array[0], "omega")
            sys.stderr.write( "Omega move: " + omegaMoveGTP)
            
            
            print >> sys.stderr, board_array[0]

            omegaLastBoardPosition = copy.deepcopy(board_array)
            
            return omegaMoveGTP
            
    #End of while omegaToPlay == True
#End of getOmegaMove()    
 

# This needs to be fixed to pass when all opponent's dead stones have
# been removed from the board.
def getOmegaMoveNoPass():
    
    
    global omegaColor
    global opponentColor
    
    global omegaLastBoardPosition
    choiceCount = 1
    
    #Board must be fed to tensorflow with black next to move regardless of
    #what color black is playing
    tensor_board = copy.deepcopy(board_array)
    if (omegaColor == WHITE):
        tensor_board *= -1
    
    #Get the prediction for next move
    test_prediction =  tf.nn.softmax(tf.matmul(tf.nn.relu(tf.matmul(tensor_board, weights_1) + biases_1), 
                          weights_2) + biases_2)
       
    del tensor_board
    
    prediction_array = test_prediction.eval(session=sess) 
    
    while True:

        if (choiceCount > 82):
            print "No more moves possible. "
            break
        
        omegaMove = np.argsort(prediction_array)[0][-choiceCount]
            
        if (omegaMove == 81):
            
            result = areThereRemainingDeadOpponentStones()
            
            if (result == True):
                choiceCount += 1
                continue
            else:
                omegaLastBoardPosition = copy.deepcopy(board_array)
                return "pass"
        
        elif (omegaMove == 82):
            
            choiceCount += 1
            continue

            
        omegaMoveGTP = arrayToGTP(omegaMove)

        
        if (board_array[0][omegaMove] != 0):

            sys.stderr.write( "Illegal omega move: point already taken: " + omegaMoveGTP)
            choiceCount += 1
            continue
        
        elif (checkMoveLegal(board_array[0], omegaMove, "omega") == False):
          
            sys.stderr.write("Illegal omega move: suicide rule violated: " + omegaMoveGTP)
            choiceCount += 1
            continue
            
        proposedPosition = copy.deepcopy(board_array)
        proposedPosition[0][omegaMove] = omegaColor
        proposedPosition[0] = removeDeadStones(proposedPosition[0], "omega")
    
        if (np.array_equal(proposedPosition[0],omegaLastBoardPosition[0])):
    
            sys.stderr.write("Illegal omega move: KO rule violation: " + omegaMoveGTP)
            choiceCount += 1
            continue
        
        else:         
            
            
            print >> sys.stderr, board_array[0]

            board_array[0][omegaMove] = omegaColor

            print >> sys.stderr, board_array[0]

            board_array[0] = removeDeadStones(board_array[0], "omega")
            sys.stderr.write( "Omega move: " + omegaMoveGTP)
            
            
            print >> sys.stderr, board_array[0]

            omegaLastBoardPosition = copy.deepcopy(board_array)
            
            return omegaMoveGTP
            
    #End of while omegaToPlay == True
#End of getOmegaMoveNoPass()   

def areThereRemainingDeadOpponentStones():
    
    
    global opponentColor
    deadStones = getGnuDeadStones()
    
    try:
        m = re.findall(r"([A-HJa-hj])([1-9])", deadStones)
        
        for deadStone in m:
            xLetter = deadStone[0]
            yNumber = deadStone[1]
            gtpMove = xLetter + str(yNumber)
            move = GTPToArray(gtpMove)
            
            if (board_array[0][move] == opponentColor):
                print >> sys.stderr, "Found remaining dead opponent stone: " + gtpMove
                return True
        
        print >> sys.stderr, "Found no remaining dead opponent stones. Can pass"
        return False
            
    except Empty:
        return False

def setGnuBoardSize():
    
    p.stdin.write('boardsize 9\n')
    # read line without blocking
    try:  
        while True:
            resp = q.get(timeout=0.2)
            if (len(resp.strip()) > 0):
                 print >> sys.stderr, "Gnu board size set to 9x9"
    except Empty:
        pass           
              
def setGnuKomi(komi):
    p.stdin.write('komi ' + str(komi) + '\n')
    # read line without blocking
    try:  
        while True:
            resp = q.get(timeout=0.2)
            if (len(resp.strip()) > 0):
               if (resp.strip() != "="):
                   print >> sys.stderr, resp
    except Empty:
        pass  
    
def playGnuMove(move):
    
    p.stdin.write(move)
        
    # read line without blocking
    try:  
        while True:
            resp = q.get(timeout=0.2)
            if ("=" in resp):
                print >> sys.stderr, "Gnu move accepted."
            elif ("illegal move" in resp):
                print >> sys.stderr,  "Illegal Gnu move '" + move +"' -> '"+ resp + "'" 
            else:
                pass
    except Empty:
        pass
    
def getGnuDeadStones():
    
    deadStones = ''
    
    p.stdin.write('final_status_list dead\n')
    # read line without blocking
    try:  
        to = 2.0
        while True:

            resp = q.get(timeout=to)
            if (len(resp.strip()) > 0):
                deadStones = deadStones + resp
            to = 0.1
            
    except Empty:
        pass 
    
    if ("=" in deadStones):
        print >> sys.stderr, "GNU found these dead stones: '" + deadStones + "'"

    return deadStones         
        
setGnuBoardSize()   
setGnuKomi(5.5)

while True:
    
    kgsInput = raw_input()
    kgsInput = str(kgsInput)
    sys.stderr.write("Input: "+kgsInput+"\n")

            
    try:                  
        m = re.match(r"boardsize ([0-9]*)", kgsInput)
        s = m.group(1)
        size = int(s)
        if (size != 9):
            sys.stdout.write('? unacceptable size\n\n')
        else:
            sys.stdout.write('= \n\n')
            setGnuBoardSize()
            prediction_array = np.zeros((1,83)).astype('float32') 
            board_array = np.zeros((1,83)).astype('float32')   
            omegaLastBoardPosition = np.zeros((1,83)).astype('float32')
        continue
    except AttributeError: 
        pass
    
    try:                  
        m = re.search(r"name", kgsInput)
        if m:
            sys.stdout.write('= NubbyBot\n\n')
            continue
    except AttributeError: 
        pass
    

    try:                  
        m = re.search(r"version", kgsInput)
        if m:
            sys.stdout.write('= Version 1\n\n')
            continue
    except AttributeError: 
        pass

    try:                  
        m = re.search(r"time_left", kgsInput)
        if m:
            sys.stdout.write('=\n\n')
            continue
    except AttributeError: 
        pass

    
    
    try:
        m = re.match(r"komi ([0-9\.]*)", kgsInput)
        s = m.group(1)
        komi = float(s)
        sys.stderr.write("Komi is: " +  str(komi))
        sys.stdout.write('=\n\n')
        continue
    except AttributeError: 
        pass 
    
    try:                  
        m = re.match(r"genmove (black|white|b|w|B|W)", kgsInput)
        color = m.group(1)
        ######################################
        #sys.stdout.write('= pass\n\n') 
        
        if (color == 'b' or color == 'black' or color == 'B'):
            omegaColor = BLACK
            opponentColor = WHITE
        elif (color == 'w' or color == 'white' or color == 'W'):
            omegaColor = WHITE
            opponentColor = BLACK
        else:
            sys.stderr.write("OMEGAGO PLAYS UNKNOWN COLOR!")
         
        move = getOmegaMove()   
        
        gnuMove = "play " + color + " " + move + "\n"
        
        playGnuMove(gnuMove)

            
        sys.stdout.write('= ' + move + '\n\n')   
        continue
    
    except AttributeError: 
        pass   
    
    try:                  
        m = re.match(r"kgs-genmove_cleanup (black|white|b|w|B|W)", kgsInput)
        color = m.group(1)
        
        sys.stderr.write("IN KGS-GENMOVE_CLEANUP **********")
        
        if (color == 'b' or color == 'black' or color == 'B'):
            omegaColor = BLACK
            opponentColor = WHITE
        elif (color == 'w' or color == 'white' or color == 'W'):
            omegaColor = WHITE
            opponentColor = BLACK
        else:
            sys.stderr.write("OMEGAGO PLAYS UNKNOWN COLOR!")
         
        move = getOmegaMoveNoPass()   
        
        gnuMove = "play " + color + " " + move + "\n"
        
        playGnuMove(gnuMove)

            
        sys.stdout.write('= ' + move + '\n\n')   
        continue
    
    except AttributeError: 
        pass      
    
    
    
    try:                  
        m = re.search(r"list_commands", kgsInput)
        if m:
            sys.stdout.write('= play\nboardsize\ngenmove\nlist_commands\nname\nquit\nversion\nkomi\nkgs-genmove_cleanup\nfinal_status_list\ntime_left\n\n')   
    
            continue
    except AttributeError: 
        pass
   
    
    try:                  
        m = re.match(r"play (black|white|b|w|B|W) (pass|PASS)", kgsInput)
        color = m.group(1)

        
        if (color == 'b' or color == 'black' or color == 'B'):
            opponentColor = BLACK
            omegaColor = WHITE
        elif (color == 'w' or color == 'white' or color == 'W'):
            opponentColor = WHITE
            omegaColor = BLACK
        else:
            sys.stderr.write("opponent PLAYS UNKNOWN COLOR!")
            
        board_array[0][81] = opponentColor

        
        playGnuMove(kgsInput + "\n\n")

        sys.stdout.write('=\n\n')
        continue
    
    except AttributeError: 
        pass
    
    try:
        m = re.search(r"final_status_list dead", kgsInput)
        if m:
            
            deadStones = getGnuDeadStones()
            sys.stdout.write(deadStones + '\n\n')
            
            continue
    except AttributeError:
        pass
    
    try:
        m = re.search(r"quit", kgsInput)
        if m:
            sys.stdout.write('=\n\n')
            
            #Reset board
            prediction_array = np.zeros((1,83)).astype('float32') 
            board_array = np.zeros((1,83)).astype('float32')   
            omegaLastBoardPosition = np.zeros((1,83)).astype('float32')  
            #break
            continue
    except AttributeError:
        pass
    
    try:                  
        m = re.match(r"play (black|white|b|w|B|W) ([A-HJa-hj])([1-9])", kgsInput)
            
        color = m.group(1)
        
        if (color == 'b' or color == 'black' or color == 'B'):
            opponentColor = BLACK
            omegaColor = WHITE
        elif (color == 'w' or color == 'white' or color == 'W'):
            opponentColor = WHITE
            omegaColor = BLACK
        else:
            sys.stderr.write("opponent PLAYS UNKNOWN COLOR!")
         
        xLetter = m.group(2)
        yNumber = m.group(3)
        
        sys.stderr.write("xLetter: " + xLetter + " yNumber: " + yNumber)
        
        gtpMove = xLetter + str(yNumber)
        opponentMove = GTPToArray(gtpMove) 

        sys.stderr.write("opponentMove: " + str(opponentMove))
            
        board_array[0][opponentMove] = opponentColor

        #sys.stderr.write("after assigning board_array position to " + str(opponentColor))

        board_array[0]= removeDeadStones(board_array[0], "opponent")   

        playGnuMove(kgsInput + "\n\n")
        
        sys.stdout.write('=\n\n')
        continue
    
    except AttributeError: 
        sys.stderr.write("Unknown, sending ?\n")
        sys.stdout.write('? unknown command\n\n')
        
        
        
        
#End of while True