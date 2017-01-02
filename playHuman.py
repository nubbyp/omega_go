#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 21 20:23:40 2016

@author: nubby

This file allows you to play a single game against your robot. 

You can set the handicap from 0 to 4 stones. You can also set who plays which color at the top of this file. 

The human needs to input the moves as grid coordinates. 

"""


import sys
import tensorflow as tf
import numpy as np
import re
import copy
from matplotlib.pyplot import ion, show
import matplotlib.pyplot as plt
import os

cwd = os.getcwd()
modelCheckpoint = cwd + '/checkpoints/model.ckpt'

BLACK = 1
WHITE = -1

humanColor = WHITE
computerColor = BLACK
# Handicap can be 0,1,2,3,4
handicap = 1

letterDict = {'a':0, 'b':1,'c':2, 'd':3, 'e':4, 'f':5,'g':6,'h':7,'i':8}
numberDict = {0:'a', 1:'b', 2:'c', 3:'d', 4:'e', 5:'f', 6:'g', 7:'h', 8:'i'}

# This should be True, except for strange debugging cases
allow_pass = True
         
#drawBoard is prettier, but this is more compact for debugging
def printBoard(array):
    print ""
    print "  abcdefghi"
    for y in range(9):
        sys.stdout.write(numberDict[y])
        sys.stdout.write(' ')
        for x in range(9):
            i=x+(y*9)
            c = '*' if array[i]<0 else ('0' if array[i]>0 else '.')
            sys.stdout.write(c)
        print ""

def drawHeatMapBoard(boardArray, predictionArray):
        
    ion()

    # create a 8" x 8" board
    fig = plt.figure(figsize=[6,6])
    fig.patch.set_facecolor((1,1,.8))
    
    ax = fig.add_subplot(111)
    
    # draw the grid
    for x in range(9):
        ax.plot([x, x], [0,8], 'k')
    for y in range(9):
        ax.plot([0, 8], [y,y], 'k')
    
    # scale the axis area to fill the whole figure
    ax.set_position([0,0,1,1])
    

    plt.xticks(range(9), [chr(97 + x) for x in xrange(9)])
    plt.yticks(range(9), [chr(97 + (8-x)) for x in xrange(9)])    
    
    ax.patch.set_facecolor((1,1,.8))
    
    # scale the plot area conveniently (the board is in 0,0..8,8)
    ax.set_xlim(-1,9)
    ax.set_ylim(-1,9)
    

    for index in range(len(predictionArray)):

    
        x = index % 9
        y = 8 - (index / 9)
        
        if (boardArray[index] == BLACK): 
            edgeColor = (.5,.5, .5)
            faceColor = 'k'
            s1, = ax.plot(x,y,'o',markersize=30, 
                          markeredgecolor=edgeColor, 
                          markerfacecolor=faceColor, 
                          markeredgewidth=2)
        elif (boardArray[index] == WHITE): 
            edgeColor = (0,0,0)
            faceColor = 'w' 
            s1, = ax.plot(x,y,'o',markersize=30, 
                          markeredgecolor=edgeColor, 
                          markerfacecolor=faceColor, 
                          markeredgewidth=2)

        heatEdgeColor = (.2,.2, .2)
        heatFaceColor = 'b'
        prediction = 120 * predictionArray[index] 

    
        if (index == 81):
            x = 0
            y = -0.5
        elif (index == 82):
            x = 1
            y = -0.5
        # draw heat map
        s2, = ax.plot(x,y,'s',markersize=prediction, 
                      markeredgecolor=heatEdgeColor, 
                      markerfacecolor=heatFaceColor, 
                      markeredgewidth=2, alpha=0.3) 
    
    #Handle resign and pass
    passRank = 100 * predictionArray[81]
    resignRank = 100 * predictionArray[82]
 
    ax.annotate("pass",
            xy=(0, -0.5), xycoords='data',
            xytext=(-1, -0.5), textcoords='data',
            arrowprops=dict(arrowstyle="->",
                            connectionstyle="arc3"),
            )
    ax.annotate("resign",
            xy=(1, -0.5), xycoords='data',
            xytext=(1.5, -0.5), textcoords='data',
            arrowprops=dict(arrowstyle="->",
                            connectionstyle="arc3"),
            )
            
    show()
    
    print "Pass rank: %.3f" % passRank
    print "Resign rank: %.3f" % resignRank
        
        

    
def checkMoveLegal(board_array, move, player):


    board_copy = copy.deepcopy(board_array)

    
    if (board_copy[move] == BLACK or board_copy[move] == WHITE):
        return False
    
    alreadyChecked = []
    group = []
    hasLiberties = False
    checkSide = 0
    
    if (player == "human"):
        board_copy[move] = humanColor
        checkSide = humanColor
    elif (player == "computer"):
        checkSide = computerColor
        board_copy[move] = computerColor


    board_copy, captures = removeDeadStones(board_copy, player)

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

    captures = 0
    board_copy = copy.deepcopy(board_array)
    
    board_copy = board_copy[:-2].astype('int')
    board_copy = board_copy.reshape(9,9)

    if (player == "human"):
        removeSide = computerColor
    elif (player == "computer"):
        removeSide = humanColor
    
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
                captures = len(group)

        #End of groupHeadFound == True

    #End of while loop until lastPointReached
    
    board_copy = board_copy.reshape(81)
    board_copy = np.append(board_copy, [0,0])
    return board_copy, captures

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

def setHandicap(handicap, board_array):
    
    print "Handicap is set to ", str(handicap)
    
    if (handicap >= 2):
        board_array[0][20] = BLACK
        board_array[0][60] = BLACK

    if (handicap >= 3):
        board_array[0][56] = BLACK    

    if (handicap >= 4):
        board_array[0][24] = BLACK

    if (handicap > 4):
        print "Maximum handicap is 4"
    
    return board_array
        
    
#############################################################################  
 
#Needed to enable somewhat graceful program ending
def everythingIsAFunction():

    
    prediction_array = np.zeros((1,83)).astype('float32') 
    board_array = np.zeros((1,83)).astype('float32')  
    
    num_nodes = 1024
    
    weights_1 = tf.Variable(tf.zeros([83, num_nodes]), name='weights_1')
    biases_1 = tf.Variable(tf.zeros([num_nodes]), name='biases_1')
    weights_2 = tf.Variable(tf.zeros([num_nodes, 83]), name='weights_2')
    biases_2 = tf.Variable(tf.zeros([83]), name='biases_2')
    
    saver = tf.train.Saver()
    sess = tf.Session()
        
    print "Loading in weights and biases. Have faith...."
    saver.restore(sess, modelCheckpoint)
    
    print "restored"
    
    humanPassed = False
    computerPassed = False
    choiceCount = 0
    
    computerToPlay = False
    humanToPlay = False
    
    if (humanColor == BLACK):
        if (handicap > 1):
            computerToPlay = True
        else:
            humanToPlay = True
        print "Human is black. Computer is white"
    else:
        if (handicap > 1):
            humanToPlay = True
        else:
            computerToPlay = True
        print "Computer is black. Human is white"
    

    if (handicap > 1):
        board_array = setHandicap(handicap, board_array)   
        drawHeatMapBoard(board_array[0], prediction_array[0])
        
    capturedHumanStones = 0
    capturedComputerStones = 0
    capturedStones = 0
    
    humansLastBoardPosition = np.zeros((1,83)).astype('float32')  
    computersLastBoardPosition = np.zeros((1,83)).astype('float32')  
       
    def lettersToArray(xLetter, yLetter): 
        x = letterDict[xLetter]
        y = letterDict[yLetter]
        return x+(y*9) 
        
    def arrayToLetters(idx):
        x = idx % 9
        y = idx / 9
        xLetter = numberDict[x]
        yLetter = numberDict[y]
        return xLetter, yLetter
        
    while True:
        
        if (humanToPlay == True):
            computerToPlay = False
        else:
            computerToPlay = True
        
        choiceCount = 1
            
        while computerToPlay == True:
            
            capturedStones = 0
            
            #Board must be fed to tensorflow with black next to move regardless of
            #what color black is playing
            tensor_board = copy.deepcopy(board_array)
            if (computerColor == WHITE):
                tensor_board *= -1
            
            #Get the prediction for next move
            test_prediction =  tf.nn.softmax(tf.matmul(tf.nn.relu(tf.matmul(tensor_board, weights_1) + biases_1), 
                                  weights_2) + biases_2)
               
            del tensor_board
            
            prediction_array = test_prediction.eval(session=sess) 

            computerMove = np.argsort(prediction_array)[0][-choiceCount]

            if (choiceCount > 82):
                print "No more moves possible. "
                computerToPlay = False
                continue
                
            elif (computerMove == 81):
                
                if allow_pass:
                    print "Computer passed"
                    drawHeatMapBoard(board_array[0], prediction_array[0])
                    print "Captured human stones: ", capturedHumanStones
                    print "Captured computer stones: ", capturedComputerStones
                    print
                    computerToPlay = False
                    if (humanPassed == True):
                        print "Game over"
                        return
                    else:
                        computerPassed = True
                        computerToPlay = False
                else:
                    print "Computer would pass, but can't, because it's not allowed to."
                    choiceCount +=1
                continue
            
            elif (computerMove == 82):
                print
                print "Computer resigned"
                
                drawHeatMapBoard(board_array[0], prediction_array[0])
                print "Captured human stones: ", capturedHumanStones
                print "Captured computer stones: ", capturedComputerStones
                print "Game over"
                return
                
                
                    
            if (board_array[0][computerMove] != 0):
                computerMoveLetters = arrayToLetters(computerMove)
                print
                print "Illegal move: point already taken: ", computerMoveLetters[0], computerMoveLetters[1]
                computerToPlay = True
                computerPassed = False
                choiceCount += 1
                continue
            elif (checkMoveLegal(board_array[0], computerMove, "computer") == False):
                computerMoveLetters = arrayToLetters(computerMove)
                print
                print "Illegal move: suicide rule violated: ", computerMoveLetters[0], computerMoveLetters[1]
                computerToPlay = True
                computerPassed = False
                choiceCount += 1
                continue
                
            proposedPosition = copy.deepcopy(board_array)
            proposedPosition[0][computerMove] = computerColor
            proposedPosition[0], capturedStones = removeDeadStones(proposedPosition[0], "computer")

            
            if (np.array_equal(proposedPosition[0],computersLastBoardPosition[0])):
                computerMoveLetters = arrayToLetters(computerMove)
                print
                print "Illegal move: KO rule violation: ", computerMoveLetters[0], computerMoveLetters[1]
                computerToPlay = True
                computerPassed = False
                choiceCount += 1

            else:         
                computerMoveLetters = arrayToLetters(computerMove)
                board_array[0][computerMove] = computerColor
                board_array[0], capturedStones = removeDeadStones(board_array[0], "computer")
                capturedHumanStones = capturedHumanStones + capturedStones
                print
                print "Computer move: " , computerMoveLetters[0], computerMoveLetters[1]
                drawHeatMapBoard(board_array[0], prediction_array[0])
                print "Captured human stones: ", capturedHumanStones
                print "Captured computer stones: ", capturedComputerStones
                computerToPlay = False
                computerPassed = False
                computersLastBoardPosition = copy.deepcopy(board_array)
        
        #End of while computerToPlay
                
        humanToPlay = True
        humanPassed = False
        while humanToPlay == True:
            
            capturedStones = 0
            
            moveParsedOK = False
            while moveParsedOK == False:
                print "Human move: Type (a-i)(a-i) to move"
                print "or type p to pass or type resign to resign."
                humanInput = raw_input()
                humanInput = str(humanInput)
                try:
                    try: 
                        try: #Passed
                            
                            match = re.match(r"(p)", humanInput)
                            p = match.group(1)
                            print "Human passed"
                            drawHeatMapBoard(board_array[0], prediction_array[0])
                            print "Captured human stones: ", capturedHumanStones
                            print "Captured computer stones: ", capturedComputerStones
                            if (computerPassed == True):
                                print "Game over"
                                return
                            else:
                                humanPassed = True
                                board_array[0][81] = humanColor
                                moveParsedOK = True
                                humanToPlay = False
                                continue
                        except: # Resign
                            match = re.match(r"(resign)", humanInput)
                            resign = match.group(1)
                            print "Human resigned"
                            drawHeatMapBoard(board_array[0], prediction_array[0])
                            print "Captured human stones: ", capturedHumanStones
                            print "Captured computer stones: ", capturedComputerStones
                            print "Game over"
                            return
                    except: #Move

                        match = re.match(r"([a-i])([a-i])", humanInput)
                        xLetter = match.group(1)
                        yLetter = match.group(2)
                        moveParsedOK = True
                        humanToPlay = False
                        humanPassed = False
                        
                        humanMove = lettersToArray(xLetter, yLetter)

                        
                        if (board_array[0][humanMove] != 0):
                            print "Illegal move: point already taken."
                            moveParsedOK = False
                            humanToPlay = True
                            continue
                        
                        if (checkMoveLegal(board_array[0], humanMove, "human") == False):
                            humanMoveLetters = arrayToLetters(humanMove)
                            print
                            print "Illegal move: suicide rule violated: ", humanMoveLetters[0], humanMoveLetters[1]
                            moveParsedOK = False
                            humanToPlay = True
                            continue
                        
                        proposedPosition = copy.deepcopy(board_array)
                        proposedPosition[0][humanMove] = humanColor
                        proposedPosition[0], capturedStones = removeDeadStones(proposedPosition[0], "human")
                        
                        if (np.array_equal(proposedPosition[0],humansLastBoardPosition[0])):
                            print "Illegal move: KO rule violation: ", xLetter, yLetter
                            moveParsedOK = False
                            humanToPlay = True

                        else:
                            board_array[0][humanMove] = humanColor
                            board_array[0], capturedStones = removeDeadStones(board_array[0], "human")   
                            capturedComputerStones = capturedComputerStones + capturedStones
                            drawHeatMapBoard(board_array[0], prediction_array[0])
                            print "Captured human stones: ", capturedHumanStones
                            print "Captured computer stones: ", capturedComputerStones
                            #Do not assign humansLastBoardPosition here. The user could undo the move.
                            #Only assign after the undo window has closed.
                            #humansLastBoardPosition = copy.deepcopy(board_array)
                except Exception as e:
                    print "Invalid entry: " , e
                    moveParsedOK = False
                    humanPassed = False
                    continue

        #End of while humanToPlay

        # Give human a chance to undo a typo
        moveParsedOK = False
        while moveParsedOK == False:
            print
            print "Type o for computer to move or u to undo your move"
            humanInput = raw_input()
            humanInput = str(humanInput)
            try:
                try: #Computer's turn
                    match = re.match(r"(o)", humanInput)
                    x = match.group(1)
                    moveParsedOK = True
                    humansLastBoardPosition = copy.deepcopy(board_array)
                    humanToPlay = False
                except: #Undo
                    match = re.match(r"(u)", humanInput)
                    x = match.group(1)
                    moveParsedOK = True  
                    board_array = copy.deepcopy(computersLastBoardPosition)
                    print "Undid human's last move."
                    drawHeatMapBoard(board_array[0], prediction_array[0])
                    capturedComputerStones = capturedComputerStones - capturedStones
                    print "Captured human stones: ", capturedHumanStones
                    print "Captured computer stones: ", capturedComputerStones

                    humanToPlay = True
                
            except: #Invalid
                print
                print "Invalid entry: Type o for computer to move or u to undo your move."
                moveParsedOK = False  


                 
everythingIsAFunction()