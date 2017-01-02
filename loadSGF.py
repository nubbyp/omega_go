#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Dec  9 08:42:33 2016

@author: nubby

This file loads in SGF (Standard Go Format) data files from five data sources openly available on the web. 
See README.md for the data source links. 

SGF files are loaded into 9x9 matrices - one matrix for each move in each game. These
are the features for the neural net training set.
A corresponding array of 9x9 matrices are generated with the next human move played saved as a single one in an 
array of zeros. This becomes the labels in the neural net training set. 

This script first sorts the SGF files downloaded by size - 9x9 games versus other games.
It then sorts the 9x9 games into strong and weak games. The definition of strong and weak 
can be adjusted of course. I use stronger than 5 kyu in the traditional ranking system or 2000 in the ELO 
ranking system. I know these two ranks do not line up but it was a function of how much data I had from 
each data source.

A recursive function checkGroup is used to establish what groups go stones belong to and then to 
determine if a group has any liberties and is thus alive or has no liberties and is thus dead. Dead
stones are removed from the matrix after each move. 

The data is increased in size 8 fold by transforming each board position around its 8 symmetries: 
rotating and flipping it. This increases the training set and also removes human bias (humans tend to
follow etiquette about where the first move should be which affects the shape of the game. The games tend
to be biased to splitting the board in the vertical dimension)

At the end the matrix data is pickled into pickle files that are 1.5 million rows long each. These pickle 
files are used in the next training step. 

"""


import numpy as np
import glob
import re
import copy
from six.moves import cPickle as pickle
import timeit
import os

cwd = os.getcwd()
path = cwd + '/data'
pickleRoot = 'pickles/pickleFile_'

kifuPath = path + '/Kifu/'
proPath = path + '/Pro/'
nngsPath = path + '/NNGS/'
top50Path = path + '/Top50/'
onlinePath = path + '/OnlineGo/'

globalCount = 1 # Number of pickle files needed
print "globalCount: ", globalCount

def getProperty(header, property):
    m = re.search(r""+property+"\[(.*?)\]", header)
    return (m.group(1))



def removeDeadStones(feature):
    
    #Board is flipped before coming in here
    #So move just played is white
    #Therefore remove black stones
    newFeature = feature[:-2].copy().astype('int')
    newFeature = newFeature.reshape(9,9)

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
                if (newFeature[row][column] == 1):
                    if ((row,column) in alreadyChecked):
                        continue
                    else:
                        groupHead = (row,column)
                        groupHeadFound = True
                        break
            if groupHeadFound == True:
                break
  
            
        if (groupHeadFound == True):
            hasLiberties = checkGroup(groupHead, alreadyChecked, group, hasLiberties, newFeature)
            if (hasLiberties == False):
                for tupe in group:
                    row = tupe[0]
                    column = tupe[1]
                    newFeature[row][column] = 0

        #End of groupHeadFound == True

    #End of while loop until lastPointReached
    
    newFeature = newFeature.reshape(81)
    newFeature = np.append(newFeature, [0,0])
    return newFeature

#End of removeDeadStones()


# A recursive function used to establish whether a group has liberties or not. 
def checkGroup(tupe, alreadyChecked, group, hasLiberties, newFeature):
    
    alreadyChecked.append(tupe)
    group.append(tupe)
    
    row = tupe[0]
    column = tupe[1]
    
    if (row < 8 and ((row+1,column) not in alreadyChecked)):
        if (newFeature[row+1][column] == 1):
            hasLiberties = checkGroup((row+1,column), alreadyChecked, group, hasLiberties, newFeature)
        elif (newFeature[row+1][column] == 0):
            hasLiberties = True

    if (row > 0 and ((row-1,column) not in alreadyChecked)):
        if (newFeature[row-1][column] == 1):
            hasLiberties = checkGroup((row-1,column), alreadyChecked, group, hasLiberties, newFeature)
        elif (newFeature[row-1][column] == 0):
            hasLiberties = True
            
    if (column < 8 and ((row,column+1) not in alreadyChecked)):
        if (newFeature[row][column+1] == 1):
            hasLiberties = checkGroup((row,column+1), alreadyChecked, group, hasLiberties, newFeature)
        elif (newFeature[row][column+1] == 0):
            hasLiberties = True            

    if (column > 0 and ((row,column-1) not in alreadyChecked)):
        if (newFeature[row][column-1] == 1):
            hasLiberties = checkGroup((row,column-1), alreadyChecked, group, hasLiberties, newFeature)
        elif (newFeature[row][column-1] == 0):
            hasLiberties = True  
            
    return hasLiberties

def randomize(dataset, labels):
  permutation = np.random.permutation(len(dataset)) 
  shuffled_dataset = dataset[permutation]
  shuffled_labels = labels[permutation]
  return shuffled_dataset, shuffled_labels



#Expand dataset size by transforming it (flip, rotate)
def transform_data(function_name, dataset, labelset):

    if (len(dataset) > 0):
            
        eighty_one_dataset = np.empty(shape = (len(dataset), 81))
        eighty_three_dataset = np.empty(shape = (len(dataset), 83))
    
        eighty_one_labelset = np.empty(shape = (len(labelset), 81))
        eighty_three_labelset = np.empty(shape = (len(labelset), 83))   
        
        for x in range(len(dataset)):
            
            passValueD = dataset[x][81]
            resignValueD = dataset[x][82]
            eighty_one_dataset[x] = dataset[x][:-2]
            temp = eighty_one_dataset[x].reshape(9,9)
            temp = function_name(temp)
            eighty_one_dataset[x] = temp.reshape(81)
            eighty_three_dataset[x] = np.append(eighty_one_dataset[x], [passValueD, resignValueD])
    
            passValueL = labelset[x][81]
            resignValueL = labelset[x][82]
            eighty_one_labelset[x] = labelset[x][:-2]
            temp2 = eighty_one_labelset[x].reshape(9,9)
            temp2 = function_name(temp2)
            eighty_one_labelset[x] = temp2.reshape(81)
            eighty_three_labelset[x] = np.append(eighty_one_labelset[x], [passValueL, resignValueL])
    
        dataset = np.concatenate((dataset, eighty_three_dataset))
        labelset = np.concatenate((labelset, eighty_three_labelset))
        
    return dataset, labelset      

def pickleFiles(features, labels):
    
    pickleFileCount = len(features)/1500000 + 1
    print "Number of pickle files needed for this sub dataset = " , pickleFileCount

    
    for i in range(pickleFileCount):
        try:
          global globalCount
          startRange = i * 1500000
          stopRange = startRange + 1500000
          print "startRange: ", startRange, " stopRange: ", stopRange
          print "pickling file: ", str(globalCount)
          pickle_file = pickleRoot + str(globalCount)  + '.pickle'
          globalCount +=1
          f = open(pickle_file, 'wb')
          save = {
            'dataset': features[startRange:stopRange],
            'labels': labels[startRange:stopRange],
        
            }
          pickle.dump(save, f, pickle.HIGHEST_PROTOCOL)
          f.close()
          
        except Exception as e:
          print('Unable to save data to pickle file: ', e)
          raise
     
def countFiles(pathName):
    count = 0
    for filename in os.listdir(pathName):
        path = os.path.join(pathName, filename)

        
        if os.path.isfile(path):
            count += 1   
    return count
    
def parseNoResignation(sgfdata) :
    
            
    gameEndedByTime = False
    gameEndedInResignation = False
    
    try:
        r = re.search(r'RE\[(.+?)\]',sgfdata)
        result = r.group(1)
        
        if (result == 'W+R'):
            #Shouldn't get here in these types of sgfs
            print "******** WHITE RESIGNED *******"
        elif (result == 'B+R'):
            #Shouldn't get here in these types of sgfs
            print "******** WHITE RESIGNED *******"
        elif (result == 'W+T' or result == 'B+T'):
            print "******** GAME ENDED BY TIME ********"
            gameEndedByTime = True
        else:
           # print "result: ", result
            # Result found so game did not end in resignation
            gameEndedInResignation = False
    except:
       # print "result not found"
        gameEndedInResignation = True
        

    return gameEndedByTime, gameEndedInResignation
    
     
def finalMoveNoResignation(gameEndedInResignation):
           
    global gameEndedInResignationFound                      
    # Add in the last board position 
    # - the final pass or resign move. 
    # The robot needs to train 
    #when to pass and when to resign
    label = np.zeros(83)

    if (gameEndedInResignation == True):
        # Board has just been flipped
        label[82]=1
        gameEndedInResignationFound += 1
    else:
        label[81]=1
   
    return label

def parseResignation(sgfdata) :
    
    try:
        gameEndedByTime = False
        blackWinByResignation = False
        whiteWinByResignation = False
        
        r = re.search(r'RE\[(.+?)\]',sgfdata)
        result = r.group(1)
                       
        if (result == 'W+R'):
            whiteWinByResignation = True
            blackWinByResignation = False
        elif (result == 'B+R'):
            whiteWinByResignation = False
            blackWinByResignation = True
        elif (result == 'B+T' or result == 'W+T'):
            gameEndedByTime = True
        else:
            blackWinByResignation = False
            whiteWinByResignation = False
    except:
     #   print "Results not found for file. "
        gameEndedByTime = True

    return gameEndedByTime, blackWinByResignation, whiteWinByResignation
    
     
def finalMoveWithResignation(blackWinByResignation, whiteWinByResignation, thisMoveColor, feature):
        
    global whiteWinByResignationFound
    global blackWinByResignationFound
                         
    # Add in the last board position 
    # - the final pass or resign move. 
    # The robot needs to train 
    #when to pass and when to resign

    label = np.zeros(83)
    
    if (whiteWinByResignation == False and blackWinByResignation == False):
     #   print "final move: pass: ", filename
            label[81]=1
    elif (whiteWinByResignation == True):
        whiteWinByResignationFound += 1
        # Board has just been flipped
        if (thisMoveColor == 'W'): # White is now -1
          #  print "final move: white is -1, black resigns: ", filename
            label[82]=1
        else:
    #        print "final move: white is 1, black resigns: ", filename
            #Need to flip board again
            feature *= -1
            label[82]= 1
    elif (blackWinByResignation == True):
        blackWinByResignationFound += 1
        # Board has just been flipped
        if (thisMoveColor == 'B'): # Black is now -1
           # print "final move: black is -1, white resigns: ", filename
            label[82]=1
        else:
         #   print "final move: black is 1, white resigns: ", filename
         #Need to flip board again
            feature *= -1
            label[82]=1
    else:
        error = "Error: unexpected else condition in determining last move of game: "
        raise error
        
    return label, feature
    
                
                
def parseMoves(nfile, labels, features, parseType) :

    global handicapFound
    global ttFound
    global gameEndedByTimeFound
    
    feature = np.zeros(83)
        
    letterDict = {'a':0, 'b':1,'c':2, 'd':3, 'e':4, 'f':5,'g':6,'h':7,'i':8}
        
    try:
        with open(nfile, 'r') as src:
            sgfdata = src.read()
            
            gameEndedByTime = False
            blackWinByResignation = False
            whiteWinByResignation = False
            gameEndedInResignation = False
            
            if (parseType == "NoResignation"):
                gameEndedByTime, gameEndedInResignation = parseNoResignation(sgfdata)
            else:
                gameEndedByTime, blackWinByResignation, whiteWinByResignation = parseResignation(sgfdata)


            #Look for handicap games
            errorInHandicap = False
            thisMoveColor = 'B'
            previousMoveColor = 'W'
            try:
                r = re.search(r'HA\[(.+?)\]',sgfdata)
                handicap = r.group(1)
                        
                handicap = int(handicap)
               # print "handicap game found: ", str(handicap), " ", nfile
                handicapFound += 1
                if (handicap > 1):
                    thisMoveColor = 'W'
                    previousMoveColor = 'B'
                    #Get where the handicap stones were placed
                    try:
                        handicapStoneLine = re.search(r'AB((\[[a-z][a-z]\])+)',sgfdata)
                        if handicapStoneLine:
                          #  print "Handicap stone line found: ", handicapStoneLine.group(1)
                            handicapStones = re.findall(r'\[([a-i])([a-i])\]',handicapStoneLine.group(1))
                            if (len(handicapStones) != handicap):
                                error =  "Error: Handicap stones expected is ", str(handicap), " but found ", len(handicapStones)
                                raise error
                            for stone in handicapStones:
                            #    print "Handicap stone: ", stone
                                xLetter = stone[0]
                                yLetter = stone[1]
                                x = letterDict[xLetter]
                                y = letterDict[yLetter]
                                # Must be -1 as first label saved is 1
                                feature[(y*9)+x] = -1 
                         #   print "feature after handicap: ", feature            
                    except Exception as e:
                        print "Error in getting handicap stones: ", e
                        errorInHandicap = True
                        raise
                        
            except:
                if (errorInHandicap == True):
                    raise
                else:
                    # print "No handicap found"
                    pass

            try:
                moveCount = 0
                matchPlay = re.findall(';[BW]\[.*?\]',sgfdata)
                for play in matchPlay:
                
                    moveCount += 1
                    
                    move = re.findall(';([BW])\[([a-i])([a-i])\]',play)
                    if move:
                   #     print "move found: ", move[0]
                        label = np.zeros(83)
    
                        thisMoveColor = move[0][0]
                        xLetter = move[0][1]
                        yLetter = move[0][2]
    
                        if (xLetter == '' or yLetter == ''):
                            print 'Blank xLetter or yLetter'
    
                        x = letterDict[xLetter]
                        y = letterDict[yLetter]
                        #print x, ' ' , y
            
                        label[(y*9)+x] = 1
                        labels.append(label)
                        copied_feature = copy.deepcopy(feature)
                        features.append(copied_feature)
                        
                        # Duplicate later moves to assist with training end game
                        if (moveCount > 30):
                          #  print "Duplicating move: ", moveCount, " ", nfile
                            labels.append(label)
                            features.append(copied_feature)
                          
                        feature[(y*9)+x] = 1 
                        if (previousMoveColor != thisMoveColor):
                            feature *= -1
                        else:
                            print "In Move: Two moves of same color back to back: ", nfile
                            print "previous Color: ", previousMoveColor
                            print "this color: ", thisMoveColor
                            print "move: ", move[0]
                            raise

                        feature = removeDeadStones(feature)
                                    
                        previousMoveColor = thisMoveColor

                    matchPass = re.findall(';([BW])\[?\]',play)
                    if matchPass:
    
                        thisMoveColor =  matchPass[0][0]
                        
                        label = np.zeros(83)
                        label[81]=1
    
                        labels.append(label)
                        copied_feature = copy.deepcopy(feature)
                        features.append(copied_feature)
                        
                        feature[81] = 1
                        
                        if (previousMoveColor != thisMoveColor):
                            feature *= -1
                        else:
                            print "In Pass: Two moves of same color back to back: ", nfile
                            print "pass move: ", matchPass[0]
                            raise

                        previousMoveColor = thisMoveColor
                        
                    matchPassAlt = re.findall(';([BW])\[tt\]',play)
                    if matchPassAlt:
    
                        ttFound += 1
                        
                        thisMoveColor =  matchPassAlt[0][0]
                        
                        label = np.zeros(83)
                        label[81]=1
    
                        labels.append(label)
                        copied_feature = copy.deepcopy(feature)
                        features.append(copied_feature)
                        
                        feature[81] = 1
                        
                        if (previousMoveColor != thisMoveColor):
                            feature *= -1
                        else:
                            print "In Pass: Two moves of same color back to back: ", nfile
                            print "pass move: ", matchPass[0]
                            raise
                            
                        previousMoveColor = thisMoveColor

            except Exception as e:
                print "Error in parsing moves: ", e, " " ,  nfile
                raise
                
            #At end of file
            # Add in the last board position - the final pass or resign move. 
            # The robot needs to train when to pass and when to resign

            #Don't do this last step if game ended by time
            if (gameEndedByTime == False):
                if (parseType == "NoResignation"):
                    label = finalMoveNoResignation(gameEndedInResignation)                
                else:
                    label, feature = finalMoveWithResignation(blackWinByResignation, whiteWinByResignation, thisMoveColor, feature)
                    
                labels.append(label)
                copied_feature = copy.deepcopy(feature)
                features.append(copied_feature)
            else:
                gameEndedByTimeFound += 1
            # end of game not ending by time - final pass/resign needed

    except Exception as e:
        raise e
    
def loadFiles(pathName, parseType, rankingSystem):

    
    sgfFiles = glob.glob(pathName + "*.sgf")

    for file in sgfFiles:
    
        if file.endswith(".sgf"):
           filename = os.path.basename(file)       
           filesize = os.path.getsize(file)
           if filesize < 180:
               os.rename(file, pathName + "too_small/" + filename)
               continue
           try:
               with open(file, 'r') as src:
    
                   sgfdata = src.read()

                   matchSize = re.search(r'SZ\[([0-9]+?)\]',sgfdata)
                   if matchSize:
                       size = matchSize.group(1)

                       if (size == '9'):
                           os.rename(file, pathName + "nine/" + filename)
                       else:
                           os.rename(file, pathName + "other/" + filename)
                   else:
                       raise

           except Exception as e:
                os.rename(file, pathName + "error/" + filename)       
                print('Error in figuring out board size: ', file, e)
    
          
    print "File count for loading files from data path: ", pathName     
    print "9x9: ", countFiles(pathName + "nine/")
    print "other: ", countFiles(pathName + "other/")
    print "Error count after sorting by size: ", countFiles(pathName + "error/")


    if (rankingSystem == "ELO" or rankingSystem == "Traditional"):
        #Don't do this for the pro games
        nine_files = glob.glob(pathName + "nine/" + "*.sgf")

        for file in nine_files:
        
            if file.endswith(".sgf"):
               filename = os.path.basename(file)
               try:
                   with open(file, 'r') as src:
                       
                       sgfdata = src.read()

                       if (rankingSystem == "ELO"):
                           #Select games where players are 1 kyu or better (2000 ELO ranking)

                           rankText = re.search(r"PB\[.*?\((.*?)\)\]", sgfdata)
                           bRank = rankText.group(1)
                           bRank = int(bRank)
                           
                           rankText = re.search(r"PW\[.*?\((.*?)\)\]", sgfdata)
                           wRank = rankText.group(1)  
                           wRank = int(wRank)
                           
                           bOK = False
                           wOK = False
                           
                           if (bRank >= 2000): # equivalent of 1 kyu
                               bOK = True
                               
                           if (wRank >= 2000):
                               wOK = True
                               
                           if (bOK and wOK):
                               os.rename(file, pathName + "nine/nine_strong/" + filename)
                           else:
                               os.rename(file,  pathName + "nine/nine_weak/" + filename)
                       else: 
                            
                           #Ranking is KYU / DAN / PRO system
                           rankText = re.search(r"BR\[([0-9]*)([kdpKDP])\]*", sgfdata)
                           bRank = rankText.group(1)
                           bKyuDan = rankText.group(2)
                        
                           rankText = re.search(r"WR\[([0-9]*)([kdpKDP])\]*", sgfdata)
                           wRank = rankText.group(1)
                           wKyuDan = rankText.group(2)
            
                           bRank = int(bRank)
                           wRank = int(wRank)
                           
                           bOK = False
                           wOK = False
                           
                           if (bKyuDan == 'd' or bKyuDan == 'D' or bKyuDan == 'P' or bKyuDan == 'p' or bRank <= 5):
                               bOK = True
                               
                           if (wKyuDan == 'd' or wKyuDan == 'D' or wKyuDan == 'P' or wKyuDan == 'p' or wRank <= 5):
                               wOK = True
                               
                           if (bOK and wOK):
                               os.rename(file, pathName + "nine/nine_strong/" + filename)
                           else:
                               os.rename(file,  pathName + "nine/nine_weak/" + filename)
        
                           
               except Exception as e:  
                    print('Error processing player rankings: ', file, " ",  e)
                    os.rename(file, pathName + "error/" + filename) 
             
                    
    
        print "Weak 9x9 games found: ", countFiles(pathName + "nine/nine_weak")
        print "Strong 9x9 games found: ",  countFiles(pathName + "nine/nine_strong")
        print "Error count after determining rank: ", countFiles(pathName + "error/")
    else:
        nine_files = glob.glob(pathName + "nine/" + "*.sgf")

        for file in nine_files:
        
            if file.endswith(".sgf"):
               filename = os.path.basename(file)
               os.rename(file, pathName + "nine/nine_strong/" + filename)
        print "Strong 9x9 games found: ",  countFiles(pathName + "nine/nine_strong")
    #End of ranking system checker
    
    labels =[]
    features = []

    i = 0
    start_time = timeit.default_timer()
            
    nine_files_strong = glob.glob(pathName +  "nine/" + "nine_strong/" +  "*.sgf")
    
    for nfile in nine_files_strong:

        if nfile.endswith(".sgf"):
           filename = os.path.basename(nfile)
        else:
            continue
        
        if (i % 500 == 0):
            print "files done: " ,  i, " time taken: " , int(timeit.default_timer() - start_time)
            start_time = timeit.default_timer()
        i = i + 1
                
        try:
            parseMoves(nfile, labels, features, parseType)
        except Exception as e:
            print('Error in reading moves in SGF file: ', nfile, e)
            os.rename(nfile, pathName + "error/" + filename) 

    labels_array = np.array(labels).astype('float32')
    features_array = np.array(features).astype('float32')
    
    print type(labels_array)
    print type(features_array)
    print labels_array.shape
    print features_array.shape
    print labels_array.ndim
    print features_array.ndim

    
    print "labels shape: " , labels_array.shape
    print "features shape: ", features_array.shape
      
    features_array, labels_array = transform_data(np.fliplr, features_array, labels_array)
    print "fliplr done"
    print "labels shape: " , labels_array.shape
    print "features shape: ", features_array.shape
    
    features_array, labels_array = transform_data(np.flipud, features_array, labels_array)
    print "flipud done"
    print "labels shape: " , labels_array.shape
    print "features shape: ", features_array.shape
    
    
    features_array, labels_array = transform_data(np.rot90, features_array, labels_array)
    print "rot90 done"
    print "labels shape: " , labels_array.shape
    print "features shape: ", features_array.shape

    print "before randomize"
    features_array, labels_array = randomize(features_array, labels_array)
    print "after randomize"
    
    pickleFiles(features_array, labels_array)

handicapFound = 0
ttFound = 0
gameEndedByTimeFound = 0
gameEndedInResignationFound  = 0
whiteWinByResignationFound = 0
blackWinByResignationFound = 0


loadFiles(kifuPath, "NoResignation", "ELO")
loadFiles(proPath, "WithResignation", "None")
loadFiles(nngsPath, "WithResignation", "Traditional")
loadFiles(onlinePath, "WithResignation", "Traditional")
loadFiles(top50Path, "NoResignation", "ELO")

print "Handicap games found: ", handicapFound
print "tt found: ", ttFound
print "gameEndedByTime: ", gameEndedByTimeFound
print "gameEndedInResignationFound : ", gameEndedInResignationFound 
print "whiteWinByResignationFound: ", whiteWinByResignationFound
print "blackWinByResignationFound: ", blackWinByResignationFound