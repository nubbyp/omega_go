## OmegaGo

OmegaGo is a neural network / deep learning trained robot that plays 9x9 games of Go. I wanted to see how well I could train a robot to play with only a fully connected neural network and no knowledge of the rules of the game of go or Monte Carlo tree search techniques. The code includes files to play games on KGS server and against GnuGo.

Credit, of course, to AlphaGo, from whose paper I learned the basics of how to break up your go game data by moves into features and labels for neural net training. This bot here does not necessarily do anything novel or new, but is a way for beginners to deep learning and neural nets to create a basic go playing robot. 

This is a series of Python code files for:

1.	Loading  SGF (Standard Go Format) files into Python, breaking them up into individual moves and converting them into a series of 9x9 matrices., taking into account the removing of dead stones from the board. An 81 length array of board positions is generated as the “features” for the neural net input and a corresponding 81 length array filled with zeros and a single 1 is generated as the “label” for the neural net output to be compared to – which is the next move the human made given that board position.  
    `loadSGF.py`
    `loadSGFMixed.py`

2.	Training the dataset on a basic two layer neural network using TensorFlow   
    `trainNeuralNet.py`

3.	Playing games of go using the checkpoint file generated in TensorFlow training to tell the computer what its next move should be. There are Python files for playing against a human, playing multi-threaded multiple games against GnuGo as a performance testing baseline, and then finally for allowing the bot to play against others on the KGS server.  To play against GnuGo and on KGS, the program speaks GTP (Go Text Protocol)  
    `playHuman.py`
    `playGnuGo.py`
    `playKGS.py`

Here are links to the data I used, all publicly available for download from the internet. From these games, my `loadSGF.py` file selects the subset of 9x9 games, and then from those, the subset of games where both players are stronger than 5 kyu or ELO 2000, depending on which Go ranking system the SGF files use. 

I create a data subdirectory off the working directory and then have 5 data sources that I save in the subdirectories named below: 
   `data/Kifu` 
   `data/NNGS`  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;https://github.com/zenon/NNGS_SGF_Archive
    `data/Pro`
    `data/Top50`
    `data/OnlineGo`  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;https://github.com/gto76/online-go-games

Under each data subdirectory create the following tree structure to enable sorting of the SGF files:
```
error
nine
	nine_weak
	nine_strong
	nine_mixed
	nine_not_needed
other
too_small
```
In addition, for the code to run, you will need the following  other subdirectories under your working directory where the Python code is stored:
```
pickles
pickles_mixed
checkpoints
trainResults
```

I have included a checkpoints directory in this github containing the needed files if you want to skip right to running your bot against a human or GnuGo and not train it yourself. 

You will need to download the GnuGo application from the internet here: https://www.gnu.org/software/gnugo/download.html

You will also need to download the gateway application kgsGTP.jar to connect your bot to KGS from here: http://files.gokgs.com/javaBin/kgsGtp-3.5.20.tar.gz

Documentation for using this Java application kgsGTP is here: http://senseis.xmp.net/?KgsGtp

I include my TensorFlow training results for various hyper parameters as well as my GnuGo playing results in the file results.txt. I ran my training runs on Amazon Web Services using a GPU. It took about 1 hour per 1 million training steps. 

