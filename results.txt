TensorFlow training results for OmegaGo:

1 million steps with gradientDescentOptimizer and Relu
Learning rate 0.04
Final test accuracy: 38.0%

1 million steps with gradientDescentOptimizer and Tanh 
Learning rate 0.04
Final test accuracy: 33.3%

1 million steps with adagradOptimizer and Relu 
Learning rate 0.04
Final test accuracy: 36.0%

2 million steps with gradientDescentOptimizer and relu
Learning rate 0.04
Final test accuracy: 37.5%

4 million steps with gradientDescentOptimizer and relu
Learning rate 0.04
Final test accuracy: 40.6%

8 million steps with gradientDescentOptimizer and relu
Learning rate 0.04
Final test accuracy: 41.4%

16 million steps with gradientDescentOptimizer and relu
Learning rate 0.04
Final test accuracy: 41.5%

16 million steps with gradientDescentOptimizer and relu
Learning rate 0.04
Larger training set that includes mixed ability games
Final test accuracy: 37.5%

NOTE: Although including mixed ability games reduced final test accuracy in the training stage, game play against GnuGo was in fact improved. Small changes in test accuracy often translated into big changes in game play performance. 

GnuGo 9x9 level 0 play results:


Runs with 4 million relu gradientDescent:

0 handicap (Omega is black)
Omega	Gnu
72	128	36%

0 handicap (Omega is white) - komi of 6.5
Omega	Gnu
60	140	30%

1 handicap (Omega is black)
Omega	Gnu
90	110	45%

2 handicap (Omega is black)
Omega	Gnu
75	25	75%


Runs with 8 million relu gradientDescent

0 handicap (Omega is black)
Omega	Gnu
80	116	41%

0 handicap (Omega is white) - komi of 6.5
Omega	Gnu
52	132	28%

1 handicap (Omega is black)
Omega	Gnu
92	106	46%

2 handicap (Omega is black)
Omega	Gnu
78	22	78%


Runs with 16 million relu gradientDescent

0 handicap (Omega is black)
Omega	Gnu
44	56	44%

0 handicap (Omega is white) - komi of 6.5
Omega	Gnu
40	60	40%

1 handicap (Omega is black)
Omega	Gnu
58	42	58%

2 handicap (Omega is black)
Omega	Gnu
75	25	75%



Runs with 16 million relu gradientDescent
Mixed ability plays

0 handicap (Omega is black)
Omega	Gnu
96	104	48%

0 handicap (Omega is white) - komi of 6.5
Omega	Gnu
106	94	53%

1 handicap (Omega is black)
Omega	Gnu
61	39	61%

2 handicap (Omega is black)
Omega	Gnu
85	15	85%



