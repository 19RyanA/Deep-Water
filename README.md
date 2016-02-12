# Deep Water

This project is for replicating Deep Mind's Atari project. I am using skflow, a library that is based on TensorFlow, a computational library for Machine Learning. It uses a simple neural network with two hidden layers of 10 units each. It implements naive Q-learning, without any of the additonal features that DeepMind themselves have used. I have yet to implement those optimizations or use an epsilon greedy policy. 

So far, I am only testing the algorithm on the game Breakout which DeepMind has talked a lot about. 

Specifically, I am using Deep Q Learning. With Arcade Learning Environment, I am able to take a grayscale image of 40 pixels by 40 pixels. The neural network uses a iterative approach to Q function approximation. 

The network saves itself after every episode to a folder named regressor. 