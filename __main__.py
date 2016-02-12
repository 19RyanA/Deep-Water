import cv2
import numpy as np
from random import randrange, randint
from ale_python_interface import ALEInterface
import skflow
import sys
g = 0.9
append = lambda x, y: np.append(x,y)
regressor = skflow.TensorFlowDNNRegressor(hidden_units=[10,10],learning_rate=0.01, verbose=randint(0,1))
regressor.fit(np.random.randn(1601, 1), append([1.], np.zeros((1600, 1))))
def Q(s, a):
    return regressor.predict(append(s, a))

def detectState(ale):
    return cv2.resize(ale.getScreenGrayscale(), (40,40))

ale = ALEInterface()
ale.loadROM("breakout.bin")
actionSet = ale.getMinimalActionSet()
while True:
    while not ale.game_over():
        if sys.argv[1] == 'disp':
            cv2.imshow('', cv2.resize(ale.getScreenRGB(), (600,600)))
            cv2.waitKey(1)
        s = cv2.resize(ale.getScreenGrayscale(), (40,40))
        s = detectState(ale)
        qvals = []
        for action in actionSet:
            qvals.append(Q(s, action)[0])
        a = actionSet[qvals.index(max(qvals))]
        X = append(s, a)
        r = ale.act(a)
        print r
        s_ = detectState(ale)
        qvals = []
        for action in actionSet:
            qvals.append(Q(s_, action)[0])
        a_ = actionSet[qvals.index(max(qvals))]
        y = r + g*Q(s_, a_)
        regressor.fit(X, y)
        # For some reason, I need this to get the game to start
        choice = randrange(len(actionSet))
        ale.act(choice)
        regressor.save('./regressor')
