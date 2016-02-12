
import cv2
import numpy as np
from random import randrange
from ale_python_interface import ALEInterface
import skflow
#
# # GAMMA = 0.9
# # dims = (40,40)
# # arr = lambda x: np.array(x)
# # append = lambda x, y: np.append(x,y)
# # def Q(state, action):
# #     return regressor.predict(append(state, action))[0]
# #
# # def Q_(s_a_t):
# #     return regressor.predict(s_a_t)
# #
regressor = skflow.TensorFlowDNNRegressor(hidden_units=[10,10],learning_rate=0.01)
# # x = np.random.randn(1, dims[0] * dims[1] + 1)
# # # y = append([1.], np.zeros(dims[0] * dims[1]))
# # y = np.zeros((1, dims[0] * dims[1] + 1))
# # print x.shape, y.shape
# # regressor.fit(x, y)
ale = ALEInterface()
ale.loadROM("breakout.bin")
actionSet = ale.getMinimalActionSet()
while not ale.game_over():
    cv2.waitKey(1)
    cv2.imshow('', cv2.resize(ale.getScreenGrayscale(), (600,600)))

#     def episode(dims):
#         X = []
#         y = []
#         gm = GameManager()
#         actionSet = gm.actionSet()[1][1:]
#         while not gm.done():
#             gm.showScreen()
#             qvalues = []
#             for action in actionSet:
#                 s_a_t = append(gm.screen(dims), arr([action]))
#                 s_a_t = np.reshape(s_a_t, (1, 1601))
#                 qvalues.append(Q_(s_a_t))
#             print qvalues
#             return
#             action_choice = actionSet[qvalues.index(max(qvalues))]
#             X.append(append(gm.screen(dims), action_choice))
#             r = gm.act(action_choice)
#             qs = []
#             for action in actionSet:
#                 qs.append(Q(gm.screen(dims), action))
#             y.append(r + GAMMA * max(qs))
#         return np.array(X), np.array(y)
#
#     @staticmethod
#     def playRandomly():
#         gm = GameManager()
#         while not gm.done():
#             actionSet = gm.actionSet()[1]
#             gm.showScreen()
#             gm.act(randrange(len(actionSet)))
#
# # def train(regressor):
# #     x_batch, y = GameManager.episode((40,40))
# #     y_batch = np.zeros((x_batch.shape[0], dims[0]*dims[1] + 1))
# #     y_batch[:,0] = y[:,0]
# #     # x_batch = y_batch = np.zeros((485,1601))
# #     print x_batch.shape, y_batch.shape
# #     regressor.fit(x_batch, y_batch)
# #     return regressor
#
# # for _ in xrange(2):
# #     x_batch, y_batch = GameManager.episode(dims)
# #     print x_batch.shape, y_batch.shape
# GameManager.playRandomly()
# # regressor.save('./regressor')
# # TODO: stack of 4 frames,



# import cv2
# import numpy as np
# import matplotlib.pyplot as plt
# from random import randrange
# from ale_python_interface import ALEInterface
# import skflow
#
# GAMMA = 0.9
#
# # Q = skflow.TensorFlowDNNRegressor(hidden_units=[10,10],learning_rate=0.01)
#
# class GameManager(object):
#     def __init__(self):
#         self.ale = ALEInterface()
#         self.ale.setInt("random_seed",123)
#         self.ale.loadROM("breakout.bin")
#
#     def actionSet(self):
#         return(self.ale.getMinimalActionSet(), self.ale.getLegalActionSet())
#
#     def act(self, choice):
#         return self.ale.act(choice)
#
#     def screen(self, dims):
#         return cv2.resize(self.ale.getScreenGrayscale(), dims)
#
#     def showScreen(self):
#         cv2.imshow('', self.screen((600,600)))
#         cv2.waitKey(1)
#
#     def done(self):
#         return self.ale.game_over()

#
#     # @staticmethod
#     # def episode(Q):
#     #     gm = GameManager()
#     #     X = []
#     #     y = []
#     #     actionset = gm.actionSet()[0]
#     #     print "Episode Started"
#     #     while not gm.done():
#     #         gm.showScreen()
#     #         values = []
#     #         for action in actionset:
#     #             values.append(Q.predict())
#     #             # values.append(Q.predict(np.array([action])))
#     #         choice = max(values)
#     #         # X.append(np.array([np.resize(gm.screen(), (40,40)).flatten(), choice]))
#     #         r = gm.act()
#
# GameManager.episode()
# # import random
# #
# # from sklearn import datasets, cross_validation, metrics
# # from sklearn import preprocessing
# #
# # import skflow
# #
# # random.seed(42)
# #
# # # Load dataset
# # boston = datasets.load_boston()
# # X, y = boston.data, boston.target
# #
# # # Split dataset into train / test
# # X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y,
# #     test_size=0.2, random_state=42)
# #
# # # scale data (training set) to 0 mean and unit Std. dev
# # scaler = preprocessing.StandardScaler()
# # X_train = scaler.fit_transform(X_train)
# #
# # # Build 2 layer fully connected DNN with 10, 10 units respecitvely.
# # Q = skflow.TensorFlowDNNQ(hidden_units=[10, 10],
# #     steps=5000, learning_rate=0.1, batch_size=1)
# #
# # # Fit
# # Q.fit(X_train, y_train)
