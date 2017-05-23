import os

location = '/home/project/emotion/facedata/'
current_location = os.getcwd()

picture_dimension = 90
learning_rate = 0.001

enable_dropout = False
dropout_probability = 0.2

enable_penalty = False
penalty = 0.0005
batch_size = 10
epoch = 1000