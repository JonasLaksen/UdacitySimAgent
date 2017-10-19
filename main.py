import csv
import numpy as np
from keras.preprocessing.image import load_img, img_to_array
from sklearn.model_selection import train_test_split

from agent import Agent


def create_samples():
    samples = []
    with open('driving_log.csv') as csvfile:
        reader = csv.reader(csvfile)
        for line in reader:
            samples.append(line)
    samples = [(img_to_array(load_img(sample[0]))/255.0, sample[3]) for sample in samples]

    samples = train_test_split(samples, test_size=.05)
    X_train, y_train, X_test, y_test = \
        [X for (X, _) in samples[0]], [y for (_, y) in samples[0]],\
        [X for (X, _) in samples[1]], [y for (_, y) in samples[1]]
    return np.asarray(X_train), np.asarray(y_train), np.asarray(X_test), np.asarray(y_test)

def test_on_sample(i, model):
    samples = []
    with open('driving_log.csv') as csvfile:
        reader = csv.reader(csvfile)
        for line in reader:
            samples.append(line)
    X = img_to_array(load_img(samples[i][0]))
    y = samples[i][3]
    print(model.predict(np.asarray([X])))
    print(y)



X_train, y_train, X_test, y_test = create_samples()
agent = Agent()
agent.build_model()
# test_on_sample(22, agent.model)
agent.train(X_train, y_train)
# agent.test(X_test, y_test)
