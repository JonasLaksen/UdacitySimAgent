from keras.layers import Conv2D, Dropout, MaxPooling2D, Flatten, Dense
from keras.models import Sequential, load_model


class Agent:
    def build_model(self, load=False):
        if load:
            self.model = load_model('model.h5')
            return
        model = Sequential([
            Conv2D(10, (5,5), input_shape=(160,320,3), activation='relu', data_format='channels_last'),
            Dropout(.2),
            Conv2D(10, (5,5), activation='relu'),
            MaxPooling2D(pool_size=(3,3)),
            Flatten(),
            Dense(20, activation='relu'),
            Dropout(.2),
            Dense(10, activation='relu'),
            Dense(1, activation='sigmoid')
        ])
        model.compile(loss='mean_squared_error', optimizer='adam')
        self.model = model

    def train(self, X, y):
        self.model.fit(X,y,epochs=10)
        self.model.save('model.h5')

    def test(self, X, y):
        print self.model.test_on_batch(X, y)
