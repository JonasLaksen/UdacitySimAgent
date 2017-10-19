from keras.layers import Conv2D, Dropout, MaxPooling2D, Flatten, Dense
from keras.models import Sequential, load_model


class Agent:
    def build_model(self, load=False):
        if load:
            self.model = load_model('model.h5')
            return
        model = Sequential([
            Conv2D(32, (5,5), input_shape=(160,320,3), activation='relu', data_format='channels_last'),
            MaxPooling2D(),
            Conv2D(32, (5,5), activation='relu'),
            MaxPooling2D(),
            Conv2D(64, (5,5), activation='relu'),
            MaxPooling2D(),
            Conv2D(128, (5,5), activation='relu'),
            MaxPooling2D(),
            Flatten(),
            Dense(1024, activation='relu'),
            Dense(512, activation='relu'),
            Dense(1, activation='sigmoid')
        ])
        model.compile(loss='mean_squared_error', optimizer='adam')
        self.model = model

    def train(self, X, y):
        self.model.fit(X,y,epochs=10)
        self.model.save('model.h5')

    def test(self, X, y):
        print self.model.test_on_batch(X, y)
