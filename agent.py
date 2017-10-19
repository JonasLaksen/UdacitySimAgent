from keras.layers import Conv2D, Dropout, MaxPooling2D, Flatten, Dense
from keras.models import Sequential, load_model
from keras.optimizers import Adam


class Agent:
    def build_model(self, load=False):
        if load:
            self.model = load_model('model.h5')
            return
        model = Sequential([
            Conv2D(32, (3,3), input_shape=(160,320,3), activation='relu', data_format='channels_last'),
            MaxPooling2D(pool_size=(3,3)),
            Conv2D(32, (3,3), activation='elu'),
            MaxPooling2D(pool_size=(3,3)),
            Conv2D(64, (3,3), activation='elu'),
            MaxPooling2D(pool_size=(3,3)),
            Conv2D(128, (3,3), activation='elu'),
            Flatten(),
            Dense(1024, activation='elu'),
            Dense(512, activation='elu'),
            Dense(1, activation='linear')
        ])
        model.compile(loss='mean_squared_error', optimizer=Adam(lr=.001))
        self.model = model

    def train(self, X, y):
        self.model.fit(X,y,epochs=10)
        self.model.save('model.h5')

    def test(self, X, y):
        print self.model.test_on_batch(X, y)
