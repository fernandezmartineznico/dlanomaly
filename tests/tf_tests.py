import pandas as pd
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

import os, sys
sys.path.insert(0, '..')
from src.utils import reset_random_seeds

def run_reproducibility(x, y):
    reset_random_seeds()

    model = Sequential()    
    model.add(Dense(32, input_shape=(x.shape[1],), activation='relu'))    
    # final layer
    model.add(Dense(1, activation='linear'))

    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(x, y, epochs=10, verbose=0)
    predictions = model.predict(x).flatten()
    loss = model.evaluate(x,  y)
    return loss 

def test_reproducibility_tf():
    random_data = np.random.normal(size=(1000, 10))
    random_df = pd.DataFrame(data=random_data, columns=['x_' + str(ii) for ii in range(10)])
    y = random_df.sum(axis=1) + np.random.normal(size=(1000))

    run_1 = run_reproducibility(random_df, y)
    run_2 = run_reproducibility(random_df, y)
    assert run_1 == run_2, "tf results not reproducible, fix seeds"
