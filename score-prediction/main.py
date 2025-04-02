import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import preprocessing
import keras
import tensorflow as tf

ipl = pd.read_csv('data.csv')
df = ipl.drop(['date', 'runs', 'wickets', 'overs', 'runs_last_5', 'wickets_last_5', 'mid', 'striker', 'non-striker'], axis=1)

X = df.drop(['total'], axis=1)
y = df['total']

from sklearn.preprocessing import LabelEncoder

venue_encoder = LabelEncoder()
batting_team_encoder = LabelEncoder()
bowling_team_encoder = LabelEncoder()
striker_encoder = LabelEncoder()
bowler_encoder = LabelEncoder()

X['venue'] = venue_encoder.fit_transform(X['venue'])
X['bat_team'] = batting_team_encoder.fit_transform(X['bat_team'])
X['bowl_team'] = bowling_team_encoder.fit_transform(X['bowl_team'])
X['batsman'] = striker_encoder.fit_transform(X['batsman'])
X['bowler'] = bowler_encoder.fit_transform(X['bowler'])

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

model = keras.Sequential([
    keras.layers.Input(shape=(X_train_scaled.shape[1],)),
    keras.layers.Dense(512, activation='relu'),
    keras.layers.Dense(216, activation='relu'),
    keras.layers.Dense(1, activation='linear'),
])

huber_loss = tf.keras.losses.Huber(delta=1.0)
model.compile(optimizer='adam', loss=huber_loss)
model.fit(X_train_scaled, y_train, epochs=50, batch_size=64, validation_data=(X_test_scaled, y_test))

model_losses = pd.DataFrame(model.history.history)
model_losses.plot()
plt.show()

predictions = model.predict(X_test_scaled)
from sklearn.metrics import mean_absolute_error, mean_squared_error
print(mean_absolute_error(y_test, predictions))

import warnings
warnings.filterwarnings('ignore')

import tkinter as tk
from tkinter import ttk

root = tk.Tk()
root.title('ipl score prediction')
root.geometry("300x250")

tk.Label(root, text="Select Venue:").pack(); venue = ttk.Combobox(root, values=df['venue'].unique().tolist()); venue.pack(); venue.current(0)
tk.Label(root, text="Select Batting Team:").pack(); batting_team = ttk.Combobox(root, values=df['bat_team'].unique().tolist()); batting_team.pack(); batting_team.current(0)
tk.Label(root, text="Select Batting Team:").pack(); bowling_team = ttk.Combobox(root, values=df['bowl_team'].unique().tolist()); bowling_team.pack(); bowling_team.current(0)
tk.Label(root, text="Select Striker:").pack(); striker = ttk.Combobox(root, values=df['batsman'].unique().tolist()); striker.pack(); striker.current(0)
tk.Label(root, text="Select Bowler:").pack(); bowler = ttk.Combobox(root, values=df['bowler'].unique().tolist()); bowler.pack(); bowler.current(0)

def predict_score():
    decoded_venue = venue_encoder.transform([venue.get()])
    decoded_batting_team = batting_team_encoder.transform([batting_team.get()])
    decoded_bowling_team = bowling_team_encoder.transform([bowling_team.get()])
    decoded_striker = striker_encoder.transform([striker.get()])
    decoded_bowler = bowler_encoder.transform([bowler.get()])

    input = np.array([decoded_venue, decoded_batting_team, decoded_bowling_team, decoded_striker, decoded_bowler])
    input = input.reshape(1, 5)
    input = scaler.transform(input)

    predict_score = model.predict(input)
    predict_score = int(predict_score[0, 0])

    output.config(text=f"{predict_score}")


tk.Button(root, text="Submit", command=predict_score).pack()
output = tk.Label(root, text="")
output.pack()

root.mainloop()