import matplotlib.pyplot as plt
import numpy
import time
import pandas as pd
import os

print("Found files: ")
file_name = None
for file in os.listdir('/logs'):
    if file.endswith('.csv'):
        file_name = file

print("Opening: ", file_name)


def get_data():
    try:
        df = pd.read_csv('../' + file_name)
    except Exception as e:
        exit(e)

    return df


def read_data():

    df = get_data()

    ser = df.iloc[:, 0]
    alt = df.iloc[:, 1]
    rew = df.iloc[:, 2]

    return alt, rew, ser


def update():

    alt, rew, ser = read_data()

    plt.plot(ser, alt, label='altitude')
    plt.xlabel("Epochs")
    plt.ylabel("Max alt / reward")
    plt.title("Sincos 2d yaw and pitch, altitude also as state")
    plt.plot(rew, label='reward')

    z = numpy.polyfit(ser, alt, 1)
    p = numpy.poly1d(z)
    plt.plot(series, p(series), "r--", color='green', label='altitude_trend')

    zr = numpy.polyfit(ser, rew, 1)
    pr = numpy.poly1d(zr)
    plt.plot(series, pr(series), "r--", label='reward_trend')

    plt.legend()

    # the line equation:
    print("alt: y=%.6fx+(%.6f)"%(z[0],z[1]))
    print("rew: y=%.6fx+(%.6f)"%(zr[0],zr[1]))


prev_series = 0

while True:
    df = get_data()

    series = df.iloc[:, 0]
    altitude = df.iloc[:, 1]
    reward = df.iloc[:, 2]

    time.sleep(2)
    new_rows_number = len(series) - prev_series

    if new_rows_number > 0:
        update()
        plt.show()
        print(df.tail(new_rows_number))
        prev_series = len(series)
