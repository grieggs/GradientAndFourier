import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys



data = pd.read_csv(sys.argv[1],header=None)

y = np.array(data[5])




grad = np.gradient(y)
fourier = np.fft.fft(y)

fig, axs = plt.subplots(3)

axs[0].set_title('Original Signal')
axs[0].plot(y)
axs[0].set_title('Gradient')
axs[1].axhline(0, linestyle='--', color='green')
axs[1].set_title('Fourier Transform')
axs[1].plot(grad)
axs[2].plot(fourier)
plt.show()
