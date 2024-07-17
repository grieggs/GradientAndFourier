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
axs[1].set_title('Gradient')
axs[1].axhline(0, linestyle='--', color='green')
axs[1].plot(grad)
# there is a lot of noise in the fourier transform that doesn't impact the signal, so I'm going to set the y-axis to a smaller range
axs[2].set_ylim([-50000,50000])
axs[2].set_title('Fourier Transform')
axs[2].plot(fourier)
plt.show()



# Commented this stuff out, you can play with changing the threshold values to see how it affects the signal.
# I would just use the fourier space as something to compare to other samples for now.

# thresh_h = 40000
# threshed_fourier = np.copy(fourier)
# threshed_fourier[fourier>thresh_h] = 0
# thresh_l = 10000
# threshed_fourier[fourier<thresh_l] = 0
# inverse_fourier = np.fft.ifft(threshed_fourier)

# fig2, axs2 = plt.subplots(3)
# axs2[0].set_title('Original Signal')
# axs2[0].plot(y)
# axs2[1].set_title('Fourier Transform, with threshold line')
# axs2[1].axhline(thresh_h, linestyle='--', color = 'red', label = 'threshold line')
# axs2[1].axhline(thresh_l, linestyle='--', color = 'red', label = 'threshold line')
# axs2[1].plot(threshed_fourier)
# axs2[2].set_title('Reverted transform, with fourier zeroed below threshold')
# axs2[2].plot(inverse_fourier)

# plt.show()
