import matplotlib.pyplot as plt
import numpy as np

# Fixing random state for reproducibility
np.random.seed(19680801)

dt = 0.05
t1 = np.arange(0.0, 20.0, dt)
t2 = np.arange(20.0,40, dt)

s1 = np.sin(5 *2* np.pi*t1 )
# s1 = np.sin(2 * np.pi * 100 * t)

s2 = np.sin(2 * np.pi *t2)

s3 = np.concatenate([s1,s2])
# create a transient "chirp"
# s2[t <= 10] = s2[12 <= t] = 0

# add some noise into the mix
# nse = 0.01 * np.random.random(size=len(t))

# x = s1 + s2 + nse  # the signal
NFFT = 102  # the length of the windowing segments
Fs = int(1.0 / dt)  # the sampling frequency

fig, (ax1, ax2,ax3,ax4) = plt.subplots(nrows=4)
ax1.plot(t1, s1)
ax2.plot(t1,s2)
t3 = np.concatenate((t1,t2))
ax3.plot(t3,s3)
Pxx, freqs, bins, im = ax4.specgram(s3,cmap='hot', NFFT=NFFT, Fs=Fs, noverlap=0,mode="psd")
# The `specgram` method returns 4 objects. They are:
# - Pxx: the periodogram
# - freqs: the frequency vector
# - bins: the centers of the time bins
# - im: the .image.AxesImage instance representing the data in the plot
print(max(s1),max(s2))
plt.savefig("spectrum/test.png")
plt.show()
