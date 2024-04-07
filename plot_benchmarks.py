import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
import pandas as pd


bins = 1000
conv = np.load("conv_timings.npy")
fft = np.load("fft_timings.npy")
scan = np.load("scan_timings.npy")

print("Conv: {}".format(conv[1:].mean()))
print("FFT: {}".format(fft[1:].mean()))
print("Scan: {}".format(scan[1:].mean()))

X = np.concatenate((conv, fft, scan))
# y = np.concatenate((np.zeros_like(conv), np.zeros_like(fft) + 1, np.zeros_like(scan) + 2))
y = np.concatenate((["conv"] * len(conv), ["fft"] * len(fft), ["scan"] * len(scan)))
X = np.stack((X, y), 1)
df = pd.DataFrame(X, columns=["Timing", "Type"])
df.Timing = pd.to_numeric(df.Timing)
sns.kdeplot(data=df, x="Timing", hue="Type", bw_adjust=0.2)  # , cut=0)
plt.show()

