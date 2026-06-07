from convolution.filters import K1, K2, K3, K4, K5, K6, convolution_grayscale
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import time

image = mpimg.imread("convolution/photo_chat.png")

t0 = time.perf_counter()
f1 = convolution_grayscale(image, K1)
t1 = time.perf_counter()
print(f"f1 done — {t1 - t0:.4f} s")

t0 = time.perf_counter()
f2 = convolution_grayscale(image, K2)
t1 = time.perf_counter()
print(f"f2 done — {t1 - t0:.4f} s")

t0 = time.perf_counter()
f3 = convolution_grayscale(image, K3)
t1 = time.perf_counter()
print(f"f3 done — {t1 - t0:.4f} s")

t0 = time.perf_counter()
f4 = convolution_grayscale(image, K4)
t1 = time.perf_counter()
print(f"f4 done — {t1 - t0:.4f} s")

t0 = time.perf_counter()
f5 = convolution_grayscale(image, K5)
t1 = time.perf_counter()
print(f"f5 done — {t1 - t0:.4f} s")

t0 = time.perf_counter()
f6 = convolution_grayscale(image, K6)
t1 = time.perf_counter()
print(f"f6 done — {t1 - t0:.4f} s")

images = [f1, f2, f3, f4, f5, f6]
names = ["f1", "f2", "f3", "f4", "f5", "f6"]

for im, name in zip(images, names):
  plt.imsave(f"outputs/{name}.png", im, cmap="gray")