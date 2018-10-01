import random
import time
import matplotlib.pyplot as plt
import sys


Size_t = []
Size_s = []
Time = []
Space = []
Size = []
for power in range(2, 10):
    size_t = 2**power
    Size.append(size_t)
    S = sys.getsizeof(random.getrandbits(size_t))
    print(S)
    Space.append(S)
    t0 = time.clock()
    for digit in xrange(1000):
        random.getrandbits(size_t) * random.getrandbits(size_t)
    time_e = (time.clock() - t0)
    Time.append(time_e)
    print(Time[-1])
    Size_t.append(Time[0] + 0.0000000055*size_t**2)
    Size_s.append(Space[0]+0.15*size_t)
plt.plot(Size, Time)
plt.plot(Size, Size_t)
plt.figure()
plt.plot(Size, Space)
plt.plot(Size, Size_s)
plt.show()
