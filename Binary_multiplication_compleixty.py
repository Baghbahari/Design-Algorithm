import random
import time
import matplotlib.pyplot as plt
import sys


def calculate_complexity():
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

    return(Size, Time, Size_t, Space, Size_s)

def plot_data(Size, Time, Size_t, Space, Size_s):
    plt.plot(Size, Time)
    plt.plot(Size, Size_t)
    plt.figure()
    plt.plot(Size, Space)
    plt.plot(Size, Size_s)
    plt.show()

def main():

    PID = 3977670
    print('My PID: ' + str(PID))
    print('The reminder of division of my PID by 4: ' + str(PID%4))
    print('The implementation is multiplication')

    Size, Time, Size_t, Space, Size_s = calculate_complexity()

    plot_data(Size, Time, Size_t, Space, Size_s)

if __name__ == '__main__':
    main()
