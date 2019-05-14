import sys, os
import numpy as np
import matplotlib.pyplot as plt
import timeit, math

def get_f_values(start_point, end_point, num_points):

    x = np.linspace(start_point, end_point, num_points)
    f_values = [i**3 for i in x]
    step_size = x[1] - x[0]

    return(f_values, step_size)

def get_rl_coeffs(index_k, index_j, alpha):

    if index_j == 0:
        return ((index_k-1)**(1-alpha)-(index_k+alpha-1)*index_k**-alpha)
    elif index_j == index_k:
        return 1
    else:
        return ((index_k-index_j+1)**(1-alpha)+(index_k-index_j-1)**(1-alpha)-2*(index_k-index_j)**(1-alpha))

def get_rl_matrix(alpha, n):

    coeff_matrix = np.zeros((n,n))
    for i in range(n):
        for j in range(i):
            coeff_matrix[i,j] = get_rl_coeffs(i,j,alpha)

    np.fill_diagonal(coeff_matrix,1)

    return coeff_matrix/math.gamma(2-alpha)


def get_rl(alpha, start_point, end_point, num_points):

    f_values, step_size = get_f_values(start_point, end_point, num_points)

    diff_int = get_rl_matrix(alpha, num_points)
    rl = step_size**-alpha*np.dot(diff_int, f_values)

    return rl

def get_alg_coeffs(alpha,n):

    coeffs = np.zeros(n+1,)
    coeffs[0] = 1

    for i in range(n):
        coeffs[i+1] = coeffs[i]*(-alpha + i)/(i+1)

    return coeffs

def get_alg(alpha, start_point, end_point, num_points):

    f_values, step_size = get_f_values(start_point, end_point, num_points)

    b_coeffs = get_alg_coeffs(alpha, num_points-1)

    b_c = np.fft.rfft(b_coeffs)
    f_c = np.fft.rfft(f_values)

    result = np.fft.irfft(f_c*b_c)*step_size**-alpha

    return result

def gen_results():
    #get_f_values(f, 0, 4, 10)
    #sys.exit()
    print('\n fractional derivative for function f(t)=t^3 at t=4')
    for d in range(1,4):
        print('\n Derivative order: '+str(0.25*d))
        d_vals = get_rl(0.25*d,0,4,500)
        print(' Riemann-Liouville algorithm: ' + str(d_vals[-1]))
        d_vals =get_alg(0.25*d,0,4,500)
        print(' The algorithm with FFT: ' + str(d_vals[-1]))

    fig0 = plt.figure()

    for d in range(0,4):
        d_vals = get_rl(0.25*d,0,4,500)
        plt.plot(d_vals)


    Time = []
    n = 500
    print('\n Running time for Riemann-Liouville algorithm ...')
    for i in range(2,n+1):
        if i%100 ==0:
            print(' number of interval points: '+str(i))
        T = timeit.timeit(lambda:get_rl(0.5,0,4,i), number=1)
        Time.append(T)

    fig1 = plt.figure()
    plt.plot(Time)
    plt.savefig('RL.png', dpi=fig1.dpi)
    #plt.show()
    #sys.exit()
    fig2 = plt.figure()
    plt.loglog(Time)
    #plt.plot(0.01*np.array(Time))
    Time = []
    n = 500
    print('\n Running time for the algorithm with FFT ...')
    for i in range(2,n+1):
        if i%100 ==0:
            print(' number of interval points: '+str(i))
        T = timeit.timeit(lambda:get_alg(0.5,0,4,i), number=2)
        Time.append(T)
        #print(i, T)
    #print(get_alg(0.5,0,4,i)[-1])
    #plt.figure()
    #plt.loglog([i for i in range(2,n)], Time)
    plt.loglog(Time)
    plt.grid()
    plt.savefig('Compare.png', dpi=fig2.dpi)
    fig3 = plt.figure()
    plt.plot(Time)
    plt.savefig('FFT_based.png', dpi=fig3.dpi)
    #plt.loglog([0.00001*i for i in range(2,n)])
    #plt.plot([i for i in range(2,n)], [0.0000001*i**2 for i in range(2,n)])
    #plt.plot([1*0.0000005*i*np.log(i) for i in range(2,n)])
    print('\n (Ctrl + \\) to exit')
    plt.show()

def main():
    gen_results()

if __name__ == '__main__':
    main()
