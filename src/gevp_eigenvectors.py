# Plymouth 2019
# GEVP analysis designed for the output file produced with HiRep glueballs_op
# reading data form out_0 and saving correlators in a dictionary (matrices)
# the key of the dictionary is delta_t = t2 - t1

from __future__ import print_function, division
#import numpy as np
import sys
sys.path.append('../libs/')
#from jackknife import *
from matrices import *


'''
filename = sys.argv[1]
t1 = int(sys.argv[2])
t2 = int(sys.argv[3])
output_file = sys.argv[4]
'''

filename = "../input/file.dat"
t1 = 2
t2 = 3
output_file = "out_0"

verbose = 0

# TODO check scientific notation and precision
# formattedNumber(number, "e")
# printf  -> %15.15e
#np.set_printoptions(linewidth=200, suppress=False)



def main():
    if (len(sys.argv) < 5):
            print("\n USAGE: python gevp_eigenvectors.py input_filename t1 t2 output_filename\n")
            exit(0)
    print("\n analyzing input file: ", filename)
    print("\n t0    td:", t1, "\t", t2, "\n")
    matrices = read_data_3(filename)
    #TODO add T read from file
    T=24
    eigenvectors_C = []
    lambda_t1_t2_C, eigenvectors_C = eigenv_on_sample(matrices, t1, t2)
    print("\n rotating correlators on the GEVP basis ...")
    C_hat = dict()
    C_hat = compute_C_hat(matrices, eigenvectors_C)
    #write_to_file(output_file, t2, mean, error)


if __name__ == '__main__':
    main()
