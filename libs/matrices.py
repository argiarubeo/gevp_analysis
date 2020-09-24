import numpy as np

t1 = 2
t2 = 3

def pretty(d, indent=0):
   for key, value in d.items():
      print('\t' * indent + str(key), end="")
      if isinstance(value, dict):
         pretty(value, indent+1)
      else:
         print('\t' * (indent+1) + str(value), "\n")

def pretty_on_file(f, d, indent=0):
   for key, value in d.items():
      f.write(str('\t' * indent + str(key) ))
      if isinstance(value, dict):
         pretty_on_file(value, indent+1)
      else:
         f.write(str('\t' * (indent+1) + str(value) ))
         f.write(str("\n"))

def print_info_array(arr):
    print ("************* info ****************")
    print ("type: ", type(arr) )
    print ("shape: ", arr.shape )
    print ("array dim: ", len(arr) )
    print ("array: \n", arr)
    print ("************* end info ****************")

def dict_times_scalar(dictionary, scal):
    ret = dict()
    for key in dictionary.keys():
        ret[key] = scal * dictionary[key]
    return ret

# it computes the mean on jack_data_ if all=True
# it computes the mean on a sample where the skip_key-th element is eliminated if all=False
def jack_mean(jack_data_, skip_key, all):
  number_meaures = len(jack_data_.keys())
  jack_sum =np.zeros(len(jack_data_[0]))
  for i in range(0, number_meaures):
      if (i == skip_key and all == False):
          continue
      jack_data_[i] = np.sort(jack_data_[i])
      #print("========= jack i ====", jack_data_[i])
      jack_sum = jack_sum + jack_data_[i]

  result = (jack_sum/number_meaures ) if all else (jack_sum/(number_meaures-1))
  #print("========= jack mean ====", result)
  return result

def jack_mean_mat(jack_data_mat, skip_key, all):
  number_meaures = len(jack_data_mat)
  shape = (len(jack_data_mat[0]), len(jack_data_mat[0]))
  jack_sum =np.zeros(shape)
  for i in range(0, number_meaures):
      if (i == skip_key and all == False):
          continue
      jack_sum = jack_sum + jack_data_mat[i]

  result = (jack_sum/number_meaures ) if all else (jack_sum/(number_meaures-1))
  return result

def jack_error(jack_data_):

  number_meaures = len(jack_data_.keys())
  avg = jack_mean(jack_data_, -1, True)
  jack_resum = 0.0

  for i in range(0, number_meaures ):
    jack_resum += ( avg - jack_mean(jack_data_, i, False) ) ** 2

  return ( jack_resum * (number_meaures - 1.0)/(number_meaures ) )** 0.5

def jack_error_mat(jack_data):
  number_meaures = len(jack_data)
  avg = jack_mean_mat(jack_data, -1, True)
  jack_resum = 0.0

  for i in range(0, number_meaures ):
    jack_resum += ( avg - jack_mean_mat(jack_data, i, False) ) ** 2

  return ( jack_resum * (number_meaures - 1.0)/(number_meaures ) )** 0.5

def jack_covariance(j1,j2):
    mean_j1 = jack_mean(j1)
    mean_j2 = jack_mean(j2)
    cov = 0.0
    n_meas = len(j1)
    if (n_meas != len(j2)):
        print("ERROR: different sizes !!! ", n_meas, ", ", len(j2))
        exit(1)
    for i in range(0, n_meas ):
        cov += (complex(j1[i])-mean_j1)*(complex(j2[i])-mean_j2)
    return cov/(n_meas*(n_meas-1.0))

def jack_correlation(j1,j2):
    return jack_covariance(j1,j2)/(jack_error(j1)*jack_error(j2))

def print_jack(jack_data_):
  n_meas = len(jack_data_)
  print("n_meas=", n_meas)
  for i in range(0,n_meas):
    print(jack_data_[i])

def mean_matrix(matrices_data, number_meaures):
    #TODO pass the dimension of the matrix instead of len(matrices_data[0][0])
    s=(len(matrices_data[0][0]), len(matrices_data[0][0]))
    mat = np.zeros(s)
    for meas_index in range (0, number_meaures):
        mat = mat + matrices_data[meas_index]
    mat = mat/number_meaures
    return mat

def normalize(v):
    norm = np.linalg.norm(v)
    #print("norm = ", norm)
    if norm == 0:
       return v
    return v / norm

def slice_vector(v,index_to_skip):
  sliced_v = []
  for i in range (0,len(v)):
    if i == index_to_skip:
      continue
    else:
     sliced_v.append(v[i])
  return np.asarray(sliced_v)

def normalize_matrix (mat):
    dim = len(mat)
    norm = 0.0
    for i in range(0, dim):
        for j in range(0, dim):
            if(i == j):
                norm = norm + abs(mat[i][j])

    mat = np.divide(mat,math.sqrt(norm))
    return mat

def eigenv_on_sample(matrices_data, t1, t2):
    number_meaures = len(matrices_data[t1])
    print(" number of measures \n",number_meaures)
    n = len(matrices_data[t1][0])
    # compute the mean of the matrix at given t over the measurements contained in the file
    mean_t0 = mean_matrix(matrices_data[t1], number_meaures)
    mean_td = mean_matrix(matrices_data[t2], number_meaures)
    # impose N is symmetric
    #N = 0.5 * (N.transpose() + N)
    # TODO hermitian in genral
    # NOT NEEDED it is by def. and there is a unit test in testing.py
    N = np.linalg.inv(mean_t0).dot(mean_td)
    N = 0.5 * (N.transpose() + N)
    print("\n mean of correlators over all measurements \n", N)

    # solve GEVP once
    lambda_t1_t2_C, eigenvectors_C = np.linalg.eig(N)

    print("\n eigenvals \n", lambda_t1_t2_C)
    for j in range (0, n):
        if(all(i < 0 for i in eigenvectors_C[j])):
            eigenvectors_C[j] = np.dot(-1.0, eigenvectors_C[j])

    # oredering eigenpairs using lambda as a key
    idx = lambda_t1_t2_C.argsort()[::-1]
    lambda_t1_t2_C = lambda_t1_t2_C[idx]
    eigenvectors_C = eigenvectors_C[:,idx]

    # normalizing eigenvectors
    for j in range (0, n):
        eigenvectors_C[j] = normalize(eigenvectors_C[j])

    print("\n eigenvectors \n", eigenvectors_C)
    return lambda_t1_t2_C, eigenvectors_C

def compute_C_hat(matrices_data, eigenvectors_C):
    C_hat = dict()
    C_hat_2 = dict()
    #TODO pass dimension of matrix instead of len(matrices_data[0][0])
    dim_mat = len(matrices_data[0][0])
    number_meaures = len(matrices_data[0])
    s=(dim_mat, dim_mat)
    mean_t0 = np.linalg.inv(mean_matrix(matrices_data[t1], number_meaures))
    mean_td = mean_matrix(matrices_data[t2], number_meaures)

    # for each value of t C(t) is rotated to C_hat(t)
    for key in matrices_data.keys():
        if (key == 0):
            continue
        for meas_index in range (0, number_meaures):
            slice_mat = slice_vector(matrices_data[key], meas_index)
            mat_mean = mean_matrix(slice_mat, number_meaures-1)
            mat_mean = mat_mean.dot(mean_t0)
            matrix_C = np.zeros(s, dtype=float)
            matrix_C_err = np.zeros(s, dtype=float)
            for i in range (0, dim_mat):
                for j in range (0, dim_mat):
                    matrix_C[i][j] = ((eigenvectors_C[i]).dot( mat_mean.dot(eigenvectors_C[j]) )  )
                    # error on the rotated matrix
                    matrix_C_err[i][j] = ((eigenvectors_C[i]).dot( matrices_data[key][meas_index].dot(eigenvectors_C[j]) )  )
            # error on the matrix not rotated(? why so small ?)
            if key in C_hat.keys():
                C_hat[key].append(matrix_C)
                C_hat_2[key].append(matrix_C_err)
            else:
                C_hat[key] = [matrix_C]
                C_hat_2[key] = [matrix_C_err]

    print("\n#t     #mean[C^-1(t0)] j_samp[C_hat(td)] diag 00         #jack error")
    for key in C_hat.keys():
        if (key == 0):
            continue
        C_hat_mean_t = mean_matrix(C_hat[key], number_meaures)
        C_hat_err = jack_error_mat(C_hat_2[key])
        print(key, "\t%.10e" %(C_hat_mean_t[0][0]), "\t\t\t\t", C_hat_err[0][0])
    print("\n#t     #mean[C^-1(t0)] j_samp[C_hat(td)] diag 11         #jack error")
    for key in C_hat.keys():
        C_hat_mean_t = mean_matrix(C_hat[key], number_meaures)
        C_mean = mean_matrix(matrices_data[key], number_meaures)
        C_hat_err = jack_error_mat(C_hat_2[key])
        print(key, "\t%.10e" %(C_hat_mean_t[1][1]), "\t\t\t\t", C_hat_err[1][1])#, "\n", C_mean[1][1], "\t\t\t", C_err[1][1])
    print("\n#t     #mean[C^-1(t0)] j_samp[C_hat(td)] off-diag        #jack error")
    for key in C_hat.keys():
        C_hat_mean_t = mean_matrix(C_hat[key], number_meaures)
        C_hat_err = jack_error_mat(C_hat_2[key])

        for i in range (0, dim_mat):
            for j in range (0, dim_mat):
                if (i != j) and (i == 1):
                    print(key, "\t%.10e" %(C_hat_mean_t[i][j]), "\t\t\t\t", C_hat_err[i][j])
    return C_hat

def divide(dividends, divisors):
    ret = dict()
    for key, dividend in dividends.items():
        if key in divisors:
            ret[key] = dividend/divisors[key]
        else:
            print ("ERROR dict have not the same size !!! ")
            ret[key] = dividend
    return ret

def divide_log(dividends, divisors):
    ret = dict()
    for key in dividends.keys():
        ret[key] = np.log (dividends[key]/divisors[key], dtype=np.complex)
    return ret

def extract_t_meas_from_line(line_text):
    splitted_line = line_text.split()
    meas = int(splitted_line[0])
    t = int(splitted_line[1])
    return t, meas

def append_data(data_list, line_text):
    splitted_line = line_text.split()
    line_list = []
    for i in range(0, len(splitted_line)):
        # TODO if want to go to complex change float
        line_list.append( float(splitted_line[i]) )
    data_list.append( line_list )
    return data_list

def read_data_3(filename):
    matrices = dict()
    with open (filename,'r') as f:
        content = f.readlines()
        data_list = []
        t = -1
        meas = -1
        read_t_meas = False
        for iline, line in enumerate(content):
            if len(line) <= 2:
                read_t_meas = True
                if(len(data_list) != 0):
                    matrix = np.array(data_list) # coverting data from list to matrix
                    data_list = []
                    if t in matrices.keys(): # Appendig to dictionary the current matrix
                        matrices[t].append(matrix)
                    else: # otherwise create
                        matrices[t] = [matrix]
                continue # skip empty line
            else:
                #print("read measure:",read_t_meas)
                if (read_t_meas == True):
                    t, meas = extract_t_meas_from_line(line)
                    read_t_meas = False
                    #print("reading data for t: ", t, "meas: ", meas)
                else:
                    #print("appending data to list ")
                    data_list = append_data(data_list, line)
                    #print(data_list)
    return matrices

def write_to_file(output_file, t2, mean, error):
    print("Appending data to file: ", output_file)
    with open (output_file,'a+') as f:
        # mean is lambda_mean
        mean = np.array(mean)
        # taking the real part
        #TODO add in testing: check that the Im part is zero within complex precision
        mean = mean.real
        error = np.array(error)
        error = error.real
        error_ord = [x for _,x in sorted(zip(mean, error))]
        mean_ord = np.sort(mean)
        f.write("# t"+"\t"+"lambda_mean"+"\t\t"+"jack_error"+"\n")
        for i in range (0, len(mean)):
            f.write(str(t2)+"\t")
            f.write(str("%.10e" %mean_ord[i])+"\t")
            f.write(str("%.10e" %error_ord[i])+"\t")
            f.write(" \n")
    f.close()

def fit_model(matrices_data, t1, t2, delta):
    return divide_log(eigenv_on_sample(matrices_data, t1, t2), eigenv_on_sample(matrices_data, t1, t2+delta) )

def chi_squared(n, y, yfit, yerr):
    s = 0
    chi=0.0
    for i in range(1, n):
        s = np.sum((yfit - y) ** 2  / yerr ** 2 )
        chi = chi + s
    return chi
