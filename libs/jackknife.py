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
