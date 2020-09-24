target = __import__("gevp_eigenvectors")
divide = target.divide
mean_matrix = target.mean_matrix
eigenv_on_sample = target.eigenv_on_sample
read_data_3 = target.read_data_3
pretty = target.pretty
compute_C_hat = target.compute_C_hat

import unittest
import numpy as np

def check_symmetric(a, rtol=1e-05, atol=1e-08):
    return np.allclose(a, np.matrix.transpose(a), rtol=rtol, atol=atol)

def check_real(a, rtol=1e-05, atol=1e-08):
    return np.allclose(0.0, a.imag, rtol=rtol, atol=atol)

class TestReadingDataFromFile(unittest.TestCase):
    def read_data(self):
        """
        Test reading data from file
        """
        #matrices = read_data_3("Corr_Total_L6_P=(0,0,0)_Irrep=EplusOhP_Irrep_ev=1x2_Charge=+.dat")
        matrices = read_data_3("../input/test4")
        print(len(matrices[0]))
        print(matrices[0][0])
        print(matrices[0][3])

class TestRotateMatrix(unittest.TestCase):
    def diag_mat(self):
        """
        Test diagonalizing matrix
        """

        mat = np.array([[1.354405e-03, 1.345483e-02], [1.345483e-02, 1.309134e-01]])
        mat_hat = []
        eigenvals, eigenvectors_C = np.linalg.eig(mat)
        matrix_C = []
        dim_mat=2

        for i in range (0, dim_mat):
            print("i = ", i)
            for j in range (0, dim_mat):
                print("j = ", j)
                matrix_C[i][j] = ((eigenvectors_C[i]).dot( mat.dot(eigenvectors_C[j]) )  )

        mat_hat.append(matrix_C)
        print("mat ")
        print(mat)
        print("rotated mat ")
        print(mat_hat)

class TestSymmetryCorrelator(unittest.TestCase):
    def test_symm_corr(self):
        """
        Test that the correlator C_ij is symmetric by def.
        C_ij = outer [O_i(t1)*, O_j(t2)] + outer [O_j(t1)*, O_i (t2)] + outer [O_i(t2)*, O_j(t1)] + outer [O_i(t1), O_j*(t2)]
        """

        t1=0
        t2=0
        #TODO read T from file
        T=24
        matrices = read_data_3("../input/test4")
        meas = 2

        for key in matrices.keys():
            for j in range (0, meas):
                result = check_symmetric( matrices[key][j], rtol=1e-02, atol=1e-03)
                if result == False:
                    print("non symmetric element t ",key,"  measurement ",j)
                self.assertEqual(result, True)

    def test_reality_eigv(self):
        """
        Test that the correlator C_ij has real eigenvalues
        within tolerance ~ 10^-12
        """
        t1=0
        t2=0
        #TODO read T from file
        T=24
        matrices = read_data_3("../input/test4")
        meas = 2 #len(matrices[T/2])
        ## test reality of eigenvalues
        for key in matrices.keys():
            for j in range (0, meas):
                dict_lambda_t1_t2, dict_lambda_t1_t2_eigenv = np.linalg.eig(matrices[key][j])
                result = check_real(dict_lambda_t1_t2_eigenv.all(), rtol=1e-12, atol=1e-11)
                self.assertEqual(result, True)


class TestDivide(unittest.TestCase):
    def test_divide(self):
        """
        Test division
        """
        dict_1 = {1:1, 2:1, 3:1}
        dict_2 = {1:2, 2:2, 3:2}
        result = divide(dict_1, dict_2)
        expected_result = {1:0.5, 2:0.5, 3:0.5}
        self.assertEqual(result, expected_result)


class TestEigenv(unittest.TestCase):
    def test_mean(self):
        """
        Test mean matrix
        """
        mat = {0:[[ 1,  1 ,  1], [ 1,  1 ,  1], [ 1,  1,  1]], 1:[[ 1,  1 ,  1], [ 1,  1 ,  1], [ 1,  1,  1]] , 2:[[ 1,  1 ,  1], [ 1,  1 ,  1], [ 1,  1,  1]], 3:[[ 1,  1 ,  1], [ 1,  1 ,  1], [ 1,  1,  1]]  }
        expected_result = [[ 1,  1 ,  1], [ 1,  1 ,  1], [ 1,  1,  1]]
        result = mean_matrix(mat, 4)
        np.testing.assert_array_equal(result, expected_result)

    def test_real_eigv_singleCorr(self):
        """
        Test that the single correlator has real eigenvalues
        """
        t1=0
        t2=0
        #TODO read T from file
        T=24
        matrices = read_data_3("../input/test4")

        if len(matrices) <= 0:
            print("ERROR: no key T/2 !!!!!! ")
            exit(-1)
        meas = 2

        for key in matrices.keys():
            for j in range (0, meas):
                result = check_symmetric( matrices[key][j], rtol=1e-02, atol=1e-03)
                if result == False:
                    print("non symmetric element t ",key,"  measurement ",j)
                self.assertEqual(result, True)

        '''
        if not np.allclose(a, np.asmatrix(corr_2pt_d[key][j]).H):
            print("expected hermitian matrix !")
        '''

        dict_lambda_t1_t2_C_tilde, dict_lambda_t1_t2_C_tilde_eigvect = eigenv_on_sample(matrices, t1, t2)




if __name__ == '__main__':
    unittest.main()
