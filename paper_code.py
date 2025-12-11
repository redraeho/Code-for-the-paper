import numpy as np
from numpy.random import normal as normal
import time
from pathlib import Path
#from numpy.linalg import norm as norm
#from scipy.stats import linregress
import matplotlib.pyplot as plt
from scipy.sparse import coo_matrix
#from scipy.sparse.linalg import expm as sparse_expm
from scipy.linalg import expm as dense_expm
#from scipy.sparse.linalg import inv as sparse_inv
#from scipy.sparse import identity

st = time.time()
N_power = 6 # Determines the maximum spatial mesh by dividing each side of (0,1)^2 by 2^N_power (or h'=2^{-N_power} or
            # h=sqrt{2}2^{-N_power}.
M_power = 12 #Determines the maximum temporal discretization by M = 2^M_power. We use 9 for the accurate weak error and 12 for the
             #strong error.
T_power = -1 # Determines T by T=2^T_power.
max_samples = 150 # Determines the number of samples used to estimate the expectation in the error.
max_samples_weak = 10000 # Determines the number of samples used to estimate the expectation in the weak error (more accurate version).
                        # Make sure to use a multiple of 10 so the final sums are saved.
M_max = 2**M_power # This is the M used in the reference solutions.
tau_min = 2**(T_power - M_power) # This is the tau used in the reference solutions.
K_power = 1 # This determines the dimension of the noise to be K = 2^{2*K_power}, or using the basis functions up to the
            # (2^{K_power},2^{K_power}) gridpoint. We use 1 and 5.
K_true = 2**(2 * K_power)
use_square_root_approximation = True # determines whether we use the C^1 Lipschitz approximation of the square root.
nonlinear_delta = 0.1 # This determines the precision of the approximation of the C^1 Lipschitz approximation of the square root.

# This is the function for the initial condition.
def initial_condition(x,y):
    return np.sin(np.pi * x) * np.sin(np.pi * y)
    
# These are the orthonormal functions in the noise indexed by the grid.
def e_func(m, n, x, y):
    return 2 * np.sin(np.pi * m * x) * np.sin(np.pi * n * y)

# This is the square root approximation function. Specifically the corresponding g in f(s)=g(s)s.
def square_root_approx(delta_param, x):
    answer = 0.0
    if np.abs(x) <= delta_param / 2:
        answer = 1 / np.sqrt(delta_param)
    elif np.abs(x) >= delta_param:
        answer = 1 / np.sqrt(np.abs(x))
    else:
        answer = -2 * np.sqrt(delta_param) * x**2 / delta_param**3 + 4 / (delta_param * np.sqrt(delta_param)) * np.abs(x) - 3 / (2 * np.sqrt(delta_param)) + np.sqrt(delta_param) / (2 * np.abs(x))
    return answer

# This is the function called, for g in f(s)=g(s)s.
def nonlinearity(x):
    answer = 0.0
    if use_square_root_approximation:
        answer = square_root_approx(nonlinear_delta, x)
    else: #put your function here. Currently the g associated to max{0,sqrt(x)}, with g(0)=0.
        if x > 0:
            answer = 1 / np.sqrt(x)
        else:
            answer = 0.0
    return answer

nonlinearity_vectorized = np.vectorize(nonlinearity, otypes=['float'])

# This is the functional that is used in the weak error.
def weak_functional(mass_matrix, x):
    return (mass_matrix.dot(x)).dot(x)
    #return np.exp(-((mass_matrix.dot(x)).dot(x)))

# This converts a grid index (m, n) to a scalar index k. Used to go between indexing of the e's and indexing of the brownian increments.
def grid_to_index(m, n):
    answer = 0
    if m > n:
        answer = (m - 1)**2 + n
    elif m < n:
        answer = (n - 1)**2 + n - 1 + m
    else:
        answer = m**2
    return answer

# Generates the mass matrix for the interior nodes for a given mesh value. Sparse format. Needed to calculate the error.
def get_2d_M_SG(N):
    entries = np.zeros(7 * N**2 - 22 * N + 17)
    rows = np.zeros(7 * N**2 - 22 * N + 17)
    columns = np.zeros(7 * N**2 - 22 * N + 17)
    index = 0
    for k in range(1,(N - 1)**2 + 1):
        entries[index] = 1 / (2 * N**2)
        rows[index] = k - 1
        columns[index] = k - 1
        index += 1
        if (k - 1) % (N - 1) != 0:
            entries[index] = 1 / (12 * N**2) 
            rows[index] = k - 1
            columns[index] = k - 2
            index += 1
            if (k - 1) // (N - 1) != N - 2:
                entries[index] = 1 / (12 * N**2) 
                rows[index] = k - 1
                columns[index] = k + N - 3
                index += 1
        if (k - 1) // (N - 1) != N - 2:
            entries[index] = 1 / (12 * N**2) 
            rows[index] = k - 1
            columns[index] = k + N - 2
            index += 1
        if k % (N - 1) != 0:
            entries[index] = 1 / (12 * N**2) 
            rows[index] = k - 1
            columns[index] = k
            index += 1
            if (k - 1) // (N - 1) != 0:
                entries[index] = 1 / (12 * N**2) 
                rows[index] = k - 1
                columns[index] = k - N + 1
                index += 1
        if (k - 1) // (N - 1) != 0:
            entries[index] = 1 / (12 * N**2) 
            rows[index] = k - 1
            columns[index] = k - N
            index += 1
    return coo_matrix((entries, (rows, columns)), shape=((N - 1)**2, (N - 1)**2)).tocsc()

# Generates the stiffness matrix for the interior nodes for a given mesh value. Sparse format.
def get_2d_S_SG(N):
    entries = np.zeros(5 * N**2 - 14 * N + 9)
    rows = np.zeros(5 * N**2 - 14 * N + 9)
    columns = np.zeros(5 * N**2 - 14 * N + 9)
    index = 0
    for k in range(1,(N - 1)**2 + 1):
        entries[index] = 4
        rows[index] = k - 1
        columns[index] = k - 1
        index += 1
        if (k - 1) % (N - 1) != 0:
            entries[index] = -1 
            rows[index] = k - 1
            columns[index] = k - 2
            index += 1
        if (k - 1) // (N - 1) != N - 2:
            entries[index] = -1 
            rows[index] = k - 1
            columns[index] = k + N - 2
            index += 1
        if k % (N - 1) != 0:
            entries[index] = -1 
            rows[index] = k - 1
            columns[index] = k
            index += 1
        if (k - 1) // (N - 1) != 0:
            entries[index] = -1
            rows[index] = k - 1
            columns[index] = k - N
            index += 1
    return coo_matrix((entries, (rows, columns)), shape=((N - 1)**2, (N - 1)**2)).tocsc()

# Generates the matrix needed to change a discretized solution in mesh size 2**-k to 2**-k_max. This is so energy calculations are
# easier.
def get_basis_change_2d(k, k_max):
    entries = np.zeros(3 * 2**(2 * (k_max - k)) * (2**k - 1)**2 + 2**(k_max - k + 1) - 2**(k_max + 1))
    rows = np.zeros(3 * 2**(2 * (k_max - k)) * (2**k - 1)**2 + 2**(k_max - k + 1) - 2**(k_max + 1))
    columns = np.zeros(3 * 2**(2 * (k_max - k)) * (2**k - 1)**2 + 2**(k_max - k + 1) - 2**(k_max + 1))
    index = 0
    for e in range(0, 2**k):
        for f in range(0, 2**(k_max - k)):
            for g in range(0, 2**k):
                for l in range(0, 2**(k_max - k)):
                    if (e == 0 and f == 0) or (g == 0 and l == 0):
                        continue
                    if f + l < 2**(k_max - k):
                        if e != 0 and g != 0:
                            entries[index] = 1 - (f + l) * 2**(k - k_max)
                            rows[index] = (e * 2**(k_max - k) + f - 1) * (2**k_max - 1) + g * 2**(k_max - k) + l - 1
                            columns[index] = (e - 1) * (2**k - 1) + g - 1
                            index += 1
                        if e != 0 and g != 2**k - 1:
                            entries[index] = l * 2**(k - k_max)
                            rows[index] = (e * 2**(k_max - k) + f - 1) * (2**k_max - 1) + g * 2**(k_max - k) + l - 1
                            columns[index] = (e - 1) * (2**k - 1) + g + 1 - 1
                            index += 1
                        if e != 2**k - 1 and g != 0:
                            entries[index] = f * 2**(k - k_max)
                            rows[index] = (e * 2**(k_max - k) + f - 1) * (2**k_max - 1) + g * 2**(k_max - k) + l - 1
                            columns[index] = e * (2**k - 1) + g - 1
                            index += 1
                    elif f + l >= 2**(k_max - k):
                        if e != 2**k - 1 and g != 2**k - 1:
                            entries[index] = (l + f) * 2**(k - k_max) - 1
                            rows[index] = (e * 2**(k_max - k) + f - 1) * (2**k_max - 1) + g * 2**(k_max - k) + l - 1
                            columns[index] = e * (2**k - 1) + g + 1 - 1
                            index += 1
                        if e != 0 and g != 2**k -1:
                            entries[index] = 1 - f * 2**(k - k_max)
                            rows[index] = (e * 2**(k_max - k) + f - 1) * (2**k_max - 1) + g * 2**(k_max - k) + l - 1
                            columns[index] = (e - 1) * (2**k - 1) + g + 1 - 1
                            index += 1
                        if e != 2**k - 1 and g != 0:
                            entries[index] = 1 - l * 2**(k - k_max)
                            rows[index] = (e * 2**(k_max - k) + f - 1) * (2**k_max - 1) + g * 2**(k_max - k) + l - 1
                            columns[index] = e * (2**k - 1) + g - 1
                            index += 1
                    else:
                        print('Something is incorrect in the basis change.')
    return coo_matrix((entries, (rows, columns)), shape=(((2**k_max - 1)**2, (2**k - 1)**2))).tocsc()

# Makes an adjusted brownian increment vector. This means we use the single brownian increments array for tau_min and make
# the corresponding Brownian increments for tau.
def make_scaled_brownians(M, all_brownian_increments):
    scaled_increments = np.zeros((K_true, M))
    for m in range(M):
        for i in range(m * M_max // M, (m + 1) * M_max // M):
            scaled_increments[:,m] += all_brownian_increments[:,i]
    return scaled_increments

# Generates the initial condition vector for a given space discretization.
def initial_vector(N):
    result = np.zeros((N - 1)**2)
    h = 1 / N
    for l in range(1, (N - 1)**2 + 1):
        result[l - 1] = initial_condition((((l - 1) % (N - 1)) + 1) * h, (((l - 1) // (N - 1)) + 1) * h)
    return result

# This generates the vector of e values for a given discretization N.
def make_e_vector_2d(N):
    result = np.zeros((K_true, (N - 1)**2))
    h = 1 / N
    for i in range(2**K_power):
        for j in range(2**K_power):
            k = grid_to_index(i + 1, j + 1)
            for l in range(1, (N - 1)**2 + 1):
                result[k - 1, l - 1] = e_func(i + 1, j + 1, (((l - 1) % (N - 1)) + 1) * h, (((l - 1) // (N - 1)) + 1) * h)
    return result

# This generates the vector of sum of brownians and e_k's.
def calculate_linear_combo(N, M, brownian_increments, e_vector):
    result = np.zeros(((N - 1)**2, M))
    for m in range(M):
        for l in range((N - 1)**2):
            result[l, m] = brownian_increments[:, m].dot(e_vector[:, l])
    return result
    

create_exponential_matrices = False
test_nonnegativity = False
create_brownian_increments = False
save_other_brownian_increments = False
create_e_vectors = False
create_e_vectors_square_sum = False
calculate_reference_solutions = False
calculate_time_discretization_errors = False
calculate_space_discretization_errors = False
strong_error_variance_analysis = False
calculate_final_errors = False
test_no_zeros_reference_solutions = False
do_variance_analysis_weak_error = False
calculate_weak_error_accurately = False
final_accurate_weak_calc = False
weak_error_space_alternative = False
weak_error_space_alternative_analysis = False
print_standard_errors_of_strong_errors = False
calculating_size_of_solutions_and_comparing_to_errors = True
plotting_strong_standard_error = False
plotting_weak_variance_analysis = False
plotting_weak_graphs = False
plotting_graphs = False
plotting_weak_lower_space = False


# The exponential matrices are created and saved for the numerical method (all time steps sizes on largest spatial mesh and vice versa).
if create_exponential_matrices:
    folder = Path('paper_results')
    for space_power in range(2, N_power):
        #temp_matrix = sparse_expm(-1 * 2**(2 * space_power + T_power - M_power) * get_2d_S_SG(2**space_power)).toarray()
        temp_matrix = dense_expm((-1 * 2**(2 * space_power + T_power - M_power) * get_2d_S_SG(2**space_power)).toarray())
        file = f'exp_matrix_space{space_power}_time{M_power - T_power}.npy'
        with open(folder / file, 'wb') as f:
            np.save(f, temp_matrix, allow_pickle=False, fix_imports=False)
        print(f'For space power {space_power} and time power {M_power - T_power}, exponential matrix is calculated and saved.')
    for time_power in range(M_power - T_power - 9, M_power - T_power + 1):
    #for time_power in range(2, 14):
        #temp_matrix = sparse_expm(-1 * 2**(2 * N_power - time_power) * get_2d_S_SG(2**N_power)).toarray()
        temp_matrix = dense_expm((-1 * 2**(2 * N_power - time_power) * get_2d_S_SG(2**N_power)).toarray())
        #temp_matrix = dense_expm((-1 * 2**(2 * 4 - time_power) * get_2d_S_SG(2**4)).toarray())
        file = f'exp_matrix_space{N_power}_time{time_power}.npy'
        #file = f'exp_matrix_space{4}_time{time_power}.npy'
        with open(folder / file, 'wb') as f:
            np.save(f, temp_matrix, allow_pickle=False, fix_imports=False)
        print(f'For space power {N_power} and time power {time_power}, exponential matrix is calculated and saved.')

# This checks whether the exponential matrices are nonnegative.
if test_nonnegativity:
    folder = Path('paper_results')
    for space_power in range(2, N_power):
        file = f'exp_matrix_space{space_power}_time{M_power - T_power}.npy'
        with open(folder / file, 'rb') as f:
            matrix = np.load(f)
        no_neg = np.count_nonzero(np.abs(matrix) - matrix)
        print(f'For space power {space_power} and time power {M_power - T_power}, exponential matrix has {no_neg} negative entries.')
    for time_power in range(M_power - T_power - 9, M_power - T_power + 1):
        file = f'exp_matrix_space{N_power}_time{time_power}.npy'
        with open(folder / file, 'rb') as f:
            matrix = np.load(f)
        no_neg = np.count_nonzero(np.abs(matrix) - matrix)
        print(f'For space power {N_power} and time power {time_power}, exponential matrix has {no_neg} negative entries.')

# Creates and saves the brownian increments for each k with the smallest time discretization. b_k,m = array[k-1,m]
if create_brownian_increments:
    folder = Path('paper_results')
    normal_samples = np.sqrt(2**(T_power - M_power)) * normal(size = (max_samples, K_true, M_max))
    file = f'brownian_increments_samples{max_samples}_K{K_true}_time{M_power - T_power}.npy'
    with open(folder / file, 'wb') as f:
        np.save(f, normal_samples, allow_pickle=False, fix_imports=False)
    print(f'Brownian increments for K = {K_true} and tau = 2^({T_power - M_power}) with {max_samples} samples generated and saved.')

# This calculates the brownians for other time step sizes and saves them.
if save_other_brownian_increments:
    folder = Path('paper_results')
    file_load = f'brownian_increments_samples{max_samples}_K{K_true}_time{M_power - T_power}.npy'
    with open(folder / file_load, 'rb') as f:
        reference_increments = np.load(f)
    for m_power in range(M_power - 9, M_power):
        M = 2**m_power
        temp_increments = np.zeros((max_samples, K_true, M))
        for sample_no in range(max_samples):
            temp_increments[sample_no, :, :] = make_scaled_brownians(M, reference_increments[sample_no, :, :])
        file_save = f'brownian_increments_samples{max_samples}_K{K_true}_time{m_power - T_power}.npy'
        with open(folder / file_save, 'wb') as f:
            np.save(f, temp_increments, allow_pickle=False, fix_imports=False)
        print(f'Brownian increments for K = {K_true} and tau = 2^({T_power - m_power}) with {max_samples} samples calculated and saved.')

# This creates the mesh point values of the e_k's for each of the different space discretizations.
if create_e_vectors:
    folder = Path('paper_results')
    for space_power in range(2, N_power + 1):
        e_vector = make_e_vector_2d(2**space_power)
        file_save = f'e_vector_K{K_true}_space{space_power}.npy'
        with open(folder / file_save, 'wb') as f:
            np.save(f, e_vector, allow_pickle=False, fix_imports=False)
        print(f'e_vector saved for space size {space_power}.')

# This creates the vector of the sum of the e_k squared at each mesh point.
if create_e_vectors_square_sum:
    folder = Path('paper_results')
    for space_power in range(2, N_power + 1):
        file_load = f'e_vector_K{K_true}_space{space_power}.npy'
        with open(folder / file_load, 'rb') as f:
            e_vector = np.load(f)
        e_vector_squared_sum = np.zeros((2**space_power - 1)**2)
        for j in range((2**space_power - 1)**2):
            for k in range(K_true):
                e_vector_squared_sum[j] += (e_vector[k, j]**2)
        file_save = f'e_vector_squared_sum_K{K_true}_space{space_power}.npy'
        with open(folder / file_save, 'wb') as f:
            np.save(f, e_vector_squared_sum, allow_pickle=False, fix_imports=False)
        print(f'e_vector squared sum saved for K = {K_true} and space power {space_power}.')

# This calculates and saves the reference solutions to our problem.
if calculate_reference_solutions:
    print('Loading saved data.')
    folder = Path('paper_results')
    file_load = f'brownian_increments_samples{max_samples}_K{K_true}_time{M_power - T_power}.npy'
    with open(folder / file_load, 'rb') as f:
        brownian_increments = np.load(f)
    file_load = f'exp_matrix_space{N_power}_time{M_power - T_power}.npy'
    with open(folder / file_load, 'rb') as f:
        exp_matrix = np.load(f)
    file_load = f'e_vector_K{K_true}_space{N_power}.npy'
    with open(folder / file_load, 'rb') as f:
        e_vector = np.load(f)
    file_load = f'e_vector_squared_sum_K{K_true}_space{N_power}.npy'
    with open(folder / file_load, 'rb') as f:
        e_vector_squared_sum = np.load(f)
    for j in range(max_samples):
        print(f"Calculating linear combo vector for {j}th reference solution.")
        linear_combo_vector = calculate_linear_combo(2**N_power, M_max, brownian_increments[j,:,:], e_vector)
        total_solution = np.zeros(((2**N_power - 1)**2, M_max + 1))
        current_solution = initial_vector(2**N_power)
        total_solution[:, 0] = current_solution
        print(f"Generating {j}th reference solution.")
        for m in range(M_max):
            g_current_solution = nonlinearity_vectorized(current_solution)
            current_solution = exp_matrix.dot(np.exp(g_current_solution * linear_combo_vector[:, m] - tau_min / 2 * g_current_solution * g_current_solution * e_vector_squared_sum) * current_solution)
            total_solution[:, m + 1] = current_solution
        file_save = f'reference_solution_{j}.npy'
        with open(folder / file_save, 'wb') as f:
            np.save(f, total_solution, allow_pickle=False, fix_imports=False)
        print(f'Reference solution {j} saved.')

# This calculates the raw error data and end time values for changing time discretization. 
if calculate_time_discretization_errors:
    print('Loading saved data.')
    folder = Path('paper_results')
    file_load = f'e_vector_K{K_true}_space{N_power}.npy'
    with open(folder / file_load, 'rb') as f:
        e_vector = np.load(f)
    file_load = f'e_vector_squared_sum_K{K_true}_space{N_power}.npy'
    with open(folder / file_load, 'rb') as f:
        e_vector_squared_sum = np.load(f)
    N = 2**N_power
    mass_matrix = get_2d_M_SG(N)
    raw_results_strong_time = np.zeros((9, 2**(M_power - 1) + 1, max_samples))
    raw_results_weak_time = np.zeros((9, max_samples, (N - 1)**2))
    for m_power in range(M_power - 9, M_power):
        print(f'Calculating raw errors for time power {m_power - T_power}.')
        M = 2**m_power
        d = M_max // M
        tau = 2**(T_power - m_power)
        file_load = f'exp_matrix_space{N_power}_time{m_power - T_power}.npy'
        with open(folder / file_load, 'rb') as f:
            exp_matrix = np.load(f)
        file_load = f'brownian_increments_samples{max_samples}_K{K_true}_time{m_power - T_power}.npy'
        with open(folder / file_load, 'rb') as f:
            brownian_increments = np.load(f)
        for j in range(max_samples):
            file_load = f'reference_solution_{j}.npy'
            with open(folder / file_load, 'rb') as f:
                reference_solution = np.load(f)
            linear_combo_vector = calculate_linear_combo(N, M, brownian_increments[j,:,:], e_vector)
            current_solution = initial_vector(N)
            total_solution = np.zeros(((N - 1)**2, M + 1))
            total_solution[:, 0] = current_solution
            for m in range(M):
                g_current_solution = nonlinearity_vectorized(current_solution)
                current_solution = exp_matrix.dot(np.exp(g_current_solution * linear_combo_vector[:, m] - tau / 2 * g_current_solution * g_current_solution * e_vector_squared_sum) * current_solution)
                total_solution[:, m + 1] = current_solution
            for m in range(M + 1):
                raw_results_strong_time[9 + m_power - M_power, m, j] = (mass_matrix.dot(reference_solution[:,m * d] - total_solution[:,m])).dot(reference_solution[:,m * d] - total_solution[:,m])
            raw_results_weak_time[9 + m_power - M_power, j, :] = total_solution[:, M]
    file_save = f'raw_results_strong_time.npy'
    with open(folder / file_save, 'wb') as f:
        np.save(f, raw_results_strong_time, allow_pickle=False, fix_imports=False)
    file_save = f'raw_results_weak_time.npy'
    with open(folder / file_save, 'wb') as f:
        np.save(f, raw_results_weak_time, allow_pickle=False, fix_imports=False)
    print(f'Raw results saved for the time discretization changes.')

# This calculates the raw error data and end time values for changing space discretization.
if calculate_space_discretization_errors:
    print('Loading saved data.')
    folder = Path('paper_results')
    mass_matrix = get_2d_M_SG(2**N_power)
    raw_results_strong_space = np.zeros((4, M_max + 1, max_samples))
    file_load = f'brownian_increments_samples{max_samples}_K{K_true}_time{M_power - T_power}.npy'
    with open(folder / file_load, 'rb') as f:
        brownian_increments = np.load(f)
    for space_power in range(N_power - 4, N_power):
        print(f'Calculating raw errors for space power {space_power}.')
        file_load = f'e_vector_K{K_true}_space{space_power}.npy'
        with open(folder / file_load, 'rb') as f:
            e_vector = np.load(f)
        file_load = f'e_vector_squared_sum_K{K_true}_space{space_power}.npy'
        with open(folder / file_load, 'rb') as f:
            e_vector_squared_sum = np.load(f)
        N = 2**space_power
        basis_change_matrix = get_basis_change_2d(space_power, N_power)
        raw_results_weak_space = np.zeros((max_samples, (N - 1)**2))
        file_load = f'exp_matrix_space{space_power}_time{M_power - T_power}.npy'
        with open(folder / file_load, 'rb') as f:
            exp_matrix = np.load(f)
        for j in range(max_samples):
            file_load = f'reference_solution_{j}.npy'
            with open(folder / file_load, 'rb') as f:
                reference_solution = np.load(f)
            linear_combo_vector = calculate_linear_combo(N, M_max, brownian_increments[j,:,:], e_vector)
            current_solution = initial_vector(N)
            total_solution = np.zeros(((N - 1)**2, M_max + 1))
            total_solution[:, 0] = current_solution
            for m in range(M_max):
                g_current_solution = nonlinearity_vectorized(current_solution)
                current_solution = exp_matrix.dot(np.exp(g_current_solution * linear_combo_vector[:, m] - tau_min / 2 * g_current_solution * g_current_solution * e_vector_squared_sum) * current_solution)
                total_solution[:, m + 1] = current_solution
            for m in range(M_max + 1):
                raw_results_strong_space[4 + space_power - N_power, m, j] = (mass_matrix.dot(reference_solution[:,m] - basis_change_matrix.dot(total_solution[:,m]))).dot(reference_solution[:,m] - basis_change_matrix.dot(total_solution[:,m]))
            raw_results_weak_space[j, :] = total_solution[:, M_max]
        file_save = f'raw_results_weak_space{space_power}.npy'
        with open(folder / file_save, 'wb') as f:
            np.save(f, raw_results_weak_space, allow_pickle=False, fix_imports=False)
    file_save = f'raw_results_strong_space.npy'
    with open(folder / file_save, 'wb') as f:
        np.save(f, raw_results_strong_space, allow_pickle=False, fix_imports=False)
    print('Raw results saved for the space discretization changes.')

# This does the variance analysis of the stronge error.
if strong_error_variance_analysis:
    folder = Path('paper_results')
    file_load = f'raw_results_strong_time.npy'
    with open(folder / file_load, 'rb') as f:
        raw_results_strong_time = np.load(f)
    file_load = f'raw_results_strong_space.npy'
    with open(folder / file_load, 'rb') as f:
        raw_results_strong_space = np.load(f)
    strong_error_time_standard_errors = np.zeros((9, 2**(M_power - 1) + 1, 2))
    strong_error_space_standard_errors = np.zeros((4, M_max + 1, 2))
    for m_power_ind in range(9):
        for m in range(2**(M_power - 9 + m_power_ind) + 1):
            strong_error_time_standard_errors[m_power_ind, m, 0] = np.average(raw_results_strong_time[m_power_ind, m, :])
            strong_error_time_standard_errors[m_power_ind, m, 1] = 1 / np.sqrt(max_samples) * np.sqrt(np.var(raw_results_strong_time[m_power_ind, m, :],ddof=1))
    strong_time_errors = np.zeros((9,2))
    for m_power_ind in range(9):
        strong_time_errors[m_power_ind,0] = np.max(strong_error_time_standard_errors[m_power_ind, :, 0])
        strong_time_errors[m_power_ind,1] = np.max(strong_error_time_standard_errors[m_power_ind, :, 1])
    for n_power_ind in range(4):
        for m in range(M_max + 1):
            strong_error_space_standard_errors[n_power_ind, m, 0] = np.average(raw_results_strong_space[n_power_ind, m, :])
            strong_error_space_standard_errors[n_power_ind, m, 1] = 1 / np.sqrt(max_samples) * np.sqrt(np.var(raw_results_strong_space[n_power_ind, m, :],ddof=1))
    strong_space_errors = np.zeros((4,2))
    for n_power_ind in range(4):
        strong_space_errors[n_power_ind,0] = np.max(strong_error_space_standard_errors[n_power_ind, :, 0])
        strong_space_errors[n_power_ind,1] = np.max(strong_error_space_standard_errors[n_power_ind, :, 1])
    file_save = 'strong_time_errors_standard_error.npy'
    with open(folder / file_save, 'wb') as f:
        np.save(f, strong_time_errors, allow_pickle=False, fix_imports=False)
    file_save = 'strong_space_errors_standard_error.npy'
    with open(folder / file_save, 'wb') as f:
        np.save(f, strong_space_errors, allow_pickle=False, fix_imports=False)
    print('Strong errors with standard error saved.')

# This calculates the final strong and weak errors and saves them in an .npy file. They are then ready to be plotted.
if calculate_final_errors:
    folder = Path('paper_results')
    file_load = f'raw_results_strong_time.npy'
    with open(folder / file_load, 'rb') as f:
        raw_results_strong_time = np.load(f)
    file_load = f'raw_results_weak_time.npy'
    with open(folder / file_load, 'rb') as f:
        raw_results_weak_time = np.load(f)
    file_load = f'raw_results_strong_space.npy'
    with open(folder / file_load, 'rb') as f:
        raw_results_strong_space = np.load(f)
    print('Calculating strong errors for time discretization.')
    strong_errors = np.zeros((2,9))
    temporary_strong_error_time = np.zeros((9, 2**(M_power - 1) + 1))
    for m_power_ind in range(9):
        for m in range(2**(M_power - 9 + m_power_ind) + 1):
            for j in range(max_samples):
                temporary_strong_error_time[m_power_ind, m] += raw_results_strong_time[m_power_ind, m, j]
            temporary_strong_error_time[m_power_ind, m] = 1 / max_samples * temporary_strong_error_time[m_power_ind, m]
            temporary_strong_error_time[m_power_ind, m] = np.sqrt(temporary_strong_error_time[m_power_ind, m])
        strong_errors[0,m_power_ind] = np.max(temporary_strong_error_time[m_power_ind, :])
    print('Calculating strong errors for space discretization.')
    temporary_strong_error_space = np.zeros((4, M_max + 1))
    for n_power_ind in range(4):
        for m in range(M_max + 1):
            for j in range(max_samples):
                temporary_strong_error_space[n_power_ind, m] += raw_results_strong_space[n_power_ind, m, j]
            temporary_strong_error_space[n_power_ind, m] = 1 / max_samples * temporary_strong_error_space[n_power_ind, m]
            temporary_strong_error_space[n_power_ind, m] = np.sqrt(temporary_strong_error_space[n_power_ind, m])
        strong_errors[1,n_power_ind] = np.max(temporary_strong_error_space[n_power_ind, :])
    file_save = f'strong_errors_time{M_power - T_power}_space{N_power}_K{K_true}_samples{max_samples}.npy'
    with open(folder / file_save, 'wb') as f:
        np.save(f, strong_errors, allow_pickle=False, fix_imports=False)
    print('Strong errors saved.')
    print('Calculating reference value for weak error.')
    weak_errors = np.zeros((2,9))
    reference_value = 0.0
    mass_matrix = get_2d_M_SG(2**N_power)
    for j in range(max_samples):
        file_load = f'reference_solution_{j}.npy'
        with open(folder / file_load, 'rb') as f:
            current_reference_solution = np.load(f)
        reference_value += weak_functional(mass_matrix, current_reference_solution[:, M_max])
    reference_value = 1 / max_samples * reference_value
    print('Calculating weak errors for time discretization.')
    for m_power_ind in range(9):
        for j in range(max_samples):
            weak_errors[0, m_power_ind] += weak_functional(mass_matrix, raw_results_weak_time[m_power_ind, j,:])
        weak_errors[0, m_power_ind] = np.abs(1 / max_samples * weak_errors[0, m_power_ind] - reference_value)
    print('Calculating weak errors for space discretization.')
    for n_power_ind in range(4):
        file_load = f'raw_results_weak_space{N_power - 4 + n_power_ind}.npy'
        with open(folder / file_load, 'rb') as f:
            temporary_weak_errors_space = np.load(f)
        mass_matrix = get_2d_M_SG(2**(N_power - 4 + n_power_ind))
        for j in range(max_samples):
            weak_errors[1, n_power_ind] += weak_functional(mass_matrix, temporary_weak_errors_space[j,:])
        weak_errors[1, n_power_ind] = np.abs(1 / max_samples * weak_errors[1, n_power_ind] - reference_value)
    file_save = f'weak_errors_time{M_power - T_power}_space{N_power}_K{K_true}_samples{max_samples}.npy'
    with open(folder / file_save, 'wb') as f:
        np.save(f, weak_errors, allow_pickle=False, fix_imports=False)
    print('Weak errors saved.')

if test_no_zeros_reference_solutions:
    folder = Path('paper_results')
    minimum_first_zero_step = M_max
    maximum_first_zero_step = 0
    minimum_all_zero_step = M_max
    maximum_all_zero_step = 0
    max_difference_first_all = 0
    for j in range(max_samples):
        file_load = f'reference_solution_{j}.npy'
        with open(folder / file_load, 'rb') as f:
            solution = np.load(f)
        existence_of_zeros = False
        for m in range(M_max + 1):
            no_zeros = solution[:, m].size - np.count_nonzero(solution[:, m])
            no_zeros_percent = no_zeros / solution[:,m].size
            if no_zeros > 0 and (not existence_of_zeros):
                print(f'For reference solution {j}, the first time step with a zero is {m}.')
                existence_of_zeros = True
                minimum_first_zero_step = min(minimum_first_zero_step, m)
                maximum_first_zero_step = max(maximum_first_zero_step, m)
                first_zero_step = m
            if no_zeros == ((2**N_power - 1)**2):
                print(f'For reference solution {j}, the first time step with all zeros is {m}.')
                difference_first_all = m - first_zero_step
                max_difference_first_all = max(max_difference_first_all, difference_first_all)
                print(f'Difference between first zeros and all zeros is {difference_first_all}.')
                minimum_all_zero_step = min(minimum_all_zero_step, m)
                maximum_all_zero_step = max(maximum_all_zero_step, m)
                break
            if m == M_max and (not existence_of_zeros):
                print(f'Solution for sample {j} remained strictly positive.')
    print(f'Maximum time step for first zeros was {maximum_first_zero_step}.')
    print(f'Minimum time step for first zeros was {minimum_first_zero_step}.')
    print(f'Maximum time step for all zeros was {maximum_all_zero_step}.')
    print(f'Minimum time step for all zeros was {minimum_all_zero_step}.')
    print(f'Maximum difference for first and all zeros was {max_difference_first_all}.')
        
        # no_zeros = solution[:,M_max].size - np.count_nonzero(solution[:,M_max])
        # no_zeros_percent = no_zeros / solution[:,M_max].size
        # print(f'Number of zeros for reference solution {j} at last time step is {no_zeros}.')
        # print(f'Percentage is {no_zeros_percent * 100}.')
        # for m in range(M_max + 1):
            # no_zeros = solution[:,m].size - np.count_nonzero(solution[:,m])
            # no_zeros_percent = no_zeros / solution[:,m].size
            # print(f'Number of zeros for reference solution {j} at time step {m} is {no_zeros}.')
            # print(f'Percentage is {no_zeros_percent * 100}.')

# This calculates the standard error of the weak error.
if do_variance_analysis_weak_error:
    folder = Path('paper_results')
    file_load = f'raw_results_weak_time.npy'
    with open(folder / file_load, 'rb') as f:
        raw_results_weak_time = np.load(f)
    data_points = np.zeros((9, max_samples))
    mass_matrix = get_2d_M_SG(2**N_power)
    print('Calculating raw weak errors for time discretization.')
    for j in range(max_samples):
        file_load = f'reference_solution_{j}.npy'
        with open(folder / file_load, 'rb') as f:
            reference_solution = np.load(f)
        for m_power_ind in range(9):
            data_points[m_power_ind,j] = weak_functional(mass_matrix, raw_results_weak_time[m_power_ind,j,:]) - weak_functional(mass_matrix,reference_solution[:,M_max])
    weak_errors_time = np.zeros((9, 2))
    for m_power_ind in range(9):
        print(f'Calculating weak errors and spread for m_power_ind {m_power_ind}')
        weak_errors_time[m_power_ind,0] = np.abs(np.average(data_points[m_power_ind,:]))
        weak_errors_time[m_power_ind,1] = 1 / np.sqrt(max_samples) * np.sqrt(np.var(data_points[m_power_ind,:],ddof=1))
    file_save = 'weak_time_errors_with_standard_error.npy'
    with open(folder / file_save, 'wb') as f:
        np.save(f, weak_errors_time, allow_pickle=False, fix_imports=False)
    print('Weak errors and standard errors saved for time discretization.')
    data_points = np.zeros((4, max_samples))
    for n_power_ind in range(4):
        print(f'Calculating weak errors data points for n_power_ind = {n_power_ind}.')
        file_load = f'raw_results_weak_space{N_power - 4 + n_power_ind}.npy'
        with open(folder / file_load, 'rb') as f:
            raw_results_weak_space = np.load(f)
        mass_matrix_rough = get_2d_M_SG(2**(N_power - 4 + n_power_ind))
        for j in range(max_samples):
            file_load = f'reference_solution_{j}.npy'
            with open(folder / file_load, 'rb') as f:
                reference_solution = np.load(f)
            data_points[n_power_ind,j] = weak_functional(mass_matrix_rough, raw_results_weak_space[j,:]) - weak_functional(mass_matrix,reference_solution[:,M_max])
    weak_errors_space = np.zeros((4,2))
    for n_power_ind in range(4):
        weak_errors_space[n_power_ind,0] = np.abs(np.average(data_points[n_power_ind,:]))
        weak_errors_space[n_power_ind,1] = 1 / np.sqrt(max_samples) * np.sqrt(np.var(data_points[n_power_ind,:],ddof=1))
    file_save = 'weak_space_errors_with_standard_error.npy'
    with open(folder / file_save, 'wb') as f:
        np.save(f, weak_errors_space, allow_pickle=False, fix_imports=False)
    print('Weak errors and standard errors saved for space discretization.')

# This calculates the weak error much more efficiently so more realizations can be used.
if calculate_weak_error_accurately:
    folder = Path('paper_results')
    # running_sums = np.zeros((2, 9))
    # running_all_values = np.zeros((2, 9, max_samples_weak))
    file_load = f'running_all_values_weak_after_sample_{5659}.npy'
    with open(folder / file_load, 'rb') as f:
       running_all_values = np.load(f)
    file_load = f'running_sums_weak_after_sample_{5659}.npy'
    with open(folder / file_load, 'rb') as f:
       running_sums = np.load(f)
    file_load = f'e_vector_K{K_true}_space{N_power}.npy'
    with open(folder / file_load, 'rb') as f:
        e_vector_finest = np.load(f)
    file_load = f'e_vector_squared_sum_K{K_true}_space{N_power}.npy'
    with open(folder / file_load, 'rb') as f:
        e_vector_squared_sum_finest = np.load(f)
    file_load = f'exp_matrix_space{N_power}_time{M_power - T_power}.npy'
    with open(folder / file_load, 'rb') as f:
        exp_matrix_finest = np.load(f)
    mass_matrix_finest = get_2d_M_SG(2**N_power)
    for j in range(5660, max_samples_weak): ###########################################
        #time_start = time.time()
        brownian_increments_finest = np.sqrt(2**(T_power - M_power)) * normal(size = (K_true, M_max))
        # Calculating reference value (finest discretization).
        linear_combo_vector = calculate_linear_combo(2**N_power, M_max, brownian_increments_finest, e_vector_finest)
        current_solution = initial_vector(2**N_power)
        for m in range(M_max):
            g_current_solution = nonlinearity_vectorized(current_solution)
            current_solution = exp_matrix_finest.dot(np.exp(g_current_solution * linear_combo_vector[:, m] - tau_min / 2 * g_current_solution * g_current_solution * e_vector_squared_sum_finest) * current_solution)
        running_all_values[1, 4, j] = weak_functional(mass_matrix_finest, current_solution)
        running_sums[1, 4] += running_all_values[1, 4, j]
        # Cycling through time discretizations.
        for m_power_ind in range(9):
            tau = 2**(-M_power + T_power + 9 - m_power_ind)
            file_load = f'exp_matrix_space{N_power}_time{M_power - T_power - 9 + m_power_ind}.npy'
            with open(folder / file_load, 'rb') as f:
                exp_matrix = np.load(f)
            brownian_increments = make_scaled_brownians(2**(M_power - 9 + m_power_ind), brownian_increments_finest)
            linear_combo_vector = calculate_linear_combo(2**N_power, 2**(M_power - 9 + m_power_ind), brownian_increments, e_vector_finest)
            current_solution = initial_vector(2**N_power)
            for m in range(2**(M_power - 9 + m_power_ind)):
                g_current_solution = nonlinearity_vectorized(current_solution)
                current_solution = exp_matrix.dot(np.exp(g_current_solution * linear_combo_vector[:, m] - tau / 2 * g_current_solution * g_current_solution * e_vector_squared_sum_finest) * current_solution)
            running_all_values[0, m_power_ind, j] = weak_functional(mass_matrix_finest, current_solution)
            running_sums[0, m_power_ind] += running_all_values[0, m_power_ind, j]
        # Cycling through space discretizations.
        for n_power_ind in range(4):
            mass_matrix = get_2d_M_SG(2**(N_power - 4 + n_power_ind))
            file_load = f'exp_matrix_space{N_power - 4 + n_power_ind}_time{M_power - T_power}.npy'
            with open(folder / file_load, 'rb') as f:
                exp_matrix = np.load(f)
            file_load = f'e_vector_K{K_true}_space{N_power - 4 + n_power_ind}.npy'
            with open(folder / file_load, 'rb') as f:
                e_vector = np.load(f)
            file_load = f'e_vector_squared_sum_K{K_true}_space{N_power - 4 + n_power_ind}.npy'
            with open(folder / file_load, 'rb') as f:
                e_vector_squared_sum = np.load(f)
            linear_combo_vector = calculate_linear_combo(2**(N_power - 4 + n_power_ind), M_max, brownian_increments_finest, e_vector)
            current_solution = initial_vector(2**(N_power - 4 + n_power_ind))
            for m in range(M_max):
                g_current_solution = nonlinearity_vectorized(current_solution)
                current_solution = exp_matrix.dot(np.exp(g_current_solution * linear_combo_vector[:, m] - tau_min / 2 * g_current_solution * g_current_solution * e_vector_squared_sum) * current_solution)
            running_all_values[1, n_power_ind, j] = weak_functional(mass_matrix, current_solution)
            running_sums[1, n_power_ind] += running_all_values[1, n_power_ind, j]
        if  ((j + 1) % 20) == 0:
            file_save = f'running_all_values_weak_after_sample_{j}.npy'
            with open(folder / file_save, 'wb') as f:
                np.save(f, running_all_values, allow_pickle=False, fix_imports=False)
            file_save = f'running_sums_weak_after_sample_{j}.npy'
            with open(folder / file_save, 'wb') as f:
                np.save(f, running_sums, allow_pickle=False, fix_imports=False)
            print(f'Raw results for weak error saved up to sample {j}.')
        #time_end = time.time()
        #print(f'Time taken to calculate sample {j} is {time_end - time_start} seconds.')

if final_accurate_weak_calc:
    folder = Path('paper_results')
    weak_errors = np.zeros((2,9))
    file_load = f'running_sums_weak_after_sample_{max_samples_weak - 1}.npy'
    with open(folder / file_load, 'rb') as f:
        running_sums = np.load(f)
    for m_power_ind in range(9):
        weak_errors[0,m_power_ind] = 1 / max_samples_weak * np.abs(running_sums[0, m_power_ind] - running_sums[1,4])
    for n_power_ind in range(4):
        weak_errors[1,n_power_ind] = 1 / max_samples_weak * np.abs(running_sums[1, n_power_ind] - running_sums[1,4])
    file_save = f'weak_errors_time{M_power - T_power}_space{N_power}_K{K_true}_samples{max_samples_weak}.npy'
    with open(folder / file_save, 'wb') as f:
        np.save(f, weak_errors, allow_pickle=False, fix_imports=False)

if weak_error_space_alternative:
    folder = Path('paper_results')
    #running_sums = np.zeros((2, 9))
    #running_all_values = np.zeros((10, max_samples_weak, 2, (2**4 - 1)**2))
    file_load = f'running_all_values_weak_space4_after_sample_{9399}.npy'
    with open(folder / file_load, 'rb') as f:
       running_all_values = np.load(f)
    #file_load = f'running_sums_weak_after_sample_{5659}.npy'
    #with open(folder / file_load, 'rb') as f:
    #   running_sums = np.load(f)
    file_load = f'e_vector_K{K_true}_space{4}.npy'
    with open(folder / file_load, 'rb') as f:
        e_vector_finest = np.load(f)
    file_load = f'e_vector_squared_sum_K{K_true}_space{4}.npy'
    with open(folder / file_load, 'rb') as f:
        e_vector_squared_sum_finest = np.load(f)
    file_load = f'exp_matrix_space{4}_time{M_power - T_power}.npy'
    with open(folder / file_load, 'rb') as f:
        exp_matrix_finest = np.load(f)
    #mass_matrix_finest = get_2d_M_SG(2**4)
    for j in range(9400, max_samples_weak):
        #time_start = time.time()
        brownian_increments_finest = np.sqrt(2**(T_power - M_power)) * normal(size = (K_true, M_max))
        # Calculating reference value (finest discretization).
        linear_combo_vector = calculate_linear_combo(2**4, M_max, brownian_increments_finest, e_vector_finest)
        current_solution = initial_vector(2**4)
        for m in range(M_max):
            g_current_solution = nonlinearity_vectorized(current_solution)
            current_solution = exp_matrix_finest.dot(np.exp(g_current_solution * linear_combo_vector[:, m] - tau_min / 2 * g_current_solution * g_current_solution * e_vector_squared_sum_finest) * current_solution)
            if ((m + 1) * 2) == M_max:
                halfway_solution = np.copy(current_solution)
        running_all_values[9, j, 0,:] = halfway_solution
        running_all_values[9, j, 1,:] = current_solution
        #running_sums[1, 4] += running_all_values[1, 4, j]
        # Cycling through time discretizations.
        for m_power_ind in range(9):
            tau = 2**(-M_power + T_power + 9 - m_power_ind)
            file_load = f'exp_matrix_space{4}_time{M_power - T_power - 9 + m_power_ind}.npy'
            with open(folder / file_load, 'rb') as f:
                exp_matrix = np.load(f)
            brownian_increments = make_scaled_brownians(2**(M_power - 9 + m_power_ind), brownian_increments_finest)
            linear_combo_vector = calculate_linear_combo(2**4, 2**(M_power - 9 + m_power_ind), brownian_increments, e_vector_finest)
            current_solution = initial_vector(2**4)
            for m in range(2**(M_power - 9 + m_power_ind)):
                g_current_solution = nonlinearity_vectorized(current_solution)
                current_solution = exp_matrix.dot(np.exp(g_current_solution * linear_combo_vector[:, m] - tau / 2 * g_current_solution * g_current_solution * e_vector_squared_sum_finest) * current_solution)
                if ((m + 1) * 2) == (2**(M_power - 9 + m_power_ind)):
                    halfway_solution = np.copy(current_solution)
            running_all_values[m_power_ind, j, 0,:] = halfway_solution
            running_all_values[m_power_ind, j, 1,:] = current_solution
            #running_sums[0, m_power_ind] += running_all_values[0, m_power_ind, j]
        if  ((j + 1) % 200) == 0:
            file_save = f'running_all_values_weak_space4_after_sample_{j}.npy'
            with open(folder / file_save, 'wb') as f:
                np.save(f, running_all_values, allow_pickle=False, fix_imports=False)
            print(f'Raw results for weak error with space 4 and halfway solution saved up to sample {j}.')
        #time_end = time.time()
        #print(f'Time taken to calculate sample {j} is {time_end - time_start} seconds.')


if weak_error_space_alternative_analysis:
    folder = Path('paper_results')
    file_load = f'running_all_values_weak_space4_after_sample_{max_samples_weak - 1}.npy'
    with open(folder / file_load, 'rb') as f:
       raw_data = np.load(f)
    mass_matrix = get_2d_M_SG(2**4)
    # For calculating the weak error.
    weak_errors_all_realisations = np.zeros((9,max_samples_weak))
    reference_values_weak = np.zeros(max_samples_weak)
    print('Calculating reference values for weak error.')
    for j in range(max_samples_weak):
        reference_values_weak[j] = weak_functional(mass_matrix, raw_data[9, j, 1, :])
    print("Calculating raw weak errors.")
    for j in range(max_samples_weak):
        for m_power_ind in range(9):
            weak_errors_all_realisations[m_power_ind, j] = weak_functional(mass_matrix, raw_data[m_power_ind, j, 1, :]) - reference_values_weak[j]
    weak_errors_final = np.zeros((9,2))
    for m_power_ind in range(9):
        print(f'Calcualting average and standard error for m_power_ind = {m_power_ind}.')
        weak_errors_final[m_power_ind, 0] = np.abs(np.average(weak_errors_all_realisations[m_power_ind,:]))
        weak_errors_final[m_power_ind, 1] = 1 / np.sqrt(max_samples_weak) * np.sqrt(np.var(weak_errors_all_realisations[m_power_ind,:],ddof=1))
    file_save = f'weak_errors_space4_standard_error.npy'
    with open(folder / file_save, 'wb') as f:
        np.save(f, weak_errors_final, allow_pickle=False, fix_imports=False)
    print('Weak errors and standard error saved.')
    # For calculating the time correlation weak error.
    weak_time_correlations_all_realisations = np.zeros((9,max_samples_weak))
    averages = np.zeros((10, 2, (2**4 - 1)**2))
    for m_power_ind in range(10):
        print(f'Calculating averages for m_power_ind = {m_power_ind}.')
        for space_ind in range((2**4 - 1)**2):
            averages[m_power_ind, 0, space_ind] = np.average(raw_data[m_power_ind, :, 0, space_ind])
            averages[m_power_ind, 1, space_ind] = np.average(raw_data[m_power_ind, :, 1, space_ind])
    reference_values_time_correlation = np.zeros(max_samples_weak)
    print('Calculating reference values for time correlation.')
    for j in range(max_samples_weak):
        reference_values_time_correlation[j] = (mass_matrix.dot(raw_data[9,j,0,:] - averages[9,0,:])).dot(raw_data[9,j,1,:] - averages[9,1,:])
    print("Calculating raw time correlation errors.")
    for j in range(max_samples_weak):
        for m_power_ind in range(9):
            weak_time_correlations_all_realisations[m_power_ind, j] = (mass_matrix.dot(raw_data[m_power_ind,j,0,:] - averages[m_power_ind,0,:])).dot(raw_data[m_power_ind,j,1,:] - averages[m_power_ind,1,:]) - reference_values_time_correlation[j]
    weak_time_correlations_final = np.zeros((9,2))
    for m_power_ind in range(9):
        print(f'Calcualting average and standard error for m_power_ind = {m_power_ind} for time correlations.')
        weak_time_correlations_final[m_power_ind, 0] = np.abs(np.average(weak_time_correlations_all_realisations[m_power_ind, :]))
        weak_time_correlations_final[m_power_ind, 1] = 1 / np.sqrt(max_samples_weak) * np.sqrt(np.var(weak_time_correlations_all_realisations[m_power_ind, :], ddof=1))
    file_save = f'weak_time_correlations_space4_standard_error.npy'
    with open(folder / file_save, 'wb') as f:
        np.save(f, weak_time_correlations_final, allow_pickle=False, fix_imports=False)
    print('Weak time correlations and standard error saved.')

if print_standard_errors_of_strong_errors:
    folder = Path('paper_results')
    file_load = 'strong_time_errors_standard_error.npy'
    with open(folder / file_load, 'rb') as f:
        strong_errors_time = np.load(f)
    file_load = 'strong_space_errors_standard_error.npy'
    with open(folder / file_load, 'rb') as f:
        strong_errors_space = np.load(f)
    max_time_error_ratio = 0.0
    max_space_error_ratio = 0.0
    for m_power_ind in range(9):
        print(f'{strong_errors_time[m_power_ind,1] / strong_errors_time[m_power_ind,0]} with error {strong_errors_time[m_power_ind,1]} and average {strong_errors_time[m_power_ind,0]}.')
        max_time_error_ratio = max(max_time_error_ratio, strong_errors_time[m_power_ind,1] / strong_errors_time[m_power_ind,0])
    for n_power_ind in range(4):
        print(f'{strong_errors_space[n_power_ind,1] / strong_errors_space[n_power_ind,0]} with error {strong_errors_space[n_power_ind,1]} and average {strong_errors_space[n_power_ind,0]}.')
        max_space_error_ratio = max(max_space_error_ratio, strong_errors_space[n_power_ind,1] / strong_errors_space[n_power_ind,0])
    # max_time_error = np.max(strong_errors_time[:,1])
    # min_time_value = np.min(strong_errors_time[:,0])
    # max_space_error = np.max(strong_errors_space[:,1])
    # min_space_value = np.min(strong_errors_space[:,0])
    print(f'Maximum standard error ratio for time dsicretizations is {max_time_error_ratio}.')
    print(f'Maximum standard error ratio for space dsicretizations is {max_space_error_ratio}.')

if calculating_size_of_solutions_and_comparing_to_errors:
    folder = Path('paper_results')
    # #Sanity check:
    # inital_solution = initial_vector(2**N_power)
    # mass_matrix = get_2d_M_SG(2**N_power)
    # print(f'Size of initial value squared norm is {weak_functional(mass_matrix, inital_solution)}.')
    # Weak Errors
    file_load = f'weak_time_errors_with_standard_error.npy'
    with open(folder / file_load, 'rb') as f:
        weak_errors = np.load(f)
    averages = np.zeros(9)
    mass_matrix = get_2d_M_SG(2**N_power)
    for j in range(max_samples):
        file_load = f'reference_solution_{j}.npy'
        with open(folder / file_load, 'rb') as f:
            reference_solution = np.load(f)
        for m_power_ind in range(9):
            averages[m_power_ind] += weak_functional(mass_matrix,reference_solution[:,M_max])
    averages = 1 / max_samples * averages
    for m_power_ind in range(9):
        print(f' For m_power_ind = {m_power_ind}, reference norm squared is {averages[m_power_ind]} and the error is {weak_errors[m_power_ind,0]}.')
    for n_power_ind in range(4):
        file_load = 'weak_space_errors_with_standard_error.npy'
        with open(folder / file_load, 'rb') as f:
            weak_errors = np.load(f)
        print(f'For n_power_ind = {n_power_ind}, reference norm squared is {averages[n_power_ind]} and the error is {weak_errors[n_power_ind,0]}.')
    # Strong Errors
    file_load = f'raw_results_strong_time.npy'
    with open(folder / file_load, 'rb') as f:
        raw_results_strong_time = np.load(f)
    file_load = f'raw_results_strong_space.npy'
    with open(folder / file_load, 'rb') as f:
        raw_results_strong_space = np.load(f)
    strong_error_time_standard_errors = np.zeros((9, 2**(M_power - 1) + 1, 2))
    strong_error_space_standard_errors = np.zeros((4, M_max + 1, 2))    
    for m_power_ind in range(9):
        for m in range(2**(M_power - 9 + m_power_ind) + 1):
            strong_error_time_standard_errors[m_power_ind, m, 0] = np.average(raw_results_strong_time[m_power_ind, m, :])
            strong_error_time_standard_errors[m_power_ind, m, 1] = 1 / np.sqrt(max_samples) * np.sqrt(np.var(raw_results_strong_time[m_power_ind, m, :],ddof=1))
    strong_time_errors_index = np.zeros(9, dtype = 'int')
    for m_power_ind in range(9):
        strong_time_errors_index[m_power_ind] = np.argmax(strong_error_time_standard_errors[m_power_ind, :, 0])
    for n_power_ind in range(4):
        for m in range(M_max + 1):
            strong_error_space_standard_errors[n_power_ind, m, 0] = np.average(raw_results_strong_space[n_power_ind, m, :])
            strong_error_space_standard_errors[n_power_ind, m, 1] = 1 / np.sqrt(max_samples) * np.sqrt(np.var(raw_results_strong_space[n_power_ind, m, :],ddof=1))
    strong_space_errors_index = np.zeros(4, dtype = 'int')
    for n_power_ind in range(4):
        strong_space_errors_index[n_power_ind] = np.argmax(strong_error_space_standard_errors[n_power_ind, :, 0])
    # Section on calculating the reference norms strong error.
    mass_matrix = get_2d_M_SG(2**N_power)
    size_of_references_time = np.zeros(9)
    size_of_references_space = np.zeros(4)
    for n_power_ind in range(4):
        for j in range(max_samples):
            file_load = f'reference_solution_{j}.npy'
            with open(folder / file_load, 'rb') as f:
                ref_sol = np.load(f)
            size_of_references_space[n_power_ind] += weak_functional(mass_matrix, ref_sol[:, strong_space_errors_index[n_power_ind]])
        size_of_references_space[n_power_ind] = 1 / max_samples * size_of_references_space[n_power_ind]
        print(f'For n_power_ind = {n_power_ind}, reference norm squared is {size_of_references_space[n_power_ind]} and the error is {strong_error_space_standard_errors[n_power_ind, strong_space_errors_index[n_power_ind], 0]}.')
    for m_power_ind in range(9):
        M = 2**(M_power - 9 + m_power_ind)
        d = M_max // M
        for j in range(max_samples):
            file_load = f'reference_solution_{j}.npy'
            with open(folder / file_load, 'rb') as f:
                ref_sol = np.load(f)
            size_of_references_time[m_power_ind] += weak_functional(mass_matrix, ref_sol[:, d * strong_time_errors_index[m_power_ind]])
        size_of_references_time[m_power_ind] = 1 / max_samples * size_of_references_time[m_power_ind]
        print(f' For m_power_ind = {m_power_ind}, reference norm squared is {size_of_references_time[m_power_ind]} and the error is {strong_error_time_standard_errors[m_power_ind, strong_time_errors_index[m_power_ind], 0]}.')
    # file_save = 'strong_time_errors_standard_error.npy'
    # with open(folder / file_save, 'wb') as f:
        # np.save(f, strong_time_errors, allow_pickle=False, fix_imports=False)
    # file_save = 'strong_space_errors_standard_error.npy'
    # with open(folder / file_save, 'wb') as f:
        # np.save(f, strong_space_errors, allow_pickle=False, fix_imports=False)
    # print('Strong errors with standard error saved.')
    

et = time.time()
print(f'Time taken: {et - st} seconds.')

if plotting_weak_variance_analysis:
    folder = Path('paper_results')
    file_load = 'weak_time_errors_with_standard_error.npy'
    with open(folder / file_load, 'rb') as f:
        weak_errors_time = np.load(f)
    file_load = 'weak_space_errors_with_standard_error.npy'
    with open(folder / file_load, 'rb') as f:
        weak_errors_space = np.load(f)
    x_values_time = np.zeros(9)
    x_values_space = np.zeros(4)
    for m_power_ind in range(9):
        x_values_time[m_power_ind] = 2**(T_power - M_power + 9 - m_power_ind)
    for n_power_ind in range(4):
        x_values_space[n_power_ind] = np.sqrt(2) * 2**(-N_power + 4 - n_power_ind)
    gradient_half_time_weak = np.array([1 / 5 * weak_errors_time[0,0] * 2**(-x / 2) for x in range(9)])
    gradient_quarter_time_weak = np.array([5 * weak_errors_time[0,0] * 2**(-x / 4) for x in range(9)])
    gradient_two_space_weak = np.array([1 / 5 * weak_errors_space[0,0] * 2**(-x * 2) for x in range(4)])
    gradient_one_space_weak = np.array([5 * weak_errors_space[0,0] * 2**(-x) for x in range(4)])
    fig_weak_time = plt.figure(1)
    ax_weak_time = fig_weak_time.add_subplot(111)
    ax_weak_time.loglog(x_values_time, weak_errors_time[:,0], c='k', marker="s", label='weak error')
    ax_weak_time.loglog(x_values_time, gradient_quarter_time_weak, c='r', linestyle="dashed", label='Slope 1/4')
    ax_weak_time.loglog(x_values_time, gradient_half_time_weak, c='b', linestyle="dashed", label='Slope 1/2')
    ax_weak_time.loglog(x_values_time, weak_errors_time[:,0] + weak_errors_time[:,1], c='g', marker="_", markersize = 15, linestyle='')
    ax_weak_time.loglog(x_values_time, weak_errors_time[:,0] - weak_errors_time[:,1], c='g', marker="_", markersize = 15, linestyle='')
    ax_weak_time.set_xlabel('τ')
    ax_weak_time.set_ylabel('weak error')
    ax_weak_time.set_title('Weak error with changing time discretization and standard error')
    ax_weak_time.legend(loc='upper left')
    fig_weak_space = plt.figure(2)
    ax_weak_space = fig_weak_space.add_subplot(111)
    ax_weak_space.loglog(x_values_space, weak_errors_space[:,0], c='k', marker="s", label='weak error')
    ax_weak_space.loglog(x_values_space, gradient_one_space_weak, c='r', linestyle="dashed", label='Slope 1')
    ax_weak_space.loglog(x_values_space, gradient_two_space_weak, c='b', linestyle="dashed", label='Slope 2')
    ax_weak_space.loglog(x_values_space, weak_errors_space[:,0] + weak_errors_space[:,1], c='g', marker="_", markersize = 15, linestyle='')
    ax_weak_space.loglog(x_values_space, weak_errors_space[:,0] - weak_errors_space[:,1], c='g', marker="_", markersize = 15, linestyle='')
    ax_weak_space.set_xlabel('h')
    ax_weak_space.set_ylabel('weak error')
    ax_weak_space.set_title('Weak error with changing space discretization and standard error')
    ax_weak_space.legend(loc='upper left')
    plt.show()

if plotting_weak_graphs:
    folder = Path('paper_results')
    file_load = f'weak_errors_time{M_power - T_power}_space{N_power}_K{K_true}_samples{max_samples_weak}.npy' ## change back to max_samples
    with open(folder / file_load, 'rb') as f:
        weak_errors = np.load(f)
    x_values_time = np.zeros(9)
    x_values_space = np.zeros(4)
    for m_power_ind in range(9):
        x_values_time[m_power_ind] = 2**(T_power - M_power + 9 - m_power_ind)
    for n_power_ind in range(4):
        x_values_space[n_power_ind] = np.sqrt(2) * 2**(-N_power + 4 - n_power_ind)
    y_values_time_weak = np.zeros(9)
    y_values_space_weak = np.zeros(4)
    for m_power_ind in range(9):
        y_values_time_weak[m_power_ind] = weak_errors[0, m_power_ind]
    for n_power_ind in range(4):
        y_values_space_weak[n_power_ind] = weak_errors[1, n_power_ind]
    gradient_half_time_weak = np.array([1 / 5 * y_values_time_weak[0] * 2**(-x / 2) for x in range(9)])
    gradient_quarter_time_weak = np.array([5 * y_values_time_weak[0] * 2**(-x / 4) for x in range(9)])
    gradient_two_space_weak = np.array([1 / 5 * y_values_space_weak[0] * 2**(-x * 2) for x in range(4)])
    gradient_one_space_weak = np.array([5 * y_values_space_weak[0] * 2**(-x) for x in range(4)])
    fig_weak_time = plt.figure(3)
    ax_weak_time = fig_weak_time.add_subplot(111)
    ax_weak_time.loglog(x_values_time, y_values_time_weak, c='k', marker="s", label='weak error')
    ax_weak_time.loglog(x_values_time, gradient_quarter_time_weak, c='r', linestyle="dashed", label='Slope 1/4')
    ax_weak_time.loglog(x_values_time, gradient_half_time_weak, c='b', linestyle="dashed", label='Slope 1/2')
    ax_weak_time.set_xlabel('tau')
    ax_weak_time.set_ylabel('weak Error')
    ax_weak_time.set_title('Weak Error with changing time discretization')
    ax_weak_time.legend(loc='upper left')
    fig_weak_space = plt.figure(4)
    ax_weak_space = fig_weak_space.add_subplot(111)
    ax_weak_space.loglog(x_values_space, y_values_space_weak, c='k', marker="s", label='weak error')
    ax_weak_space.loglog(x_values_space, gradient_one_space_weak, c='r', linestyle="dashed", label='Slope 1')
    ax_weak_space.loglog(x_values_space, gradient_two_space_weak, c='b', linestyle="dashed", label='Slope 2')
    ax_weak_space.set_xlabel('h')
    ax_weak_space.set_ylabel('weak Error')
    ax_weak_space.set_title('Weak Error with changing space discretization')
    ax_weak_space.legend(loc='upper left')
    plt.show()


# This plots the errors.
if plotting_graphs:
    folder = Path('paper_results')
    file_load = f'strong_errors_time{M_power - T_power}_space{N_power}_K{K_true}_samples{max_samples}.npy'
    with open(folder / file_load, 'rb') as f:
        strong_errors = np.load(f)
    file_load = f'weak_errors_time{M_power - T_power}_space{N_power}_K{K_true}_samples{max_samples}.npy'
    with open(folder / file_load, 'rb') as f:
        weak_errors = np.load(f)
    x_values_time = np.zeros(9)
    x_values_space = np.zeros(4)
    for m_power_ind in range(9):
        x_values_time[m_power_ind] = 2**(T_power - M_power + 9 - m_power_ind)
    for n_power_ind in range(4):
        x_values_space[n_power_ind] = np.sqrt(2) * 2**(-N_power + 4 - n_power_ind)
    y_values_time_strong = np.zeros(9)
    y_values_time_weak = np.zeros(9)
    y_values_space_strong = np.zeros(4)
    y_values_space_weak = np.zeros(4)
    for m_power_ind in range(9):
        y_values_time_strong[m_power_ind] = strong_errors[0, m_power_ind]
        y_values_time_weak[m_power_ind] = weak_errors[0, m_power_ind]
    for n_power_ind in range(4):
        y_values_space_strong[n_power_ind] = strong_errors[1, n_power_ind]
        y_values_space_weak[n_power_ind] = weak_errors[1, n_power_ind]
    gradient_half_time_strong = np.array([1 / 3 * y_values_time_strong[0] * 2**(-x / 2) for x in range(9)])
    gradient_quarter_time_strong = np.array([3 * y_values_time_strong[0] * 2**(-x / 4) for x in range(9)])
    gradient_half_time_weak = np.array([1 / 5 * y_values_time_weak[0] * 2**(-x / 2) for x in range(9)])
    gradient_quarter_time_weak = np.array([5 * y_values_time_weak[0] * 2**(-x / 4) for x in range(9)])
    gradient_two_space_strong = np.array([1 / 3 * y_values_space_strong[0] * 2**(-x * 2) for x in range(4)])
    gradient_one_space_strong = np.array([3 * y_values_space_strong[0] * 2**(-x) for x in range(4)])
    gradient_two_space_weak = np.array([1 / 5 * y_values_space_weak[0] * 2**(-x * 2) for x in range(4)])
    gradient_one_space_weak = np.array([5 * y_values_space_weak[0] * 2**(-x) for x in range(4)])
    fig_strong_time = plt.figure(1)
    ax_strong_time = fig_strong_time.add_subplot(111)
    ax_strong_time.loglog(x_values_time, y_values_time_strong, c='k', marker="s", label='strong error')
    ax_strong_time.loglog(x_values_time, gradient_quarter_time_strong, c='r', linestyle="dashed", label='Slope 1/4')
    ax_strong_time.loglog(x_values_time, gradient_half_time_strong, c='b', linestyle="dashed", label='Slope 1/2')
    ax_strong_time.set_xlabel('τ')
    ax_strong_time.set_ylabel('strong error')
    ax_strong_time.set_title('Strong error with changing time discretization')
    ax_strong_time.legend(loc='upper left')
    fig_strong_space = plt.figure(2)
    ax_strong_space = fig_strong_space.add_subplot(111)
    ax_strong_space.loglog(x_values_space, y_values_space_strong, c='k', marker="s", label='strong error')
    ax_strong_space.loglog(x_values_space, gradient_one_space_strong, c='r', linestyle="dashed", label='Slope 1')
    ax_strong_space.loglog(x_values_space, gradient_two_space_strong, c='b', linestyle="dashed", label='Slope 2')
    ax_strong_space.set_xlabel('h')
    ax_strong_space.set_ylabel('strong error')
    ax_strong_space.set_title('Strong error with changing space discretization')
    ax_strong_space.legend(loc='upper left')
    fig_weak_time = plt.figure(3)
    ax_weak_time = fig_weak_time.add_subplot(111)
    ax_weak_time.loglog(x_values_time, y_values_time_weak, c='k', marker="s", label='weak error')
    ax_weak_time.loglog(x_values_time, gradient_quarter_time_weak, c='r', linestyle="dashed", label='Slope 1/4')
    ax_weak_time.loglog(x_values_time, gradient_half_time_weak, c='b', linestyle="dashed", label='Slope 1/2')
    ax_weak_time.set_xlabel('τ')
    ax_weak_time.set_ylabel('weak Error')
    ax_weak_time.set_title('Weak Error with changing time discretization')
    ax_weak_time.legend(loc='upper left')
    fig_weak_space = plt.figure(4)
    ax_weak_space = fig_weak_space.add_subplot(111)
    ax_weak_space.loglog(x_values_space, y_values_space_weak, c='k', marker="s", label='weak error')
    ax_weak_space.loglog(x_values_space, gradient_one_space_weak, c='r', linestyle="dashed", label='Slope 1')
    ax_weak_space.loglog(x_values_space, gradient_two_space_weak, c='b', linestyle="dashed", label='Slope 2')
    ax_weak_space.set_xlabel('h')
    ax_weak_space.set_ylabel('weak Error')
    ax_weak_space.set_title('Weak Error with changing space discretization')
    ax_weak_space.legend(loc='upper left')
    plt.show()
    
if plotting_strong_standard_error:
    folder = Path('paper_results')
    file_load = 'strong_time_errors_standard_error.npy'
    with open(folder / file_load, 'rb') as f:
        strong_errors_time = np.load(f)
    file_load = 'strong_space_errors_standard_error.npy'
    with open(folder / file_load, 'rb') as f:
        strong_errors_space = np.load(f)
    x_values_time = np.zeros(9)
    x_values_space = np.zeros(4)
    for m_power_ind in range(9):
        x_values_time[m_power_ind] = 2**(T_power - M_power + 9 - m_power_ind)
    for n_power_ind in range(4):
        x_values_space[n_power_ind] = np.sqrt(2) * 2**(-N_power + 4 - n_power_ind)
    gradient_one_time_strong = np.array([1 / 5 * strong_errors_time[0,0] * 2**(-x) for x in range(9)])
    gradient_half_time_strong = np.array([5 * strong_errors_time[0,0] * 2**(-x / 2) for x in range(9)])
    gradient_four_space_strong = np.array([1 / 5 * strong_errors_space[0,0] * 2**(-x * 4) for x in range(4)])
    gradient_two_space_strong = np.array([5 * strong_errors_space[0,0] * 2**(-x * 2) for x in range(4)])
    fig_strong_time = plt.figure(1)
    ax_strong_time = fig_strong_time.add_subplot(111)
    ax_strong_time.loglog(x_values_time, strong_errors_time[:,0], c='k', marker="s", label='strong error squared')
    ax_strong_time.loglog(x_values_time, gradient_half_time_strong, c='r', linestyle="dashed", label='Slope 1/2')
    ax_strong_time.loglog(x_values_time, gradient_one_time_strong, c='b', linestyle="dashed", label='Slope 1')
    ax_strong_time.loglog(x_values_time, strong_errors_time[:,0] + strong_errors_time[:,1], c='g', marker="_", markersize = 15, linestyle='')
    ax_strong_time.loglog(x_values_time, strong_errors_time[:,0] - strong_errors_time[:,1], c='g', marker="_", markersize = 15, linestyle='')
    ax_strong_time.set_xlabel('τ')
    ax_strong_time.set_ylabel('strong error squared')
    ax_strong_time.set_title('Strong error squared with changing time discretization and standard error')
    ax_strong_time.legend(loc='upper left')
    fig_strong_space = plt.figure(2)
    ax_strong_space = fig_strong_space.add_subplot(111)
    ax_strong_space.loglog(x_values_space, strong_errors_space[:,0], c='k', marker="s", label='strong error squared')
    ax_strong_space.loglog(x_values_space, gradient_two_space_strong, c='r', linestyle="dashed", label='Slope 2')
    ax_strong_space.loglog(x_values_space, gradient_four_space_strong, c='b', linestyle="dashed", label='Slope 4')
    ax_strong_space.loglog(x_values_space, strong_errors_space[:,0] + strong_errors_space[:,1], c='g', marker="_", markersize = 15, linestyle='')
    ax_strong_space.loglog(x_values_space, strong_errors_space[:,0] - strong_errors_space[:,1], c='g', marker="_", markersize = 15, linestyle='')
    ax_strong_space.set_xlabel('h')
    ax_strong_space.set_ylabel('strong error squared')
    ax_strong_space.set_title('Strong error squared with changing space discretization and standard error')
    ax_strong_space.legend(loc='upper left')
    plt.show()

if plotting_weak_lower_space:
    folder = Path('paper_results')
    file_load = 'weak_errors_space4_standard_error.npy'
    with open(folder / file_load, 'rb') as f:
        y_values_weak = np.load(f)
    file_load = 'weak_time_correlations_space4_standard_error.npy'
    with open(folder / file_load, 'rb') as f:
        y_values_time_correlation = np.load(f)
    x_values_time = np.zeros(9)
    for m_power_ind in range(9):
        x_values_time[m_power_ind] = 2**(T_power - M_power + 9 - m_power_ind)
    gradient_half_weak = np.array([1 / 5 * np.abs(y_values_weak[0,0]) * 2**(-x / 2) for x in range(9)])
    gradient_quarter_weak = np.array([5 * np.abs(y_values_weak[0,0]) * 2**(-x / 4) for x in range(9)])
    gradient_half_time_correlation = np.array([1 / 5 * np.abs(y_values_time_correlation[0,0]) * 2**(-x / 2) for x in range(9)])
    gradient_quarter_time_correlation = np.array([5 * np.abs(y_values_time_correlation[0,0]) * 2**(-x / 4) for x in range(9)])
    fig_weak = plt.figure(1)
    ax_weak = fig_weak.add_subplot(111)
    ax_weak.loglog(x_values_time, y_values_weak[:,0], c='k', marker="s", label='weak error')
    ax_weak.loglog(x_values_time, gradient_quarter_weak, c='r', linestyle="dashed", label='Slope 1/4')
    ax_weak.loglog(x_values_time, gradient_half_weak, c='b', linestyle="dashed", label='Slope 1/2')
    ax_weak.loglog(x_values_time, y_values_weak[:,0] + y_values_weak[:,1], c='g', marker="_", markersize = 15, linestyle='')
    ax_weak.loglog(x_values_time, y_values_weak[:,0] - y_values_weak[:,1], c='g', marker="_", markersize = 15, linestyle='')
    ax_weak.set_xlabel('τ')
    ax_weak.set_ylabel('weak error')
    ax_weak.set_title('Weak error with changing time discretization and standard error')
    ax_weak.legend(loc='upper left')
    fig_time_correlation = plt.figure(2)
    ax_time_correlation = fig_time_correlation.add_subplot(111)
    ax_time_correlation.loglog(x_values_time, y_values_time_correlation[:,0], c='k', marker="s", label='weak time correlation error')
    ax_time_correlation.loglog(x_values_time, gradient_quarter_time_correlation, c='r', linestyle="dashed", label='Slope 1/4')
    ax_time_correlation.loglog(x_values_time, gradient_half_time_correlation, c='b', linestyle="dashed", label='Slope 1/2')
    ax_time_correlation.loglog(x_values_time, y_values_time_correlation[:,0] + y_values_time_correlation[:,1], c='g', marker="_", markersize = 15, linestyle='')
    ax_time_correlation.loglog(x_values_time, y_values_time_correlation[:,0] - y_values_time_correlation[:,1], c='g', marker="_", markersize = 15, linestyle='')
    ax_time_correlation.set_xlabel('τ')
    ax_time_correlation.set_ylabel('weak time correlation error')
    ax_time_correlation.set_title('Weak time correlation error with changing time discretization and standard error')
    ax_time_correlation.legend(loc='upper left')
    plt.show()

# def get_2d_matrix_to_be_inverted(N,tau):
    # return identity((N - 1)**2, format='csc') + tau * N**2 * get_2d_S_SG(N)

#### There's a problem with the nonnegativity!!!!!! Checking if dense_expm helps. If it does change something, then see what happens
#### for much smaller tau (like 2^-26 or something). Dense_expm does help!!!! Should redo everything.
# testing = False
# if testing:
    # folder = Path('paper_results')
    # file_load = 'reference_solution_3.npy'
    # with open(folder / file_load, 'rb') as f:
        # test = np.load(f)
    # print(np.isnan(np.sum(test)))
    # print(np.min(test))
    # print(np.max(test))
    # st = time.time()
    # folder = Path('paper_results')
    # file = f'exp_matrix_space{6}_time{12}.npy'
    # with open(folder / file, 'rb') as f:
        # matrix = np.load(f)
    # no_neg = np.count_nonzero(np.abs(matrix) - matrix)
    # print(f'For space power {6} and time power {12}, exponential matrix using sparse has {no_neg} negative entries.')
    # file = f'exp_matrix_space{6}_time{13}.npy'
    # with open(folder / file, 'rb') as f:
        # matrix = np.load(f)
    # no_neg = np.count_nonzero(np.abs(matrix) - matrix)
    # print(f'For space power {6} and time power {13}, exponential matrix using sparse has {no_neg} negative entries.')
    # et = time.time()
    # print(f'{et - st} s')
    # matrix = dense_expm((-1 * 2**(2 * 6 - 12) * get_2d_S_SG(2**6)).toarray())
    # no_neg = np.count_nonzero(np.abs(matrix) - matrix)
    # print(f'For space power {6} and time power {12}, exponential matrix using dense has {no_neg} negative entries.')
    # et = time.time()
    # print(f'{et - st} s')
    # matrix = dense_expm((-1 * 2**(2 * 6 - 13) * get_2d_S_SG(2**6)).toarray())
    # no_neg = np.count_nonzero(np.abs(matrix) - matrix)
    # print(f'For space power {6} and time power {13}, exponential matrix using dense has {no_neg} negative entries.')
    # et = time.time()
    # print(f'{et - st} s')
    
    #test_matrix = (sparse_inv(get_2d_matrix_to_be_inverted(2**N_power, 2**(12) * tau_min))).toarray()
    # test_matrix = sparse_expm(-1 * 2**(2 * 5 - 52) * get_2d_S_SG(2**5)).toarray()
    # print(f'Maximum is {np.max(test_matrix)}.')
    # print(f'Minimum is {np.min(test_matrix)}.')
    # folder = Path('paper_results')
    # file = f'exp_matrix_space{6}_time{12}.npy'
    # with open(folder / file, 'rb') as f:
        # matrix = np.load(f)
    # print(f'Maximum is {np.max(matrix)}.')
    # print(f'Minimum is {np.min(matrix)}.')
    # file = f'exp_matrix_space{6}_time{13}.npy'
    # with open(folder / file, 'rb') as f:
        # matrix = np.load(f)
    # print(f'Maximum is {np.max(matrix)}.')
    # print(f'Minimum is {np.min(matrix)}.')