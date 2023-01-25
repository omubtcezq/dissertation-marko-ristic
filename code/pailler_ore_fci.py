import numpy as np

def omega_exact(sensor_traces):
    inv_sum = sum((1/x for x in sensor_traces))
    weights = []
    for trace in sensor_traces:
        weights.append((1/trace)/inv_sum)
    return weights

def omega_2sen(tr1, tr2):
	return (1/tr1)/(1/tr1+1/tr2)

def copy_mat_into_mat(in_mat, out_mat, in_dim_i, in_dim_j, out_start_i, out_start_j):
	for i in range(in_dim_i):
		for j in range(in_dim_j):
			out_mat[out_start_i+i][out_start_j+j] = in_mat[i][j]

def copy_vec_into_mat(in_vec, out_mat, in_dim, out_start_i, out_start_j, as_col_vec=True):
	if as_col_vec:
		for i in range(in_dim):
			out_mat[out_start_i+i][out_start_j] = in_vec[i]
	else:
		for j in range(in_dim):
			out_mat[out_start_i][out_start_j+j] = in_vec[j]

def copy_vec_into_vec(in_vec, out_vec, in_dim, out_start):
	for i in range(in_dim):
		out_vec[out_start+i] = in_vec[i]

def omega_partials(sensor_traces):
	n = len(sensor_traces)

	# Linear system dimensions
	res = np.zeros(n*(n-1))
	mat = np.zeros((n*(n-1),n+(n-1)*(n-1)))

	# Get computable point, trivial points and differences
	for i in range(n-1):
		omega = omega_2sen(sensor_traces[i], sensor_traces[i+1])
		point = np.array([0]*i + [omega, 1-omega] + [0]*(n-2-i))
		known_points = []
		for j in range(n):
			if j==i or j==i+1:
				continue
			p = np.zeros(n)
			p[j] = 1
			known_points.append(p)
		directions = [p-point for p in known_points]
		#print(point,'')
		#print(known_points,'')
		#print(directions,'')

		# Populate linear system matrices
		copy_vec_into_vec(-point, res, n, i*n)
		copy_mat_into_mat(-np.eye(n), mat, n, n, i*n, 0)
		for ind,d in enumerate(directions):
			copy_vec_into_mat(d, mat, n, i*n, n+(n-2)*i+ind)
	#print(res,'')
	#print(mat,'')

	# Solve system
	sol_vec, err, rnk, sing = np.linalg.lstsq(mat, res)
	#print(err,')
	return sol_vec[:n]

rand_tests = []
for i in range(100):
	traces = [np.random.randint(1,256) for _ in range(4)]
	true_weights = omega_exact(traces)
	weights_to_check = omega_partials(traces)
	res = np.allclose(true_weights, weights_to_check)
	print("Test %d" % i)
	print("True           :", true_weights)
	print("To check       :", weights_to_check)
	print("Allclose       :", res)
	rand_tests.append(res)
print("Passed all tests:", np.all(rand_tests))