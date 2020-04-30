import numpy as np


def mc_volume(H, num_samples):
    num_constraints, num_vars = H.shape
    
    r = 1 - 2*np.random.rand(num_vars, num_samples)
    #print(r)
    bool_check = np.dot(H, r)>0
    check_cols = np.sum(bool_check.astype(int), axis=0)
    intersection = np.sum(check_cols == num_constraints)
    good_samples = r[:,check_cols == num_constraints]
    return intersection / num_samples, good_samples


def test_cvx_scipy():
    H_cvxopt = np.array([[-0.9,       -2.0853279, -0.81     ],
            [-0.9,       2.507031,  0.9     ]])

    H_scipy = np.array([[ 0.,     -0.6561,  0.    ],
            [-0.9,       2.507031,  0.9     ],
            [ 0.,       2.14659, -0.9    ]])

    num_samps = 50000000
    print("cvx")
    vol, s = mc_volume(H_cvxopt, num_samps)
    print(vol)
    #print(s.transpose() > 0)
    print("scipy")
    vol, s = mc_volume(H_scipy, num_samps)
    print(vol)
#print(s.transpose())

bad_lp_constraints =  [[ 9.00000000e-01,0.00000000e+00, -1.71000000e+00,8.10000000e-01, 0.00000000e+00],
 [-8.10000000e-01,9.00000000e-01,0.00000000e+00,8.10000000e-01, -9.00000000e-01],
 [ 0.00000000e+00,0.00000000e+00,9.00000000e-01,0.00000000e+00,-9.00000000e-01],
 [-9.00000000e-01,0.00000000e+00, -8.10000000e-01,1.70999918e+00, 0.00000000e+00],
 [ 0.00000000e+00, -1.71000000e+00,1.71000000e+00,0.00000000e+00, 0.00000000e+00],
 [ 0.00000000e+00, -8.10000000e-01,1.71000000e+00,0.00000000e+00,-9.00000000e-01],
 [ 0.00000000e+00, -9.00000000e-01,0.00000000e+00,0.00000000e+00, 9.00000000e-01],
 [ 9.00000000e-01, -8.10000000e-01,0.00000000e+00,8.10000000e-01,-9.00000000e-01],
 [ 9.00000000e-01,0.00000000e+00,0.00000000e+00,0.00000000e+00,-9.00000000e-01],
 [-8.10000000e-01,9.00000000e-01, -7.29000000e-01,1.53900000e+00,-9.00000000e-01],
 [-9.00000000e-01,0.00000000e+00,0.00000000e+00,8.99999179e-01, 0.00000000e+00],
 [ 9.00000000e-01, -1.71000000e+00,8.10000000e-01,0.00000000e+00, 0.00000000e+00],
 [ 9.00000000e-02,0.00000000e+00,8.10000000e-02,7.29000000e-01, -9.00000000e-01],
 [ 9.00000000e-01,0.00000000e+00,8.10000000e-01, -7.38747908e-07,-1.71000000e+00],
 [-8.10000000e-01,0.00000000e+00,0.00000000e+00,1.70999918e+00,-9.00000000e-01],
 [-8.10000000e-01, -9.00000000e-01,0.00000000e+00,1.70999918e+00, 0.00000000e+00],
 [ 0.00000000e+00, -8.10000000e-01,0.00000000e+00,1.70999918e+00,-9.00000000e-01],
 [-9.00000000e-01,0.00000000e+00,9.00000000e-02,8.09999179e-01, 0.00000000e+00],
 [-8.10000000e-01,0.00000000e+00,1.71000000e-01,1.53899918e+00,-9.00000000e-01]]
lp_constraints = np.array(bad_lp_constraints)
print(mc_volume(lp_constraints, 100000)[0])
input()


#TODO: create a brute force checker for the lp redundancy removal, 
# we can compare volumes for different approaches, also see whether true reward lies in intersection
# also can take a sample of points in the intersection, run value iteration and check if same policy

H = [[ 0.9,  0.,   0.,  -0.9],
[ 0.81000533,  0.,         -0.81,        0.        ],
[-7.37099464,  0.,          8.27099179, -0.9       ],
[ 0.,         -0.9,         0.89999918,  0.        ],
[ 0.72899716,  0.,          0.171,      -0.9       ],
[ 0.9, -0.9,  0.,   0. ],
[ 0.819,  0.,     0.081, -0.9  ],
[-8.17030556e-06,  0.00000000e+00,  9.00000000e-01, -9.00000000e-01],
[ 8.99999996, -0.9,        -8.09999261,  0.        ]]

eval = [ 0.94137693, -0.9161511,   0.858183,    0.35572327]

true = [ 0.92287664, -0.56020092,  0.81459249, -0.26578537]

#check if in BEC

bool_check = np.dot(H, eval)>0
print(bool_check)
check_cols = np.sum(bool_check.astype(int), axis=0)
print(check_cols)
intersection = np.sum(check_cols == len(H))
print(intersection)

bool_check = np.dot(H, true)>0
print(bool_check)
check_cols = np.sum(bool_check.astype(int), axis=0)
print(check_cols)
intersection = np.sum(check_cols == len(H))
print(intersection)
