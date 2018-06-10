from numpy import eye, array, dot
import cv2

# Camera projection matrices
P1 = eye(4)
P2 = array([[ 0.878, -0.01 ,  0.479, -1.995],
            [ 0.01 ,  1.   ,  0.002, -0.226],
            [-0.479,  0.002,  0.878,  0.615],
            [ 0.   ,  0.   ,  0.   ,  1.   ]])

# Homogeneous arrays
a3xN = array([[ 0.091,  0.167,  0.231,  0.083,  0.154],
              [ 0.364,  0.333,  0.308,  0.333,  0.308],
              [ 1.   ,  1.   ,  1.   ,  1.   ,  1.   ]])

b3xN = array([[ 0.42 ,  0.537,  0.645,  0.431,  0.538],
              [ 0.389,  0.375,  0.362,  0.357,  0.345],
              [ 1.   ,  1.   ,  1.   ,  1.   ,  1.   ]])

# The cv2 method
X = cv2.triangulatePoints( P1[:3], P2[:3], a3xN[:2], b3xN[:2] )
# Remember to divide out the 4th row. Make it homogeneous
X /= X[3]
# Recover the origin arrays from PX
x1 = dot(P1[:3],X)
x2 = dot(P2[:3],X)
# Again, put in homogeneous form before using them
x1 /= x1[2]
x2 /= x2[2]
 
#print('X\n', X)
#print('x1\n', x1)
#print('x2\n', x2)


in_mat = array([[ 338.,    0.,  320.],
                [   0.,  338.,  169.],
                [   0.,    0.,    1.]])

M_l = array([[-0.80043717, -0.43424157,  0.41320044,  0.3158204 ],
             [-0.4342354 , -0.05511935, -0.89911149, -0.68739911],
             [ 0.41320692, -0.89910851, -0.14444349,  0.6540183 ]])

M_r = array([[-0.80049903, -0.43498174,  0.41230109,  0.73909118],
             [-0.43400221, -0.05372092, -0.89930871, -1.42759566],
             [ 0.41333206, -0.89883533, -0.14577949,  1.17646534]])

P_l = dot(in_mat, M_l)
P_r = dot(in_mat, M_r)

px_l = array([[ 139.72270203,    4.29182386],
              [ 143.67126465,    8.48717308],
              [ 139.62565613,   10.48939228]])

px_r = array([[ 138.25082397,    5.02436066],
              [ 142.21923828,    9.47432137],
              [ 138.17907715,   11.47253132]])

X = cv2.triangulatePoints(P_l, P_r, px_l.T, px_r.T)

# Remember to divide out the 4th row. Make it homogeneous
X /= X[3]
# Recover the origin arrays from PX
x1 = dot(P_l,X)
x2 = dot(P_r,X)

# Again, put in homogeneous form before using them
x1 /= x1[2]
x2 /= x2[2]
 
print('X\n', X)
print('x1\n', x1)
print('x2\n', x2)

