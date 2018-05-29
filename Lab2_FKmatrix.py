# Template for multiplying two matrices

import matlib
import math
import sys

# Use help(math) to see what functions
# the math library contains


def cos(x):
    return math.cos(math.radians(x))


def sin(x):
    return math.sin(math.radians(x))


def dh_params_to_matrix(dh):
    theta = dh[0]
    d = dh[1]
    a = dh[2]
    alpha = dh[3]
    return [
        [cos(theta), -sin(theta)*cos(alpha), sin(theta)*sin(alpha), a*cos(theta)],
        [sin(theta), cos(theta)*cos(alpha), -cos(theta)*sin(alpha), a*sin(theta)],
        [0,          sin(alpha),            cos(alpha),             d],
        [0,          0,                     0,                      1]
    ]


theta1 = int(sys.argv[1])
theta2 = int(sys.argv[2])
theta3 = int(sys.argv[3])

dh0 = [0, 0.1, 0, -90]
dh1 = [theta1, -0.098, 0, 90]
dh2 = [theta2, 0, 0.105, -90]
dh3 = [-theta3, -0.015, 0.1137, 90]

a0 = dh_params_to_matrix(dh0)
a1 = dh_params_to_matrix(dh1)
a2 = dh_params_to_matrix(dh2)
a3 = dh_params_to_matrix(dh3)
tool_offset = [
    [1, 0, 0, 0],
    [0, 1, 0, 0],
    [0, 0, 1, -0.01231],
    [0, 0, 0, 1]
]

inter1 = matlib.matmul(a0, a1)
inter2 = matlib.matmul(inter1, a2)
inter3 = matlib.matmul(inter2, a3)
final = matlib.matmul(inter3, tool_offset)

print 'RShoulderPitch =', theta1
print 'RShoulderRoll =', theta2
print 'RElbowRoll =', theta3
print 'H ='
# for i in range(4):
#     print ('%.4f\n%.4f\n%.4f\n%.4f\n' % (final[0][i], final[1][i], final[2][i], final[3][i]))

matlib.matprint(final, '%8.3f')
