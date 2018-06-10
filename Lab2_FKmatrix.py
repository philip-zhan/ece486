import matlib
import math
import sys
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

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


def generate_transformation_matrix(shoulder_pitch, shoulder_roll, elbow_roll):
    dh0 = [0, 0.1, 0, -90]
    dh1 = [shoulder_pitch, -0.098, 0, 90]
    dh2 = [shoulder_roll, 0, 0.105, -90]
    dh3 = [-elbow_roll, -0.015, 0.1137, 90]

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
    return final


def plot_workspace(is_2d):
    shoulder_pitch_range = np.arange(-119.5, 119.5, 2)
    shoulder_roll_range = np.arange(-76, 18, 2)
    elbow_roll_range = np.arange(2, 88.5, 2)

    transformation_matrices = []
    for shoulder_pitch in shoulder_pitch_range:
        for elbow_roll in elbow_roll_range:
            if is_2d:
                transformation_matrices.append(generate_transformation_matrix(shoulder_pitch, 0, elbow_roll))
            else:
                for shoulder_roll in shoulder_roll_range:
                    transformation_matrices.append(generate_transformation_matrix(shoulder_pitch, shoulder_roll, elbow_roll))

    xs = [matrix[0][3] for matrix in transformation_matrices]
    ys = [matrix[1][3] for matrix in transformation_matrices]
    zs = [matrix[2][3] for matrix in transformation_matrices]

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.plot(xs, ys, zs)
    plt.show()


def main():
    if len(sys.argv) == 4:
        shoulder_pitch = int(sys.argv[1])
        shoulder_roll = int(sys.argv[2])
        elbow_roll = int(sys.argv[3])
        transformation_matrix = generate_transformation_matrix(shoulder_pitch, shoulder_roll, elbow_roll)
        print 'RShoulderPitch =', shoulder_pitch
        print 'RShoulderRoll =', shoulder_roll
        print 'RElbowRoll =', elbow_roll
        print 'H ='
        matlib.matprint(transformation_matrix, '%8.3f')
    else:
        # plot_workspace(True)
        plot_workspace(False)


if __name__ == '__main__':
    main()
