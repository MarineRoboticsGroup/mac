import numpy as np
from collections import namedtuple
import matplotlib.pyplot as plt
from scipy.sparse import csr_matrix, coo_matrix
from typing import List, Tuple, Union
import networkx as nx
from mac.utils import Edge

# Define RelativePoseMeasurement container
RelativePoseMeasurement = namedtuple('RelativePoseMeasurement',
                                     ['i', 'j', 't', 'R', 'kappa', 'tau'])

def rot2D_from_theta(theta: float) -> np.ndarray:
    """
    Simply builds a 2D rotation matrix:

        | cos(theta) -sin(theta) |
        | sin(theta)  cos(theta) |

    from a floating point angle `theta`.

    Args:
        theta (float): Angle in radians.

    Returns:
        np.ndarray: 2D rotation matrix.
    """
    c = np.cos(theta)
    s = np.sin(theta)
    return np.array([[c, -s], [s, c]])


# TODO @kevin what is the type of `poses`? Is it list of np.ndarray?
def se2poses_to_x(poses) -> np.ndarray:
    """
    Takes list of N SE(2) poses and returns a numpy array formatted as:
    [R1 R2 R3 ... RN t1 t2 t3 ... tN]
    where Ri is a matrix in SO(2) and ti is a 2D vector

    Args:
        poses (??): List of SE(2) poses.

    Returns:
        np.ndarray: Numpy array of SE(2) poses.
    """
    d = 2
    N = len(poses)
    offset = N
    X = np.zeros([d, N*(d+1)])
    for (i,pose) in enumerate(poses):
        X[:,i] = pose[:d, d]
        X[:,(offset + i*d):(offset + i*d + d)] = pose[:d, :d]
    return X


# TODO @kevin what is the type of `pose`? Is it np.ndarray?
def Rt_from_pose(pose) -> Tuple[np.ndarray, np.ndarray]:
    """
    Returns the rotation matrix and translation vector from a pose.

    Args:
        pose (np.ndarray): Pose matrix.

    Returns:
        Tuple[np.ndarray, np.ndarray]: Tuple containing the rotation matrix and translation vector.
    """
    # TODO should have an assert that the pose is 2D (matrix is correctly shaped)

    Xmat = se2poses_to_x([pose])
    return rotations_from_variable_matrix(Xmat), translations_from_variable_matrix(Xmat)

def plot_poses(xhat: np.ndarray, measurements: List[RelativePoseMeasurement],
                show: bool=True, color: str='b', alpha: float=0.25) -> None:
    """Plots the edges indicated by the given measurements, with node vertices
    determined by xhat. In english: plots the estimated pose graph.

    Args:
        xhat (np.ndarray): the variable matrix containing the poses to plot
        measurements (List[RelativePoseMeasurement]): the list of measurements
        show (bool, optional): whether to plot now. Defaults to True.
        color (str, optional): the plotting color of the measurements. Defaults to 'b'.
        alpha (float, optional): transparency. Defaults to 0.25.
    """
    fig = plt.figure()
    t_hat = translations_from_variable_matrix(xhat)
    Rhat = rotations_from_variable_matrix(xhat)

    d, n = t_hat.shape
    if d == 3:
        ax = fig.gca(projector='3d')
    else:
        ax = fig.add_subplot(1,1,1)

    # 'Unrotate' the vectors by pre-multiplying by the inverse of the first
    # orientation estimate
    t_hat_rotated = Rhat[0:d, 0:d].transpose().dot(t_hat)

    # Translate the resulting vectors to the origin
    t_hat_anchored = t_hat_rotated - np.mat(t_hat_rotated[:, 0]).transpose()

    # print(t_hat_anchored.shape)
    # print(t_hat_anchored)

    x = t_hat_anchored[0, :]
    y = t_hat_anchored[1, :]
    # print(x)
    # print(y)

    # Plot odometric links (this should skip lines that don't make sense?)
    ax.plot(
        x.transpose(),
        y.transpose(), alpha=1.0, color=color, linewidth=0.5)  # matplotlib.pyplot hates row vectors, so we transpose

    # print(x.shape)
    # plt.scatter(
    #     list(x.flatten()),
    #     list(y.flatten()))  # matplotlib.pyplot hates row vectors, so we transpose

    # For now, we don't change linestyles or alpha
    for measurement in measurements:
        id1 = measurement.i
        id2 = measurement.j

        x_pair = [t_hat_anchored[0, id1], t_hat_anchored[0, id2]]
        y_pair = [t_hat_anchored[1, id1], t_hat_anchored[1, id2]]

        if abs(id1 - id2) > 1:
            # This is a loop closure measurement
            ax.plot(x_pair, y_pair, alpha=alpha, color=color, linewidth=0.5)
        # else:
        #     plt.plot(x_pair, y_pair, color='r')


    # Sometimes matplotlib really grinds my gears
    # See: https://github.com/matplotlib/matplotlib/issues/17172#issuecomment-830139107
    if d == 3:
        ax.set_box_aspect([ub - lb for lb, ub in (getattr(ax, f'get_{a}lim')() for a in 'xyz')])
    else:
        plt.gca().set_aspect('equal')
    ax.set_axis_off()

    if show:
        plt.show()


def quat2rot(q: Union[np.ndarray, List]) -> np.ndarray:
    """
    Converts a quaternion q = [qw, qx, qy, qz] to a 3D rotation matrix

    Args:
        q (Union[np.ndarray, List]): Quaternion to convert.

    Returns:
        np.ndarray: 3D rotation matrix.
    """

    s = np.zeros(4)
    qw = q[0]
    qx = q[1]
    qy = q[2]
    qz = q[3]

    R = np.zeros((3,3))
    R[0,0] = qx**2 - qy**2 - qz**2 + qw**2
    R[0,1] = 2.0*(qx * qy - qz * qw)
    R[0,2] = 2.0*(qx * qz + qy * qw)

    R[1,0] = 2.0*(qx*qy + qz*qw)
    R[1,1] = -qx**2 + qy**2 - qz**2 + qw**2
    R[1,2] = 2.0*(qy*qz - qx * qw)

    R[2,0] = 2.0*(qx*qz - qy*qw)
    R[2,1] = 2.0*(qy*qz + qx*qw)
    R[2,2] = -qx**2 -qy**2 + qz**2 + qw**2
    return R

def read_g2o_file(filename: str) -> Tuple[List[RelativePoseMeasurement], int]
    """
    Parses the file with path `filename`, interpreting it as a g2o file and
    returning the set of measurements it contains and the number of poses
    Only works with 2D problems. 3D pose graph parsing is not implemented, but
    can be without too much trouble

    # TODO @kevin it seems like you now have 3D pose graphs, so this should be updated?

    Args:
        filename (str): Path to the g2o file.

    Returns:
        Tuple[List[RelativePoseMeasurement], int]: A tuple containing the list
        of measurements and the number of poses in the file.
    """
    measurements = []
    with open(filename, 'r') as infile:
        num_poses = 0
        for line in infile:
            parsed_line = line.split(' ', )

            # Clean up output
            while '\n' in parsed_line:
                parsed_line.remove('\n')
            while '' in parsed_line:
                parsed_line.remove('')

            if len(parsed_line) == 0:
                continue

            token = parsed_line[0]
            if (token == "EDGE_SE3:QUAT"):
                # This is a 3D pose measurements
                """The g2o format specifies a 3D relative pose measurement in the
                following form:
                EDGE_SE3:QUAT id1 id2 dx dy dz dqx dqy dqz dqw
                I11 I12 I13 I14 I15 I16
                    I22 I23 I24 I25 I26
                        I33 I34 I35 I36
                            I44 I45 I46
                                I55 I56
                                    I66
                """
                # Extract formatted output
                i, j, dx, dy, dz, dqx, dqy, dqz, dqw, I11, I12, I13, I14, I15, I16, I22, I23, I24, I25, I26, I33, I34, I35, I36, I44, I45, I46, I55, I56, I66 = map(float, parsed_line[1:])

                # Fill in elements of this measurement

                # Clean up pose ids
                i = int(i)
                j = int(j)

                # Raw measurements
                t = np.array([dx, dy, dz])

                # Reconstruct quaternion for relative measurement
                q = np.array([dqw, dqx, dqy, dqz])
                q = q / np.linalg.norm(q)

                R = quat2rot(q)

                meas_info = np.array([[I11, I12, I13, I14, I15, I16],
                                       [I12, I22, I23, I24, I25, I26],
                                      [I13, I23, I33, I34, I35, I36],
                                      [I14, I24, I34, I44, I45, I46],
                                      [I15, I25, I35, I45, I55, I56],
                                      [I16, I26, I36, I46, I56, I66]])

                tau = 3.0 / np.trace(np.linalg.inv(meas_info[0:2, 0:2]))
                kappa = 3.0 / (2.0 * np.trace(np.linalg.inv(meas_info[3:5, 3:5])))

                measurement = RelativePoseMeasurement(i=i,
                                                      j=j,
                                                      t=t,
                                                      R=R,
                                                      tau=tau,
                                                      kappa=kappa)
                max_pair = max(i, j)
                num_poses = max(num_poses, max_pair)

                measurements.append(measurement)
            if (token == "EDGE_SE2"):
                # This is a 2D pose measurement
                """ The g2o format specifies a 2D relative pose measurement in the following
                form:
                EDGE_SE2 id1 id2 dx dy dtheta I11 I12 I13 I22 I23 I33
                """

                # Extract formatted output
                i, j, dx, dy, dtheta, I11, I12, I13, I22, I23, I33 = map(
                    float, parsed_line[1:])

                # Fill in elements of this measurement

                # Clean up pose ids
                i = int(i)
                j = int(j)

                # Raw measurements
                t = np.array([dx, dy])
                R = rot2D_from_theta(dtheta)

                tran_cov = np.array([[I11, I12], [I12, I22]])
                tau = 2.0 / np.trace(np.linalg.inv(tran_cov))
                kappa = I33
                measurement = RelativePoseMeasurement(i=i,
                                                      j=j,
                                                      t=t,
                                                      R=R,
                                                      tau=tau,
                                                      kappa=kappa)
                max_pair = max(i, j)
                num_poses = max(num_poses, max_pair)

                measurements.append(measurement)

    # Account for zero-based indexing
    num_poses = num_poses + 1

    return measurements, num_poses

def translations_from_variable_matrix(xhat: np.ndarray) -> np.ndarray:
    """extracts the translations from the variable matrix xhat

    Args:
        xhat (np.ndarray): the variable matrix

    Returns:
        np.ndarray: the translations stacked on top of each other
    """
    d, cols = xhat.shape
    n = int(cols / (d + 1))
    print(n)
    return xhat[:, 0:n]


def rotations_from_variable_matrix(xhat: np.ndarray) -> np.ndarray:
    """Returns the columns associated with rotations in the variable matrix
    #TODO @kevin: check that the above statement is correct

    Args:
        xhat (np.ndarray): the variable matrix

    Returns:
        np.ndarray: the rotations stacked on top of each other
    """
    d, cols = xhat.shape
    n = int(cols / (d + 1))

    # TODO if we are just going from 'n' to end of matrix it might be cleaner to
    # just write `xhat[:, n:]`
    return xhat[:, n:(d + 1) * n]

def rpm_to_mac(measurements: List[RelativePoseMeasurement]) -> List[Edge]:
    """Converts a list of relative pose measurements into a list of edges of a
    graph weighted by the rotation measurement concentration parameters (kappas)
    of the measurements.

    Args:
        measurements (List[RelativePoseMeasurement]): the list of relative pose measurements

    Returns:
        List[Edge]: the list of edges
    """
    edges = []
    for meas in measurements:
        edge = Edge(meas.i, meas.j, meas.kappa)
        edges.append(edge)
    return edges

#TODO @kevin: is this dead code? It looks like the function is unfinished
def nx_to_rpm(G: nx.Graph) -> List[RelativePoseMeasurement]:
    measurements = []
    for edge in G.edges():
        t = np.zeros(3)
        R = np.eye(3)
        meas = RelativePoseMeasurement(edge[0], edge[1], t, R, 1.0, 1.0)
        measurements.append(meas)
    return measurements


def rpm_to_nx(measurements: List[RelativePoseMeasurement]) -> nx.Graph:
    """Convert a list of relative pose measurements into a networkx graph
    weighted by the rotation measurement concentration parameters (kappas)
    of the measurements.

    Args:
        measurements (List[RelativePoseMeasurement]): the list of relative pose measurements

    Returns:
        nx.Graph: the graph
    """
    G = nx.Graph()
    for meas in measurements:
        G.add_edge(meas.i, meas.j, weight=meas.kappa)
    return G
