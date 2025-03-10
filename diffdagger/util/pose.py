from scipy.spatial.transform import Rotation as R
import numpy as np
import itertools
from numpy.linalg import norm


def q_mult(q1, q2):

    w1, x1, y1, z1 = np.split(q1, q1.shape[-1], axis=-1)
    w2, x2, y2, z2 = np.split(q2, q2.shape[-1], axis=-1)

    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 + y1 * w2 + z1 * x2 - x1 * z2
    z = w1 * z2 + z1 * w2 + x1 * y2 - y1 * x2
    return np.concatenate((w, x, y, z), axis=-1)


def q_conjugate(q):
    w, x, y, z = np.split(q, q.shape[-1], axis=-1)
    return np.concatenate((w, -x, -y, -z), axis=-1)


def rotations(array):
    for x, y, z in itertools.permutations([0, 1, 2]):
        for sx, sy, sz in itertools.product([-1, 1], repeat=3):
            rotation_matrix = np.zeros((3, 3))
            rotation_matrix[0, x] = sx
            rotation_matrix[1, y] = sy
            rotation_matrix[2, z] = sz
            if np.linalg.det(rotation_matrix) == 1:
                yield np.matmul(rotation_matrix, array)


def quat_rotations():
    for x, y, z in itertools.permutations([0, 1, 2]):
        for sx, sy, sz in itertools.product([-1, 1], repeat=3):
            rotation_matrix = np.zeros((3, 3))
            rotation_matrix[0, x] = sx
            rotation_matrix[1, y] = sy
            rotation_matrix[2, z] = sz
            if np.linalg.det(rotation_matrix) == 1:
                quat = R.from_matrix(rotation_matrix)
                yield quat.as_quat()


all_rotations = np.array(list(quat_rotations()))


def get_delta_ori_aligned2(env, obj_num, dist_norm_threshold=0.13):
    tcp_q = R.from_quat(env.robot.get_ee_orientation())
    rotated_tcp_q = tcp_q * R.from_quat([1, 0, 0, 0])
    obj_ori = env.sim.get_base_orientation(f"object{obj_num}")
    ori_indices = np.argsort(
        [
            1
            - np.dot(
                (R.from_quat(obj_ori) * R.from_quat(rotation)).as_quat(),
                rotated_tcp_q.as_quat(),
            )
            ** 2
            for rotation in all_rotations
        ]
    )
    dist = (
        env.sim.get_base_position("object1")[:2]
        - env.sim.get_base_position("object2")[:2]
    )
    dist_norm = norm(dist)
    dist_axis = dist / dist_norm
    # print('start', end =" ")
    for i, rotation in enumerate(all_rotations[ori_indices]):
        rotated_ori = R.from_quat(obj_ori) * R.from_quat(rotation)
        if rotated_ori.as_matrix()[2, 2] < 2 ** (-0.5):
            continue
        if dist_norm > dist_norm_threshold:
            break
        obj_axis = rotated_ori.as_matrix()[:2, 1] / norm(rotated_ori.as_matrix()[:2, 1])
        if abs(np.dot(obj_axis, dist_axis)) > 2 ** (-0.5):
            continue
        break

    ori = (rotated_ori * R.from_quat([1, 0, 0, 0]) * tcp_q ** (-1)).as_quat()
    return ori


def get_delta_ori_aligned3(env, obj_num):
    other_obj_nums = [num for num in range(1, 4) if num != obj_num]
    tcp_q = R.from_quat(env.robot.get_ee_orientation())
    rotated_tcp_q = tcp_q * R.from_quat([1, 0, 0, 0])
    obj_ori = env.sim.get_base_orientation(f"object{obj_num}")
    closer_other_obj_num = (
        other_obj_nums[0]
        if norm(
            env.sim.get_base_position(f"object{obj_num}")
            - env.sim.get_base_position(f"object{other_obj_nums[0]}")
        )
        < norm(
            env.sim.get_base_position(f"object{obj_num}")
            - env.sim.get_base_position(f"object{other_obj_nums[1]}")
        )
        else other_obj_nums[1]
    )
    ori_indices = np.argsort(
        [
            1
            - np.dot(
                (R.from_quat(obj_ori) * R.from_quat(rotation)).as_quat(),
                rotated_tcp_q.as_quat(),
            )
            ** 2
            for rotation in all_rotations
        ]
    )
    dist = (
        env.sim.get_base_position(f"object{obj_num}")[:2]
        - env.sim.get_base_position(f"object{closer_other_obj_num}")[:2]
    )
    dist_norm = norm(dist)
    dist_axis = dist / dist_norm

    for i, rotation in enumerate(all_rotations[ori_indices]):
        rotated_ori = R.from_quat(obj_ori) * R.from_quat(rotation)
        if rotated_ori.as_matrix()[2, 2] < 2 ** (-0.5):
            continue
        if dist_norm > 0.12:
            break
        obj_axis = rotated_ori.as_matrix()[:2, 1] / norm(rotated_ori.as_matrix()[:2, 1])
        if abs(np.dot(obj_axis, dist_axis)) > 2 ** (-0.5):
            continue
        break

    ori = (rotated_ori * R.from_quat([1, 0, 0, 0]) * tcp_q ** (-1)).as_quat()
    return ori


def get_delta_default_ori(env, obj_num):
    tcp_q = R.from_quat(env.robot.get_ee_orientation())
    identity_ori = R.from_quat([1, 0, 0, 0])
    ori_indices = np.argsort(
        [
            1
            - np.dot((identity_ori * R.from_quat(rotation)).as_quat(), tcp_q.as_quat())
            ** 2
            for rotation in all_rotations
        ]
    )
    for i, rotation in enumerate(all_rotations[ori_indices]):
        rotated_ori = identity_ori * R.from_quat(rotation)
        if rotated_ori.as_matrix()[2, 2] > -(2 ** (-0.5)):
            continue
        break
    ori = (rotated_ori * tcp_q ** (-1)).as_quat()
    return ori


def get_delta_cur_ori(env, obj_num):
    tcp_q = R.from_quat(env.robot.get_ee_orientation())
    # rotvec = R.from_quat([0,0,0,1]).as_rotvec()
    # ori = R.from_rotvec(rotvec).as_quat()
    # return ori
    return R.from_quat([0, 0, 0, 1]).as_quat()


def get_obj_pos(env, obj_num):
    return env.sim.get_base_position(f"object{obj_num}")


def get_target_goal_pos(env, obj_num):
    return env.task.goal[3 * (obj_num - 1) : 3 * obj_num]


def get_center_grip_pos(env):
    robot_pos = env.robot.get_ee_position()
    grip_axis = (
        R.from_quat(env.robot.get_ee_orientation()).as_matrix().flatten()[[1, 4, 7]]
    )
    robot_pos += (
        (
            env.robot.sim.get_joint_angle(
                env.robot.body_name, env.robot.fingers_indices[1]
            )
            - env.robot.sim.get_joint_angle(
                env.robot.body_name, env.robot.fingers_indices[0]
            )
        )
        / 2
        * grip_axis
    )
    return robot_pos
