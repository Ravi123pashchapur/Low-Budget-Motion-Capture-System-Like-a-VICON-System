##################################################################################
# @author: Ravi Ashok Pashchapur & Yuxi Chen                                     #
# @Date: 19/04/2022                                                              #
# @Project: Low-cost Multi-object Positioning System with Optical Sensor Fusion  #
# @Licence: MIT                                                                  #                           
##################################################################################

import math
import numpy as np


# Calibrate camera heading and elevation
def check_calibration(cam_pos, origin_pos):

    pos_diff = origin_pos - cam_pos

    c_phi = math.atan2(pos_diff[0], pos_diff[1])
    c_theta = math.atan2(pos_diff[2], np.linalg.norm(pos_diff[:2]))
    c_angle = np.array([c_phi, c_theta])

    return c_angle


# Generate camera LOS
def get_los(cam_pos, cam_los_phi, cam_los_theta):

    d = 100  # distance between 2 points

    cam_los = np.zeros((2, 3))
    cam_los[0] = cam_pos

    cam_los[1, 0] = cam_pos[0] + d * math.sin(cam_los_phi) * math.cos(cam_los_theta)
    cam_los[1, 1] = cam_pos[1] + d * math.cos(cam_los_phi) * math.cos(cam_los_theta)
    cam_los[1, 2] = cam_pos[2] + d * math.sin(cam_los_theta)

    return cam_los


# Calculate intersection point and weight for the two given LOS
def get_intersection(cam_los_a, cam_los_b):

    int_data = np.zeros(4)

    if np.isnan(cam_los_a).any() or np.isnan(cam_los_b).any():
        return int_data

    p_1 = cam_los_a[0]
    p_2 = cam_los_a[1]
    p_3 = cam_los_b[0]
    p_4 = cam_los_b[1]

    p_13 = p_1 - p_3
    p_43 = p_4 - p_3
    p_21 = p_2 - p_1

    d_1343 = p_13[0] * p_43[0] + p_13[1] * p_43[1] + p_13[2] * p_43[2]
    d_4321 = p_43[0] * p_21[0] + p_43[1] * p_21[1] + p_43[2] * p_21[2]
    d_1321 = p_13[0] * p_21[0] + p_13[1] * p_21[1] + p_13[2] * p_21[2]
    d_4343 = p_43[0] * p_43[0] + p_43[1] * p_43[1] + p_43[2] * p_43[2]
    d_2121 = p_21[0] * p_21[0] + p_21[1] * p_21[1] + p_21[2] * p_21[2]

    top_1 = d_1343 * d_4321 - d_1321 * d_4343
    bot_1 = d_2121 * d_4343 - d_4321 * d_4321

    if bot_1 == 0:
        return int_data

    mu_a = top_1 / bot_1
    mu_b = (d_1343 + d_4321 * mu_a) / d_4343

    if (mu_a < 0) or (mu_b < 0):
        return int_data

    p_a = p_1 + mu_a * p_21
    p_b = p_3 + mu_b * p_43

    int_xyz = np.mean([p_a, p_b], axis=0)
    int_w = 1 / np.linalg.norm(p_a - p_b)
    int_data = np.append(int_xyz, int_w)

    return int_data


# Update position
def update_position(cam_pos, cam_los_phi, cam_los_theta):

    cam_num = len(cam_pos)
    color_num = len(cam_los_phi[0])

    cam_los = np.zeros((cam_num, color_num, 2, 3))

    for cam_i in range(cam_num):
        for color_i in range(color_num):
            cam_los[cam_i, color_i] = get_los(cam_pos[cam_i],
                                              cam_los_phi[cam_i, color_i],
                                              cam_los_theta[cam_i, color_i])

    int_num = int((cam_num ** 2 - cam_num) / 2)
    int_xyz = np.zeros((int_num, color_num, 3))
    int_w = np.zeros((int_num, color_num, 3))
    int_i = 0

    for cam_i in range(cam_num - 1):
        for cam_j in range(cam_i + 1, cam_num):
            for color_i in range(color_num):
                int_data = get_intersection(cam_los[cam_i, color_i], cam_los[cam_j, color_i])
                int_xyz[int_i, color_i] = int_data[:3]
                int_w[int_i, color_i, :] = int_data[3]

            int_i += 1

    int_w_total = np.sum(int_w, axis=0)
    int_w_total[int_w_total == 0] = float('nan')
    int_w = int_w / int_w_total
    color_pos = np.sum(int_xyz * int_w, axis=0)

    return color_pos
