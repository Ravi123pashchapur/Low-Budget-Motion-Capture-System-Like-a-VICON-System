##################################################################################
# @author: Ravi Ashok Pashchapur & Yuxi Chen                                     #
# @Date: 19/04/2022                                                              #
# @Project: Low-cost Multi-object Positioning System with Optical Sensor Fusion  #
# @Licence: MIT                                                                  #                           
##################################################################################

import cv2
import math
import numpy as np
import matplotlib.pyplot as plt
import update_position as up


# Define observer

ob_en = True
ob_id = 0


# Define cameras

pixel_x = 640
pixel_y = 480

fov_x = math.radians(36)
fov_y = math.radians(27.8)

cam_num = 3

cam_id = [2, 3, 1]

cam_pos = np.zeros((cam_num, 3))
cam_pos[0] = [0, 0, 2.04]
cam_pos[1] = [3, 0, 2.04]
cam_pos[2] = [6, 0, 2.04]

origin_pos = np.zeros((cam_num, 3))
origin_pos[0] = [6, 14.95, 0]
origin_pos[1] = [6, 14.95, 0]
origin_pos[2] = [6, 14.95, 0]

cam_phi = np.zeros((cam_num, 1))
cam_theta = np.zeros((cam_num, 1))

for cam_i in range(cam_num):
    cam_angle = up.check_calibration(cam_pos[cam_i], origin_pos[cam_i])
    cam_phi[cam_i, 0] = cam_angle[0]
    cam_theta[cam_i, 0] = cam_angle[1]


# Define colors

color_num = 2

color = np.zeros((color_num, 2, 3))  # (colors, min and max range, HSV)
color[0] = [[170, 100, 20], [180, 255, 255]]  # red
color[1] = [[50, 120, 20], [80, 255, 255]]  # green

color_plot = ['#ff0000', '#00ff00']


# Initialise cameras

ob_cap = cv2.VideoCapture(ob_id)

cap = []

for cam_i in range(cam_num):
    cap.append(cv2.VideoCapture(cam_id[cam_i]))


# Initialise figure

x_data = []
y_data = []
z_data = []

for color_i in range(color_num):
    x_data.append([])
    y_data.append([])
    z_data.append([])

fig3d = plt.figure()
ax3d = fig3d.add_subplot(projection='3d')

fig = plt.figure()
x_plot = fig.add_subplot(311)
y_plot = fig.add_subplot(312)
z_plot = fig.add_subplot(313)


# Update continuously

while True:
    frame = []
    frame_hsv = []
    data = []

    for cam_i in range(cam_num):
        frame.append(cap[cam_i].read()[1])
        frame_hsv.append(cv2.cvtColor(frame[cam_i], cv2.COLOR_BGR2HSV))

        for color_i in range(color_num):
            mask = cv2.inRange(frame_hsv[cam_i], color[color_i, 0], color[color_i, 1])
            cont = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]

            if len(cont) != 0:
                cont_area = np.zeros(len(cont))

                for cont_i in range(len(cont)):
                    cont_area[cont_i] = cv2.contourArea(cont[cont_i])

                cont_max_i = np.argmax(cont_area)

                if cont_area[cont_max_i] > 50:
                    x, y, w, h = cv2.boundingRect(cont[cont_max_i])
                    cv2.rectangle(frame[cam_i], (x, y), (x + w, y + h), [0, 0, 0], 3)
                    cv2.circle(frame[cam_i], (int(x + 0.5 * w), int(y + 0.5 * h)), 5, [0, 0, 255], -1)
                    offset = ((x + 0.5 * w) / pixel_x - 0.5, 0.5 - (y + 0.5 * h) / pixel_y)
                    data.append((cam_i, color_i, offset[0], offset[1]))

    cam_los_phi_offset = np.zeros((cam_num, color_num))
    cam_los_phi_offset[:] = float('nan')
    cam_los_theta_offset = np.zeros((cam_num, color_num))
    cam_los_theta_offset[:] = float('nan')

    if len(data) != 0:
        for i_data in data:
            cam_los_phi_offset[i_data[0], i_data[1]] = i_data[2] * fov_x
            cam_los_theta_offset[i_data[0], i_data[1]] = i_data[3] * fov_y

    cam_los_phi = cam_phi + cam_los_phi_offset
    cam_los_theta = cam_theta + cam_los_theta_offset

    color_pos = up.update_position(cam_pos, cam_los_phi, cam_los_theta)

    print(color_pos)

    ax3d.clear()
    ax3d.set_xlim((-2, 8))
    ax3d.set_ylim((0, 17))
    ax3d.set_zlim((0, 4))
    ax3d.set_title('3D Space')
    ax3d.set_xlabel('X Axis (m)')
    ax3d.set_ylabel('Y Axis (m)')
    ax3d.set_zlabel('Z Axis (m)')

    x_plot.clear()
    y_plot.clear()
    z_plot.clear()
    x_plot.set_title('XYZ Data Plot')
    z_plot.set_xlabel('Time Step')
    x_plot.set_ylabel('X Axis (m)')
    y_plot.set_ylabel('Y Axis (m)')
    z_plot.set_ylabel('Z Axis (m)')

    for color_i in range(color_num):

        if len(x_data[color_i]) > 50:
            x_data[color_i].pop(0)
            y_data[color_i].pop(0)
            z_data[color_i].pop(0)

        if not np.isnan(color_pos[color_i, 0]):
            x_data[color_i].append(color_pos[color_i, 0])
            y_data[color_i].append(color_pos[color_i, 1])
            z_data[color_i].append(color_pos[color_i, 2])

            ax3d.scatter(color_pos[color_i, 0], color_pos[color_i, 1], color_pos[color_i, 2], c=color_plot[color_i])
            ax3d.plot(x_data[color_i], y_data[color_i], z_data[color_i], c=color_plot[color_i])

            x_plot.plot(x_data[color_i], color_plot[color_i])
            y_plot.plot(y_data[color_i], color_plot[color_i])
            z_plot.plot(z_data[color_i], color_plot[color_i])

    if ob_en:
        ob_frame = ob_cap.read()[1]
        cam_stack = np.vstack((np.hstack((ob_frame, frame[0])), np.hstack((frame[1], frame[2]))))
        cv2.imshow('Cameras', cam_stack)
    else:
        for cam_i in range(cam_num):
            cv2.imshow(('Camera %d' % cam_i), frame[cam_i])

    plt.show(block=False)
    plt.pause(0.1)
