import cv2
import IPython
import numpy as np
import os
import pyquaternion as quat
import yaml

from rpg_common_py import pose
import rpg_datasets_py.base as base
from rpg_datasets_py.utils.symlink import symlink


def intrinsicsToK(fxfypxpy):
    K = np.eye(3)
    K[:2, :2] = np.diag(fxfypxpy[:2])
    K[:2, 2] = fxfypxpy[2:]
    return K


class Cam(object):
    def __init__(self, euroc_root, cam_i):
        self.root = os.path.join(euroc_root, 'cam%d' % cam_i)
        yam = yaml.load(file(os.path.join(self.root, 'sensor.yaml'), 'r'))
        self.T_B_C = pose.fromMatrix(
            np.array(yam['T_BS']['data']).reshape((4, 4)))
        self.resolution = np.array(yam['resolution'])
        intr = np.array(yam['intrinsics'])
        self.K = intrinsicsToK(intr)
        assert yam['distortion_model'] == 'radial-tangential'
        self.dist = np.array(yam['distortion_coefficients'])
        self.map1 = None
        self.map2 = None

    def setUndistortRectifyMap(self, R, P):
        self.map1, self.map2 = cv2.initUndistortRectifyMap(
            self.K, self.dist, R, P, tuple(self.resolution), cv2.CV_32FC1)


class EurocSeq(base.RectifiedStereoSequence):
    def __init__(self, sequence_id):
        root = os.path.join(symlink('euroc'), sequence_id, 'mav0')

        lr_images = [None, None]
        for i in [0, 1]:
            image_folder = os.path.join(root, 'rect%d' % i)
            if not os.path.exists(image_folder):
                raise Exception(
                    'Need to call euroc.undistort() on this sequence first!')
            K = np.loadtxt(os.path.join(image_folder, 'K.txt'))
            image_folder_contents = sorted(os.listdir(image_folder))
            if i == 0:
                image_times = [int(j[:-4]) for j in
                               image_folder_contents if j.endswith('.png')]
            lr_images[i] = [os.path.join(image_folder, j) for j in
                            image_folder_contents if j.endswith('.png')]
        assert len(lr_images[0]) == len(lr_images[1])
        left_images = lr_images[0]
        right_images = lr_images[1]

        gt_path = os.path.join(root, 'state_groundtruth_estimate0')
        gt_data_file = os.path.join(gt_path, 'data.csv')
        gt_data = np.loadtxt(gt_data_file, skiprows=1, delimiter=',')

        # Only interestd in poses at image times:
        vicon_filter = []
        for image_time in image_times:
            vicon_filter.append(np.argmin(
                np.abs(gt_data[:, 0] - image_time)))
        gt_data = gt_data[vicon_filter, :]

        T_W_GT = [pose.Pose(
            quat.Quaternion(row[4:8]).rotation_matrix, row[1:4].reshape((3, 1)))
            for row in gt_data]

        gt_extr_file = os.path.join(gt_path, 'sensor.yaml')
        yam = yaml.load(file(gt_extr_file, 'r'))
        T_B_GT_arr = np.array(yam['T_BS']['data']).reshape((4, 4))
        T_B_GT = pose.fromApproximateMatrix(T_B_GT_arr)
        T_GT_B = T_B_GT.inverse()

        # TODO: Dataset-dependent?
        cams = [Cam(root, i) for i in [0, 1]]
        T_W_C = [i * T_GT_B * cams[0].T_B_C for i in T_W_GT]

        baseline = (cams[0].T_B_C.inverse() * cams[1].T_B_C).t[0]

        print(image_folder)
        base.RectifiedStereoSequence.__init__(
            self, left_images, right_images, K, K, T_W_C, baseline, 'euroc',
            sequence_id)


def undistort(sequence_id):
    root = os.path.join(symlink('euroc'), sequence_id, 'mav0')

    cams = [Cam(root, i) for i in [0, 1]]
    assert np.all(cams[0].resolution == cams[1].resolution)

    T_C1_C0 = cams[1].T_B_C.inverse() * cams[0].T_B_C

    R1, R2, P1, P2, _, _, _ = cv2.stereoRectify(
        cams[0].K, cams[0].dist, cams[1].K, cams[1].dist,
        tuple(cams[0].resolution), T_C1_C0.R, T_C1_C0.t, alpha=0)

    cams[0].setUndistortRectifyMap(R1, P1)
    cams[1].setUndistortRectifyMap(R2, P2)

    new_K = [P1[:3, :3], P2[:3, :3]]

    assert np.all(new_K[0] == new_K[1])

    for i in [0, 1]:
        image_folder = os.path.join(root, 'cam%d' % i, 'data')
        out_folder = os.path.join(root, 'rect%d' % i)
        if not os.path.exists(out_folder):
            os.makedirs(out_folder)

        np.savetxt(os.path.join(out_folder, 'K.txt'), new_K[i])

        image_folder_contents = sorted(os.listdir(image_folder))
        in_images = [os.path.join(image_folder, j) for j in
                     image_folder_contents if j.endswith('.png')]
        out_images = [os.path.join(out_folder, j) for j in
                      image_folder_contents if j.endswith('.png')]

        for j in range(len(in_images)):
            print('%d/%d' % (j, len(in_images)))
            img = cv2.imread(in_images[j])
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            undimg = cv2.remap(
                img, cams[i].map1, cams[i].map2, cv2.INTER_LINEAR)
            cv2.imwrite(out_images[j], undimg)
