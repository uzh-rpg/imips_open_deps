# Copyright (C) 2019 Titus Cieslewski, RPG, University of Zurich, Switzerland
#   You can contact the author at <titus at ifi dot uzh dot ch>
# Copyright (C) 2019 Davide Scaramuzza, RPG, University of Zurich, Switzerland
#
# This file is part of imips_open_deps.
#
# imips_open_deps is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# imips_open_deps is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with imips_open_deps. If not, see <http:#www.gnu.org/licenses/>.

# Robotcar is a special one because it needs tons of preprocessing.
# For that, code credit also goes to Amadeus Oertel.

import copy
import cv2
import itertools
import matplotlib.pyplot as plt
import multiprocessing
import numpy as np
import os
import PIL
import scipy.spatial.distance as scidist
import skimage.measure
import sys

from rpg_common_py import pose
import rpg_datasets_py.base as base
from rpg_datasets_py.utils.symlink import symlink
import rpg_datasets_py.utils.path as utils_path


class CroppedGraySequence(base.RectifiedStereoSequence):
    def __init__(self, seq_id):
        seq_folder = os.path.join(symlink('robotcar_cropped_gray'), seq_id)
        if not os.path.exists(seq_folder):
            raise Exception('Need to call makeCroppedGraySequence(\'%s\') to '
                            'preprocess the raw data!' % seq_id)

        ims_dir = os.path.join(seq_folder, 'rect')
        left_ims, right_ims = utils_path.imagesFromSubdirs(
            ims_dir, extension='.png')
        assert len(left_ims) == len(right_ims)
        assert len(left_ims) > 0

        k_files = [os.path.join(seq_folder, '%s_K.txt' % i)
                   for i in ['left', 'right']]
        Ks = [np.loadtxt(k_file) for k_file in k_files]

        T_W_C_file = os.path.join(seq_folder, 'T_W_C.txt')
        T_W_C_serialized = np.loadtxt(T_W_C_file)
        T_W_C = [pose.fromMatrix(np.reshape(T_W_C_serialized[i, :], (4, 4)))
                 for i in range(T_W_C_serialized.shape[0])]

        base.RectifiedStereoSequence.__init__(
            self, left_ims, right_ims, Ks[0], Ks[1], T_W_C, 0.24,
            'rc', seq_id)


# Load extra modules.
sdk_dir = os.path.join(symlink('robotcar'), 'robotcar-dataset-sdk')
sdk_py_dir = os.path.join(sdk_dir, 'python')
cam_model_dir = os.path.join(sdk_dir, 'models')
sys.path.append(sdk_py_dir)
assert os.path.exists(sdk_py_dir)
import camera_model
import image
import interpolate_poses
import transform


class ParallelImageConverter(object):
    def __init__(self, model, in_dir, out_dir, images):
        self._model = model
        self._in_dir = in_dir
        self._out_dir = out_dir
        self._images = images

    def __call__(self, i):
        out = os.path.join(self._out_dir, self._images[i])
        if os.path.exists(out):
            return
        in_image = os.path.join(self._in_dir, self._images[i])
        img = image.load_image(in_image, self._model)
        img = img[:820, :, :]
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        img = skimage.measure.block_reduce(
            img, block_size=(2, 2), func=np.mean).astype('uint8')
        im = PIL.Image.fromarray(img)
        im.save(out)
        print(i)


def makeCroppedGraySequence(seq_id, num_threads=8):
    src_folder = os.path.join(symlink('robotcar'), seq_id)
    dst_folder = os.path.join(symlink('robotcar_cropped_gray'), seq_id)
    if not os.path.exists(dst_folder):
        os.makedirs(dst_folder)

    # Convert images.
    stereo_dir = os.path.join(src_folder, 'stereo')

    for leftright in ['left', 'right']:
        in_dir = os.path.join(stereo_dir, leftright)
        images = utils_path.imagesFromDir(in_dir, extension='.png')
        im_names = [os.path.basename(i) for i in images]
        cam_model = camera_model.CameraModel(cam_model_dir, in_dir)
        out_dir = os.path.join(dst_folder, 'rect', leftright)
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)

        p = multiprocessing.Pool(num_threads)
        print('Converting %s...' % leftright)
        p.map(ParallelImageConverter(
            cam_model, in_dir, out_dir, im_names), range(len(images)))

        # K files
        # Image downscaling affects camera intrinsics
        f = [0.5 * x for x in cam_model.focal_length]
        c = [0.5 * x for x in cam_model.principal_point]
        K = np.array([[f[0], 0, c[0]],
                      [0, f[1], c[1]],
                      [0, 0, 1]])
        np.savetxt(os.path.join(dst_folder, '%s_K.txt' % leftright), K)

    times = [int(i[:16]) for i in im_names]

    # Get poses everywhere.
    T_W_C_file = os.path.join(dst_folder, 'T_W_C.txt')
    if not os.path.exists(T_W_C_file):
        print('Interpolating poses...')
        insext = np.loadtxt(os.path.join(
            sdk_dir, 'extrinsics', 'ins.txt')).tolist()
        mtx = transform.build_se3_transform(insext)
        # Cx: Oxford style camera, where x looks into the image plane.
        T_I_Cx = pose.fromMatrix(np.asarray(mtx)).inverse()
        T_Cx_C = pose.yRotationDeg(90) * pose.zRotationDeg(90)
        T_I_C = T_I_Cx * T_Cx_C

        # T_W_I0: Because in Oxford, z looks down.
        T_W_I0 = pose.xRotationDeg(180)

        ins_path = os.path.join(src_folder, 'gps', 'ins.csv')
        T_I0_I = interpolate_poses.interpolate_ins_poses(
            ins_path, times, times[0])

        T_I0_I = [pose.fromMatrix(np.asarray(i)) for i in T_I0_I]

        T_W_C = [T_W_I0 * i * T_I_C for i in T_I0_I]
        T_W_C_serialized = np.array([i.asArray().ravel() for i in T_W_C])
        np.savetxt(T_W_C_file, T_W_C_serialized)
