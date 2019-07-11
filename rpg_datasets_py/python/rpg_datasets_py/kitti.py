import numpy as np
import os

from rpg_common_py import pose
import rpg_datasets_py.base as base
from rpg_datasets_py.utils.symlink import symlink
import rpg_datasets_py.utils.path as utils_path


class KittiSeq(base.RectifiedStereoSequence):
    def __init__(self, sequence_id):
        seq_folder = os.path.join(symlink('kitti'), sequence_id)
        images, right_images = utils_path.imagesFromSubdirs(
            seq_folder, sub_names=['image_0', 'image_1'], extension='.png')
        assert len(images) == len(right_images)
        assert len(images) > 0

        poses = np.loadtxt(os.path.join(
            symlink('kitti'), 'poses', sequence_id + '.txt'))
        assert poses.shape[0] == len(images)

        T_W_C = [pose.fromMatrix(np.reshape(
            poses[i, :], (3, 4))) for i in
            range(poses.shape[0])]

        calibs = np.loadtxt(os.path.join(seq_folder, 'calib.txt'),
                            usecols=range(1, 13))
        assert len(calibs) == 4

        left_K = np.reshape(calibs[0], [3, 4])[:3, :3]
        right_K = np.reshape(calibs[1], [3, 4])[:3, :3]

        baseline = 0.54

        base.RectifiedStereoSequence.__init__(
            self, images, right_images, left_K, right_K, T_W_C, baseline,
            'kt', sequence_id)


def split(tvt):
    """ Split originally defined in SIPS paper. Each split has consistent
    resolution. """
    if tvt == 'training':
        return ['06', '08', '09', '10']
    elif tvt == 'validation':
        return ['05']
    else:
        assert tvt == 'testing'
        return ['00']
