import cv2
import hickle as hkl
import IPython
import numpy as np
import os
import scipy.io

from rpg_common_py import pose

import rpg_datasets_py.base as base
from rpg_datasets_py.utils.symlink import symlink


class Calibration(object):
    def __init__(self):
        mat_path = os.path.join(symlink('dtu'), 'calibrationFile.mat')
        self.mat = scipy.io.loadmat(mat_path)

    def getK(self, half):
        c = self.mat['cc']
        f = self.mat['fc']
        if half:
            c = c / 2.
            f = f / 2.
        K = np.eye(3)
        K[:2, :2] = np.diag(f.flatten())
        K[:2, 2] = c.flatten()
        return K

    def getT_W_C(self):
        Rs = [self.mat['Rc_%d' % i] for i in range(1, 120)]
        ts = [self.mat['Tc_%d' % i] for i in range(1, 120)]
        T_C_W = [pose.Pose(R, t) for R, t in zip(Rs, ts)]
        return [i.inverse() for i in T_C_W]


def pointCloud(seq_i):
    mat_path = os.path.join(symlink('dtu'), 'reconstructions',
                            'Clean_Reconstruction_%02d.mat' % seq_i)
    mat = scipy.io.loadmat(mat_path)
    return np.hstack((mat['pts3D_near'][:3, :], mat['pts3D_far'][:3, :]))


def project(P_C, K):
    assert P_C.shape[0] == 3
    p_C = np.dot(K, P_C)
    return p_C[:2] / p_C[2, :]


def inside(rc, im):
    return 0 <= rc[0] < im.shape[0] and 0 <= rc[1] < im.shape[1]


class SameLightSequence(base.RectifiedMonoSequence):
    def __init__(self, seq_i, light_i, half=True):
        half_full = 'half' if half else 'full'
        seq_path = os.path.join(symlink('dtu'), half_full, 'SET%03d' % seq_i)
        im_names = ['Img%03d_%02d.bmp' % (i, light_i) for i in range(1, 120)]
        im_paths = [os.path.join(seq_path, i) for i in im_names]

        calib = Calibration()

        base.RectifiedMonoSequence.__init__(
            self, im_paths, calib.getK(half), calib.getT_W_C(),
            'dtu_%s' % half_full, '%03d_%02d' % (seq_i, light_i))
        self.T_C_W = [i.inverse() for i in self.T_W_C]

        self.points = pointCloud(seq_i)
        point_ids_file = os.path.join(seq_path, 'point_ids.hkl')
        if not os.path.exists(point_ids_file):
            self.point_ids = [
                self.calculateObservedPoints(i) for i in range(119)]
            hkl.dump(self.point_ids, open(point_ids_file, 'w'))
        else:
            self.point_ids = hkl.load(open(point_ids_file, 'r'))

    def calculateObservedPoints(self, frame_i):
        print('Observed points %d...' % frame_i)
        P_C = self.T_C_W[frame_i] * self.points
        p_C = project(P_C, self.K)

        depths = np.zeros((600, 800))
        ids = -np.ones((600, 800), dtype=int)
        for i in range(p_C.shape[1]):
            rc = [int(p_C[1, i]), int(p_C[0, i])]
            if inside(rc, depths):
                if P_C[2, i] < depths[rc[0], rc[1]] or \
                        depths[rc[0], rc[1]] == 0.:
                    depths[rc[0], rc[1]] = P_C[2, i]
                    ids[rc[0], rc[1]] = i

        return ids

    def depthImage(self, frame_i):
        P_C = self.T_C_W[frame_i] * self.points

        depths = np.zeros((600, 800))
        for r in range(600):
            for c in range(800):
                if self.point_ids[frame_i][r, c] != -1:
                    depths[r, c] = P_C[2, self.point_ids[frame_i][r, c]]

        return depths

    def getCorrespondence(self, point_rc, src, dst):
        i = self.point_ids[src][point_rc[0], point_rc[1]]
        if i < 0:
            return None
        P_C = self.T_C_W[dst] * self.points[:, i:i+1]
        return np.flip(project(P_C, self.K).astype(int).flatten())

    def correspondenceExample(self):
        n_pts = 20
        pts = (np.random.random((n_pts, 2)) * np.array((600, 800))).astype(int)
        corrs = np.zeros((n_pts, 2), dtype=int)
        corr_valid = np.ones(n_pts, dtype=bool)
        for i in range(n_pts):
            corr = self.getCorrespondence(pts[i], 0, 118)
            if corr is None:
                corr_valid[i] = False
            else:
                corrs[i] = corr
        pts = pts[corr_valid, :]
        corrs = corrs[corr_valid, :]
        ims = [cv2.imread(self.images[i]) for i in [0, 118]]
        ims = [cv2.cvtColor(i, cv2.COLOR_BGR2RGB) for i in ims]
        assert ims[0].shape[2] == 3
        render = np.concatenate(ims, axis=1)
        for i in range(len(pts)):
            cv2.line(render, tuple(pts[i, [1, 0]]), tuple(
                corrs[i, [1, 0]] + np.array([800, 0])), (255, 0, 0), 2,
                     cv2.LINE_AA)
        return render
