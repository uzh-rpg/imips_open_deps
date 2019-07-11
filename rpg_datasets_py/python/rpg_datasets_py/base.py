import numpy as np


class ImageSequence(object):
    def __init__(self, images, set_name, seq_name):
        self.images = images
        self.set_name = set_name
        self.seq_name = seq_name

    def __len__(self):
        return len(self.images)

    def __getitem__(self, item):
        return self.images[item]

    def name(self):
        return '%s%s' % (self.set_name, self.seq_name)


class RectifiedMonoSequence(ImageSequence):
    def __init__(self, images, K, T_W_C, set_name, seq_name):
        ImageSequence.__init__(self, images, set_name, seq_name)
        self.K = K
        self.T_W_C = T_W_C

    def positions(self):
        return np.array([np.ravel(i.t) for i in self.T_W_C])

    def getT_A_B(self, indices):
        return self.T_W_C[indices[0]].inverse() * self.T_W_C[indices[1]]


class RectifiedStereoSequence(RectifiedMonoSequence):
    def __init__(self, left_images, right_images, left_K, right_K, T_W_C_left,
                 baseline, set_name, seq_name):
        RectifiedMonoSequence.__init__(
            self, left_images, left_K, T_W_C_left, set_name, seq_name)
        self.right_images = right_images
        self.right_K = right_K
        self.baseline = baseline


tvt = ['training', 'validation', 'testing']
