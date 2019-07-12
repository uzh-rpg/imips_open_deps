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
