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

import os

import rpg_datasets_py.base as base
import rpg_datasets_py.utils.path as utils_path
from rpg_datasets_py.utils.symlink import symlink


class Sequence(base.ImageSequence):
    def __init__(self, seq_id):
        seq_path = os.path.join(symlink('tum_mono'), 'sequence_%s' % seq_id)
        images = utils_path.imagesFromDir(os.path.join(seq_path, 'rect'))
        base.ImageSequence.__init__(self, images, 'tm', seq_id)
