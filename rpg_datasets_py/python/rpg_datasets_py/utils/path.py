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


def imagesFromDir(root, extension='.jpg'):
    dir_contents = sorted(os.listdir(root))
    return [os.path.join(root, i) for i in dir_contents
            if i.endswith(extension)]


def imagesFromSubdirs(
        root, sub_names=['left', 'right'], extension='.jpg'):
    dirs = [os.path.join(root, i) for i in sub_names]
    return [imagesFromDir(i, extension=extension) for i in dirs]