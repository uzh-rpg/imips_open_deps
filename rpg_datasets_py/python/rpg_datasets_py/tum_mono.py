import os

import rpg_datasets_py.base as base
import rpg_datasets_py.utils.path as utils_path
from rpg_datasets_py.utils.symlink import symlink


class Sequence(base.ImageSequence):
    def __init__(self, seq_id):
        seq_path = os.path.join(symlink('tum_mono'), 'sequence_%s' % seq_id)
        images = utils_path.imagesFromDir(os.path.join(seq_path, 'rect'))
        base.ImageSequence.__init__(self, images, 'tm', seq_id)
