import copy
import cv2
import numpy as np
import os


from rpg_datasets_py.utils.symlink import symlink


class HPatchPair(object):
    def __init__(self, im0, im1, homography, seqname, indices):
        self.im = [im0, im1]
        self.H = homography
        self.seqname = seqname
        self.indices = indices

    def correspondences(self, pixels_xy, inv=False):
        # Pixels given and returned in x,y, col-major
        if not inv:
            H = self.H
        else:
            H = np.linalg.inv(self.H)
        hom_tf = np.dot(H, np.vstack(
            (pixels_xy, np.ones((1, pixels_xy.shape[1])))))
        return hom_tf[:2, :] / hom_tf[2, :]

    def name(self):
        return '%s %d %d' % (self.seqname, self.indices[0], self.indices[1])


class HPatchSeq(object):

    def __init__(self, name, images, homographies):
        assert len(homographies) == len(images) - 1
        self.name = name
        self.images = images  # 1, 2, ... 6
        self.homographies = homographies  # 1-2, 1-3, ... 1-6

    def downSample(self):
        for i in range(len(self.images)):
            self.images[i] = cv2.pyrDown(self.images[i])

        H_half = np.array([[.5, 0, 0], [0, 0.5, 0], [0, 0, 1]])
        H_double = np.array([[2, 0, 0], [0, 2, 0], [0, 0, 1]])

        for i in range(len(self.homographies)):
            self.homographies[i] = np.dot(
                H_half, np.dot(self.homographies[i], H_double))

    def write(self, root_dir):
        seq_dir = os.path.join(root_dir, self.name)
        os.makedirs(seq_dir)

        def fname(name):
            return os.path.join(seq_dir, name)

        for i in range(len(self.images)):
            cv2.imwrite(fname('%d.pgm' % (i + 1)), self.images[i])
        for i in range(len(self.homographies)):
            np.savetxt(fname('H_1_%d' % (i + 2)), self.homographies[i])

    def pairHomography(self, indices1):
        if indices1[0] == 1:
            return self.homographies[indices1[1] - 2]
        else:
            return np.dot(self.homographies[indices1[1] - 2],
                          np.linalg.inv(self.homographies[indices1[0] - 2]))

    def getPair(self, index):
        pairs1 = [(i, j) for i in range(1, 7) for j in range(i + 1, 7)]
        pair1 = pairs1[index]
        return HPatchPair(self.images[pair1[0] - 1],
                          self.images[pair1[1] - 1],
                          self.pairHomography(pair1), self.name, pair1)


def fromFolder(folder, ext='ppm'):
    name = os.path.basename(folder)
    im_paths = [os.path.join(folder, '%d.%s' % (i, ext)) for i in range(1, 7)]
    images = [cv2.imread(i, cv2.IMREAD_GRAYSCALE) for i in im_paths]
    H_paths = [os.path.join(folder, 'H_1_%d' % i) for i in range(2, 7)]
    homographies = [np.loadtxt(i) for i in H_paths]
    return HPatchSeq(name, images, homographies)


def createMinHpatches():
    root = symlink('hpatches')
    all_folders = sorted(os.listdir(root))

    v_folders = [os.path.join(root, i) for i in all_folders
                 if i.startswith('v_')]
    sequences = [fromFolder(i) for i in v_folders]

    print('Original resolutions:')
    for seq in sequences:
        print('%s: %d x %d' % (seq.name, seq.images[0].shape[0],
                               seq.images[0].shape[1]))

    for seq in sequences:
        while np.any(np.array(seq.images[0].shape) > 1000):
            seq.downSample()

    print('')
    print('New resolutions:')
    for seq in sequences:
        print('%s: %d x %d' % (seq.name, seq.images[0].shape[0],
                               seq.images[0].shape[1]))

    dest = symlink('min_hpatches')
    os.makedirs(dest)
    for seq in sequences:
        seq.write(dest)


class HPatches(object):

    def __init__(self, tvt, use_min=False):
        if use_min:
            root = symlink('min_hpatches')
            if not os.path.exists(root):
                raise Exception('Create min_hpatches with createMinHpatches()')
            ext = 'pgm'
        else:
            root = symlink('hpatches')
            ext = 'ppm'

        folders = split(root, tvt)
        self.folder_names = [os.path.basename(path) for path in folders]

        self._seqs = []
        for folder in folders:
            self._seqs.append(fromFolder(folder, ext))

        self._pairs_per_seq = 15

        print('Hpatches for %s has length %d' % (tvt, len(self)))

    def __getitem__(self, i):
        return self._seqs[i / self._pairs_per_seq].getPair(
            i % self._pairs_per_seq)

    def __len__(self):
        return len(self._seqs) * self._pairs_per_seq

    def getRandomDataPoint(self):
        return np.random.choice(self)

    def subSampled(self, num_seqs):
        result = copy.deepcopy(self)
        result.folder_names = result.folder_names[:num_seqs]
        result._seqs = result._seqs[:num_seqs]
        return result


def split(root, tvt):
    all_folders = sorted(os.listdir(symlink(root)))

    # https://github.com/hpatches/hpatches-benchmark/blob/master/tasks/splits/splits.json
    test_folders = [
        "v_courses", "v_coffeehouse", "v_abstract", "v_feast",
        "v_woman", "v_talent", "v_tabletop", "v_bees", "v_strand",
        "v_fest", "v_yard", "v_underground", "v_azzola", "v_eastsouth",
        "v_yuri", "v_soldiers", "v_man", "v_pomegranate",
        "v_birdwoman", "v_busstop"]

    if tvt == 'testing':
        return [os.path.join(root, x) for x in test_folders]
    else:
        folders = [os.path.join(root, x) for x in all_folders
                   if x[0] == 'v' and x not in test_folders]
        if tvt == 'validation':
            return folders[:5]
        else:
            assert tvt == 'training'
            return folders[5:]
