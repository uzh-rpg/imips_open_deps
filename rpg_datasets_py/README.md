# rpg_datasets_py for IMIPS

## Linking datasets

### EuRoC

Download at least `V1_01_easy` from https://projects.asl.ethz.ch/datasets/doku.php?id=kmavvisualinertialdatasets . Then, in **this** directory, do:
```bash
ln -s your/path/to/euroc .
```
Undistort the images by running the following in python:
```python
import rpg_datasets_py.euroc
rpg_datasets_py.euroc.undistort('V1_01_easy')
```

### HPatches

Download http://icvl.ee.ic.ac.uk/vbalnt/hpatches/hpatches-sequences-release.tar.gz and also create a folder min_hpatches. Then, in **this** directory, do:
```bash
ln -s your/path/to/hpatches .
ln -s your/path/to/min_hpatches .
```
Finally, parse the downscaled `min_hpatches` by running the following in python:
```python
import rpg_datasets_py.hpatches
rpg_datasets_py.hpatches.createMinHpatches()
```

### KITTI

Download at least `00` (testing) and `05` (validation) from grayscale KITTI: http://www.cvlibs.net/datasets/kitti/eval_odometry.php . Then, in **this** directory, do:
```bash
ln -s your/path/to/kitti .
```

### Robotcar (SIPs only)

Download at least `2014-07-14-14-49-50` (Bumblebee, GPS/INS) from https://robotcar-dataset.robots.ox.ac.uk/datasets/ and the [robotcar-dataset-sdk](https://github.com/ori-mrg/robotcar-dataset-sdk/releases). Both should be contained in a folder called `robotcar`. Then, in **this** directory, do:
```bash
ln -s your/path/to/robotcar .
```

### TUM mono

Download at least sequences `01, 02, 03, 48, 49, 50` from https://vision.in.tum.de/data/datasets/mono-dataset and undistort them using https://github.com/tum-vision/mono_dataset_code . Then, in **this** directory, do:
```bash
ln -s your/path/to/tum_mono .
```
