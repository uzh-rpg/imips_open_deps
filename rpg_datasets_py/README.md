# rpg_datasets_py for IMIPS

## Linking up datasets

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

### TUM mono
