# rpg_datasets_py
A more pythonic version of rpg_datasets

Unlike [rpg_datasets](https://github.com/uzh-rpg/rpg_datasets), this allows direct pythonic instantiation of any dataset class:
```python
from rpg_datasets_py.hpatches import HPatches

dataset = HPatches('training')
```

Like in rpg_datasets, **strictly no dataset data** should reside here. Instead, the main approach should be `.gitignore`d symlinks that need to be created once (so you can store the actual data anywhere you like). The symlinks should be located in the root directory of this package (`ln -s source destination`, see table below). Then, a **working-directory-independent** function to locate them is provided in [symlink.py](python/rpg_datasets_py/utils/symlink.py).

## Contributing

* Every new dataset should result in *exactly one* python file in `python/rpg_datasets_py`. Check out the utilities in `python/rpg_datasets_py/utils` and add new ones if necessary.
* Adhere to the aforementioned symlink approach.
* Consider using the base classes in `python/rpg_datasets_py/base.py`.

## Symlinks

| Dataset | URL | Command |
| ------- | --- | ------- |
| Hpatches | http://icvl.ee.ic.ac.uk/vbalnt/hpatches/hpatches-sequences-release.tar.gz | `ln -s /home/titus/data/hpatches-sequences-release hpatches` |
| KITTI | http://www.cvlibs.net/datasets/kitti/eval_odometry.php | `ln -s /home/titus/data/kitti .` |
| Robotcar | https://robotcar-dataset.robots.ox.ac.uk/ | `ln -s /home/titus/data/robotcar .` |

For Robotcar, the SDK from https://github.com/ori-drs/robotcar-dataset-sdk additionally needs to be cloned into the robotcar folder.
