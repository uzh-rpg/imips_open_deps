import os


def imagesFromDir(root, extension='.jpg'):
    dir_contents = sorted(os.listdir(root))
    return [os.path.join(root, i) for i in dir_contents
            if i.endswith(extension)]


def imagesFromSubdirs(
        root, sub_names=['left', 'right'], extension='.jpg'):
    dirs = [os.path.join(root, i) for i in sub_names]
    return [imagesFromDir(i, extension=extension) for i in dirs]