from glob import glob
import os
import os.path as osp
import glog as log
import json


def main():
    imagenet_dirname = 'data/imagenet'
    output_dirname = 'data/imagenet/valv2'
    imagenetv2_raw_dirname = 'data/imagenetv2-raw'
    imagenetv2_keys = ['matched-frequency', 'threshold0.7', 'topimages']

    # copy imagenet val into output directory
    log.info('Copy imagenet val images into output directory')
    fnames = glob(osp.join(imagenet_dirname, 'val', '*', '*.JPEG'))
    assert len(fnames) == 50000
    for fname in fnames:
        split = fname.split('/')
        output_fname = osp.join(output_dirname, split[-2], split[-1])
        if not osp.exists(output_fname):
            os.makedirs(osp.dirname(output_fname), exist_ok=True)
            shell_cmd = 'ln -s {} {}'.format(osp.realpath(fname), output_fname)
            os.system(shell_cmd)

    # copy imagenetv2 into output directory
    log.info('Copy imagenetv2 images into output directory')
    with open(osp.join(imagenet_dirname, 'imagenet_class_index.json'), 'r') as f:
        class_mapping = json.load(f)

    for class_index in class_mapping.keys():
        wnid = class_mapping[class_index][0]
        for key in imagenetv2_keys:
            fnames = glob(osp.join(imagenetv2_raw_dirname, 'imagenetv2-{}'.format(key), class_index, '*.jpeg'))
            assert len(fnames) == 10
            for fname in fnames:
                split = fname.split('/')
                fid = split[-1].split('.')[0]
                output_fname = osp.join(output_dirname, wnid,
                        'imagenetv2-{}-{}.JPEG'.format(key, fid))
                if not osp.exists(output_fname):
                    os.makedirs(osp.dirname(output_fname), exist_ok=True)
                    shell_cmd = 'ln -s {} {}'.format(osp.realpath(fname), output_fname)
                    os.system(shell_cmd)

    log.info('Done')


if __name__ == '__main__':
    main()

