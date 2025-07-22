import os.path as osp

import mmcv
import numpy as np
from mmcv.parallel import DataContainer as DC
from torch.utils.data import Dataset
import torch
import torch.nn.functional as F
import torchvision.utils
import random

from .registry import DATASETS
from .transforms import ImageTransform, Numpy2Tensor
from .utils import to_tensor, random_scale
from .extra_aug import ExtraAugmentation
import cv2


@DATASETS.register_module
class CustomDataset(Dataset):
    """Custom dataset for detection.

    Annotation format:
    [
        {
            'filename': 'a.jpg',
            'width': 1280,
            'height': 720,
            'ann': {
                'labels': <np.ndarray> (n, )
            }
        },
        ...
    ]

    The `ann` field is optional for testing.
    """

    CLASSES = None

    def __init__(self,
                 ann_file,
                 img_prefix,
                 img_scale,
                 img_norm_cfg,
                 LT_ann_file=None,
                 multiscale_mode='value',
                 size_divisor=None,
                 flip_ratio=0,
                 extra_aug=None,
                 resize_keep_ratio=True,
                 test_mode=False,
                 class_split=None,
                 see_only=set(),
                 save_info=False,
                 use_splicemix=True,
                 splicemix_prob=0.3,
                 splicemix_mode='Default',
                 splicemix_grids=('1x1', '2x2'),
                 splicemix_n_grids=(1, 1)):

        # prefix of images path
        self.img_prefix = img_prefix
        # self.single_label = True if 'Imagenet' in ann_file else False
        self.single_label = False

        # load annotations (and proposals)
        if LT_ann_file is not None:
            self.img_infos = self.load_annotations(ann_file, LT_ann_file)
        else:
            self.img_infos = self.load_annotations(ann_file)
        self.ann_file = ann_file
        self.see_only = see_only
        # filter images with no annotation during training
        if not test_mode and 'width' in self.img_infos[0].keys():
            min_size = 32
            valid_inds = []
            for i, img_info in enumerate(self.img_infos):
                if min(img_info['width'], img_info['height']) >= min_size:
                    valid_inds.append(i)
            # valid_inds = self._filter_imgs()

            self.img_infos = [self.img_infos[i] for i in valid_inds]
        # (long_edge, short_edge) or [(long1, short1), (long2, short2), ...]
        self.img_scales = img_scale if isinstance(img_scale,
                                                  list) else [img_scale]
        assert mmcv.is_list_of(self.img_scales, tuple)
        # normalization configs
        self.img_norm_cfg = img_norm_cfg

        # multi-scale mode (only applicable for multi-scale training)
        self.multiscale_mode = multiscale_mode
        assert multiscale_mode in ['value', 'range']

        # flip ratio
        self.flip_ratio = flip_ratio
        assert flip_ratio >= 0 and flip_ratio <= 1
        # padding border to ensure the image size can be divided by
        # size_divisor (used for FPN)
        self.size_divisor = size_divisor

        # in test mode or not
        self.test_mode = test_mode

        # set group flag for the sampler
        if not self.test_mode:
            self._set_group_flag()

        # transforms
        self.img_transform = ImageTransform(
            size_divisor=self.size_divisor, **self.img_norm_cfg)
        self.numpy2tensor = Numpy2Tensor()

        # if use extra augmentation
        if extra_aug is not None:
            self.extra_aug = ExtraAugmentation(**extra_aug)
        else:
            self.extra_aug = None

        # image rescale if keep ratio
        self.resize_keep_ratio = resize_keep_ratio

        if class_split is not None:
            self.class_split = mmcv.load(class_split)

        self.save_info = save_info

        # SpliceMix配置
        self.use_splicemix = use_splicemix
        if self.use_splicemix:
            self.splicemix_prob = splicemix_prob
            self.splicemix_mode = splicemix_mode
            self.splicemix_grids = splicemix_grids
            self.splicemix_n_grids = splicemix_n_grids
            self.splicemix_config = {
                'Default': False,
                'Mini': False,
                'config_default': {'1x2': .7, '2x2': .3, '2x3': .0, 'drop_rate': .3}
            }
            if '--' in splicemix_mode:
                str_list = splicemix_mode.split('--')
                for s in str_list[1:]:
                    exec(f"self.splicemix_config['{s.split('=')[0]}'] = {s.split('=')[1]}")

    def __len__(self):
        return len(self.img_infos)

    def load_annotations(self, ann_file, LT_ann_file=None):
        return mmcv.load(ann_file)

    def get_index_dic(self, list=False, get_labels=False):
        """ build a dict with class as key and img_ids as values
        :return: dict()
        """
        if self.single_label:
            return

        num_classes = len(self.get_ann_info(0)['labels'])
        gt_labels = []
        idx2img_id = []
        img_id2idx = dict()
        co_labels = [[] for _ in range(num_classes)]
        condition_prob = np.zeros([num_classes, num_classes])

        if list:
            index_dic = [[] for i in range(num_classes)]
        else:
            index_dic = dict()
            for i in range(num_classes):
                index_dic[i] = []

        for i, img_info in enumerate(self.img_infos):
            img_id = img_info['id']
            label = self.get_ann_info(i)['labels']
            gt_labels.append(label)
            idx2img_id.append(img_id)
            img_id2idx[img_id] = i
            for idx in np.where(np.asarray(label) == 1)[0]:
                index_dic[idx].append(i)
                co_labels[idx].append(label)

        for cla in range(num_classes):
            cls_labels = co_labels[cla]
            num = len(cls_labels)
            condition_prob[cla] = np.sum(np.asarray(cls_labels), axis=0) / num

        ''' save original dataset statistics, run once!'''
        if self.save_info:
            self._save_info(gt_labels, img_id2idx, idx2img_id, condition_prob)

        if get_labels:
            return index_dic, co_labels
        else:
            return index_dic

    def get_class_instance_num(self):
        gt_labels = []
        for i, img_info in enumerate(self.img_infos):
            ann = self.get_ann_info(i)['labels']
            gt_labels.append(ann)
        class_instance_num = np.sum(gt_labels, 0)
        return class_instance_num

    def get_ann_info(self, idx):
        return self.img_infos[idx]['ann']

    def _filter_imgs(self, min_size=32):
        """Filter images too small."""
        valid_inds = []
        for i, img_info in enumerate(self.img_infos):
            if min(img_info['width'], img_info['height']) >= min_size:
                valid_inds.append(i)
        return valid_inds

    def _set_group_flag(self):
        """Set flag according to image aspect ratio.

        Images with aspect ratio greater than 1 will be set as group 1,
        otherwise group 0.
        """
        self.flag = np.zeros(len(self), dtype=np.uint8)
        for i in range(len(self)):
            if 'width' not in self.img_infos[0].keys():
                self.flag[i] = i % 2
                continue
            img_info = self.img_infos[i]
            if img_info['width'] / img_info['height'] > 1:
                self.flag[i] = 1
            elif img_info['width'] / img_info['height'] == 1:
                self.flag[i] = i % 2

    def _rand_another(self, idx):
        pool = np.where(self.flag == self.flag[idx])[0]
        return np.random.choice(pool)

    def __getitem__(self, idx):
        if self.test_mode:
            return self.prepare_test_img(idx)
        while True:
            data = self.prepare_train_img(idx)
            if data is None:
                idx = self._rand_another(idx)
                continue
            return data

    def _apply_splicemix(self, img, gt_labels):
        """应用SpliceMix数据增强"""
        if not self.use_splicemix or np.random.rand() > self.splicemix_prob:
            return img, gt_labels

        # 添加batch维度
        img = img.unsqueeze(0)  # [1, C, H, W]
        gt_labels = gt_labels.unsqueeze(0)  # [1, num_classes]

        if self.splicemix_config['Mini']:
            return self._splicemix_minimalism(img, gt_labels)

        # 默认模式配置
        if self.splicemix_config['Default']:
            coin = random.random()
            coin_dp = random.random()
            # 修改：确保至少有一个网格
            self.splicemix_n_grids = [max(1, img.shape[0] // 4)]

            if coin > self.splicemix_config['config_default']['1x2']:
                n_drop = 1 if coin_dp < self.splicemix_config['config_default']['drop_rate'] else 0
                self.splicemix_grids = [f'1x2-{n_drop}', ]
            elif coin > self.splicemix_config['config_default']['2x2']:
                n_drop = random.sample(range(1, 4), 1)[0] if coin_dp < self.splicemix_config['config_default'][
                    'drop_rate'] else 0
                self.splicemix_grids = [f'2x2-{n_drop}', ]
            elif coin > self.splicemix_config['config_default']['2x3']:
                n_drop = random.sample(range(1, 6), 1)[0] if coin_dp < self.splicemix_config['config_default'][
                    'drop_rate'] else 0
                self.splicemix_grids = [f'2x3-{n_drop}', ]

        # 执行网格混合
        bs = img.shape[0]
        mix_ind = torch.zeros((bs), device=img.device)
        # 修改：确保有足够的随机索引
        rand_ind = np.asarray([random.sample(range(bs), bs) for _ in range(max(10, bs))]).reshape(-1)

        for g, ng in zip(self.splicemix_grids, self.splicemix_n_grids):
            g_row, g_col = [int(t) if '-' not in t else t.split('-') for t in g.split('x')]
            (g_col, n_drop) = [int(t) for t in g_col] if type(g_col) is list else (g_col, 0)
            g = g_row * g_col

            # 修改：确保ng至少为1且不超过可用样本数
            if ng == 0:
                ng = max(1, bs // g) if len(self.splicemix_grids) == 1 else 1
            ng = min(ng, bs // g)  # 确保不超过可用样本数

            if ng * g > len(rand_ind):
                # 如果随机索引不够，重新生成
                rand_ind = np.asarray(
                    [random.sample(range(bs), bs) for _ in range(max(10, (ng * g + bs - 1) // bs))]).reshape(-1)

            rand_ind_g = rand_ind[:ng * g]
            rand_ind = rand_ind[ng * g:]

            # 执行混合
            inputs_mix_g, targets_mix_g = self._mix_fn(
                img[rand_ind_g], gt_labels[rand_ind_g],
                g_row=g_row, g_col=g_col, n_grid=ng, n_drop=n_drop)

            img = torch.cat([img, inputs_mix_g], dim=0)
            gt_labels = torch.cat([gt_labels, targets_mix_g], dim=0)

        # 移除batch维度
        return img[0], gt_labels[0]

    def _mix_fn(self, inputs, targets, g_row, g_col, n_grid, n_drop=0):
        """执行网格混合的核心函数"""
        bs, c, h, w = inputs.shape
        g = g_row * g_col

        # 检查输入有效性
        if bs == 0 or g_col == 0:
            return inputs, targets

        if n_drop > 0:
            drop_rand_ind = np.asarray(
                [random.sample(range(i * g, (i + 1) * g), min(n_drop, g)) for i in range(n_grid)]).reshape(-1)
            drop_mask = torch.ones((bs, 1, 1, 1), device=inputs.device)
            drop_mask[drop_rand_ind] = 0
            inputs = inputs * drop_mask

        # 降采样
        inputs = F.interpolate(inputs, (h // g_row, w // g_col), mode='bilinear', align_corners=True)

        # 网格排列
        try:
            inputs_mix = torchvision.utils.make_grid(inputs, nrow=g_col, padding=0)
            inputs_mix = inputs_mix.split(h // g_row * g_row, dim=1)
            inputs_mix = torch.stack(inputs_mix, dim=0)
        except Exception as e:
            print(f"Error in make_grid: inputs.shape={inputs.shape}, g_col={g_col}, g_row={g_row}, n_grid={n_grid}")
            return inputs, targets

        # 恢复原始大小
        if (inputs_mix.shape[-2], inputs_mix.shape[-1]) != (h, w):
            inputs_mix = F.interpolate(inputs_mix, (h, w), mode='bilinear', align_corners=True)

        # 标签混合
        if n_drop > 0:
            drop_mask = drop_mask.squeeze(-1).squeeze(-1).squeeze(-1)
            targets = targets * drop_mask
        targets_mix = targets.view(n_grid, g, -1).sum(1)
        targets_mix[targets_mix > 0] = 1

        return inputs_mix, targets_mix

    def _splicemix_minimalism(self, X, Y):
        """极简模式的SpliceMix"""
        g_row, g_col = 2, 2
        B, C, H, W = X.shape
        ng = B // (g_row * g_col) * (g_row * g_col)
        Omega = random.sample(range(B), B // ng)

        X_ds = F.interpolate(X[Omega], (H // g_row, W // g_col), mode='bilinear', align_corners=True)
        X_ = torchvision.utils.make_grid(X_ds, nrow=g_col, padding=0)
        X_ = X_.split(H, dim=1)
        X_ = torch.stack(X_, dim=0)

        Y_ = Y[Omega].view(ng, g_row * g_col, -1).sum(1)
        Y_[Y_ > 0] = 1

        X_hat = torch.cat((X, X_), dim=0)
        Y_hat = torch.cat((Y, Y_), dim=0)

        return X_hat[0], Y_hat[0]

    def prepare_train_img(self, idx):
        img_info = self.img_infos[idx]
        # load image
        img = mmcv.imread(osp.join(self.img_prefix, img_info['filename']))
        ann = self.get_ann_info(idx)
        gt_labels = ann['labels']

        # extra augmentation
        if self.extra_aug is not None:
            img, gt_labels = self.extra_aug(img, gt_labels)

        # apply transforms
        flip = True if np.random.rand() < self.flip_ratio else False
        img_scale = random_scale(self.img_scales, self.multiscale_mode)
        img, img_shape, pad_shape, scale_factor = self.img_transform(
            img, img_scale, flip, keep_ratio=self.resize_keep_ratio)
        img = img.copy()

        # 应用SpliceMix增强
        if self.use_splicemix:
            img = torch.from_numpy(img)
            gt_labels = torch.from_numpy(gt_labels)
            img, gt_labels = self._apply_splicemix(img, gt_labels)
            img = img.numpy()
            gt_labels = gt_labels.numpy()

        ori_shape = (img_info['height'], img_info['width'], 3)
        img_meta = dict(
            ori_shape=ori_shape,
            img_shape=img_shape,
            pad_shape=pad_shape,
            scale_factor=scale_factor,
            flip=flip)
        data = dict(
            img=DC(to_tensor(img), stack=True),
            img_meta=DC(img_meta, cpu_only=True),
            gt_labels=to_tensor(gt_labels))
        return data

    def prepare_test_img(self, idx):
        """Prepare an image for testing (multi-scale and flipping)"""
        img_info = self.img_infos[idx]
        img = mmcv.imread(osp.join(self.img_prefix, img_info['filename']))

        def prepare_single(img, scale, flip):
            _img, img_shape, pad_shape, scale_factor = self.img_transform(
                img, scale, flip, keep_ratio=self.resize_keep_ratio)
            _img = to_tensor(_img)
            _img_meta = dict(
                ori_shape=(img_info['height'], img_info['width'], 3),
                img_shape=img_shape,
                pad_shape=pad_shape,
                scale_factor=scale_factor,
                flip=flip)
            return _img, _img_meta

        imgs = []
        img_metas = []
        proposals = []
        for scale in self.img_scales:
            _img, _img_meta = prepare_single(img, scale, False)
            imgs.append(_img)
            img_metas.append(DC(_img_meta, cpu_only=True))
            if self.flip_ratio > 0:
                _img, _img_meta = prepare_single(img, scale, True)
                imgs.append(_img)
                img_metas.append(DC(_img_meta, cpu_only=True))

        data = dict(img=imgs, img_meta=img_metas)
        return data

    def prepare_raw_img(self, idx):
        img_info = self.img_infos[idx]
        img = mmcv.imread(osp.join(self.img_prefix, img_info['filename']))

        _img, img_shape, pad_shape, scale_factor = self.img_transform(
            img, self.img_scales[0], flip=False, keep_ratio=self.resize_keep_ratio)
        img_meta = dict(
            ori_shape=(img_info['height'], img_info['width'], 3),
            img_shape=img_shape,
            pad_shape=pad_shape,
            scale_factor=scale_factor,
            flip=False)

        data = dict(img=img, img_meta=img_meta)
        return data

    def _save_info(self, gt_labels, img_id2idx, idx2img_id, condition_prob):
        '''save info for later training'''
        ''' save original gt_labels '''
        save_data = dict(gt_labels=gt_labels, img_id2idx=img_id2idx, idx2img_id=idx2img_id)
        if 'coco' in self.ann_file:
            # path = 'mllt/appendix/coco/terse_gt_2017_test.pkl'
            path = 'mllt/appendix/coco/terse_gt_2017.pkl'
        elif 'VOC' in self.ann_file:
            path = 'mllt/appendix/VOCdevkit/terse_gt_2012.pkl'
            # path = 'mllt/appendix/VOCdevkit/terse_gt_2007_test.pkl'
        else:
            raise NameError

        if not osp.exists(path):
            mmcv.dump(save_data, path)
            print('key info saved at {}!'.format(path))
        else:
            print('already exist, wont\'t overwrite!')

        ''' save long tail information '''
        class_freq = np.sum(gt_labels, axis=0)
        # print(np.mean(class_freq), np.var(class_freq/len(gt_labels)))
        neg_class_freq = np.shape(gt_labels)[0] - class_freq
        save_data = dict(gt_labels=gt_labels, class_freq=class_freq, neg_class_freq=neg_class_freq
                         , condition_prob=condition_prob)
        if 'coco' in self.ann_file:
            # long-tail coco
            path = 'mllt/appendix/coco/longtail2017/class_freq.pkl'
            # full coco
            # path = 'mllt/appendix/coco/class_freq.pkl'
        elif 'VOC' in self.ann_file:
            # long-tail VOC
            path = 'mllt/appendix/VOCdevkit/longtail2012/class_freq.pkl'
            # full VOC
            # path = 'mllt/appendix/VOCdevkit/class_freq.pkl'
        else:
            raise NameError

        if not osp.exists(path):
            mmcv.dump(save_data, path)
            print('key info saved at {}!'.format(path))
        else:
            print('already exist, wont\'t overwrite!')
        exit()