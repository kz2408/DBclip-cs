from __future__ import division
import math
import torch
import numpy as np
from mmcv.runner import get_dist_info
from torch.utils.data import Sampler
from torch.utils.data import DistributedSampler as _DistributedSampler
import random


class DistributedSampler(_DistributedSampler):

    def __init__(self, dataset, num_replicas=None, rank=None, shuffle=True):
        super().__init__(dataset, num_replicas=num_replicas, rank=rank)
        self.shuffle = shuffle

    def __iter__(self):
        # deterministically shuffle based on epoch
        if self.shuffle:
            g = torch.Generator()
            g.manual_seed(self.epoch)
            indices = torch.randperm(len(self.dataset), generator=g).tolist()
        else:
            indices = torch.arange(len(self.dataset)).tolist()

        # add extra samples to make it evenly divisible
        indices += indices[:(self.total_size - len(indices))]
        assert len(indices) == self.total_size

        # subsample
        indices = indices[self.rank:self.total_size:self.num_replicas]
        assert len(indices) == self.num_samples

        return iter(indices)


class GroupSampler(Sampler):

    def __init__(self, dataset, samples_per_gpu=1):
        assert hasattr(dataset, 'flag')
        self.dataset = dataset
        self.samples_per_gpu = samples_per_gpu
        self.flag = dataset.flag.astype(np.int64)
        self.epoch = 0

        # self.flag = (torch.randn(len(dataset)) > 0).type(torch.int64)

        self.group_sizes = np.bincount(self.flag)
        if min(self.group_sizes) < self.samples_per_gpu:
            for i in range(len(self.flag)):
                self.flag[i] = i % 2
            self.group_sizes = np.bincount(self.flag)
            print('\033[1;35m >>> group sampler randomly aranged!\033[0;0m')

        self.num_samples = 0
        for i, size in enumerate(self.group_sizes):
            self.num_samples += int(np.ceil(
                size / self.samples_per_gpu)) * self.samples_per_gpu

    def __iter__(self):
        indices = []
        for i, size in enumerate(self.group_sizes):
            if size == 0:
                continue
            indice = np.where(self.flag == i)[0]
            assert len(indice) == size
            np.random.shuffle(indice)
            num_extra = int(np.ceil(size / self.samples_per_gpu)
                            ) * self.samples_per_gpu - len(indice)
            indice = np.concatenate([indice, indice[:num_extra]])
            indices.append(indice)
        indices = np.concatenate(indices)
        indices = [
            indices[i * self.samples_per_gpu:(i + 1) * self.samples_per_gpu]
            for i in np.random.permutation(
                range(len(indices) // self.samples_per_gpu))
        ]
        indices = np.concatenate(indices)
        indices = indices.astype(np.int64).tolist()
        assert len(indices) == self.num_samples
        return iter(indices)

    def __len__(self):
        return self.num_samples

    def set_epoch(self, epoch):
        self.epoch = epoch


class DistributedGroupSampler(Sampler):
    """Sampler that restricts data loading to a subset of the dataset.
    It is especially useful in conjunction with
    :class:`torch.nn.parallel.DistributedDataParallel`. In such case, each
    process can pass a DistributedSampler instance as a DataLoader sampler,
    and load a subset of the original dataset that is exclusive to it.
    .. note::
        Dataset is assumed to be of constant size.
    Arguments:
        dataset: Dataset used for sampling.
        num_replicas (optional): Number of processes participating in
            distributed training.
        rank (optional): Rank of the current process within num_replicas.
    """

    def __init__(self,
                 dataset,
                 samples_per_gpu=1,
                 num_replicas=None,
                 rank=None):
        _rank, _num_replicas = get_dist_info()
        if num_replicas is None:
            num_replicas = _num_replicas
        if rank is None:
            rank = _rank
        self.dataset = dataset
        self.samples_per_gpu = samples_per_gpu
        self.num_replicas = num_replicas
        self.rank = rank
        self.epoch = 0

        assert hasattr(self.dataset, 'flag')
        self.flag = self.dataset.flag
        self.group_sizes = np.bincount(self.flag)

        self.num_samples = 0
        for i, j in enumerate(self.group_sizes):
            self.num_samples += int(
                math.ceil(self.group_sizes[i] * 1.0 / self.samples_per_gpu /
                          self.num_replicas)) * self.samples_per_gpu
        self.total_size = self.num_samples * self.num_replicas

    def __iter__(self):
        # deterministically shuffle based on epoch
        g = torch.Generator()
        g.manual_seed(self.epoch)

        indices = []
        for i, size in enumerate(self.group_sizes):
            if size > 0:
                indice = np.where(self.flag == i)[0]
                assert len(indice) == size
                indice = indice[list(torch.randperm(int(size),
                                                    generator=g))].tolist()
                extra = int(
                    math.ceil(
                        size * 1.0 / self.samples_per_gpu / self.num_replicas)
                ) * self.samples_per_gpu * self.num_replicas - len(indice)
                indice += indice[:extra]
                indices += indice

        assert len(indices) == self.total_size

        indices = [
            indices[j] for i in list(
                torch.randperm(
                    len(indices) // self.samples_per_gpu, generator=g))
            for j in range(i * self.samples_per_gpu, (i + 1) *
                           self.samples_per_gpu)
        ]

        # subsample
        offset = self.num_samples * self.rank
        indices = indices[offset:offset + self.num_samples]
        assert len(indices) == self.num_samples

        return iter(indices)

    def __len__(self):
        return self.num_samples

    def set_epoch(self, epoch):
        self.epoch = epoch


class FastRandomIdentitySampler(Sampler):
    def __init__(self, dataset, num_classes=16, num_instances=5, select_classes=4, select_instances=1,
                 samples_per_gpu=4):

        self.epoch = 0

        self.index_dic = dataset.get_index_dic()
        self.pids = list(self.index_dic.keys())
        self.num_classes = num_classes
        self.num_instances = num_instances
        self.samples_per_gpu = samples_per_gpu

        self.select_classes = select_classes
        self.select_instances = select_instances

        assert self.num_classes == len(self.pids)
        assert select_classes * select_instances == samples_per_gpu
        assert int(self.num_classes % select_classes) == 0
        # assert int(self.num_instances % select_instances) == 0
        self.instance_per_class = int(np.ceil(self.num_instances / self.select_instances)) * self.select_instances

    def __len__(self):
        return self.num_classes * self.instance_per_class

    def __iter__(self):

        rand_seed = self.epoch
        random.seed(rand_seed)
        np.random.seed(rand_seed)

        for k, v in self.index_dic.items():
            replace = True if self.instance_per_class > len(v) else False
            # todo: reduce repeated elements as much as possible!
            if replace:
                v.extend(np.random.choice(v, size=self.instance_per_class - self.num_instances, replace=False).tolist())
                v = np.asarray(v)
            else:
                v = np.random.choice(v, size=self.instance_per_class, replace=False)
            np.random.shuffle(v)
            self.index_dic[k] = v
        devide = int(self.instance_per_class / self.select_instances)
        indices = torch.randperm(self.num_classes)
        ret = []
        for d in range(devide):
            for i in indices:
                this = self.index_dic[int(i)][d * self.select_instances:(d + 1) * self.select_instances]
                ret.extend(this.tolist())

        # print('Done data sampling')
        return iter(ret)

    def set_epoch(self, epoch):
        self.epoch = epoch


class RandomCycleIter:

    def __init__(self, data_list, test_mode=False):
        self.data_list = list(data_list)
        self.length = len(self.data_list)
        self.i = self.length - 1
        self.test_mode = test_mode

    def __iter__(self):
        return self

    def __next__(self):
        self.i += 1

        if self.i == self.length:
            self.i = 0
            if not self.test_mode:
                random.shuffle(self.data_list)

        return self.data_list[self.i]


def class_aware_sample_generator(cls_iter, data_iter_list, n, num_samples_cls=1):
    i = 0
    j = 0
    while i < n:

        #         yield next(data_iter_list[next(cls_iter)])

        if j >= num_samples_cls:
            j = 0

        if j == 0:
            temp_tuple = next(zip(*[data_iter_list[next(cls_iter)]] * num_samples_cls))
            yield temp_tuple[j]
        else:
            yield temp_tuple[j]

        i += 1
        j += 1


class ClassAwareSampler(Sampler):

    def __init__(self, data_source, num_samples_cls=3, reduce=4):
        random.seed(0)
        torch.manual_seed(0)
        num_classes = len(np.unique(data_source.CLASSES))

        self.epoch = 0

        self.class_iter = RandomCycleIter(range(num_classes))
        # cls_data_list = [list() for _ in range(num_classes)]
        '''
        labels = [ i for i in range(num_classes)]
        for i, label in enumerate(labels):
            cls_data_list[label].append(i)'''
        self.cls_data_list, self.gt_labels = data_source.get_index_dic(list=True, get_labels=True)

        self.num_classes = len(self.cls_data_list)
        self.data_iter_list = [RandomCycleIter(x) for x in self.cls_data_list]  # repeated
        self.num_samples = int(max([len(x) for x in self.cls_data_list]) * len(
            self.cls_data_list) / reduce)  # attention, ~ 1500(person) * 80
        self.num_samples_cls = num_samples_cls
        print('>>> Class Aware Sampler Built! Class number: {}, reduce {}'.format(num_classes, reduce))

    def __iter__(self):
        return class_aware_sample_generator(self.class_iter, self.data_iter_list,
                                            self.num_samples, self.num_samples_cls)

    def __len__(self):
        return self.num_samples

    def set_epoch(self, epoch):
        self.epoch = epoch

    def get_sample_per_class(self):
        condition_prob = np.zeros([self.num_classes, self.num_classes])
        sample_per_cls = np.asarray([len(x) for x in self.gt_labels])
        rank_idx = np.argsort(-sample_per_cls)

        for i, cls_labels in enumerate(self.gt_labels):
            num = len(cls_labels)
            condition_prob[i] = np.sum(np.asarray(cls_labels), axis=0) / num

        sum_prob = np.sum(condition_prob, axis=0)
        need_sample = sample_per_cls / sum_prob
        import matplotlib.pyplot as plt
        fig = plt.figure()
        ax1 = fig.add_subplot(2, 1, 1)
        ax1.bar(range(self.num_classes), sum_prob[rank_idx], alpha=0.5, color='green', label='sum_j( p(i|j) )')
        plt.legend()
        plt.hlines(1, 0, self.num_classes, linestyles='dashed', color='r', linewidth=1)
        ax2 = fig.add_subplot(2, 1, 2)
        # ax2.bar(range(self.num_classes), need_sample[rank_idx], alpha = 0.5, label='need_avg')
        ax2.bar(range(self.num_classes), sample_per_cls[rank_idx], alpha=0.5, label='ori_distribution')
        plt.legend()
        plt.savefig('./coco_resample_deduce.jpg')
        print('saved at ./coco_resample_deduce.jpg')
        print(np.min(sum_prob), np.max(need_sample))
        exit()


class ClassWeightedSampler(Sampler):
    """
    类别加权重采样器 (Class-Weighted Resampling)
    基于类别频率计算样本权重，对尾部类别的样本赋予更高的采样权重

    公式:
    - 类别权重: μc = (N / Nc)^s (s为软化系数)
    - 样本权重: 样本包含的所有类别权重之和
    """

    def __init__(self, data_source, weight_power=1.0, oversample_factor=4.0):
        """
        Args:
            data_source: 数据集对象，需要有 get_ann_info 方法
            weight_power: 权重软化系数（相当于超参数s）
                          = 1.0: 保持原始权重
                          < 1.0: 软化权重差异（推荐 0.5-0.7）
                          > 1.0: 放大权重差异
        """
        random.seed(0)
        torch.manual_seed(0)

        # 保存参数
        self.data_source = data_source
        self.weight_power = weight_power
        self.oversample_factor = oversample_factor
        self.epoch = 0

        # 增加采样数量
        self.num_classes = len(data_source.CLASSES)
        self.num_samples_per_epoch = len(data_source)
        self.num_samples = int(self.num_samples_per_epoch * oversample_factor)

        # 获取每个样本的标签
        print('>>> Collecting sample labels...')
        self.sample_labels = []
        for idx in range(self.num_samples_per_epoch):
            ann_info = data_source.get_ann_info(idx)
            labels = ann_info['labels']
            self.sample_labels.append(labels)

        # 计算权重
        self.class_weights = self._calculate_class_weights()
        self.sample_weights = self._calculate_sample_weights()

        # 打印统计信息
        self._print_statistics()

    def _calculate_class_weights(self):
        """
        计算类别权重: μc = (N / Nc)^weight_power

        Returns:
            class_weights: torch.Tensor, shape (num_classes,)
        """
        total_samples = self.num_samples_per_epoch
        class_weights = torch.zeros(self.num_classes, dtype=torch.float32)

        # 统计每个类别出现在多少个样本中
        class_counts = torch.zeros(self.num_classes)
        for labels in self.sample_labels:
            for class_id in range(self.num_classes):
                if labels[class_id] == 1:
                    class_counts[class_id] += 1

        # 计算类别权重，并应用 weight_power
        for class_id in range(self.num_classes):
            count = class_counts[class_id]
            if count > 0:
                # 应用软化系数: (N/Nc)^s
                raw_weight = total_samples / count
                class_weights[class_id] = raw_weight ** self.weight_power
            else:
                class_weights[class_id] = 0.0

        return class_weights

    def _calculate_sample_weights(self):
        """
        计算样本权重: 样本包含的所有类别权重之和

        Returns:
            sample_weights: torch.Tensor, shape (num_samples,)
        """
        sample_weights = torch.zeros(self.num_samples_per_epoch, dtype=torch.float32)

        for sample_idx, labels in enumerate(self.sample_labels):
            # 遍历该样本的所有类别
            for class_id in range(self.num_classes):
                if labels[class_id] == 1:
                    # 累加该类别的权重
                    sample_weights[sample_idx] += self.class_weights[class_id]

        # 归一化：除以最大权重，使权重范围在 [0, 1]
        if torch.max(sample_weights) > 0:
            sample_weights = sample_weights / torch.max(sample_weights)

        return sample_weights

    def _print_statistics(self):

        # 统计类别分布
        class_counts = torch.zeros(self.num_classes)
        for labels in self.sample_labels:
            for class_id in range(self.num_classes):
                if labels[class_id] == 1:
                    class_counts[class_id] += 1
        print('=' * 60)
        print('>>> Class-Weighted Sampler Built (Enhanced)!')
        print(f'    Number of classes: {self.num_classes}')
        print(f'    Original dataset size: {self.num_samples_per_epoch}')
        print(f'    Samples per epoch: {self.num_samples} ({self.oversample_factor}x)')
        print(f'    Weight power (s): {self.weight_power}')
        print(f'    Class weight range: [{torch.min(self.class_weights):.4f}, {torch.max(self.class_weights):.4f}]')
        print(f'    Sample weight range: [{torch.min(self.sample_weights):.4f}, {torch.max(self.sample_weights):.4f}]')

        # 找出头部和尾部类别
        sorted_indices = torch.argsort(class_counts, descending=True)
        head_classes = sorted_indices[:3]
        tail_classes = sorted_indices[-3:]

        print('\n    Top 3 head classes (most samples):')
        for idx in head_classes:
            idx = int(idx)
            count = int(class_counts[idx])
            weight = self.class_weights[idx].item()
            class_name = self.data_source.CLASSES[idx] if hasattr(self.data_source, 'CLASSES') else f'Class_{idx}'
            print(f'      {class_name}: {count} samples, weight={weight:.4f}')

        print('\n    Top 3 tail classes (fewest samples):')
        for idx in tail_classes:
            idx = int(idx)
            count = int(class_counts[idx])
            weight = self.class_weights[idx].item()
            expected_times = self.sample_weights[self.sample_labels == 1].sum() * self.oversample_factor / count
            class_name = self.data_source.CLASSES[idx] if hasattr(self.data_source, 'CLASSES') else f'Class_{idx}'
            print(
                f'      {class_name}: {count} samples, weight={weight:.4f}, , expected ~{expected_times:.1f}x per epoch')

        print('=' * 60)

    def __iter__(self):
        """
        返回一个迭代器，基于样本权重进行有放回采样
        """
        # 使用 PyTorch 的 multinomial 进行加权采样
        indices = torch.multinomial(
            self.sample_weights,
            num_samples=self.num_samples,
            replacement=True
        ).tolist()

        return iter(indices)

    def __len__(self):
        return self.num_samples

    def set_epoch(self, epoch):
        self.epoch = epoch