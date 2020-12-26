import numpy as np
import configs as cfg


def calc_semantic_segmentation_confusion(pred_labels, gt_labels):
    """建立混淆矩阵,以方便计算PA,MPA,Iou等各个指标"""
    confusion = np.zeros((cfg.n_class, cfg.n_class))
    for pred_label, gt_label in zip(pred_labels, gt_labels):
        # 判断混淆矩阵是否是2维
        if pred_label.ndim != 2 or gt_label.ndim != 2:
            raise ValueError('ndim of labels should be two.')

        # 判断 输入输出尺寸一样
        if pred_label.shape != gt_label.shape:
            raise ValueError('Shape of ground truth and prediction should'
                             ' be same.')

        # 打成一维向量
        pred_label = pred_label.flatten()
        gt_label = gt_label.flatten()

        # 校验一下 保证 gt_label都是大于等于0的，以保证计算没有错误
        mask = gt_label >= 0

        # 统计正确和错误分类个数 放在对应位置上（  n×label + pred）
        confusion += np.bincount(cfg.n_class * gt_label[mask].astype(int) + pred_label[mask],
                                    minlength=cfg.n_class ** 2).reshape((cfg.n_class, cfg.n_class))

    return confusion


# 计算交并比
def calc_semantic_segmentation_iou(confusion):
    """通过混淆矩阵计算交并比。
    混淆矩阵中  对角线为分类正确的个数，其余的均为分类错误
              所有行，表示标签中的类别个数
              所有列，表示预测值的类别个数
              交并比 = 正确的个数/(对角线上行的和+列的和 - 对角线上的值)
    """

    union_area = confusion.sum(axis=1) + confusion.sum(axis=0) - np.diag(confusion)
    iou = np.diag(confusion) / union_area
    # 将背景的交并比省去
    return iou[1:]


def eval_semantic_segmentation(pred_labels, gt_labels):
    """验证语义分割"""
    confusion = calc_semantic_segmentation_confusion(pred_labels, gt_labels)

    iou = calc_semantic_segmentation_iou(confusion)

    # 计算 PA值
    pixel_accuracy = np.diag(confusion).sum() / confusion.sum()

    # 计算每个分类的PA值
    # class_accuracy = np.diag(confusion)/ (np.sum(confusion, axis=1) + 1e-10)

    return {'iou': iou,
            'miou': np.nanmean(iou),
            'PA': pixel_accuracy,
            #"class_accuracy": class_accuracy,
            #"mean_class_accuracy": np.nanmean(class_accuracy[1:])
            }
