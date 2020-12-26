import torch.nn as nn
import torch
import torch.nn.functional as F
from torch import optim
from torch.utils.data import DataLoader
from uilts.parse_cfg import parse_json
from CustomDataset import CamvidDataset
from evalution import eval_semantic_segmentation
from models.UNet_Res import UNet
import configs as cfg
from tqdm import tqdm
from visdom import Visdom
import shutil, time
# device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


def train():
    # 数据可视化
    viz = Visdom()
    assert viz.check_connection()
    viz.close()

    viz.line([0.], [0.], win='train_loss', opts=dict(title='train loss'))
    viz.line([[0.0, 0.0, 0.0, 0.0]], [0.], win='test', opts=dict(title='test loss&acc.',
                                                       legend=['acc', 'miou', 'loss', 'train_loss']))

    # 打印增强信息
    augments_info = parse_json(cfg.json_path)['train']['pipeline']
    augments_info_txt = ""
    for augment_info in augments_info:
        if "type" in augment_info.keys():
            for key,value in augment_info.items():
                augments_info_txt += "%s:%s  " % (key, value)
            augments_info_txt += "\n"
    print(augments_info_txt)

    # 加载数据
    train_data = CamvidDataset(cfg.train_path, cfg.train_label_path, cfg.json_path, mode="train")
    val_data = CamvidDataset(cfg.val_path, cfg.val_label_path, cfg.json_path, mode="val")

    train_db = DataLoader(train_data, batch_size=cfg.batch_size, shuffle=True)
    val_db = DataLoader(val_data, batch_size=cfg.batch_size)

    # 模型和损失函数
    model = UNet(3, cfg.n_class)
    # model.load_state_dict(torch.load("weights/best.pth"))
    criterion = nn.NLLLoss()         # 表达式 loss(input, class) = -input[class]
    optimizer = optim.Adam(model.parameters(), lr=cfg.lr)

    best = 0.
    net = model.train()

    global_step = 0
    for epoch in range(cfg.epochs):

        if epoch // int(cfg.epochs*0.8) == 0 and cfg.epochs != 0:
            for param_group in optimizer.param_groups:
                param_group['lr'] = cfg.lr * 0.1
        if epoch // int(cfg.epochs*0.9) == 0 and cfg.epochs != 0:
            for param_group in optimizer.param_groups:
                param_group['lr'] = cfg.lr * 0.01

        train_loss = 0
        train_acc = 0
        train_miou = 0

        nLen = len(train_db)
        batch_bar = tqdm(enumerate(train_db), total=nLen)
        for i, (img, label) in batch_bar:
            img = img
            label = label    # [batch, 256, 256]
            out = net(img)      # [batch, 3, 256, 256]
            out = torch.nn.functional.log_softmax(out, dim=1)

            loss = criterion(out, label)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            batch_loss = loss.item()
            train_loss += batch_loss

            # 评估
            pre_label = out.max(dim=1)[1].data.cpu().numpy()
            true_label = label.data.cpu().numpy()

            eval_metrix = eval_semantic_segmentation(pre_label, true_label)
            train_acc += eval_metrix['PA']
            train_miou += eval_metrix['miou']
            global_step += 1
            batch_bar.set_description('|batch[{}/{}]|train_loss {: .8f}|'.format(i + 1, len(train_db), batch_loss))
            viz.line([batch_loss], [global_step], win='train_loss', update='append', opts=dict(legend=['loss']))

            # if i+1 == nLen:
        epoch_train_acc = train_acc / nLen
        epoch_train_miou = train_miou / nLen
        epoch_train_loss = train_loss / nLen
        metric_description = '|Train Acc|: {:.5f}|Train Mean IOU|: {:.5f}|Train loss|: {:.5f}'.format(
            epoch_train_acc, epoch_train_miou, epoch_train_loss)
        # batch_bar.write(metric_description)
        print(metric_description)
        if best <= epoch_train_miou:
            torch.save(net.state_dict(), 'best.pth')
            best = epoch_train_miou

        # 验证集
        epoch_val_acc, epoch_val_miou, epoch_val_loss = evaluate(model, val_db, criterion)
        print('|test Acc|: {:.5f}|test Mean IOU|: {:.5f}|'.format(epoch_val_acc, epoch_val_miou))
        viz.line([[epoch_val_acc, epoch_val_miou, epoch_val_loss, epoch_train_loss]], [epoch], win='test', update='append',
                 opts=dict(title='test loss&acc.', legend=['acc', 'miou', 'val_loss', 'train_loss']))

    # 验证测试集
    model.load_state_dict(torch.load('best.pth'))
    test_data = CamvidDataset(cfg.test_path, cfg.test_label_path, cfg.json_path, mode="test")
    test_db = DataLoader(test_data, batch_size=cfg.batch_size)

    epoch_test_acc, epoch_test_miou, epoch_test_loss = evaluate(model, test_db, criterion)
    epoch_val_acc, epoch_val_miou, epoch_val_loss = evaluate(model, val_db, criterion)


    miou = (epoch_val_miou+ epoch_test_miou) * 0.5
    shutil.copy(cfg.json_path, 'configs/' + 'CamVidEpoch%d_miou_=%.5f_%s.json' %
                (cfg.epochs, miou, time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())))


    augments_info_txt += 'epoch=%d, best_miou=%.4f, val_miou=%.4f, miou=%.4f' % \
                         (cfg.epochs, epoch_val_miou, epoch_test_miou, miou)

    viz.text(augments_info_txt)
    print(augments_info_txt)


def evaluate(model, val_db, criterion):
    net = model.eval()
    eval_loss = 0
    eval_acc = 0
    eval_miou = 0
    nLen = len(val_db)
    tset_bar = tqdm(enumerate(val_db), total=nLen)
    for i, (img, label) in tset_bar:
        img = img
        label = label

        out = net(img)
        out = F.log_softmax(out)
        loss = criterion(out, label)
        eval_loss += loss.item()
        # 评估
        pre_label = out.max(dim=1)[1].data.cpu().numpy()
        true_label = label.data.cpu().numpy()

        eval_metrix = eval_semantic_segmentation(pre_label, true_label)
        eval_acc += eval_metrix['PA']
        eval_miou += eval_metrix['miou']

        tset_bar.set_description('|batch[{}/{}]|test_loss {: .8f}|'.format(i + 1, nLen, loss.item()))

    epoch_test_acc = eval_acc / nLen
    epoch_test_miou = eval_miou / nLen
    epoch_eval_loss = eval_loss/ nLen

    return epoch_test_acc, epoch_test_miou, epoch_eval_loss


if __name__ == "__main__":
    train()

