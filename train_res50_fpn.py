"""
以ResNet50 + FPN 作为backbone进行训练
"""
import sys

import torch
import transforms
from network_files.faster_rcnn_framework import FasterRCNN, FastRCNNPredictor
from backbone.resnet50_fpn_model import resnet50_fpn_backbone
from my_dataset import VOC2012DataSet
from train_utils import train_eval_utils as utils
import os
import warnings

warnings.filterwarnings('ignore')


# 保存控制台内容
class Logger(object):
    def __init__(self, fileN="Default.log"):
        self.terminal = sys.stdout
        self.log = open(fileN, "a")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        pass


sys.stdout = Logger('./log/log.txt')


def create_model(num_classes):
    backbone = resnet50_fpn_backbone()
    model = FasterRCNN(backbone=backbone, num_classes=91)
    # 载入预训练模型权重
    weights_dict = torch.load("./backbone/fasterrcnn_resnet50_fpn.pth")
    missing_keys, unexpected_keys = model.load_state_dict(weights_dict, strict=False)
    if len(missing_keys) != 0 or len(unexpected_keys) != 0:
        print("missing_keys: ", missing_keys)
        print("unexpected_keys: ", unexpected_keys)

    # get number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    return model


def main(parser_data):
    device = torch.device(parser_data.device if torch.cuda.is_available() else "cpu")
    print('use device', device)
    data_transform = {
        "train": transforms.Compose([transforms.ToTensor(),
                                     transforms.RandomHorizontalFlip(0.5)]),
        "val": transforms.Compose([transforms.ToTensor()])
    }

    VOC_root = parser_data.data_path
    # load train data set
    train_data_set = VOC2012DataSet(VOC_root, data_transform["train"], True)
    # 注意这里的collate_fn是自定义的，因为读取的数据包括image和targets，不能直接使用默认的方法合成batch
    train_data_loader = torch.utils.data.DataLoader(train_data_set,
                                                    batch_size=parser_data.batch_size,
                                                    shuffle=True,
                                                    num_workers=parser_data.num_workers,
                                                    collate_fn=utils.collate_fn)

    # load validation data set
    val_data_set = VOC2012DataSet(VOC_root, data_transform["val"], False)
    val_data_set_loader = torch.utils.data.DataLoader(val_data_set,
                                                      batch_size=parser_data.batch_size,
                                                      shuffle=False,
                                                      num_workers=parser_data.num_workers,
                                                      collate_fn=utils.collate_fn)

    # create model num_classes equal background + 20 classes
    model = create_model(num_classes=21)
    # print(model)

    model.to(device)

    # define optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    if parser_data.opt == "SGD":
        optimizer = torch.optim.SGD(params, lr=parser_data.lr,
                                    momentum=0.9, weight_decay=0.0005)
    else:
        optimizer = torch.optim.Adam(params, lr=parser_data.lr)
    # learning rate scheduler
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                   step_size=5,
                                                   gamma=0.33)

    # 如果指定了上次训练保存的权重文件地址，则接着上次结果接着训练
    if parser_data.resume != "":
        checkpoint = torch.load(parser_data.resume)
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        parser_data.start_epoch = checkpoint['epoch'] + 1
        print("the training process from epoch{}...".format(parser_data.start_epoch))

    for epoch in range(parser_data.start_epoch, parser_data.epochs):
        # train for one epoch, printing every 10 iterations
        utils.train_one_epoch(model, optimizer, train_data_loader,
                              device, epoch, print_freq=50, warmup=True)
        # update the learning rate
        lr_scheduler.step()

        # evaluate on the test dataset
        utils.evaluate(model, val_data_set_loader, device=device)

        # save weights
        save_files = {
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'lr_scheduler': lr_scheduler.state_dict(),
            'epoch': epoch}
        torch.save(save_files, "./save_weights/resNetFpn-model-{}.pth".format(epoch))

    # model.eval()
    # x = [torch.rand(3, 300, 400), torch.rand(3, 400, 400)]
    # predictions = model(x)
    # print(predictions)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description=__doc__)

    # 训练设备类型
    parser.add_argument('--device', default='cuda:0', help='device')
    # 训练数据集的根目录
    parser.add_argument('--data-path', default='./', help='dataset')
    # 文件保存地址
    parser.add_argument('--output-dir', default='./save_weights', help='path where to save')
    # 若需要接着上次训练，则指定上次训练保存权重文件地址
    parser.add_argument('--resume', default='', type=str, help='resume from checkpoint')
    # 指定接着从哪个epoch数开始训练
    parser.add_argument('--start_epoch', default=0, type=int, help='start epoch')
    # 训练的总epoch数
    parser.add_argument('--epochs', default=300, type=int, metavar='N',
                        help='number of total epochs to run')

    # 训练的学习率
    parser.add_argument('--lr', default='0.0001', type=float, help='learning rate')

    # batch_size

    parser.add_argument('--batch_size', default='8', type=int, help='batch_size nums')

    # nums_works

    parser.add_argument('--num_workers', default='8', type=int, help='num_workers')

    # 选择优化器
    parser.add_argument('--opt', default='SGD', type=str, help='optimizer')

    args = parser.parse_args()
    print(args)

    # 检查保存权重文件夹是否存在，不存在则创建
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    main(args)
