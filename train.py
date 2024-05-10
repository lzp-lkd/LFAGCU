import os
import argparse
import math
import sys
import pandas as pd
import seaborn as sns
import numpy as np

import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms, datasets
import torch.optim.lr_scheduler as lr_scheduler
from tqdm import tqdm
from my_dataset import MyDataSet
from model import LFAGCU_s as create_model
from utils import read_split_data, train_one_epoch, evaluate
from sklearn.metrics import confusion_matrix, accuracy_score, recall_score, f1_score,precision_score
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
'''
使用t-SNE进行降维并绘制t-SNE图
'''
def plot_tsne(y_true, y_pred, gxy_labels):
    df = pd.DataFrame({'True Label': y_true, 'Predicted Label': y_pred})

    # 保留每个类别的所有点
    sampled_df = df

    # 使用t-SNE进行降维，设置perplexity参数为一个较小的值
    tsne = TSNE(n_components=2, random_state=42, init='random', learning_rate=50, perplexity=7)
    tsne_result = tsne.fit_transform(sampled_df[['True Label', 'Predicted Label']])

    # 绘制t-SNE图
    sns.set(style="whitegrid")
    plt.figure(figsize=(10, 8))  # 调整图的大小
    sns.scatterplot(x=tsne_result[:, 0], y=tsne_result[:, 1], hue=sampled_df['True Label'], palette='viridis', s=250)
    plt.xlabel('t-SNE Dimension 1', fontsize=30)  # 增大横坐标字体
    plt.ylabel('t-SNE Dimension 2', fontsize=30)  # 增大纵坐标字体
    plt.xticks(fontsize=30)  # 增大刻度标签字体
    plt.yticks(fontsize=30)  # 增大刻度标签字体
    # plt.title('t-SNE Plot', fontsize=16)  # 增大标题字体


    # Create a custom legend with the provided labels
    """    RSSCN7 data    """
    legend_labels = ["aGrass",
                  "bField",
                  "cIndustry",
                  "dRiverLake",
                  "eForest",
                  "fResident",
                  "gParking"]

    """    WHU_RS19 data    """
    # legend_labels = [
    #     "Airport", "Beach", "Bridge", "Commercial", "Desert", "Farmland", "Football Field",
    #     "Forest", "Industrial", "Meadow", "Mountain", "Park", "Parking", "Pond", "Port",
    #     "Railway Station", "Residential", "River", "Viaduct"
    # ]
    """
    UCMerced_LandUse data
    """
    legend_labels = [
        "agricultural",
        "airplane",
        "baseballdiamond",
        "beach",
        "buildings",
        "chaparral",
        "denseresidential",
        "forest",
        "freeway",
        "golfcourse",
        "harbor",
        "intersection",
        "mediumresidential",
        "mobilehomepark",
        "overpass",
        "parkinglot",
        "river",
        "runway",
        "sparseresidential",
        "storagetanks",
        "tenniscourt",
    ]
    handles = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=c, markersize=10) for c in
               sns.color_palette('viridis', len(legend_labels))]
    # plt.legend(handles, legend_labels, fontsize=14, loc='lower right')
    plt.legend(handles, legend_labels, fontsize=14)

    plt.tight_layout()
    # plt.legend(fontsize=30, loc='upper left')  # 增大图例字体
    plt.subplots_adjust(bottom=0.15, left=0.15)  # 调整底部和左边的空白
    model_name = 'LFAGCU_s'
    plt.savefig(
        os.path.join(
            r'F:\pythonproject\deep-learning-for-image-processing-master\pytorch_classification\Test14_MobileViT\UCMerced_LandUse',
            'UCMerced_LandUse_' + model_name + '_tsne.png'),
        dpi=300)
    plt.show()

def predict_model(model,test_loader):
    """
    Predict test data
    """
    # evaluation
    model.eval()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # empty lists for results
    y_true = []
    y_pred = []
    test_bar = tqdm(test_loader, file=sys.stdout)
    for step, data in enumerate(test_bar):
        images, labels = data

        with torch.no_grad():
            # predict class
            pred_logits = model(images.to(device))
            _, predict = torch.max(pred_logits.detach(), dim=1)
            y_true += torch.squeeze(labels.cpu()).tolist()
            y_pred += torch.squeeze(predict).tolist()
    #
    # print_res = "class: {}   prob: {:.3}".format(class_indict[str(predict_cla)],
    #                                              predict[predict_cla].numpy())
    return y_true, y_pred

def main(args):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    if os.path.exists("./weights") is False:
        os.makedirs("./weights")

    tb_writer = SummaryWriter()

    # train_images_path, train_images_label, val_images_path, val_images_label = read_split_data(args.data_path)

    img_size = 224
    data_transform = {
        "train": transforms.Compose([transforms.RandomResizedCrop(img_size),
                                     transforms.RandomHorizontalFlip(),
                                     transforms.ToTensor(),
                                     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]),
        "val": transforms.Compose([transforms.Resize(int(img_size * 1.143)),
                                   transforms.CenterCrop(img_size),
                                   transforms.ToTensor(),
                                   transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]),
        "test": transforms.Compose([transforms.Resize(int(img_size * 1.143)),
                                   transforms.CenterCrop(img_size),
                                   transforms.ToTensor(),
                                   transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])    }

    data_root = os.path.abspath(os.path.join(os.getcwd(), "../.."))  # get data root path
    image_path = os.path.join(data_root, "data_set", "UCMerced_LandUse")  # flower data set path
    assert os.path.exists(image_path), "{} path does not exist.".format(image_path)

    # # 实例化训练数据集
    # train_dataset = MyDataSet(images_path=train_images_path,
    #                           images_class=train_images_label,
    #                           transform=data_transform["train"])
    #
    # # 实例化验证数据集
    # val_dataset = MyDataSet(images_path=val_images_path,
    #                         images_class=val_images_label,
    #                         transform=data_transform["val"])

    # 实例化训练数据集
    train_dataset = datasets.ImageFolder(root=os.path.join(image_path,
                                                           "train"),
                                         transform=data_transform["train"])
    train_num = len(train_dataset)

    # 实例化验证数据集
    val_dataset = datasets.ImageFolder(root=os.path.join(image_path, "val"),
                                       transform=data_transform["val"])
    val_num = len(val_dataset)

    # 实例化test数据集
    test_dataset = datasets.ImageFolder(root=os.path.join(image_path, "test"),
                                        transform=data_transform["test"])
    test_num = len(test_dataset)


    batch_size = args.batch_size
    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])  # number of workers
    print('Using {} dataloader workers every process'.format(nw))

    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=batch_size,
                                               shuffle=True,
                                               pin_memory=True,
                                               num_workers=nw,
                                               # collate_fn=train_dataset.collate_fn
                                               )

    val_loader = torch.utils.data.DataLoader(val_dataset,
                                             batch_size=batch_size,
                                             shuffle=False,
                                             pin_memory=True,
                                             num_workers=nw,
                                             # collate_fn=val_dataset.collate_fn
                                             )
    test_loader = torch.utils.data.DataLoader(test_dataset,
                                             batch_size=batch_size,
                                             shuffle=False,
                                             pin_memory=True,
                                             num_workers=nw,
                                             # collate_fn=val_dataset.collate_fn
                                              )
    print("using {} images for training, {} images for validation, {} images for testdation.".format(train_num,
                                                                                                     val_num,
                                                                                                     test_num))

    model = create_model(num_classes=args.num_classes).to(device)

    if args.weights != "":
        assert os.path.exists(args.weights), "weights file: '{}' not exist.".format(args.weights)
        weights_dict = torch.load(args.weights, map_location=device)
        weights_dict = weights_dict["model"] if "model" in weights_dict else weights_dict
        # 删除有关分类类别的权重
        for k in list(weights_dict.keys()):
            if "classifier" in k:
                del weights_dict[k]
        print(model.load_state_dict(weights_dict, strict=False))

    if args.freeze_layers:
        for name, para in model.named_parameters():
            # 除head外，其他权重全部冻结
            if "classifier" not in name:
                para.requires_grad_(False)
            else:
                print("training {}".format(name))

    pg = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.AdamW(pg, lr=args.lr, weight_decay=1E-2)
    lf = lambda x: ((1 + math.cos(x * math.pi / args.epochs)) / 2) * (1 - args.lrf) + args.lrf  # cosine
    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)

    best_acc = 0.0
    best_epoch = -1
    early_stop_epochs = 15
    model_name = 'LFAGCU_s'
    save_path = 'UCMerced_LandUse' + model_name + '.pth'

    '''
    训练
    '''
    for epoch in range(args.epochs):
        # train
        train_loss, train_acc = train_one_epoch(model=model,
                                                optimizer=optimizer,
                                                data_loader=train_loader,
                                                device=device,
                                                epoch=epoch)

        scheduler.step()

        # validate
        val_loss, val_acc = evaluate(model=model,
                                     data_loader=val_loader,
                                     device=device,
                                     epoch=epoch)

        tags = ["train_loss", "train_acc", "val_loss", "val_acc", "learning_rate"]
        tb_writer.add_scalar(tags[0], train_loss, epoch)
        tb_writer.add_scalar(tags[1], train_acc, epoch)
        tb_writer.add_scalar(tags[2], val_loss, epoch)
        tb_writer.add_scalar(tags[3], val_acc, epoch)
        tb_writer.add_scalar(tags[4], optimizer.param_groups[0]["lr"], epoch)

        # torch.save(model.state_dict(), "./weights/model-{}.pth".format(epoch))
        ## Early stopping
        if (epoch + 1) - best_epoch >= early_stop_epochs:
            print("Early stopping... (Model did not improve after {} epochs)".format(early_stop_epochs))
            break

        val_accurate = val_acc
        if val_accurate > best_acc:
            best_acc = val_accurate
            best_epoch = epoch + 1
            torch.save(model.state_dict(), save_path)

    print('Finished Training')
    save_path=r'F:\pythonproject\deep-learning-for-image-processing-master\pytorch_classification\Test14_MobileViT\UCMerced_LandUse\UCMerced_LandUse_LFAGCU_s.pth'
    model.load_state_dict(torch.load(save_path))

    y_true, y_pred = predict_model(model, test_loader)

    """
    RSSCN7 data
    """
    # gxy_labels = ["aGrass",
    #               "bField",
    #               "cIndustry",
    #               "dRiverLake",
    #               "eForest",
    #               "fResident",
    #               "gParking"]

    """
    WHU_RS19 data
    """
    # gxy_labels = ["Airport",
    #               "Beach",
    #               "Bridge",
    #               "Commercial",
    #               "Desert",
    #               "Farmland",
    #               "footballField",
    #               "Forest",
    #               "Industrial",
    #               "Meadow",
    #               "Mountain",
    #               "Park",
    #               "Parking",
    #               "Pond",
    #               "Port",
    #               "railwayStation",
    #               "Residential",
    #               "River",
    #               "Viaduct",]

    """
    UCMerced_LandUse data
    """
    gxy_labels = [
        "agricultural",
        "airplane",
        "baseballdiamond",
        "beach",
        "buildings",
        "chaparral",
        "denseresidential",
        "forest",
        "freeway",
        "golfcourse",
        "harbor",
        "intersection",
        "mediumresidential",
        "mobilehomepark",
        "overpass",
        "parkinglot",
        "river",
        "runway",
        "sparseresidential",
        "storagetanks",
        "tenniscourt",
    ]

    # 绘制t-SNE图

    plot_tsne(y_true, y_pred, gxy_labels)


    '''
    #混淆矩阵
    WHU_RS19的配色是            YlGnBu      figsize=(18, 18)
    RSSCN7的配色是              RdBu_r,     figsize=(9, 9)
    UCMerced_LandUse的配色是    YlOrBr      figsize=(18, 18)
    '''
    plt.rcParams["font.family"] = "Times New Roman"
    cm = confusion_matrix(y_true, y_pred, normalize='true')
    cm_df = pd.DataFrame(cm, index=gxy_labels, columns=gxy_labels)
    # Choose a suitable color palette
    color_palette = sns.color_palette("YlOrBr")
    # Create a custom color palette
    colors = [(0.85, 0.85, 0.85)] + sns.color_palette("YlOrBr", n_colors=100)
    cmap = sns.blend_palette(colors, as_cmap=True)

    # Create a mask for values that are equal to 0
    mask = np.isclose(cm_df, 0.00)

    sns.set(font_scale=2)
    fig = plt.figure(figsize=(18, 18))
    ax = sns.heatmap(cm_df, annot=True, fmt=".2f", cmap=cmap, cbar=False, annot_kws={"size": 28}, mask=mask)
    ax.grid(True, which="both", linewidth=2)
    # Increase font size for x and y tick labels
    ax.set_xticklabels(ax.get_xticklabels(), fontsize=30)
    ax.set_yticklabels(ax.get_yticklabels(), fontsize=30)

    # Adjust the layout to make the labels extend beyond the grid cells
    plt.tight_layout()
    plt.savefig(
        os.path.join(
            r'F:\pythonproject\deep-learning-for-image-processing-master\pytorch_classification\Test14_MobileViT\UCMerced_LandUse',
            'UCMerced_LandUse_' + model_name + '_cm.png'),
        dpi=300)
    plt.show()

    '''Precision精确率'''
    # Precision精确率
    precision = precision_score(y_true, y_pred, average='macro')
    print("Precision = {:.4f}\n".format(precision))

    # Recall召回率
    recall = recall_score(y_true, y_pred, average='macro')
    print("Recall = {:.4f}\n".format(recall))

    # accuracy
    acc = accuracy_score(y_true, y_pred)
    print("Accuracy = {:.4f}\n".format(acc))

    # f1 score
    F1 = f1_score(y_true, y_pred, average='macro')
    print("F1 score = {:.4f}\n".format(F1))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_classes', type=int, default=21)
    parser.add_argument('--epochs', type=int, default=120)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=0.0002)
    parser.add_argument('--lrf', type=float, default=0.01)

    # 数据集所在根目录
    # https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz
    # parser.add_argument('--data-path', type=str,
    #                     default="/data/flower_photos")

    # 预训练权重路径，如果不想载入就设置为空字符
    parser.add_argument('--weights', type=str, default='./LFAGCU_s.pt',
                        help='initial weights path')
    # 是否冻结权重
    parser.add_argument('--freeze-layers', type=bool, default=False)
    parser.add_argument('--device', default='cuda:0', help='device id (i.e. 0 or 0,1 or cpu)')

    opt = parser.parse_args()

    main(opt)
