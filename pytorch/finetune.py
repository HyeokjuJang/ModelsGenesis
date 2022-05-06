import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import *
from torch.optim.lr_scheduler import StepLR
from mri_dataloader import (
    MRI_AD_CN_Dataset_from_file_list,
    MRI_AD_CN_Dataset,
    MRI_AD_CN_MCI_Dataset,
    MRI_AD_CN_Yaware_Dataset,
    MRIDataset,
)
import os
import random
import wandb
import unet3d
from scipy.stats import norm
from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import KFold, StratifiedShuffleSplit
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize
import numpy as np

seed = 2022

lr = 0.001
batch_size = 16
n_epochs = 10
gpu = 0
mci = True
n_class = 2
workers = 4 * torch.cuda.device_count()

# gamma for StepLR
gamma = 0.5

# Beta1 hyperparam for Adam optimizers
beta1 = 0.9

# Create CNN Model
class TargetNet(nn.Module):
    def __init__(self, base_model, in_channel=[16, 32, 64, 128], out_channel=1):
        super(TargetNet, self).__init__()
        self.base_model = base_model
        
        self.conv_layer1 = self._conv_layer_set(in_channel[0], 32, 3, 1)
        self.conv_layer2 = self._conv_layer_set(in_channel[1] * 2, 64, 3, 1)
        self.conv_layer3 = self._conv_layer_set(in_channel[2] * 2, 128, 3, 1)
        self.conv_layer4 = self._conv_layer_set(in_channel[3] * 2, 16, 3, 1)
        self.avg_pool = nn.AvgPool3d(3)
        self.fc_layer = nn.Sequential(
            # nn.Linear(128, 256),
            # nn.Linear(192, 256),  # for (121, 121, 145)
            nn.Linear(432, 256),  # for (144, 148, 176)
            nn.BatchNorm1d(256),
            nn.LeakyReLU(),
            nn.Dropout(p=0.3),
            nn.Linear(256, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(),
            nn.Dropout(p=0.3),
            nn.Linear(256, out_channel),
            # nn.Sigmoid(),
        )
        self.all_models = [
            self.conv_layer1,
            self.conv_layer2,
            self.conv_layer3,
            self.conv_layer4,
            self.fc_layer,
        ]

    def _conv_layer_set(self, in_c, out_c, kernel_size=3, stride=1):
        conv_layer = nn.Sequential(
            nn.MaxPool3d((2, 2, 2)),
            nn.Conv3d(
                in_c,
                out_c,
                kernel_size=(kernel_size, kernel_size, kernel_size),
                padding=1,
                stride=stride,
            ),
            nn.BatchNorm3d(out_c),
            nn.LeakyReLU(),
        )
        return conv_layer

    def forward(self, x):
        batch_size = x[0].shape[0]
        # Set 1
        out0 = self.conv_layer1(x[3])
        out1 = self.conv_layer2(torch.cat([out0, x[2]], dim=1))
        out2 = self.conv_layer3(torch.cat([out1, x[1]], dim=1))
        out = self.conv_layer4(torch.cat([out2, x[0]], dim=1))
        # print(out.shape)  # torch.Size([8, 16, 7, 7, 9])
        out = self.avg_pool(out)
        # print(out.shape)  # torch.Size([8, 16, 2, 2, 3])
        out = out.view(out.size(0), -1)
        # print(out.shape)  # torch.Size([8, 192])
        out = self.fc_layer(out)

        return out

    def reset_parameters(self):
        for m in self.all_models:
            for layer in m.children():
                if hasattr(layer, "reset_parameters"):
                    layer.reset_parameters()

def train_worker(
    gpu,
    n_gpus,
    batch_size=8,
    num_worker=8,
    num_epochs=30,
    block_method="block",
    base_trainable=False,
    pretrain_path=None,
    load_path=None,
    dataset_path="mri_norm_talairach",
    is_wandb=False,
    exp_id="",
    base_trainable_after_n_epoch=-1,
    mci=True,
    y_aware_test=False,
    densenet=False,
    n_classes=2,
    legacy=False,
):
    best_loss = 10000
    best_f1 = 0
    best_epoch = 0
    best_fold = 0
    best_acc = 0
    best_auc_macro = 0
    best_auc_micro = 0

    if is_wandb and gpu == 0:
        job_name = "Training-block_{}-gpu_{}-batch_{}-worker_{}-epochs_{}_id_{}".format(
            block_method, n_gpus, batch_size, workers, num_epochs, exp_id
        )
        if block_method == "guided":
            job_name += "-alpha_{}".format(pretrain_path.split("_")[-1].split(".pt")[0])
        wandb.init(entity="entaline", project="brain-3d-inpainting", name=job_name)

    _batch_size = int(batch_size / n_gpus)
    _num_worker = int(num_worker / n_gpus)

    # load AD CN dataset
    ad_train_path = os.path.join(dataset_path, "ad")
    cn_train_path = os.path.join(dataset_path, "cn")
    ad_test_path = os.path.join(dataset_path, "ad_test")
    cn_test_path = os.path.join(dataset_path, "cn_test")
    if mci:
        mci_train_path = os.path.join(dataset_path, "mci")
        mci_test_path = os.path.join(dataset_path, "mci_test")
        train_datasets = MRI_AD_CN_MCI_Dataset(
            ad_path=ad_train_path,
            cn_path=cn_train_path,
            mci_path=mci_train_path,
            resize=None,
            n_classes=n_classes,
        )
        test_dataset = MRI_AD_CN_MCI_Dataset(
            ad_path=ad_test_path,
            cn_path=cn_test_path,
            mci_path=mci_test_path,
            resize=None,
            n_classes=n_classes,
        )
    else:
        train_datasets = MRI_AD_CN_Dataset(
            ad_path=ad_train_path, cn_path=cn_train_path, resize=None
        )
        test_dataset = MRI_AD_CN_Dataset(
            ad_path=ad_test_path, cn_path=cn_test_path, resize=None
        )

    sss = StratifiedShuffleSplit(n_splits=5, test_size=0.2, random_state=6)

    for fold_id, (train_index, valid_index) in enumerate(
        sss.split(train_datasets.x_data_file_list, train_datasets.y_data)
    ):
        print("Fold:", fold_id)

        base_model = unet3d.UNet3D(finetune=True)

        #Load pre-trained weights
        weight_dir = pretrain_path
        checkpoint = torch.load(weight_dir)
        state_dict = checkpoint['state_dict']
        unParalled_state_dict = {}
        for key in state_dict.keys():
            unParalled_state_dict[key.replace("module.", "")] = state_dict[key]
        base_model.load_state_dict(unParalled_state_dict)
        fullModel = TargetNet(base_model, out_channel=n_classes)
        fullModel.cuda(gpu)
        target_model = nn.DataParallel(target_model, device_ids = [i for i in range(torch.cuda.device_count())])
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(target_model.parameters(), lr, momentum=0.9, weight_decay=0.0, nesterov=False)
        
        torch.cuda.set_device(gpu)

        if y_aware_test:
            train_dataset = MRI_AD_CN_Yaware_Dataset(
                ad_path=[ad_train_path, ad_test_path],
                cn_path=[cn_train_path, cn_test_path],
                fold=fold_id,
                train=True,
            )
            valid_dataset = MRI_AD_CN_Yaware_Dataset(
                ad_path=[ad_train_path, ad_test_path],
                cn_path=[cn_train_path, cn_test_path],
                fold=fold_id,
                train=False,
            )
        else:
            train_dataset = MRI_AD_CN_Dataset_from_file_list(
                [
                    train_datasets.x_data_file_list[train_idx]
                    for train_idx in train_index
                ],
                [train_datasets.y_data[train_idx] for train_idx in train_index],
            )
            valid_dataset = MRI_AD_CN_Dataset_from_file_list(
                [
                    train_datasets.x_data_file_list[valid_idx]
                    for valid_idx in valid_index
                ],
                [train_datasets.y_data[valid_idx] for valid_idx in valid_index],
            )

        train_sampler = None

        # Create the dataloader
        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=_batch_size,
            shuffle=(train_sampler is None),
            num_workers=_num_worker,
            pin_memory=True,
            sampler=train_sampler,
            drop_last=True,
        )
        valid_loader = torch.utils.data.DataLoader(
            valid_dataset,
            batch_size=_batch_size,
            shuffle=True,
            num_workers=_num_worker,
            drop_last=True,
        )
        test_loader = torch.utils.data.DataLoader(
            test_dataset, batch_size=_batch_size, shuffle=False, num_workers=_num_worker
        )

        # Initialize Loss function
        criterion = nn.CrossEntropyLoss().cuda(gpu)
        # criterion = nn.KLDivLoss(reduction='mean').cuda(gpu)
        # criterion = nn.L1Loss().cuda(gpu)
        # criterion = nn.NLLLoss().cuda(gpu)
        # criterion = dice_loss

        # Setup Adam optimizers for both G and D
        optimizer = Adam(
            fullModel.parameters(),
            lr=lr,
            betas=(beta1, 0.999),
            eps=1e-08,
            weight_decay=0.01,
        )

        if is_wandb and gpu == 0:
            wandb.watch(fullModel, log="all")

        scheduler = StepLR(optimizer, step_size=2, gamma=gamma)

        for epoch in range(1, num_epochs + 1):
            if train_sampler is not None:
                train_sampler.set_epoch(epoch)

            if (
                base_trainable_after_n_epoch > 0
                and epoch > base_trainable_after_n_epoch
            ):
                fullModel.base_trainable = True

            # train start
            train(
                fullModel,
                train_loader,
                criterion,
                optimizer,
                epoch,
                gpu,
                n_gpus,
                is_wandb,
                dry_run=False,
                mci=mci,
                n_class=n_classes,
            )

            # evaluation
            val_loss, val_correct, val_f1, val_auc = evaluate(
                fullModel,
                gpu,
                n_gpus,
                valid_loader,
                is_wandb,
                isTest=False,
                n_classes=n_classes,
            )

            # compare to best
            is_best = False
            if val_loss < best_loss:
                best_loss = val_loss
                best_epoch = epoch
                best_f1 = val_f1
                best_fold = fold_id
                best_acc = val_correct
                best_auc_macro = val_auc["macro"]
                best_auc_micro = val_auc["micro"]
                is_best = True

            # log for validation
            if gpu == 0:
                print(
                    "\nValid evaluate set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%), F1 Score: {:.4f}, macro AUC: {:.4f}, micro AUC: {:.4f}\n".format(
                        val_loss,
                        val_correct,
                        len(valid_loader.dataset),
                        100.0 * val_correct / len(valid_loader.dataset),
                        val_f1,
                        val_auc["macro"],
                        val_auc["micro"],
                    )
                )
                print(
                    "current best loss: {:.4f}, acc: {:.2f}, f1: {:.4f}, macroAUC: {:.4f}, microAUC: {:.4f}".format(
                        best_loss,
                        100.0 * best_acc / len(valid_loader.dataset),
                        best_f1,
                        best_auc_macro,
                        best_auc_micro,
                    )
                )
                if is_wandb:
                    wandb.log(
                        {
                            "Valid Average Loss": val_loss,
                            "Valid Accuracy": val_correct / len(valid_loader.dataset),
                            "Valid F1 Score": val_f1,
                            "Valid macro AUC": val_auc["macro"],
                            "Valid micro AUC": val_auc["micro"],
                        }
                    )
                if is_best:
                    print("new best, saving model")
                    save_train(
                        fullModel,
                        optimizer,
                        block_method,
                        val_correct / len(valid_loader.dataset),
                        epoch,
                        n_gpus,
                        is_wandb,
                        is_best,
                    )
            # update lr
            scheduler.step()

    if gpu == 0:
        print(
            "Testing with best fold {}, valid {} epoch version".format(
                best_fold, best_epoch
            )
        )

    # redefine model for load from state_dict
    fullModel = TargetNet(base_model, out_channel=n_classes)
    fullModel.cuda(gpu)

    # load best model
    map_location = {"cuda:%d" % 0: "cuda:%d" % gpu}
    fullModel.load_state_dict(
        torch.load(
            os.path.join("models", "%s-%s-best.pt" % ("train", block_method)),
            map_location=map_location,
        )["state_dict"]
    )
    test_loss, test_correct, test_f1, test_auc = evaluate(
        fullModel, gpu, n_gpus, test_loader, is_wandb, isTest=True, n_classes=n_classes
    )

    if gpu == 0:
        print(
            "\nTest evaluate set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%), F1 Score: {:.4f}, Macro AUC: {:.4f}, micro AUC: {:.4f}\n".format(
                test_loss,
                test_correct,
                len(test_loader.dataset),
                100.0 * test_correct / len(test_loader.dataset),
                test_f1,
                test_auc["macro"],
                test_auc["micro"],
            )
        )
        if is_wandb:
            wandb.log(
                {
                    "Test Average Loss": test_loss,
                    "Test Accuracy": test_correct / len(test_loader.dataset),
                    "Test F1 Score": test_f1,
                    "Test macro AUC": test_auc["macro"],
                    "Test micro AUC": test_auc["micro"],
                }
            )
            wandb.finish()


def train(
    model,
    train_loader,
    criterion,
    optimizer,
    epoch,
    gpu,
    n_gpus,
    is_wandb,
    dry_run=False,
    mci=True,
    n_class=2,
):
    model.train()
    for i, (data, target) in enumerate(train_loader, 0):
        # load data
        data = data.cuda(gpu, non_blocking=True)
        target = target.cuda(gpu, non_blocking=True)

        # load balanced weight from data
        batch_size = target.size(0)
        weights = []
        for y_t in range(n_class):
            num_of_sums = torch.sum(target == y_t)
            if num_of_sums == 0.0:
                num_of_sums = batch_size
            weights.append(batch_size / num_of_sums)

        criterion = nn.CrossEntropyLoss(weight=torch.tensor(weights)).cuda(gpu)
        data.requires_grad = True
        model.zero_grad()
        output = model(data).view(batch_size, -1)

        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        if (i % 10 == 0 or (i == len(train_loader) - 1)) and gpu == 0:
            print(
                "Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}".format(
                    epoch,
                    i * len(data),
                    len(train_loader.dataset),
                    100.0 * i / len(train_loader),
                    loss.item(),
                )
            )
            if is_wandb:
                wandb.log({"loss": loss.item()})
            if dry_run:
                break


def evaluate(model, gpu, n_gpus, test_loader, is_wandb, isTest=False, n_classes=2):
    model.eval()
    test_loss = 0
    correct = 0
    f1_true = None
    f1_pred = None
    out_score = None
    with torch.no_grad():
        for data, target in test_loader:
            data = data.cuda(gpu, non_blocking=True)
            target = target.cuda(gpu, non_blocking=True)
            output = model(data).view(target.shape[0], -1)
            test_loss += F.cross_entropy(
                output, target, reduction="sum"
            ).item()  # sum up batch loss

            # cross entropy
            score, pred = torch.max(output, 1)
            # bianry
            # pred = (output > 0.5).float()

            # accuracy
            correct += pred.eq(target.view_as(pred)).sum().item()

            if f1_true == None:
                f1_true = target
            else:
                f1_true = torch.cat([f1_true, target], dim=0)

            if f1_pred == None:
                f1_pred = pred
            else:
                f1_pred = torch.cat([f1_pred, pred], dim=0)

            if out_score == None:
                out_score = output
            else:
                out_score = torch.cat([out_score, output], dim=0)

    # compute AUC
    label_binary = label_binarize(f1_true.detach().cpu().numpy(), classes=[0, 1, 2])

    label_binary = label_binary[:, :n_classes]
    out_score = out_score.detach().cpu().numpy()

    # Compute ROC curve and ROC area for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(label_binary[:, i], out_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Compute micro-average ROC curve and ROC area
    fpr["micro"], tpr["micro"], _ = roc_curve(label_binary.ravel(), out_score.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

    # First aggregate all false positive rates
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))

    # Then interpolate all ROC curves at this points
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(n_classes):
        mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])

    # Finally average it and compute AUC
    mean_tpr /= n_classes

    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

    # print results
    print("macro AUC: {}, micro AUC: {}".format(roc_auc["macro"], roc_auc["micro"]))
    for i in range(n_classes):
        print("class {} AUC: {}".format(i, roc_auc[i]))

    # f1 score
    f1 = f1_score(f1_true.cpu(), f1_pred.cpu(), average="weighted")
    print(classification_report(f1_true.cpu(), f1_pred.cpu()))

    test_loss /= len(test_loader.dataset)

    return test_loss, correct, f1, roc_auc

def save_train(
    fullModel, optim, block_method, val_acc, epoch, n_gpus, is_wandb, is_best
):
    fullModel_state_dict = (
        fullModel.module.state_dict() if n_gpus > 1 else fullModel.state_dict()
    )
    checkpoint = {
        "state_dict": fullModel_state_dict,
        "block_method": block_method,
        "epoch": epoch,
        "optim": optim,
    }
    # save model
    filename = "%s-%s-val_acc_%.2f-epoch_%d.pt" % (
        "train",
        block_method,
        100 * val_acc,
        epoch,
    )
    torch.save(checkpoint, os.path.join("models", filename))
    if is_wandb:
        # save model to wandb
        torch.save(checkpoint, os.path.join(wandb.run.dir, filename))
    if is_best:
        torch.save(
            checkpoint,
            os.path.join("models", "%s-%s-best.pt" % ("train", block_method)),
        )

def main():
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
    train_id = "genesis_train"
    
    train_worker(
         0, # gpu
        1, # n_gpus
        batch_size, # batch_size
        4, # num_worker
        10, # num_epochs
        'genesis', # block_method
        False, # base_trainable
        'pretrained_weights/Genesis_Chest_CT.pt', # pretrain_path
        None, # load_path
        os.path.join("data2", 'adni', "adni_144_148_176"), # dataset_path
        False, # is_wandb
        train_id,# exp_id
        -1, # base_trainable_after_n_epoch -1 is not trainable if > 0, make trainable after num, even base_trainable is false
        mci, # MCI
        False, # y_aware_test
        False, # densenet,
        n_class, # number of classes
        True, # legacy top model
    )


if __name__ == "__main__":
    main()