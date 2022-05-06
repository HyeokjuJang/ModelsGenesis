import torch
import os
import numpy as np
import math
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F


class MRI_AD_CN_Dataset_from_file_list(Dataset):
    def __init__(self, X, y, resize=None):
        self.x_data_file_list = X
        self.y_data = y
        self.resize = resize

    def __len__(self):
        return len(self.x_data_file_list)

    def __getitem__(self, idx):
        # mgz read
        # img = nib.load(self.x_data_file_list[idx])
        # x = torch.tensor(img.get_fdata(), dtype=torch.float32).unsqueeze(0) / 256

        # npy read
        img = np.load(self.x_data_file_list[idx])
        img = ((img - img.min()) / (img.max() - img.min())).squeeze()
        x = torch.tensor(img, dtype=torch.float32).unsqueeze(0)
        y = torch.tensor(self.y_data[idx], dtype=torch.int64).type(torch.LongTensor)

        if self.resize is not None:
            x = resize_3d_tensor(x.unsqueeze(0), out_cube=self.resize).squeeze(0)
        return x, y


class MRI_AD_CN_Dataset(Dataset):
    def __init__(self, ad_path, cn_path, resize=None):
        # mgz list
        # l = os.listdir(ad_path)
        # self.ad_file_list = [
        #     os.path.join(ad_path, f, "mri", "norm.talairach.mgz") for f in l
        # ]
        # l = os.listdir(cn_path)
        # self.cn_file_list = [
        #     os.path.join(cn_path, f, "mri", "norm.talairach.mgz") for f in l
        # ]
        # self.x_data_file_list = [*self.ad_file_list, *self.cn_file_list]

        # npy list
        ad_list = os.listdir(ad_path)
        cn_list = os.listdir(cn_path)
        self.ad_file_list = [os.path.join(ad_path, file) for file in ad_list]
        self.cn_file_list = [os.path.join(cn_path, file) for file in cn_list]
        self.x_data_file_list = [*self.ad_file_list, *self.cn_file_list]

        self.y_data = [
            *(1 for _ in range(len(self.ad_file_list))),
            *(0 for _ in range(len(self.cn_file_list))),
        ]
        self.resize = resize

    def __len__(self):
        return len(self.x_data_file_list)

    def __getitem__(self, idx):
        # mgz read
        # img = nib.load(self.x_data_file_list[idx])
        # x = torch.tensor(img.get_fdata(), dtype=torch.float32).unsqueeze(0) / 256

        # npy read
        img = np.load(self.x_data_file_list[idx])
        img = ((img - img.min()) / (img.max() - img.min())).squeeze()
        x = torch.tensor(img, dtype=torch.float32).unsqueeze(0)
        y = torch.tensor(self.y_data[idx], dtype=torch.int64).type(torch.LongTensor)

        if self.resize is not None:
            x = resize_3d_tensor(x.unsqueeze(0), out_cube=self.resize).squeeze(0)
        return x, y


class MRI_AD_CN_MCI_Dataset(Dataset):
    def __init__(self, ad_path, cn_path, mci_path, resize=None, n_classes=2):
        # mgz list
        # l = os.listdir(ad_path)
        # self.ad_file_list = [
        #     os.path.join(ad_path, f, "mri", "norm.talairach.mgz") for f in l
        # ]
        # l = os.listdir(cn_path)
        # self.cn_file_list = [
        #     os.path.join(cn_path, f, "mri", "norm.talairach.mgz") for f in l
        # ]
        # self.x_data_file_list = [*self.ad_file_list, *self.cn_file_list]

        # npy list
        ad_list = os.listdir(ad_path)
        cn_list = os.listdir(cn_path)
        mci_list = os.listdir(mci_path)
        self.ad_file_list = [os.path.join(ad_path, file) for file in ad_list]
        self.cn_file_list = [os.path.join(cn_path, file) for file in cn_list]
        self.mci_file_list = [os.path.join(mci_path, file) for file in mci_list]
        self.x_data_file_list = [
            *self.ad_file_list,
            *self.cn_file_list,
            *self.mci_file_list,
        ]
        mci_label = 1
        if n_classes == 3:
            mci_label = 2
        self.y_data = [
            *(1 for _ in range(len(self.ad_file_list))),
            *(0 for _ in range(len(self.cn_file_list))),
            *(mci_label for _ in range(len(self.mci_file_list))),
        ]
        self.resize = resize

    def __len__(self):
        return len(self.x_data_file_list)

    def __getitem__(self, idx):
        # mgz read
        # img = nib.load(self.x_data_file_list[idx])
        # x = torch.tensor(img.get_fdata(), dtype=torch.float32).unsqueeze(0) / 256

        # npy read
        img = np.load(self.x_data_file_list[idx])
        img = ((img - img.min()) / (img.max() - img.min())).squeeze()
        x = torch.tensor(img, dtype=torch.float32).unsqueeze(0)
        y = torch.tensor(self.y_data[idx], dtype=torch.int).type(torch.LongTensor)

        if self.resize is not None:
            x = resize_3d_tensor(x.unsqueeze(0), out_cube=self.resize).squeeze(0)
        return x, y


class MRI_AD_CN_Yaware_Dataset(Dataset):
    def __init__(
        self,
        ad_path,
        cn_path,
        fold=0,
        train=True,
        json_path="D:\\mri\\adni-1st-session_5foldCV_N100.json",
        resize=None,
    ):
        # read json
        import json

        with open(json_path, "r") as f:
            json_data = json.load(f)
        # npy list
        ad_list = os.listdir(ad_path[0])
        cn_list = os.listdir(cn_path[0])
        self.ad_file_list = [os.path.join(ad_path[0], file) for file in ad_list]
        self.cn_file_list = [os.path.join(cn_path[0], file) for file in cn_list]

        ad_list += os.listdir(ad_path[1])
        cn_list += os.listdir(cn_path[1])
        self.ad_file_list += [os.path.join(ad_path[1], file) for file in ad_list]
        self.cn_file_list += [os.path.join(cn_path[1], file) for file in cn_list]
        self.x_data_file_list = [*self.ad_file_list, *self.cn_file_list]

        self.y_data = [
            *(1 for _ in range(len(self.ad_file_list))),
            *(0 for _ in range(len(self.cn_file_list))),
        ]
        self.resize = resize

        # filter json list
        json_list = json_data["fold{}".format(fold)]
        train_list = json_list["train"]
        test_list = json_list["test"]
        x_data_p_id_list = []

        for x_data in self.x_data_file_list:
            x_data_p_id_list.append(
                "".join(x_data.split("ADNI_ADNI_")[1].split("_")[0:3])
            )

        tmp_x_data_filtered = []
        tmp_y_data_filtered = []

        if train:
            filter_list = train_list
        else:
            filter_list = test_list

        self.nono_list = []
        for t_data in filter_list:
            try:
                idx = x_data_p_id_list.index(t_data)
                tmp_x_data_filtered.append(self.x_data_file_list[idx])
                tmp_y_data_filtered.append(self.y_data[idx])
            except:
                self.nono_list.append(t_data)

        self.x_data_file_list = tmp_x_data_filtered
        self.y_data = tmp_y_data_filtered

    def __len__(self):
        return len(self.x_data_file_list)

    def __getitem__(self, idx):
        # mgz read
        # img = nib.load(self.x_data_file_list[idx])
        # x = torch.tensor(img.get_fdata(), dtype=torch.float32).unsqueeze(0) / 256

        # npy read
        img = np.load(self.x_data_file_list[idx])
        img = ((img - img.min()) / (img.max() - img.min())).squeeze()
        x = torch.tensor(img, dtype=torch.float32).unsqueeze(0)
        y = torch.tensor(self.y_data[idx], dtype=torch.int64).type(torch.LongTensor)

        if self.resize is not None:
            x = resize_3d_tensor(x.unsqueeze(0), out_cube=self.resize).squeeze(0)
        return x, y


class MRIDataset(Dataset):
    def __init__(self, datapath=None, resize=64, block_method="block"):
        if datapath is None:
            datapath = os.path.join(
                "/data2", "mri_norm_talairach", "mri_norm_talairach"
            )
        dir_list = os.listdir(datapath)
        # npy list
        self.x_data_file_list = [os.path.join(datapath, file) for file in dir_list]
        # mgz list
        # self.x_data_file_list = [
        #     os.path.join(datapath, _dir, file)
        #     for _dir in dir_list
        #     for file in os.listdir(os.path.join(datapath, _dir))
        #     if file.endswith(".mgz")
        # ]
        self.resize = resize
        self.block = block_method

    def __len__(self):
        return len(self.x_data_file_list)

    def __getitem__(self, idx):
        # npy load
        img = np.load(self.x_data_file_list[idx])
        x = ((img - img.min()) / (img.max() - img.min())).squeeze()

        # # mgz load
        # img = nib.load(self.x_data_file_list[idx])
        # x = img.get_fdata() / 255

        mask = None
        if self.block == "block":
            masked, mask = static_block_masking(np.copy(x))
            x_mask = torch.tensor(masked, dtype=torch.float32).unsqueeze(0)

        elif self.block == "random_block":
            masked, mask = random_block_masking(np.copy(x))
            x_mask = torch.tensor(masked, dtype=torch.float32).unsqueeze(0)

        elif self.block == "random_region":
            masked, mask = random_region_masking(np.copy(x))
            x_mask = torch.tensor(masked, dtype=torch.float32).unsqueeze(0)

        elif self.block == "guided":
            x_mask = torch.tensor(np.copy(x), dtype=torch.float32).unsqueeze(0)

        x = torch.tensor(x, dtype=torch.float32).unsqueeze(0)
        if mask is not None:
            mask = torch.tensor(mask, dtype=torch.float32).unsqueeze(0)

        if self.resize is not None and x.shape[2] != self.resize:
            x = resize_3d_tensor(x.unsqueeze(0), out_cube=self.resize).squeeze(0)
            mask = resize_3d_tensor(mask.unsqueeze(0), out_cube=self.resize).squeeze(0)
            if self.block is not None:
                x_mask = resize_3d_tensor(
                    x_mask.unsqueeze(0), out_cube=self.resize
                ).squeeze(0)

        if self.block is not None and mask is not None:
            return x, x_mask, mask
        elif self.block == "guided":
            return x, x_mask
        else:
            return x


def resize_3d_tensor(x, in_cube=256, out_cube=64):
    # only cube can resized
    N, C, H, W = 1, in_cube, in_cube, in_cube

    d = torch.linspace(-1, 1, out_cube)
    meshx, meshy, meshz = torch.meshgrid((d, d, d), indexing="ij")
    grid = torch.stack((meshx, meshy, meshz), 3)
    grid = grid.unsqueeze(0)  # add batch dim

    out = F.grid_sample(x, grid, align_corners=True)
    out = torch.transpose(out, 2, 4)
    return out


def static_block_masking(x, size=1 / 3):
    _x, _y, _z = x.shape
    if size == 1 / 2:
        _x_0 = math.floor(_x / 4)
        _x_1 = math.floor(3 * _x / 4)
        _x_2 = _x_1 - _x_0
        _y_0 = math.floor(_y / 4)
        _y_1 = math.floor(3 * _y / 4)
        _y_2 = _y_1 - _y_0
        _z_0 = math.floor(_z / 4)
        _z_1 = math.floor(3 * _z / 4)
        _z_2 = _z_1 - _z_0
    elif size == 1 / 3:
        _x_0 = math.floor(_x / 3)
        _x_1 = math.floor(2 * _x / 3)
        _x_2 = _x_1 - _x_0
        _y_0 = math.floor(_y / 3)
        _y_1 = math.floor(2 * _y / 3)
        _y_2 = _y_1 - _y_0
        _z_0 = math.floor(_z / 3)
        _z_1 = math.floor(2 * _z / 3)
        _z_2 = _z_1 - _z_0

    x[_x_0:_x_1, _y_0:_y_1, _z_0:_z_1] = np.zeros((_x_2, _y_2, _z_2))  # or ones
    mask = np.zeros_like(x)
    mask[_x_0:_x_1, _y_0:_y_1, _z_0:_z_1] = np.ones((_x_2, _y_2, _z_2))
    return x, mask


def random_block_masking(x, size=1 / 9):
    _x, _y, _z = x.shape
    threshold = (_x * _y * _z) * size / 3
    done = 0
    mask = np.zeros_like(x)
    while mask.sum() < threshold and done < 100:
        x_base, y_base, z_base = np.random.normal(0.45, 0.2, 3)
        _x_0 = np.clip(math.floor(_x * x_base), 0, _x - 1)
        _x_1 = np.clip(math.floor(_x * (size + x_base)), 0, _x - 1)
        _x_2 = _x_1 - _x_0
        _y_0 = np.clip(math.floor(_y * y_base), 0, _y - 1)
        _y_1 = np.clip(math.floor(_y * (size + y_base)), 0, _y - 1)
        _y_2 = _y_1 - _y_0
        _z_0 = np.clip(math.floor(_z * z_base), 0, _z - 1)
        _z_1 = np.clip(math.floor(_z * (size + z_base)), 0, _z - 1)
        _z_2 = _z_1 - _z_0

        done += 1

        # if brain area is too small, skip
        if (
            np.count_nonzero(x[_x_0:_x_1, _y_0:_y_1, _z_0:_z_1])
            < (_x_2 * _y_2 * _z_2) / 27
        ):
            continue
        else:
            mask[_x_0:_x_1, _y_0:_y_1, _z_0:_z_1] = np.ones(
                (_x_2, _y_2, _z_2)
            )  # or ones
    return x * (1.0 - mask), mask


def random_region_masking(x, size=1 / 27):
    _x, _y, _z = x.shape
    mask = np.zeros_like(x)
    x_base, y_base, z_base = np.random.normal(0.5, 0.2, 3)
    x_next, y_next, z_next = np.random.normal(0.0, 0.2, 3)
    x_next = x_base + (x_next)
    y_next = y_base + (y_next)
    z_next = y_base + (z_next)
    threshold = _x * _y * _z * (size)
    done = 0
    while mask.sum() < threshold and done < 100:
        if (
            x_base == x_next
            or y_base == y_next
            or z_base == z_next
            or x_base > 1
            or x_base < 0
            or y_base > 1
            or y_base < 0
            or z_base > 1
            or z_next < 0
        ):
            # prepare next step
            x_base, y_base, z_base = np.random.normal(0.5, 0.2, 3)
            x_next, y_next, z_next = np.random.normal(0.0, 0.1, 3)
            x_next = x_base + (x_next)
            y_next = y_base + (y_next)
            z_next = y_base + (z_next)
            continue
        _x_0 = np.clip(math.floor(_x * np.min((x_base, x_next))), 0, _x - 1)
        _x_1 = np.clip(math.floor(_x * np.max((x_base, x_next))), 0, _x - 1)
        _x_2 = _x_1 - _x_0
        _y_0 = np.clip(math.floor(_y * np.min((y_base, y_next))), 0, _y - 1)
        _y_1 = np.clip(math.floor(_y * np.max((y_base, y_next))), 0, _y - 1)
        _y_2 = _y_1 - _y_0
        _z_0 = np.clip(math.floor(_z * np.min((z_base, z_next))), 0, _z - 1)
        _z_1 = np.clip(math.floor(_z * np.max((z_base, z_next))), 0, _z - 1)
        _z_2 = _z_1 - _z_0

        done += 1

        # if brain area is too small, skip
        if (
            np.count_nonzero(x[_x_0:_x_1, _y_0:_y_1, _z_0:_z_1])
            < (_x_2 * _y_2 * _z_2) / 27
        ):
            x_base = -1  # for reset value
            continue
        else:
            mask[_x_0:_x_1, _y_0:_y_1, _z_0:_z_1] = np.ones((_x_2, _y_2, _z_2))

        # prepare next step
        x_base = x_next
        y_base = x_next
        z_base = x_next
        x_next, y_next, z_next = np.random.normal(0.0, 0.2, 3)
        x_next = x_base + (x_next)
        y_next = y_base + (y_next)
        z_next = y_base + (z_next)

    return x * (1.0 - mask), mask