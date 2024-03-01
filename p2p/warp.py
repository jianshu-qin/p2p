import numpy as np
import pandas as pd
import torch
from tqdm import tqdm


def read_csv_correspondence(fn):
    data = pd.read_csv(fn)
    ret = []
    for i, row in data.iterrows():
        ret.append([int(x) for x in row])
    return ret

def get_correspondence_from_densepose(self, x_t, df1="", df2="", labels=0):
    if not df1:
        df1 = pd.read_csv('/raid/cvg_data/ECCV2024/code/vid2densepose/csv/00000000.csv')
    if not df2:
        df2 = pd.read_csv('/raid/cvg_data/ECCV2024/code/vid2densepose/csv/00000050.csv')
    if not labels:
        labels = 2
    tensor1 = torch.tensor(df1[df1['i'] == labels].values,dtype=torch.float16)
    tensor2 = torch.tensor(df2[df2['i'] == labels].values,dtype=torch.float16)

    # 获取tensor形状
    shape1 = tensor1.shape
    shape2 = tensor2.shape

    # 初始化保存结果的tensor
    result_tensor = torch.zeros(shape1[0] , 11, dtype=torch.float16)

    # 计算距离并保存结果
    for i in tqdm(range(shape1[0]), desc="Processing", unit="row"):
        # 获取满足条件的selected_tensor
        selected_tensor = tensor2[tensor2[:, 2] == tensor1[i, 2], :]

        # 计算最小距离的索引
        distances = torch.norm(tensor1[i, 3:] - selected_tensor[:, 3:], dim=1)
        min_index = torch.argmin(distances).item()

        # 在selected_tensor中找到对应的索引
        corresponding_index_in_tensor2 = torch.nonzero(tensor2[:, 2] == tensor1[i, 2]).squeeze(1)[min_index].item()

        # 将结果保存到result_tensor
        result_tensor[i, :] = torch.cat((tensor1[i, :], tensor2[corresponding_index_in_tensor2, :], distances[min_index].unsqueeze(0)), dim=0)
    # 提取指定列
    int_columns = torch.cat([result_tensor[:, 0:2], result_tensor[:, 3:5]], dim=1)
    # cols: x1, y1, x2, y2
    return int_columns

if __name__ == "__main__":
    fn = "/home/jianshu/code/prompt_travel/stylize/warp_test/0_50_mean.csv"
    l = read_csv_correspondence(fn)
    print(l)