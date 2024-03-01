import numpy as np
import pandas as pd
import torch
import torch.cuda
from tqdm import tqdm


def get_ref_tar(df1, df2, num):
    tensor1 = torch.tensor(df1[df1['i'] == 2].values,dtype=torch.float16)
    print(tensor1)
    print(tensor1.shape)
    tensor2 = torch.tensor(df2[df2['i'] == 2].values,dtype=torch.float16)
    print(tensor2)
    print(tensor2.shape)


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
    int_columns = torch.cat([torch.floor_divide(result_tensor[:, 0:2], 8) , torch.floor_divide(result_tensor[:, 5:7], 8)], dim=1)


    # 转换为 Pandas DataFrame
    df_int = pd.DataFrame(int_columns.numpy().astype('int32'), columns=['col1', 'col2', 'col3', 'col4'])


    # 保存为 CSV 文件
    df_int.to_csv('../stylize/warp_test/0_50.csv', index=False)

    # 读取CSV文件
    df = pd.read_csv("../stylize/warp_test/0_50.csv")

    # 找到前两列完全相同的行
    unique_rows = df[df.duplicated(subset=['col1', 'col2'], keep=False)].drop_duplicates(subset=['col1', 'col2'], keep='first')

    # 计算平均值并替换后两列的值
    unique_rows['col3'] = unique_rows.groupby(['col1', 'col2'])['col3'].transform('mean')
    unique_rows['col4'] = unique_rows.groupby(['col1', 'col2'])['col4'].transform('mean')

    # 保存结果到新的CSV文件
    unique_rows.to_csv(f"../stylize/warp_test/0_{num}_mean_body.csv", index=False)


# 读取两个CSV文件
for i in range(1, 16):
    num = 50 + i
    df1 = pd.read_csv(f'/raid/cvg_data/ECCV2024/code/vid2densepose/csv/000000{50+i}.csv')
    df2 = pd.read_csv('/raid/cvg_data/ECCV2024/code/vid2densepose/csv/00000000.csv')
    get_ref_tar(df1, df2, num)