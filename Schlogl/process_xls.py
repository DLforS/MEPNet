#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 23 14:54:54 2024

@author: yangwuyue
"""
import pandas as pd

# 加载min_max_values_1e4.txt以确定共同的x范围
min_max_values_path = 'min_max_values_1e4.txt'
min_max_values_df = pd.read_csv(min_max_values_path, header=None, sep='\s+', names=['min', 'max'])

# 确定共同的x范围
common_x_min = min_max_values_df['min'].min()
common_x_max = min_max_values_df['max'].max()

# 生成共同的x范围列表
common_x_range = list(range(int(common_x_min), int(common_x_max) + 1))

# 加载all_histogram_values_1e4.txt
histogram_values_path = 'all_histogram_values_1e4.txt'
histogram_values_df = pd.read_csv(histogram_values_path, header=None, sep='\s+')

# 初始化一个新的DataFrame，用于存储重新处理的数据
processed_histogram_df = pd.DataFrame(columns=common_x_range)

# 遍历每一行，将数据映射到共同的x范围
for index, row in histogram_values_df.iterrows():
    # 获取当前行的最小和最大x值
    row_min = min_max_values_df.loc[index, 'min']
    row_max = min_max_values_df.loc[index, 'max']
    
    # 生成当前行的x范围列表
    row_x_range = list(range(int(row_min), int(row_max) + 1))
    
    # 创建一个临时的字典，用于存储当前行的数据，映射到共同的x范围
    temp_dict = {x: 0 for x in common_x_range}  # 初始化所有共同x范围的值为0
    
    # 更新当前行的数据到临时字典
    for x, value in zip(row_x_range, row):
        temp_dict[x] = value
    
    # 将临时字典添加到processed_histogram_df中
    processed_histogram_df = processed_histogram_df.append(temp_dict, ignore_index=True)

# 将处理后的数据保存到CSV文件中
processed_histogram_path = 'processed_histogram_values_1e4.csv'
processed_histogram_df.to_csv(processed_histogram_path, index=False)


