# 算法执行器模型接口说明

## 概述

本文档详细描述了智能数据分析系统中算法执行器(`AlgorithmExecutor`)包含的所有模型接口。系统目前包含7个稳定的模型，每个模型都具备完善的数据验证和清洗功能。

---

## 模型列表

1. [描述统计](#1-描述统计-descriptive-statistics)
2. [最后点击归因](#2-最后点击归因-last-click-attribution)
3. [首次点击归因](#3-首次点击归因-first-click-attribution)
4. [Markov渠道模型](#4-markov渠道模型-markov-channel-model)
5. [Markov吸收链模型](#5-markov吸收链模型-markov-absorption-model)
6. [多维度归因分析](#6-多维度归因分析-multi-dimension-attribution)
7. [相关性分析](#7-相关性分析-correlation-analysis)
8. [时间趋势分析](#8-时间趋势分析-time-trend-analysis)
9. [分类分析](#9-分类分析-categorical-analysis)

---

## 1. 描述统计 (Descriptive Statistics)

### 功能描述
- 对数据进行基础统计分析，包括数据概览、数值列统计、分类列统计
- 提供数据质量评估和基础特征描述
- 适用于任何类型的数据集，无需特定字段要求

### 输入接口
```python
{
    'data': pd.DataFrame,  # 原始数据表
    'params': {}  # 无需额外参数
}
```

### 输出接口
```python
{
    'basic_stats': {
        'total_records': int,        # 总记录数
        'total_columns': int,        # 总列数
        'memory_usage_mb': float,    # 内存使用量(MB)
        'duplicate_records': int,    # 重复记录数
        'missing_values_total': int  # 总缺失值数
    },
    'numerical_stats': {             # 数值列统计
        'column_name': {
            'count': int,            # 非空值数量
            'mean': float,           # 均值
            'std': float,            # 标准差
            'min': float,            # 最小值
            'max': float,            # 最大值
            'median': float,         # 中位数
            'q25': float,            # 25%分位数
            'q75': float             # 75%分位数
        }
    },
    'categorical_stats': {           # 分类列统计
        'column_name': {
            'unique_count': int,     # 唯一值数量
            'most_common': str,      # 最常见值
            'most_common_count': int, # 最常见值出现次数
            'top_5_values': dict     # 前5个值及其频次
        }
    },
    'data_cleaning_info': dict,      # 数据清洗信息
    'data_warnings': list           # 数据警告信息
}
```

---

## 2. 最后点击归因 (Last Click Attribution)

### 功能描述
- 将用户转化归因给用户路径中的最后一个接触渠道
- 适用于评估渠道的直接转化效果
- 基于用户行为路径的时间序列分析

### 输入接口
```python
{
    'data': pd.DataFrame,  # 原始数据表
    'params': {
        'user_id_column': str,      # 用户ID列名
        'channel_column': str,      # 渠道列名
        'event_time_column': str    # 事件时间列名
    }
}
```

### 输出接口
```python
{
    'attribution_method': 'last_click',
    'total_users': int,             # 总用户数
    'channel_attribution': {         # 渠道归因结果
        'channel_name': int         # 渠道名: 归因次数
    },
    'conversion_rates': {           # 渠道转化率
        'channel_name': float       # 渠道名: 转化率
    },
    'top_channels': [               # 前5个渠道
        ('channel_name', int)       # (渠道名, 归因次数)
    ],
    'data_cleaning_info': {         # 数据清洗信息
        'original_shape': tuple,    # 原始数据形状
        'cleaned_shape': tuple,     # 清洗后数据形状
        'removed_rows': int,        # 删除的行数
        'removed_missing': int,     # 删除的缺失值数
        'unique_users': int,        # 唯一用户数
        'unique_channels': int      # 唯一渠道数
    },
    'data_warnings': list          # 数据警告信息
}
```

---

## 3. 首次点击归因 (First Click Attribution)

### 功能描述
- 将用户转化归因给用户路径中的第一个接触渠道
- 适用于评估渠道的获客效果
- 识别用户首次接触的渠道

### 输入接口
```python
{
    'data': pd.DataFrame,  # 原始数据表
    'params': {
        'user_id_column': str,      # 用户ID列名
        'channel_column': str,      # 渠道列名
        'event_time_column': str    # 事件时间列名
    }
}
```

### 输出接口
```python
{
    'attribution_method': 'first_click',
    'total_users': int,             # 总用户数
    'channel_attribution': {         # 渠道归因结果
        'channel_name': int         # 渠道名: 归因次数
    },
    'conversion_rates': {           # 渠道转化率
        'channel_name': float       # 渠道名: 转化率
    },
    'top_channels': [               # 前5个渠道
        ('channel_name', int)       # (渠道名, 归因次数)
    ],
    'data_cleaning_info': dict,     # 数据清洗信息
    'data_warnings': list          # 数据警告信息
}
```

---

## 4. Markov渠道模型 (Markov Channel Model)

### 功能描述
- 基于马尔可夫链的渠道归因分析
- 计算渠道间的转移概率和移除效果
- 适用于复杂的多渠道归因场景
- 考虑渠道间的相互影响

### 输入接口
```python
{
    'data': pd.DataFrame,  # 原始数据表
    'params': {
        'user_id_column': str,      # 用户ID列名
        'channel_column': str,      # 渠道列名
        'datetime_columns': list    # 时间列列表(可选)
    }
}
```

### 输出接口
```python
{
    'attribution_method': 'markov_channel',
    'total_users': int,             # 总用户数
    'unique_channels': int,         # 唯一渠道数
    'transition_matrix_shape': tuple, # 转移矩阵形状
    'removal_effects': {            # 移除效果
        'state_0': float,           # 状态0的移除效果(%)
        'state_1': float            # 状态1的移除效果(%)
    },
    'top_removal_effects': [        # 前5个移除效果
        ('state_name', float)       # (状态名, 移除效果)
    ],
    'data_cleaning_info': dict,     # 数据清洗信息
    'data_warnings': list          # 数据警告信息
}
```

---

## 5. Markov吸收链模型 (Markov Absorption Model)

### 功能描述
- 基于马尔可夫吸收链的归因分析
- 计算用户从初始状态到吸收状态（转化/不转化）的概率
- 分析用户路径的吸收时间和转化概率
- 适用于评估渠道的长期转化效果

### 输入接口
```python
{
    'data': pd.DataFrame,  # 原始数据表
    'params': {
        'user_id_column': str,      # 用户ID列名
        'channel_column': str,      # 渠道列名
        'datetime_columns': list    # 时间列列表(可选)
    }
}
```

### 输出接口
```python
{
    'attribution_method': 'markov_absorption',
    'total_users': int,             # 总用户数
    'unique_channels': int,         # 唯一渠道数
    'absorption_matrix_shape': tuple, # 吸收矩阵形状
    'absorption_probabilities': {   # 吸收概率
        'Conversion': float,        # 转化概率
        'No_Conversion': float,     # 不转化概率
        'channel_name': float       # 各渠道的吸收概率
    },
    'average_absorption_time': float, # 平均吸收时间
    'top_absorption_states': [      # 前5个吸收状态
        ('state_name', float)       # (状态名, 吸收概率)
    ],
    'data_cleaning_info': dict,     # 数据清洗信息
    'data_warnings': list          # 数据警告信息
}
```

---

## 6. 多维度归因分析 (Multi-Dimension Attribution)

### 功能描述
- 基于多个时间维度的归因分析
- 分析不同时间节点下的渠道分布和转化效果
- 提供更全面的归因视角
- 支持多时间维度的综合分析

### 输入接口
```python
{
    'data': pd.DataFrame,  # 原始数据表
    'params': {
        'user_id_column': str,      # 用户ID列名
        'channel_column': str,      # 渠道列名
        'datetime_columns': list    # 时间列列表
    }
}
```

### 输出接口
```python
{
    'attribution_method': 'multi_dimension',
    'total_users': int,             # 总用户数
    'dimensions_analyzed': int,     # 分析的维度数
    'dimension_results': {          # 各维度分析结果
        'datetime_column': {        # 时间列名
            'total_records': int,   # 总记录数
            'unique_users': int,    # 唯一用户数
            'unique_channels': int, # 唯一渠道数
            'channel_distribution': { # 渠道分布
                'channel_name': int # 渠道名: 记录数
            }
        }
    },
    'data_cleaning_info': {         # 数据清洗信息
        'original_shape': tuple,    # 原始数据形状
        'cleaned_shape': tuple,     # 清洗后数据形状
        'removed_rows': int,        # 删除的行数
        'removed_missing': int,     # 删除的缺失值数
        'unique_users': int,        # 唯一用户数
        'unique_channels': int,     # 唯一渠道数
        'available_datetime_columns': list, # 可用的时间列
        'total_datetime_columns': int       # 总时间列数
    },
    'data_warnings': list          # 数据警告信息
}
```

---

## 6. 相关性分析 (Correlation Analysis)

### 功能描述
- 分析数值变量间的相关性关系
- 识别强相关变量对，帮助理解数据特征
- 支持Pearson相关系数计算
- 自动识别强相关关系(相关系数≥0.7)

### 输入接口
```python
{
    'data': pd.DataFrame,  # 原始数据表
    'params': {
        'columns': list    # 需要分析的数值列名列表
    }
}
```

### 输出接口
```python
{
    'correlation_matrix': dict,     # 相关系数矩阵
    'strong_correlations': [        # 强相关关系列表
        {
            'variable1': str,       # 变量1名称
            'variable2': str,       # 变量2名称
            'correlation': float,   # 相关系数
            'strength': str         # 相关强度('strong'/'moderate')
        }
    ],
    'total_correlations': int,      # 强相关关系总数
    'data_cleaning_info': {         # 数据清洗信息
        'original_shape': tuple,    # 原始数据形状
        'cleaned_shape': tuple,     # 清洗后数据形状
        'removed_rows': int,        # 删除的行数
        'removed_missing': int,     # 删除的缺失值数
        'available_columns': list   # 可用的数值列
    },
    'data_warnings': list          # 数据警告信息
}
```

---

## 7. 时间趋势分析 (Time Trend Analysis)

### 功能描述
- 分析数值变量在不同时间维度上的变化趋势
- 识别时间序列模式和周期性特征
- 计算线性趋势的统计指标
- 支持多个时间维度的并行分析

### 输入接口
```python
{
    'data': pd.DataFrame,  # 原始数据表
    'params': {
        'datetime_columns': list,   # 日期时间列列表
        'numeric_columns': list     # 数值列列表
    }
}
```

### 输出接口
```python
{
    'datetime_column': {            # 时间列名
        'time_stats': {             # 时间统计
            'start_date': str,      # 开始日期
            'end_date': str,        # 结束日期
            'total_days': int,      # 总天数
            'records_per_day': float # 日均记录数
        },
        'numeric_trends': {         # 数值列趋势
            'numeric_column': {     # 数值列名
                'slope': float,     # 趋势斜率
                'intercept': float, # 截距
                'r_squared': float, # 决定系数
                'p_value': float,   # P值
                'trend_direction': str,  # 趋势方向('increasing'/'decreasing')
                'trend_strength': str    # 趋势强度('strong'/'moderate'/'weak')
            }
        }
    },
    'data_cleaning_info': {         # 数据清洗信息
        'available_datetime_columns': list, # 可用的时间列
        'available_numeric_columns': list,  # 可用的数值列
        'total_datetime_columns': int,      # 总时间列数
        'total_numeric_columns': int,       # 总数值列数
        'processed_columns': int            # 处理的列数
    },
    'data_warnings': list          # 数据警告信息
}
```

---

## 8. 分类分析 (Categorical Analysis)

### 功能描述
- 分析分类变量的分布特征和统计信息
- 识别主要分类和多样性指标
- 评估分类数据的完整性和一致性
- 适用于任何包含分类字段的数据集

### 输入接口
```python
{
    'data': pd.DataFrame,  # 原始数据表
    'params': {
        'categorical_columns': list  # 需要分析的分类列名列表
    }
}
```

### 输出接口
```python
{
    'column_name': {                # 分类列名
        'total_records': int,       # 总记录数
        'unique_categories': int,   # 唯一分类数
        'missing_count': int,       # 缺失值数量
        'missing_rate': float,      # 缺失率
        'diversity_index': float,   # 多样性指标
        'value_counts': dict,       # 各分类的频次统计
        'proportions': dict,        # 各分类的占比
        'major_categories': dict,   # 主要分类(占比>5%)
        'top_5_categories': dict    # 前5个分类
    },
    'data_cleaning_info': dict,     # 数据清洗信息
    'data_warnings': list          # 数据警告信息
}
```

---

## 通用输出字段

所有模型都会在结果中包含以下通用字段：

```python
{
    'algorithm_name': str,          # 算法名称
    'execution_timestamp': str,     # 执行时间戳
    'data_cleaning_info': dict,     # 数据清洗统计信息
    'data_warnings': list          # 数据质量警告信息
}
```

## 错误处理

当模型执行失败时，返回：

```python
{
    'error': str,                   # 错误信息
    'algorithm_name': str,          # 算法名称
    'execution_timestamp': str      # 执行时间戳
}
```

## 数据验证和清洗

每个模型都包含专门的数据验证和清洗功能：

### 验证内容
- 必需列的存在性检查
- 数据类型验证
- 数据量验证
- 缺失值处理

### 清洗策略
- 自动删除缺失值
- 数据类型转换
- 异常值处理
- 数据质量统计

### 清洗信息
每个模型都会返回详细的数据清洗统计信息，包括：
- 原始数据形状
- 清洗后数据形状
- 删除的行数和缺失值数
- 可用的列信息
- 数据质量警告

---

## 使用示例

```python
from core.algorithm_executor import AlgorithmExecutor

# 创建算法执行器
executor = AlgorithmExecutor()

# 执行最后点击归因
params = {
    'user_id_column': 'order_number',
    'channel_column': 'big_channel_name',
    'event_time_column': 'order_create_time'
}

result = executor.execute_algorithm('最后点击归因', data, params)

# 检查结果
if 'error' not in result:
    print(f"总用户数: {result['total_users']}")
    print(f"渠道归因: {result['channel_attribution']}")
    print(f"数据清洗信息: {result['data_cleaning_info']}")
else:
    print(f"执行失败: {result['error']}")
```

---

## 版本信息

- **文档版本**: 2.0
- **系统版本**: 智能数据分析系统 v2.0
- **更新时间**: 2025年8月20日
- **模型数量**: 9个稳定模型
- **支持格式**: Excel(.xlsx, .xls), CSV(.csv) 