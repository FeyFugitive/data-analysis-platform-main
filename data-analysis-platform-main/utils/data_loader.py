#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Data Loading Module
Supports loading Excel and CSV files
"""

import pandas as pd
import os
from typing import Optional, Union
import logging

class DataLoader:
    """Data Loader"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
    def load_data(self, file_path: str, sheet_name: Union[str, int] = 0) -> Optional[pd.DataFrame]:
        """
        Load data file
        
        Args:
            file_path: File path
            sheet_name: Worksheet name or index (Excel files)
            
        Returns:
            pd.DataFrame: Loaded data, returns None on failure
        """
        try:
            if not os.path.exists(file_path):
                self.logger.error(f"File does not exist: {file_path}")
                return None
            
            file_extension = os.path.splitext(file_path)[1].lower()
            
            if file_extension in ['.xlsx', '.xls']:
                return self._load_excel(file_path, sheet_name)
            elif file_extension == '.csv':
                return self._load_csv(file_path)
            else:
                self.logger.error(f"Unsupported file format: {file_extension}")
                return None
                
        except Exception as e:
            self.logger.error(f"Data loading failed: {str(e)}")
            return None
    
    def _load_excel(self, file_path: str, sheet_name: Union[str, int] = 0) -> Optional[pd.DataFrame]:
        """加载Excel文件"""
        try:
            # 尝试读取指定的工作表
            data = pd.read_excel(file_path, sheet_name=sheet_name)
            
            # 检查返回的数据类型
            if isinstance(data, dict):
                # 如果返回的是字典，说明有多个工作表，取第一个
                if sheet_name in data:
                    data = data[sheet_name]
                else:
                    # 取第一个工作表
                    first_sheet = list(data.keys())[0]
                    data = data[first_sheet]
                    self.logger.warning(f"工作表 '{sheet_name}' 不存在，使用工作表: {first_sheet}")
            
            if data is not None and hasattr(data, 'shape'):
                self.logger.info(f"成功加载Excel文件: {file_path}, 工作表: {sheet_name}, 形状: {data.shape}")
                return data
            else:
                self.logger.error(f"无法读取Excel文件的工作表: {sheet_name}")
                return None
                
        except Exception as e:
            self.logger.error(f"加载Excel文件失败: {str(e)}")
            return None
    
    def _load_csv(self, file_path: str) -> Optional[pd.DataFrame]:
        """加载CSV文件"""
        # 尝试不同的编码格式
        encodings = ['utf-8', 'gbk', 'gb2312', 'latin-1', 'cp1252']
        
        for encoding in encodings:
            try:
                data = pd.read_csv(file_path, encoding=encoding)
                self.logger.info(f"成功加载CSV文件: {file_path}, 编码: {encoding}, 形状: {data.shape}")
                return data
            except UnicodeDecodeError:
                continue
            except Exception as e:
                self.logger.warning(f"使用编码 {encoding} 加载CSV失败: {str(e)}")
                continue
        
        self.logger.error(f"无法使用任何编码格式加载CSV文件: {file_path}")
        return None
    
    def get_sheet_names(self, file_path: str) -> list:
        """
        获取Excel文件的工作表名称列表
        
        Args:
            file_path: Excel文件路径
            
        Returns:
            list: 工作表名称列表
        """
        try:
            if not file_path.lower().endswith(('.xlsx', '.xls')):
                return []
            
            excel_file = pd.ExcelFile(file_path)
            return excel_file.sheet_names
            
        except Exception as e:
            self.logger.error(f"获取工作表名称失败: {str(e)}")
            return []
    
    def validate_data(self, data: pd.DataFrame) -> bool:
        """
        验证数据有效性
        
        Args:
            data: 数据表
            
        Returns:
            bool: 数据是否有效
        """
        if data is None or data.empty:
            self.logger.error("数据为空")
            return False
        
        if len(data.columns) == 0:
            self.logger.error("数据没有列")
            return False
        
        self.logger.info(f"数据验证通过: {data.shape}")
        return True 