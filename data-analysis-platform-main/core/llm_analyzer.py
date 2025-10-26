#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LLM Intelligent Analysis Module - Dedicated for Attribution Analysis
LLM-based data understanding and attribution algorithm recommendation
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Tuple, Optional
import logging
from datetime import datetime
import os
import glob
import asyncio

# Import LLM client
from .llm_client import LLMClient, LLMAnalyzerClient

class ModelSchema:
    """Model Schema Definition"""
    
    def __init__(self, name: str, description: str, priority: int = 1):
        self.name = name
        self.description = description
        self.priority = priority
        self.required_fields = {}
        self.optional_fields = {}
        self.field_patterns = {}
        self.data_types = {}
    
    def add_required_field(self, field_name: str, patterns: List[str], data_types: List[str] = None):
        """Add required field"""
        self.required_fields[field_name] = patterns
        self.field_patterns[field_name] = patterns
        if data_types:
            self.data_types[field_name] = data_types
    
    def add_optional_field(self, field_name: str, patterns: List[str], data_types: List[str] = None):
        """Add optional field"""
        self.optional_fields[field_name] = patterns
        self.field_patterns[field_name] = patterns
        if data_types:
            self.data_types[field_name] = data_types
    
    def set_priority(self, priority: int):
        """Set priority"""
        self.priority = priority

class LLMAnalyzer:
    """LLM Intelligent Analyzer - Dedicated for Attribution Analysis"""
    
    def __init__(self, llm_client=None, enable_llm=False, llm_config=None):
        """
        Initialize LLM analyzer
        
        Args:
            llm_client: LLM client (optional, for actual LLM calls)
            enable_llm: Whether to enable LLM functionality
            llm_config: LLM configuration dictionary
        """
        self.enable_llm = enable_llm
        self.llm_client = llm_client
        self.llm_analyzer_client = None
        
        # If LLM is enabled and no client is provided, create default client
        if self.enable_llm and not self.llm_client:
            if llm_config:
                self.llm_client = LLMClient(llm_config)
            else:
                # Use default configuration
                self.llm_client = LLMClient()
            self.llm_analyzer_client = LLMAnalyzerClient(self.llm_client)
        
        self.logger = logging.getLogger(__name__)
        self.model_schemas = {}
        self._register_default_models()
    
    def _register_default_models(self):
        """Register default analysis model schemas"""
        # 1. Last click attribution model
        last_click_model = ModelSchema("last_click_attribution", "Last Click Attribution", priority=1)
        last_click_model.add_required_field("user_id", ["user_id", "userid", "customer_id", "customerid", "order_number"])
        last_click_model.add_required_field("channel", ["channel", "source", "medium", "campaign", "big_channel_name"])
        last_click_model.add_required_field("timestamp", ["timestamp", "event_time", "eventtime", "time", "create_time", "payment_time", "order_create_time"])
        last_click_model.add_optional_field("conversion", ["conversion", "purchase", "order", "status", "blind_lock_status"])
        self.model_schemas["last_click_attribution"] = last_click_model
        
        # 2. First click attribution model
        first_click_model = ModelSchema("first_click_attribution", "First Click Attribution", priority=1)
        first_click_model.add_required_field("user_id", ["user_id", "userid", "customer_id", "customerid", "order_number"])
        first_click_model.add_required_field("channel", ["channel", "source", "medium", "campaign", "big_channel_name"])
        first_click_model.add_required_field("timestamp", ["timestamp", "event_time", "eventtime", "time", "create_time", "payment_time", "order_create_time"])
        first_click_model.add_optional_field("conversion", ["conversion", "purchase", "order", "status", "blind_lock_status"])
        self.model_schemas["first_click_attribution"] = first_click_model
        
        # 3. Markov channel model
        markov_channel_model = ModelSchema("markov_channel_model", "Markov Channel Model", priority=1)
        markov_channel_model.add_required_field("user_id", ["user_id", "userid", "customer_id", "customerid", "order_number"])
        markov_channel_model.add_required_field("channel", ["channel", "source", "medium", "campaign", "big_channel_name"])
        markov_channel_model.add_required_field("timestamp", ["timestamp", "event_time", "eventtime", "time", "create_time", "payment_time", "order_create_time"])
        markov_channel_model.add_optional_field("conversion", ["conversion", "purchase", "order", "status", "blind_lock_status"])
        self.model_schemas["markov_channel_model"] = markov_channel_model
        
        # 4. Markov absorption chain model
        markov_absorption_model = ModelSchema("markov_absorption_model", "Markov Absorption Chain Model", priority=1)
        markov_absorption_model.add_required_field("user_id", ["user_id", "userid", "customer_id", "customerid", "order_number"])
        markov_absorption_model.add_required_field("channel", ["channel", "source", "medium", "campaign", "big_channel_name"])
        markov_absorption_model.add_required_field("timestamp", ["timestamp", "event_time", "eventtime", "time", "create_time", "payment_time", "order_create_time"])
        markov_absorption_model.add_optional_field("conversion", ["conversion", "purchase", "order", "status", "blind_lock_status"])
        self.model_schemas["markov_absorption_model"] = markov_absorption_model
        
        # 5. Correlation analysis model
        correlation_model = ModelSchema("correlation_analysis", "Correlation Analysis", priority=2)
        correlation_model.add_required_field("numeric_field", ["count", "amount", "number", "score", "rate", "current_retained_locked_count", "waiting_deposit_count", "waiting_lock_count", "current_period_refund_count", "retained_locked2delivery_count", "non_current_month_retained_locked_count", "locked_order_count", "delivered_locked_order_count", "lock_base_is_create_wechat_group", "is_create_wechat_group"])
        self.model_schemas["correlation_analysis"] = correlation_model
        
        # 6. Time series analysis model
        time_series_model = ModelSchema("time_series_analysis", "Time Series Analysis", priority=3)
        time_series_model.add_required_field("timestamp", ["timestamp", "event_time", "eventtime", "time", "时间", "日期", "create_time", "payment_time", "order_create_time"])
        time_series_model.add_required_field("numeric_field", ["count", "amount", "number", "score", "rate", "waiting_deposit_count"])
        self.model_schemas["time_series_analysis"] = time_series_model
        
        # 7. Categorical analysis model
        categorical_model = ModelSchema("categorical_analysis", "Categorical Analysis", priority=4)
        categorical_model.add_required_field("categorical_field", ["type", "category", "status", "hold_type", "user_type"])
        self.model_schemas["categorical_analysis"] = categorical_model
        
        # 8. Descriptive statistics model
        descriptive_model = ModelSchema("descriptive_statistics", "Descriptive Statistics", priority=5)
        # Descriptive statistics does not require specific fields
        self.model_schemas["descriptive_statistics"] = descriptive_model
        
        self.logger.info(f"Registered {len(self.model_schemas)} default model schemas")
    
    def _extract_data_schema(self, data: pd.DataFrame, schema_path: str = None) -> Dict[str, Any]:
        """Extract current data schema"""
        schema = {}
        field_mapping = self._load_field_mapping(schema_path)
        
        for column in data.columns:
            field_info = field_mapping.get(column, {})
            
            # Process sample values to ensure JSON serialization
            sample_values = data[column].head(3).tolist()
            serializable_samples = []
            for val in sample_values:
                if pd.isna(val):
                    serializable_samples.append(None)
                elif isinstance(val, pd.Timestamp):
                    serializable_samples.append(val.isoformat())
                else:
                    serializable_samples.append(str(val))
            
            schema[column] = {
                'data_type': field_info.get('data_type', ''),
                'comment': field_info.get('comment', ''),
                'column_name': column,
                'sample_values': serializable_samples,
                'data_quality_score': self._calculate_data_quality_score(data[column])
            }
        
        return schema
    
    def _calculate_data_quality_score(self, series: pd.Series) -> float:
        """Calculate data quality score"""
        try:
            # Missing rate
            missing_rate = series.isnull().sum() / len(series)
            
            # Unique value ratio
            unique_ratio = series.nunique() / len(series)
            
            # Data type consistency
            type_consistency = 1.0 if series.dtype in ['object', 'int64', 'float64', 'datetime64[ns]'] else 0.5
            
            # Non-null value count
            non_null_count = series.count()
            
            # Comprehensive score
            score = (1 - missing_rate) * 0.4 + min(unique_ratio, 1.0) * 0.3 + type_consistency * 0.2 + min(non_null_count / 1000, 1.0) * 0.1
            
            return round(score, 3)
        except:
            return 0.5
    
    def _calculate_match_score(self, column_name: str, field_info: Dict, patterns: List[str], sample_data: pd.Series = None) -> float:
        """计算字段匹配分数 - 增强版，结合语义和数据特征"""
        score = 0.0
        column_lower = column_name.lower()
        
        # 1. 语义相似性匹配 (40%)
        semantic_score = self._calculate_semantic_similarity(column_name, field_info, patterns)
        score += semantic_score * 0.4
        
        # 2. 数据特征匹配 (35%) - 如果有样本数据
        if sample_data is not None:
            data_feature_score = self._analyze_data_features(sample_data, patterns)
            score += data_feature_score * 0.35
        
        # 3. 传统模式匹配 (25%) - 作为兜底
        pattern_score = self._calculate_pattern_score(column_name, field_info, patterns)
        score += pattern_score * 0.25
        
        return min(score, 1.0)
    
    def _calculate_semantic_similarity(self, column_name: str, field_info: Dict, patterns: List[str]) -> float:
        """计算语义相似性"""
        score = 0.0
        
        # 字段名分词
        column_tokens = self._tokenize_field_name(column_name)
        
        # 与模式进行语义匹配
        for pattern in patterns:
            pattern_tokens = self._tokenize_field_name(pattern)
            similarity = self._calculate_token_similarity(column_tokens, pattern_tokens)
            score = max(score, similarity)
        
        # 注释语义匹配
        comment = field_info.get('comment', '')
        if comment:
            for pattern in patterns:
                if pattern.lower() in comment.lower():
                    score = max(score, 0.8)
        
        return score
    
    def _analyze_data_features(self, sample_data: pd.Series, patterns: List[str]) -> float:
        """分析数据特征匹配度"""
        score = 0.0
        
        # 数据分布特征
        uniqueness_ratio = sample_data.nunique() / len(sample_data)
        missing_ratio = sample_data.isnull().sum() / len(sample_data)
        
        # 根据模式推断期望的数据特征
        for pattern in patterns:
            pattern_lower = pattern.lower()
            
            # 用户ID模式
            if 'user' in pattern_lower and 'id' in pattern_lower:
                if uniqueness_ratio > 0.95:  # 接近唯一标识符
                    score = max(score, 0.8)
            
            # 时间模式
            elif any(time_word in pattern_lower for time_word in ['time', 'date', 'timestamp']):
                if pd.api.types.is_datetime64_any_dtype(sample_data):
                    score = max(score, 0.9)
            
            # 数值模式
            elif any(num_word in pattern_lower for num_word in ['count', 'amount', 'number', 'score', 'rate']):
                if pd.api.types.is_numeric_dtype(sample_data):
                    score = max(score, 0.8)
            
            # 分类模式
            elif any(cat_word in pattern_lower for cat_word in ['type', 'category', 'status', 'name']):
                if uniqueness_ratio < 0.1:  # 分类变量
                    score = max(score, 0.7)
        
        # 数据质量加分
        if missing_ratio < 0.1:
            score += 0.1
        
        return min(score, 1.0)
    
    def _calculate_pattern_score(self, column_name: str, field_info: Dict, patterns: List[str]) -> float:
        """传统模式匹配（兜底方法）"""
        score = 0.0
        column_lower = column_name.lower()
        
        # 1. 字段名模式匹配 (50%)
        for pattern in patterns:
            if pattern.lower() in column_lower:
                score += 0.5
                break
        
        # 2. 注释匹配 (30%)
        comment = field_info.get('comment', '').lower()
        for pattern in patterns:
            if pattern.lower() in comment:
                score += 0.3
                break
        
        # 3. 数据类型匹配 (20%)
        data_type = field_info.get('data_type', '').lower()
        if any(dt in data_type for dt in ['datetime', 'timestamp']) and any(dt in patterns for dt in ['time', 'date', 'timestamp']):
            score += 0.2
        elif any(dt in data_type for dt in ['int', 'bigint', 'float', 'decimal']) and any(dt in patterns for dt in ['count', 'amount', 'number', 'score', 'rate']):
            score += 0.2
        elif any(dt in data_type for dt in ['varchar', 'char', 'text']) and any(dt in patterns for dt in ['type', 'category', 'status', 'name']):
            score += 0.2
        
        return min(score, 1.0)
    
    def _tokenize_field_name(self, field_name: str) -> List[str]:
        """字段名分词"""
        import re
        
        # 分割下划线
        tokens = field_name.lower().split('_')
        
        # 处理驼峰命名
        camel_case_tokens = []
        for token in tokens:
            camel_case_tokens.extend(re.findall(r'[a-z]+|[A-Z][a-z]*', token))
        
        # 过滤空字符串
        tokens = [t for t in camel_case_tokens if t]
        
        return tokens
    
    def _calculate_token_similarity(self, tokens1: List[str], tokens2: List[str]) -> float:
        """计算token相似性"""
        if not tokens1 or not tokens2:
            return 0.0
        
        # 计算Jaccard相似性
        set1 = set(tokens1)
        set2 = set(tokens2)
        
        intersection = len(set1.intersection(set2))
        union = len(set1.union(set2))
        
        if union == 0:
            return 0.0
        
        return intersection / union
    
    def _find_multiple_numeric_matches(self, patterns: List[str], data_schema: Dict[str, Any], data: pd.DataFrame = None) -> List[str]:
        """查找多个数值字段匹配"""
        matches = []
        
        for column_name, field_info in data_schema.items():
            # 检查是否为数值类型
            data_type = field_info.get('data_type', '').lower()
            if 'bigint' in data_type or 'int' in data_type:
                # 获取样本数据用于智能匹配
                sample_data = data[column_name] if data is not None and column_name in data.columns else None
                # 计算匹配分数
                score = self._calculate_match_score(column_name, field_info, patterns, sample_data)
                if score > 0.3:  # 匹配阈值
                    matches.append(column_name)
        
        # 按匹配分数排序，返回前几个
        return matches[:5]  # 最多返回5个数值字段
    
    def _find_best_match(self, field_name: str, patterns: List[str], data_schema: Dict, data: pd.DataFrame = None) -> Optional[str]:
        """找到最佳匹配的字段"""
        best_match = None
        best_score = 0.0
        
        for column_name, field_info in data_schema.items():
            # 获取样本数据用于智能匹配
            sample_data = data[column_name] if data is not None and column_name in data.columns else None
            score = self._calculate_match_score(column_name, field_info, patterns, sample_data)
            if score > best_score:
                best_score = score
                best_match = column_name
        
        return best_match if best_score > 0.3 else None
    
    def match_models(self, data: pd.DataFrame, schema_path: str = None) -> List[Dict[str, Any]]:
        """匹配当前数据Schema与各模型Schema"""
        self.logger.info("开始模型匹配")
        
        # 提取当前数据Schema
        data_schema = self._extract_data_schema(data, schema_path)
        self.logger.info(f"当前数据Schema包含 {len(data_schema)} 个字段")
        
        results = []
        
        for model_name, model_schema in self.model_schemas.items():
            self.logger.info(f"对比模型: {model_name}")
            
            matched_fields = {}
            missing_fields = []
            matched_required = 0
            total_required = len(model_schema.required_fields)
            
            # 匹配必需字段
            for field_name, patterns in model_schema.required_fields.items():
                if field_name == 'numeric_field' and model_name == 'correlation_analysis':
                    # 对于相关性分析，需要找到多个数值字段
                    numeric_matches = self._find_multiple_numeric_matches(patterns, data_schema, data)
                    if len(numeric_matches) >= 2:  # 至少需要2个数值字段
                        matched_fields[field_name] = numeric_matches
                        matched_required += 1
                    else:
                        missing_fields.append(field_name)
                else:
                    best_match = self._find_best_match(field_name, patterns, data_schema, data)
                    if best_match:
                        matched_fields[field_name] = best_match
                        matched_required += 1
                    else:
                        missing_fields.append(field_name)
            
            # 匹配可选字段
            for field_name, patterns in model_schema.optional_fields.items():
                best_match = self._find_best_match(field_name, patterns, data_schema, data)
                if best_match:
                    matched_fields[field_name] = best_match
            
            # 计算匹配度
            executable = matched_required == total_required
            match_rate = matched_required / total_required if total_required > 0 else 1.0
            
            result = {
                'model_name': model_name,
                'model_description': model_schema.description,
                'priority': model_schema.priority,
                'executable': executable,
                'match_rate': match_rate,
                'field_mapping': matched_fields,
                'missing_fields': missing_fields
            }
            
            results.append(result)
            
            if executable:
                self.logger.info(f"模型 {model_name} 可执行，匹配度: {match_rate:.2%}")
                self.logger.info(f"字段映射: {matched_fields}")
            else:
                self.logger.info(f"模型 {model_name} 不可执行，缺失字段: {missing_fields}")
        
        # 按优先级和匹配度排序
        results.sort(key=lambda x: (x['priority'], -x['match_rate']))
        
        self.logger.info(f"模型匹配完成，共匹配 {len(results)} 个模型")
        return results
    
    def quick_examination(self, data: pd.DataFrame, sample_size: int = 20, schema_path: str = None) -> Dict[str, Any]:
        """
        快速体检（仅20行样本）
        
        Args:
            data: 输入数据
            sample_size: 样本大小
            
        Returns:
            Dict[str, Any]: 体检结果
        """
        try:
            self.logger.info(f"开始快速体检，样本大小: {sample_size}")
            
            # 取样本数据
            sample_data = data.head(sample_size)
            
            # 识别列类型
            column_types = self._identify_column_types(sample_data, schema_path)
            
            # 计算缺失率
            missing_rates = self._calculate_missing_rates(sample_data)
            
            # 计算可解析日期比例
            date_parse_rates = self._calculate_date_parse_rates(sample_data)
            
            # 识别归因分析关键字段
            attribution_fields = self._identify_attribution_fields(sample_data, column_types, schema_path)
            
            examination_result = {
                'sample_size': sample_size,
                'column_types': column_types,
                'missing_rates': missing_rates,
                'date_parse_rates': date_parse_rates,
                'attribution_fields': attribution_fields,
                'examination_timestamp': datetime.now().isoformat()
            }
            
            self.logger.info("快速体检完成")
            return examination_result
            
        except Exception as e:
            self.logger.error(f"快速体检失败: {str(e)}")
            return {'error': str(e)}
    
    def analyze_data_structure(self, data: pd.DataFrame, schema_path: str = None) -> Dict[str, Any]:
        """
        分析数据结构（兼容方法）
        
        Args:
            data: 输入数据
            schema_path: 字段说明文件路径
            
        Returns:
            Dict[str, Any]: 分析结果
        """
        # 调用快速体检方法
        return self.quick_examination(data, schema_path=schema_path)
    
    def _identify_column_types(self, data: pd.DataFrame, schema_path: str = None) -> Dict[str, str]:
        """识别列类型 - 基于字段说明文档"""
        column_types = {}
        
        # 尝试加载字段说明文档
        field_mapping = self._load_field_mapping(schema_path)
        
        for col in data.columns:
            col_lower = col.lower()
            
            # 基于字段说明文档进行识别
            if col in field_mapping:
                field_info = field_mapping[col]
                # 传入样本数据进行动态识别
                column_types[col] = self._map_field_to_type(field_info, data[col])
            else:
                # 如果字段说明文档中没有，使用原有的模式匹配作为备选
                column_types[col] = self._fallback_column_type_identification(col, data[col])
        
        return column_types
    
    def _load_field_mapping(self, schema_path: str = None) -> Dict[str, Dict[str, str]]:
        """加载字段说明文档"""
        field_mapping = {}
        
        try:
            # 确定字段说明文件路径
            field_desc_path = self._find_schema_file(schema_path)
            
            if field_desc_path and os.path.exists(field_desc_path):
                field_df = pd.read_excel(field_desc_path)
                
                for _, row in field_df.iterrows():
                    column_name = row['Column Name']
                    data_type = row['Data Type']
                    comment = row['Comment']
                    
                    field_mapping[column_name] = {
                        'column_name': column_name,  # 添加column_name字段
                        'data_type': data_type,
                        'comment': comment
                    }
                
                self.logger.info(f"成功加载字段说明文档: {field_desc_path}，共{len(field_mapping)}个字段")
            else:
                self.logger.warning("未找到字段说明文档，将使用备选识别方法")
                
        except Exception as e:
            self.logger.warning(f"加载字段说明文档失败: {str(e)}，将使用备选识别方法")
        
        return field_mapping
    
    def _find_schema_file(self, schema_path: str = None) -> str:
        """
        智能查找字段说明文件
        
        Args:
            schema_path: 用户指定的schema文件路径
            
        Returns:
            str: 找到的schema文件路径，如果没找到则返回None
        """
        # 如果用户明确指定了schema文件路径，直接使用
        if schema_path:
            if os.path.exists(schema_path):
                return schema_path
            else:
                self.logger.warning(f"指定的schema文件不存在: {schema_path}")
                return None
        
        # 如果没有指定，智能查找可能的schema文件
        search_directories = [
            'data',           # 当前目录下的data文件夹
            '.',              # 当前目录
            '../data',        # 上级目录的data文件夹
            '..',             # 上级目录
        ]
        
        # 可能的文件名模式
        schema_patterns = [
            '*字段说明*.xlsx',
            '*field*description*.xlsx', 
            '*schema*.xlsx',
            '*metadata*.xlsx',
            '*说明*.xlsx',
            '*description*.xlsx',
            '*字段*.xlsx'
        ]
        
        for directory in search_directories:
            if not os.path.exists(directory):
                continue
                
            for pattern in schema_patterns:
                search_pattern = os.path.join(directory, pattern)
                matching_files = glob.glob(search_pattern)
                
                if matching_files:
                    # 如果找到多个文件，优先选择包含"字段说明"的文件
                    for file in matching_files:
                        if '字段说明' in os.path.basename(file):
                            self.logger.info(f"智能找到字段说明文件: {file}")
                            return file
                    
                    # 如果没有包含"字段说明"的，选择第一个
                    selected_file = matching_files[0]
                    self.logger.info(f"智能找到可能的字段说明文件: {selected_file}")
                    return selected_file
        
        # 作为最后的备选，尝试原来的硬编码路径（但会给出警告）
        fallback_path = 'data/Copy of 整车订单状态指标表字段说明.xlsx'
        if os.path.exists(fallback_path):
            self.logger.warning(f"使用兜底的字段说明文件: {fallback_path} (建议明确指定schema文件路径)")
            return fallback_path
        
        return None
    
    def _map_field_to_type(self, field_info: Dict[str, str], sample_data: pd.Series = None) -> str:
        """动态字段类型识别 - 基于字段特征而非硬编码名称"""
        data_type = field_info.get('data_type', '').lower()
        comment = field_info.get('comment', '').lower()
        column_name = field_info.get('column_name', '').lower()
        
        # 1. 基于数据类型的初步判断
        if 'datetime' in data_type or 'timestamp' in data_type:
            return 'datetime'
        elif 'bigint' in data_type or 'int' in data_type or 'float' in data_type:
            return 'number'
        
        # 2. 动态语义分析
        if sample_data is not None:
            return self._dynamic_semantic_analysis(column_name, comment, sample_data)
        
        # 3. 基于注释的语义分析
        return self._analyze_comment_semantics(comment, column_name)
    
    def _dynamic_semantic_analysis(self, column_name: str, comment: str, sample_data: pd.Series) -> str:
        """动态语义分析"""
        # 分析字段名语义
        name_semantics = self._analyze_name_semantics(column_name)
        
        # 分析注释语义
        comment_semantics = self._analyze_comment_semantics(comment, column_name)
        
        # 分析数据内容特征
        content_features = self._analyze_content_features(sample_data)
        
        # 综合判断
        return self._classify_field_type(name_semantics, comment_semantics, content_features)
    
    def _analyze_name_semantics(self, column_name: str) -> Dict[str, float]:
        """分析字段名语义"""
        column_lower = column_name.lower()
        
        # 定义语义关键词（可扩展）
        semantic_patterns = {
            'user_id': ['user', 'customer', 'client', 'member', 'id', 'number', 'code', 'order'],
            'channel': ['channel', 'source', 'medium', 'platform', 'path', 'way', 'route'],
            'timestamp': ['time', 'date', 'created', 'updated', 'timestamp', 'when'],
            'conversion': ['conversion', 'purchase', 'order', 'transaction', 'sale', 'buy'],
            'category': ['type', 'category', 'class', 'group', 'status', 'level']
        }
        
        scores = {}
        for field_type, keywords in semantic_patterns.items():
            score = sum(1 for keyword in keywords if keyword in column_lower)
            scores[field_type] = score / len(keywords) if keywords else 0
        
        return scores
    
    def _analyze_comment_semantics(self, comment: str, column_name: str) -> Dict[str, float]:
        """分析注释语义"""
        comment_lower = comment.lower()
        
        semantic_patterns = {
            'user_id': ['用户', '客户', '订单', '编号', 'id', 'number'],
            'channel': ['渠道', '来源', '平台', '路径', 'channel', 'source'],
            'timestamp': ['时间', '日期', '创建', '更新', 'time', 'date'],
            'conversion': ['转化', '购买', '交易', '订单', 'conversion', 'purchase'],
            'category': ['类型', '分类', '状态', '级别', 'type', 'category']
        }
        
        scores = {}
        for field_type, keywords in semantic_patterns.items():
            score = sum(1 for keyword in keywords if keyword in comment_lower)
            scores[field_type] = score / len(keywords) if keywords else 0
        
        return scores
    
    def _analyze_content_features(self, sample_data: pd.Series) -> Dict[str, Any]:
        """分析数据内容特征"""
        features = {
            'unique_ratio': sample_data.nunique() / len(sample_data),
            'is_timestamp': self._is_timestamp_format(sample_data),
            'is_id_format': self._is_id_format(sample_data),
            'is_numeric': self._is_numeric_format(sample_data),
            'is_categorical': self._is_categorical_format(sample_data)
        }
        return features
    
    def _is_timestamp_format(self, sample_data: pd.Series) -> bool:
        """检测是否为时间戳格式"""
        try:
            # 尝试解析为时间格式
            pd.to_datetime(sample_data.head(10), errors='coerce')
            return True
        except:
            return False
    
    def _is_id_format(self, sample_data: pd.Series) -> bool:
        """检测是否为ID格式"""
        sample_str = sample_data.astype(str).head(10)
        # ID通常具有特定模式：字母数字组合，长度一致
        return all(len(s) > 5 and s.isalnum() for s in sample_str)
    
    def _is_numeric_format(self, sample_data: pd.Series) -> bool:
        """检测是否为数值格式"""
        try:
            pd.to_numeric(sample_data.head(10), errors='coerce')
            return True
        except:
            return False
    
    def _is_categorical_format(self, sample_data: pd.Series) -> bool:
        """检测是否为分类格式"""
        unique_ratio = sample_data.nunique() / len(sample_data)
        return unique_ratio < 0.5  # 分类变量通常唯一值比例较低
    
    def _classify_field_type(self, name_semantics: Dict, comment_semantics: Dict, content_features: Dict) -> str:
        """综合分类字段类型"""
        # 综合评分
        scores = {}
        for field_type in ['user_id', 'channel', 'timestamp', 'conversion', 'category']:
            name_score = name_semantics.get(field_type, 0)
            comment_score = comment_semantics.get(field_type, 0)
            content_score = self._get_content_score(field_type, content_features)
            
            # 加权平均
            scores[field_type] = name_score * 0.4 + comment_score * 0.3 + content_score * 0.3
        
        # 返回得分最高的类型
        best_type = max(scores, key=scores.get)
        return best_type if scores[best_type] > 0.3 else 'unknown'
    
    def _get_content_score(self, field_type: str, content_features: Dict) -> float:
        """获取内容特征得分"""
        if field_type == 'timestamp' and content_features.get('is_timestamp'):
            return 1.0
        elif field_type == 'user_id' and content_features.get('is_id_format'):
            return 1.0
        elif field_type == 'category' and content_features.get('is_categorical'):
            return 1.0
        elif field_type in ['user_id', 'channel', 'conversion'] and content_features.get('unique_ratio', 0) > 0.8:
            return 0.8
        return 0.0
    
    def _fallback_column_type_identification(self, col: str, series: pd.Series) -> str:
        """备选的列类型识别方法"""
        col_lower = col.lower()
        
        # 检测用户ID列
        if any(pattern in col_lower for pattern in ['user_id', 'userid', 'customer_id', 'customerid', '用户id', '客户id', 'lead_id', 'clue_id']):
            return 'user_id'
        # 检测渠道列
        elif any(pattern in col_lower for pattern in ['channel', 'source', 'medium', 'campaign', '渠道', '来源']):
            return 'channel'
        # 检测事件时间列
        elif any(pattern in col_lower for pattern in ['event_time', 'eventtime', 'time', '时间', '日期', 'create_time', 'payment_time']):
            return 'event_time'
        # 检测转化列
        elif any(pattern in col_lower for pattern in ['conversion', 'purchase', 'order', '转化', '购买', '订单', 'status']):
            return 'conversion'
        # 检测日期列
        elif self._is_datetime_column(series):
            return 'datetime'
        # 检测数值列
        elif self._is_numeric_column(series):
            return 'number'
        # 检测分类列
        elif self._is_categorical_column(series):
            return 'category'
        # 默认为文本列
        else:
            return 'text'
    
    def _is_datetime_column(self, series: pd.Series) -> bool:
        """检测是否为日期列"""
        # 检查数据类型
        if pd.api.types.is_datetime64_any_dtype(series):
            return True
        
        # 尝试转换为时间类型
        try:
            pd.to_datetime(series.head(10), errors='raise')
            return True
        except:
            pass
        
        return False
    
    def _is_numeric_column(self, series: pd.Series) -> bool:
        """检测是否为数值列"""
        if pd.api.types.is_numeric_dtype(series):
            return True
        
        # 尝试转换为数值
        try:
            pd.to_numeric(series.head(10), errors='raise')
            return True
        except:
            pass
        
        return False
    
    def _is_categorical_column(self, series: pd.Series) -> bool:
        """检测是否为分类列"""
        # 如果唯一值数量较少且不是数值，可能是分类
        unique_ratio = series.nunique() / len(series)
        if unique_ratio < 0.5 and not pd.api.types.is_numeric_dtype(series):
            return True
        
        return False
    
    def _calculate_missing_rates(self, data: pd.DataFrame) -> Dict[str, float]:
        """计算缺失率"""
        missing_rates = {}
        for col in data.columns:
            missing_count = data[col].isnull().sum()
            missing_rates[col] = missing_count / len(data)
        
        return missing_rates
    
    def _calculate_date_parse_rates(self, data: pd.DataFrame) -> Dict[str, float]:
        """计算可解析日期比例"""
        date_parse_rates = {}
        
        for col in data.columns:
            try:
                # 尝试解析日期
                parsed_dates = pd.to_datetime(data[col], errors='coerce')
                valid_dates = parsed_dates.notna().sum()
                date_parse_rates[col] = valid_dates / len(data)
            except:
                date_parse_rates[col] = 0.0
        
        return date_parse_rates
    
    def _identify_attribution_fields(self, data: pd.DataFrame, column_types: Dict[str, str], schema_path: str = None) -> Dict[str, Any]:
        """识别归因分析关键字段 - 基于字段说明文档"""
        attribution_fields = {}
        
        # 加载字段说明文档
        field_mapping = self._load_field_mapping(schema_path)
        
        # 识别用户ID字段
        user_id_cols = [col for col, col_type in column_types.items() if col_type == 'user_id']
        if user_id_cols:
            attribution_fields['user_id_column'] = user_id_cols[0]
            attribution_fields['unique_users'] = data[user_id_cols[0]].nunique()
        else:
            # 尝试从字段说明文档中识别用户相关字段
            for col, field_info in field_mapping.items():
                if col in data.columns:
                    comment = field_info.get('comment', '').lower()
                    if any(keyword in comment for keyword in ['用户', '客户', '线索', '主理']):
                        attribution_fields['user_id_column'] = col
                        attribution_fields['unique_users'] = data[col].nunique()
                        break
        
        # 识别渠道字段
        channel_cols = [col for col, col_type in column_types.items() if col_type == 'channel']
        if channel_cols:
            attribution_fields['channel_column'] = channel_cols[0]
            attribution_fields['unique_channels'] = data[channel_cols[0]].nunique()
        else:
            # 尝试从字段说明文档中识别渠道相关字段
            for col, field_info in field_mapping.items():
                if col in data.columns:
                    comment = field_info.get('comment', '').lower()
                    if '渠道' in comment:
                        attribution_fields['channel_column'] = col
                        attribution_fields['unique_channels'] = data[col].nunique()
                        break
        
        # 识别事件时间字段
        event_time_cols = [col for col, col_type in column_types.items() if col_type == 'event_time']
        if event_time_cols:
            attribution_fields['event_time_column'] = event_time_cols[0]
        else:
            # 尝试从字段说明文档中识别时间相关字段
            time_fields = []
            for col, field_info in field_mapping.items():
                if col in data.columns:
                    comment = field_info.get('comment', '').lower()
                    data_type = field_info.get('data_type', '').lower()
                    if 'datetime' in data_type and any(keyword in comment for keyword in ['时间', '日期', '创建', '支付']):
                        time_fields.append(col)
            
            if time_fields:
                # 优先选择创建时间作为事件时间
                preferred_fields = ['order_create_time', 'clue_create_time', 'wish_create_time']
                for preferred in preferred_fields:
                    if preferred in time_fields:
                        attribution_fields['event_time_column'] = preferred
                        break
                else:
                    attribution_fields['event_time_column'] = time_fields[0]
        
        # 识别转化字段 - order_number既是用户ID也是转化字段
        if attribution_fields.get('user_id_column') == 'order_number':
            attribution_fields['conversion_column'] = 'order_number'
        else:
            # 尝试从字段说明文档中识别转化相关字段
            for col, field_info in field_mapping.items():
                if col in data.columns:
                    comment = field_info.get('comment', '').lower()
                    if any(keyword in comment for keyword in ['订单', '锁单', '交付', '状态']):
                        attribution_fields['conversion_column'] = col
                        break
        
        # 识别日期字段（用于时间序列分析）
        datetime_cols = [col for col, col_type in column_types.items() if col_type == 'datetime']
        if datetime_cols:
            attribution_fields['datetime_columns'] = datetime_cols
        
        # 识别数值字段（用于相关性分析）
        numeric_cols = [col for col, col_type in column_types.items() if col_type == 'number']
        if numeric_cols:
            attribution_fields['numeric_columns'] = numeric_cols
        
        # 识别分类字段（用于组间差异分析）
        categorical_cols = [col for col, col_type in column_types.items() if col_type == 'category']
        if categorical_cols:
            attribution_fields['categorical_columns'] = categorical_cols
        
        return attribution_fields
    
    def recommend_algorithms(self, examination_result: Dict[str, Any], 
                           full_data: pd.DataFrame, schema_path: str = None) -> List[Dict[str, Any]]:
        """
        推荐归因分析算法（使用Schema对比方式）
        
        Args:
            examination_result: 体检结果
            full_data: 完整数据
            
        Returns:
            List[Dict[str, Any]]: 算法推荐列表
        """
        try:
            self.logger.info("开始归因算法推荐（Schema对比方式）")
            
            # 使用Schema对比方式匹配模型
            model_matches = self.match_models(full_data, schema_path)
            
            recommendations = []
            
            # 将Schema匹配结果转换为算法推荐格式
            for match in model_matches:
                if match['executable']:
                    # 构建推荐理由
                    field_mapping = match['field_mapping']
                    reason_parts = []
                    
                    for field_name, mapped_field in field_mapping.items():
                        reason_parts.append(f"{field_name}({mapped_field})")
                    
                    reason = f"Schema匹配成功: {', '.join(reason_parts)}"
                    
                    # 构建算法参数
                    params = {}
                    for field_name, mapped_field in field_mapping.items():
                        if field_name == 'user_id':
                            params['user_id_column'] = mapped_field
                        elif field_name == 'channel':
                            params['channel_column'] = mapped_field
                        elif field_name == 'timestamp':
                            params['event_time_column'] = mapped_field
                        elif field_name == 'conversion':
                            params['conversion_column'] = mapped_field
                        elif field_name == 'numeric_field':
                            # 相关性分析需要 'columns' 参数，需要多个数值列
                            if match['model_name'] == 'correlation_analysis':
                                if isinstance(mapped_field, list):
                                    params['columns'] = mapped_field
                                else:
                                    if 'columns' not in params:
                                        params['columns'] = []
                                    params['columns'].append(mapped_field)
                            else:
                                params['numeric_columns'] = [mapped_field]
                        elif field_name == 'categorical_field':
                            params['categorical_columns'] = [mapped_field]
                    
                    # 为时间趋势分析添加额外的参数
                    if match['model_name'] == 'time_series_analysis':
                        # 确保有日期列和数值列
                        if 'timestamp' in field_mapping and 'numeric_field' in field_mapping:
                            params['datetime_columns'] = [field_mapping['timestamp']]
                            params['numeric_columns'] = [field_mapping['numeric_field']]
                    
                    # 根据模型名称映射到算法名称
                    algorithm_name_map = {
                        'last_click_attribution': '最后点击归因',
                        'first_click_attribution': '首次点击归因',
                        'markov_channel_model': 'Markov渠道模型',
                        'markov_absorption_model': 'Markov吸收链模型',
                        'correlation_analysis': '相关性分析',
                        'time_series_analysis': '时间趋势分析',
                        'categorical_analysis': '分类分析',
                        'descriptive_statistics': '描述统计'
                    }
                    
                    algorithm_name = algorithm_name_map.get(match['model_name'], match['model_name'])
                    
                    recommendations.append({
                        'algorithm': algorithm_name,
                        'reason': reason,
                        'confidence': match['match_rate'],
                        'executable': True,
                        'params': params,
                        'priority': match['priority']
                    })
                else:
                    # 不可执行的模型
                    algorithm_name_map = {
                        'last_click_attribution': '最后点击归因',
                        'first_click_attribution': '首次点击归因',
                        'markov_channel_model': 'Markov渠道模型',
                        'markov_absorption_model': 'Markov吸收链模型',
                        'correlation_analysis': '相关性分析',
                        'time_series_analysis': '时间趋势分析',
                        'categorical_analysis': '分类分析',
                        'descriptive_statistics': '描述统计'
                    }
                    
                    algorithm_name = algorithm_name_map.get(match['model_name'], match['model_name'])
                    
                    recommendations.append({
                        'algorithm': algorithm_name,
                        'reason': f"Schema匹配失败: 缺少字段 {', '.join(match['missing_fields'])}",
                        'confidence': 0.0,
                        'executable': False,
                        'priority': match['priority']
                    })
            
            # 按优先级排序
            recommendations.sort(key=lambda x: x.get('priority', 999))
            
            self.logger.info(f"Schema对比推荐完成，共推荐 {len(recommendations)} 个算法")
            return recommendations
            
        except Exception as e:
            self.logger.error(f"算法推荐失败: {str(e)}")
            return [{'error': str(e)}]

            
        except Exception as e:
            self.logger.error(f"归因算法推荐失败: {str(e)}")
            return []
    
    def should_execute_algorithms(self, recommendations: List[Dict[str, Any]], 
                                confidence_threshold: float = 0.8) -> Tuple[bool, List[Dict[str, Any]]]:
        """
        判断是否应该执行算法
        
        Args:
            recommendations: 算法推荐列表
            confidence_threshold: 置信度阈值
            
        Returns:
            Tuple[bool, List[Dict[str, Any]]]: (是否执行, 执行列表)
        """
        # 筛选置信度≥阈值的可执行算法
        executable_algorithms = [
            rec for rec in recommendations 
            if rec['confidence'] >= confidence_threshold and rec['executable']
        ]
        
        # 检查是否有算法冲突
        has_conflicts = self._check_algorithm_conflicts(executable_algorithms)
        
        if executable_algorithms and not has_conflicts:
            return True, executable_algorithms
        else:
            return False, recommendations
    
    def _check_algorithm_conflicts(self, algorithms: List[Dict[str, Any]]) -> bool:
        """检查算法冲突"""
        # 简单的冲突检查逻辑
        algorithm_names = [alg['algorithm'] for alg in algorithms]
        
        # 检查是否有重复的算法类型
        if len(algorithm_names) != len(set(algorithm_names)):
            return True
        
        return False 

    def _calculate_semantic_match_score(self, column_name: str, field_info: Dict, schema_model_requirements: Dict, sample_data: pd.Series = None) -> float:
        """
        基于语义和数据特征的智能字段匹配
        
        Args:
            column_name: 数据列名
            field_info: 字段信息（包含注释、数据类型等）
            schema_model_requirements: 模型要求的字段规范
            sample_data: 样本数据，用于数据特征分析
            
        Returns:
            float: 匹配分数 (0.0-1.0)
        """
        score = 0.0
        
        # 1. 语义相似性匹配 (40%)
        semantic_score = self._calculate_semantic_similarity(column_name, field_info, schema_model_requirements)
        score += semantic_score * 0.4
        
        # 2. 数据特征匹配 (35%)
        if sample_data is not None:
            data_feature_score = self._analyze_data_features(sample_data, schema_model_requirements)
            score += data_feature_score * 0.35
        
        # 3. 数据类型兼容性 (15%)
        type_compatibility_score = self._check_type_compatibility(field_info, schema_model_requirements)
        score += type_compatibility_score * 0.15
        
        # 4. 业务逻辑匹配 (10%)
        business_logic_score = self._check_business_logic(column_name, field_info, schema_model_requirements)
        score += business_logic_score * 0.1
        
        return min(score, 1.0)
    
    def _calculate_semantic_similarity(self, column_name: str, field_info: Dict, patterns: List[str]) -> float:
        """计算语义相似性"""
        score = 0.0
        
        # 字段名分词
        column_tokens = self._tokenize_field_name(column_name)
        
        # 与模式进行语义匹配
        for pattern in patterns:
            pattern_tokens = self._tokenize_field_name(pattern)
            similarity = self._calculate_token_similarity(column_tokens, pattern_tokens)
            score = max(score, similarity)
        
        # 注释语义匹配
        comment = field_info.get('comment', '')
        if comment:
            for pattern in patterns:
                if pattern.lower() in comment.lower():
                    score = max(score, 0.8)
        
        return score
    
    def _analyze_data_features(self, sample_data: pd.Series, patterns: List[str]) -> float:
        """分析数据特征匹配度"""
        score = 0.0
        
        # 数据分布特征
        uniqueness_ratio = sample_data.nunique() / len(sample_data)
        missing_ratio = sample_data.isnull().sum() / len(sample_data)
        
        # 根据模式推断期望的数据特征
        for pattern in patterns:
            pattern_lower = pattern.lower()
            
            # 用户ID模式
            if 'user' in pattern_lower and 'id' in pattern_lower:
                if uniqueness_ratio > 0.95:  # 接近唯一标识符
                    score = max(score, 0.8)
            
            # 时间模式
            elif any(time_word in pattern_lower for time_word in ['time', 'date', 'timestamp']):
                if pd.api.types.is_datetime64_any_dtype(sample_data):
                    score = max(score, 0.9)
            
            # 数值模式
            elif any(num_word in pattern_lower for num_word in ['count', 'amount', 'number', 'score', 'rate']):
                if pd.api.types.is_numeric_dtype(sample_data):
                    score = max(score, 0.8)
            
            # 分类模式
            elif any(cat_word in pattern_lower for cat_word in ['type', 'category', 'status', 'name']):
                if uniqueness_ratio < 0.1:  # 分类变量
                    score = max(score, 0.7)
        
        # 数据质量加分
        if missing_ratio < 0.1:
            score += 0.1
        
        return min(score, 1.0)
    
    def _check_type_compatibility(self, field_info: Dict, requirements: Dict) -> float:
        """检查数据类型兼容性"""
        score = 0.0
        
        data_type = field_info.get('data_type', '').lower()
        required_types = requirements.get('compatible_types', [])
        
        # 类型映射
        type_mapping = {
            'datetime': ['timestamp', 'date', 'time'],
            'numeric': ['int', 'bigint', 'float', 'decimal', 'number'],
            'string': ['varchar', 'char', 'text', 'string'],
            'boolean': ['bool', 'boolean', 'bit']
        }
        
        for req_type in required_types:
            if req_type.lower() in data_type:
                score = 1.0
                break
            elif req_type in type_mapping:
                if any(t in data_type for t in type_mapping[req_type]):
                    score = 0.8
                    break
        
        return score
    
    def _check_business_logic(self, column_name: str, field_info: Dict, requirements: Dict) -> float:
        """检查业务逻辑匹配"""
        score = 0.0
        
        # 业务规则匹配
        business_rules = requirements.get('business_rules', {})
        
        # 例如：用户ID应该包含数字
        if 'user_id' in column_name.lower():
            if 'numeric_content' in business_rules:
                score += 0.5
        
        # 例如：时间字段应该有时间格式
        if any(time_word in column_name.lower() for time_word in ['time', 'date', 'created', 'updated']):
            if 'datetime_format' in business_rules:
                score += 0.5
        
        return score
    
    def _tokenize_field_name(self, field_name: str) -> List[str]:
        """字段名分词"""
        # 处理下划线、驼峰命名等
        import re
        
        # 分割下划线
        tokens = field_name.lower().split('_')
        
        # 处理驼峰命名
        camel_case_tokens = []
        for token in tokens:
            camel_case_tokens.extend(re.findall(r'[a-z]+|[A-Z][a-z]*', token))
        
        # 过滤空字符串
        tokens = [t for t in camel_case_tokens if t]
        
        return tokens
    
    def _calculate_token_similarity(self, tokens1: List[str], tokens2: List[str]) -> float:
        """计算token相似性"""
        if not tokens1 or not tokens2:
            return 0.0
        
        # 计算Jaccard相似性
        set1 = set(tokens1)
        set2 = set(tokens2)
        
        intersection = len(set1.intersection(set2))
        union = len(set1.union(set2))
        
        if union == 0:
            return 0.0
        
        return intersection / union
    
    def _calculate_text_similarity(self, text1: str, text2: str) -> float:
        """计算文本相似性（简化版）"""
        # 这里可以使用更复杂的文本相似性算法
        # 如余弦相似性、编辑距离等
        
        tokens1 = text1.lower().split()
        tokens2 = text2.lower().split()
        
        return self._calculate_token_similarity(tokens1, tokens2)
    
    def _analyze_data_distribution(self, sample_data: pd.Series) -> Dict:
        """分析数据分布特征"""
        analysis = {}
        
        if pd.api.types.is_numeric_dtype(sample_data):
            analysis['type'] = 'numeric'
            analysis['min'] = sample_data.min()
            analysis['max'] = sample_data.max()
            analysis['mean'] = sample_data.mean()
        elif pd.api.types.is_datetime64_any_dtype(sample_data):
            analysis['type'] = 'datetime'
            analysis['min'] = sample_data.min()
            analysis['max'] = sample_data.max()
        else:
            analysis['type'] = 'categorical'
            analysis['unique_count'] = sample_data.nunique()
        
        return analysis
    
    def _detect_data_format(self, sample_data: pd.Series) -> str:
        """检测数据格式"""
        if pd.api.types.is_datetime64_any_dtype(sample_data):
            return 'datetime'
        elif pd.api.types.is_numeric_dtype(sample_data):
            return 'numeric'
        else:
            # 检查是否是邮箱、手机号等格式
            sample_str = sample_data.astype(str).iloc[0] if len(sample_data) > 0 else ''
            
            if '@' in sample_str and '.' in sample_str:
                return 'email'
            elif len(sample_str) == 11 and sample_str.isdigit():
                return 'phone'
            else:
                return 'text'
    
    async def analyze_with_llm(self, data: pd.DataFrame, schema_path: str = None) -> Dict[str, Any]:
        """
        使用LLM进行智能分析（以字段说明为主，LLM为辅）
        
        Args:
            data: 输入数据
            schema_path: 字段说明文件路径
            
        Returns:
            Dict: LLM分析结果
        """
        if not self.enable_llm or not self.llm_analyzer_client:
            self.logger.warning("LLM功能未启用或LLM客户端未初始化")
            return {
                'success': False,
                'error': 'LLM功能未启用',
                'fallback_analysis': self.analyze_data_structure(data, schema_path)
            }
        
        try:
            self.logger.info("开始使用LLM进行智能分析（以字段说明为主）")
            
            # 加载字段说明映射
            field_mapping = self._load_field_mapping(schema_path)
            
            # 调用LLM分析器客户端，传递字段说明映射
            result = await self.llm_analyzer_client.analyze_data_structure(data, field_mapping)
            
            if result['success']:
                self.logger.info("LLM分析成功")
                return result
            else:
                self.logger.warning(f"LLM分析失败: {result.get('error', '未知错误')}")
                # 返回基于字段说明的备用分析结果
                fallback_result = result.get('fallback_analysis', {})
                if not fallback_result:
                    # 如果没有fallback_analysis，使用基于字段说明的分析
                    fallback_result = self._analyze_with_field_mapping(data, field_mapping)
                
                return {
                    'success': False,
                    'error': result.get('error', 'LLM分析失败'),
                    'fallback_analysis': fallback_result,
                    'llm_response': result.get('raw_response', '')
                }
                
        except Exception as e:
            self.logger.error(f"LLM分析异常: {str(e)}")
            return {
                'success': False,
                'error': str(e),
                'fallback_analysis': self._analyze_with_field_mapping(data, field_mapping) if 'field_mapping' in locals() else self.analyze_data_structure(data, schema_path)
            }
    
    def _analyze_with_field_mapping(self, data: pd.DataFrame, field_mapping: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        基于字段说明文档进行基础分析（与LLM客户端保持一致）
        
        Args:
            data: 输入数据
            field_mapping: 字段说明映射
            
        Returns:
            Dict: 基于字段说明的分析结果
        """
        from .llm_client import LLMAnalyzerClient
        
        # 创建一个临时的LLM客户端来复用分析逻辑
        temp_client = LLMAnalyzerClient(None)  # 传入None作为LLM客户端
        return temp_client._analyze_with_field_mapping(data, field_mapping)
    
    def analyze_data_structure_with_llm(self, data: pd.DataFrame, schema_path: str = None) -> Dict[str, Any]:
        """
        同步版本的LLM分析（内部使用asyncio）
        
        Args:
            data: 输入数据
            schema_path: 字段说明文件路径
            
        Returns:
            Dict: 分析结果
        """
        if not self.enable_llm:
            # 如果不启用LLM，使用原有的规则引擎分析
            return self.analyze_data_structure(data, schema_path)
        
        try:
            # 使用asyncio运行异步方法
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            result = loop.run_until_complete(self.analyze_with_llm(data, schema_path))
            loop.close()
            return result
        except Exception as e:
            self.logger.error(f"同步LLM分析失败: {str(e)}")
            return {
                'success': False,
                'error': str(e),
                'fallback_analysis': self.analyze_data_structure(data, schema_path)
            } 