#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LLM Client - Supports multiple LLM API calls
"""

import asyncio
import aiohttp
import json
import ssl
import logging
from typing import Dict, Any, Optional, List
from datetime import datetime
import pandas as pd


class LLMClient:
    """LLM Client - Supports multiple LLM APIs"""
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize LLM client
        
        Args:
            config: Configuration dictionary containing API keys and other information
        """
        self.config = config or {}
        self.logger = logging.getLogger(__name__)
        
        # Default configuration
        self.default_config = {
            'api_key': None,  # User must provide API key
            'provider': 'dashscope',  # Default provider
            'model_name': 'qwen-turbo',
            'timeout': 300,
            'max_tokens': 4000,
            'temperature': 0.7,
            'top_p': 0.8
        }
        
        # Merge configuration
        self.config = {**self.default_config, **self.config}
        
        # Validate API key
        if not self.config.get('api_key'):
            provider = self.config.get('provider', 'dashscope')
            env_var_map = {
                'dashscope': 'DASHSCOPE_API_KEY',
                'openai': 'OPENAI_API_KEY'
            }
            env_var = env_var_map.get(provider, 'API_KEY')
            raise ValueError(f"API key is required. Please provide your {provider} API key via --llm-api-key parameter or {env_var} environment variable.")
        
        # Setup API endpoint
        self._setup_api_endpoint()
    
    def _setup_api_endpoint(self):
        """Setup API endpoint"""
        provider = self.config.get('provider', 'dashscope')
        
        if provider == 'dashscope':
            self.api_endpoint = 'https://dashscope.aliyuncs.com/api/v1/services/aigc/text-generation/generation'
        elif provider == 'openai':
            self.api_endpoint = 'https://api.openai.com/v1/chat/completions'
        else:
            raise ValueError(f"Unsupported provider: {provider}")
    
    async def call_llm(self, prompt: str, system_prompt: str = None, **kwargs) -> Dict[str, Any]:
        """
        Call LLM API
        
        Args:
            prompt: User prompt
            system_prompt: System prompt
            **kwargs: Other parameters
            
        Returns:
            Dict: Dictionary containing response content and metadata
        """
        try:
            provider = self.config.get('provider', 'dashscope')
            
            if provider == 'dashscope':
                return await self._call_dashscope(prompt, system_prompt, **kwargs)
            elif provider == 'openai':
                return await self._call_openai(prompt, system_prompt, **kwargs)
            else:
                raise ValueError(f"Unsupported provider: {provider}")
                
        except Exception as e:
            self.logger.error(f"LLM call failed: {str(e)}")
            return {
                'success': False,
                'error': str(e),
                'content': '',
                'metadata': {}
            }
    
    async def _call_dashscope(self, prompt: str, system_prompt: str = None, **kwargs) -> Dict[str, Any]:
        """Call Alibaba Cloud Dashscope API"""
        try:
            api_key = self.config['api_key']
            model_name = self.config['model_name']
            endpoint = self.api_endpoint
            
            # 构建消息
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": prompt})
            
            # 构建请求体
            payload = {
                "model": model_name,
                "input": {
                    "messages": messages
                },
                "parameters": {
                    "max_tokens": kwargs.get('max_tokens', self.config['max_tokens']),
                    "temperature": kwargs.get('temperature', self.config['temperature']),
                    "top_p": kwargs.get('top_p', self.config['top_p']),
                    "result_format": "message"
                }
            }
            
            headers = {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json"
            }
            
            # 发送请求
            timeout = aiohttp.ClientTimeout(total=self.config['timeout'])
            
            # 创建SSL上下文，禁用证书验证
            ssl_context = ssl.create_default_context()
            ssl_context.check_hostname = False
            ssl_context.verify_mode = ssl.CERT_NONE
            
            connector = aiohttp.TCPConnector(ssl=ssl_context)
            async with aiohttp.ClientSession(timeout=timeout, connector=connector) as session:
                async with session.post(endpoint, json=payload, headers=headers) as response:
                    if response.status == 200:
                        result = await response.json()
                        
                        # 解析响应
                        if 'output' in result and 'choices' in result['output']:
                            content = result['output']['choices'][0]['message']['content']
                        else:
                            content = result.get('output', {}).get('text', '')
                        
                        return {
                            'success': True,
                            'content': content,
                            'metadata': {
                                'provider': 'dashscope',
                                'model': model_name,
                                'timestamp': datetime.now().isoformat(),
                                'raw_response': result
                            }
                        }
                    else:
                        error_text = await response.text()
                        raise Exception(f"API调用失败: {response.status} - {error_text}")
                        
        except Exception as e:
            self.logger.error(f"Dashscope API调用失败: {str(e)}")
            raise
    
    async def _call_openai(self, prompt: str, system_prompt: str = None, **kwargs) -> Dict[str, Any]:
        """调用OpenAI API"""
        try:
            api_key = self.config['api_key']
            model_name = self.config['model_name']
            endpoint = self.api_endpoint
            
            # 构建消息
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": prompt})
            
            # 构建请求体
            payload = {
                "model": model_name,
                "messages": messages,
                "max_tokens": kwargs.get('max_tokens', self.config['max_tokens']),
                "temperature": kwargs.get('temperature', self.config['temperature']),
                "top_p": kwargs.get('top_p', self.config['top_p'])
            }
            
            headers = {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json"
            }
            
            # 发送请求
            timeout = aiohttp.ClientTimeout(total=self.config['timeout'])
            
            # 创建SSL上下文，禁用证书验证
            ssl_context = ssl.create_default_context()
            ssl_context.check_hostname = False
            ssl_context.verify_mode = ssl.CERT_NONE
            
            connector = aiohttp.TCPConnector(ssl=ssl_context)
            async with aiohttp.ClientSession(timeout=timeout, connector=connector) as session:
                async with session.post(endpoint, json=payload, headers=headers) as response:
                    if response.status == 200:
                        result = await response.json()
                        
                        # 解析响应
                        if 'choices' in result and len(result['choices']) > 0:
                            content = result['choices'][0]['message']['content']
                        else:
                            content = ''
                        
                        return {
                            'success': True,
                            'content': content,
                            'metadata': {
                                'provider': 'openai',
                                'model': model_name,
                                'timestamp': datetime.now().isoformat(),
                                'raw_response': result
                            }
                        }
                    else:
                        error_text = await response.text()
                        raise Exception(f"API调用失败: {response.status} - {error_text}")
                        
        except Exception as e:
            self.logger.error(f"OpenAI API调用失败: {str(e)}")
            raise
    
    def set_config(self, config: Dict[str, Any]):
        """更新配置"""
        self.config.update(config)
    
    def get_supported_models(self) -> List[str]:
        """获取支持的模型列表"""
        provider = self.config.get('provider', 'dashscope')
        
        if provider == 'dashscope':
            return ['qwen-turbo', 'qwen-plus', 'qwen-max', 'qwen-max-longcontext']
        elif provider == 'openai':
            return ['gpt-3.5-turbo', 'gpt-4', 'gpt-4-turbo', 'gpt-4o', 'gpt-4o-mini']
        else:
            return []


class LLMAnalyzerClient:
    """LLM分析器客户端 - 专门用于数据分析场景"""
    
    def __init__(self, llm_client: LLMClient = None):
        """
        初始化LLM分析器客户端
        
        Args:
            llm_client: LLM客户端实例
        """
        self.llm_client = llm_client or LLMClient()
        self.logger = logging.getLogger(__name__)
    
    async def analyze_data_structure(self, data: pd.DataFrame, field_mapping: Dict[str, Any] = None, sample_size: int = 50) -> Dict[str, Any]:
        """
        分析数据结构并推荐分析模型（以字段说明为主，LLM为辅）
        
        Args:
            data: 输入数据
            field_mapping: 字段说明映射（从字段说明文档加载）
            sample_size: 样本大小
            
        Returns:
            Dict: 分析结果和模型推荐
        """
        try:
            # 1. 首先基于字段说明进行基础分析
            base_analysis = self._analyze_with_field_mapping(data, field_mapping)
            
            # 2. 准备样本数据
            sample_data = self._prepare_sample_data(data, sample_size)
            
            # 3. 构建基于字段说明的增强提示词
            prompt = self._build_enhanced_analysis_prompt(data, sample_data, field_mapping, base_analysis)
            
            # 4. 系统提示词 - 强调以字段说明为主
            system_prompt = """你是一个专业的数据分析专家。请基于提供的字段说明文档进行智能分析。
字段说明文档是主要依据，请优先考虑字段的业务含义和数据类型。
在此基础上，结合数据特征推荐最适合的分析模型。"""
            
            # 5. 调用LLM进行增强分析
            result = await self.llm_client.call_llm(prompt, system_prompt)
            
            if result['success']:
                # 6. 解析响应并与字段说明分析结果合并
                llm_analysis = self._parse_analysis_response(result['content'], data)
                enhanced_result = self._merge_field_and_llm_analysis(base_analysis, llm_analysis)
                
                return {
                    'success': True,
                    'analysis': enhanced_result,
                    'raw_response': result['content'],
                    'metadata': result['metadata']
                }
            else:
                # 7. LLM失败时，返回基于字段说明的分析结果
                self.logger.warning(f"LLM分析失败: {result.get('error', '未知错误')}，使用字段说明分析结果")
                return {
                    'success': False,
                    'error': result.get('error', 'LLM调用失败'),
                    'fallback_analysis': base_analysis
                }
                
        except Exception as e:
            self.logger.error(f"数据分析失败: {str(e)}")
            # 8. 异常时也返回基于字段说明的分析结果
            base_analysis = self._analyze_with_field_mapping(data, field_mapping)
            return {
                'success': False,
                'error': str(e),
                'fallback_analysis': base_analysis
            }
    
    def _prepare_sample_data(self, data: pd.DataFrame, sample_size: int) -> List[Dict[str, Any]]:
        """准备样本数据"""
        if len(data) <= sample_size:
            return data.to_dict('records')
        
        # 随机采样
        sample_data = data.sample(min(sample_size, len(data)))
        return sample_data.to_dict('records')
    
    def _analyze_with_field_mapping(self, data: pd.DataFrame, field_mapping: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        基于字段说明文档进行基础分析
        
        Args:
            data: 输入数据
            field_mapping: 字段说明映射
            
        Returns:
            Dict: 基于字段说明的分析结果
        """
        analysis_result = {
            'field_analyses': [],
            'model_recommendations': [],
            'data_quality_assessment': {},
            'business_insights': []
        }
        
        if not field_mapping:
            self.logger.warning("未提供字段说明映射，使用基础分析")
            return analysis_result
        
        # 分析每个字段
        for col in data.columns:
            field_info = field_mapping.get(col, {})
            col_data = data[col]
            
            # 字段分析
            field_analysis = {
                'field_name': col,
                'data_type': str(col_data.dtype),
                'business_meaning': field_info.get('comment', ''),
                'field_type': field_info.get('data_type', ''),
                'missing_ratio': col_data.isnull().sum() / len(col_data),
                'unique_ratio': col_data.nunique() / len(col_data)
            }
            
            # 数值型字段的额外统计
            if pd.api.types.is_numeric_dtype(col_data):
                field_analysis.update({
                    'min_value': col_data.min(),
                    'max_value': col_data.max(),
                    'mean_value': col_data.mean(),
                    'std_value': col_data.std()
                })
            
            analysis_result['field_analyses'].append(field_analysis)
        
        # 基于字段说明推荐模型
        recommendations = self._recommend_models_by_field_mapping(data, field_mapping)
        analysis_result['model_recommendations'] = recommendations
        
        # 数据质量评估
        analysis_result['data_quality_assessment'] = self._assess_data_quality(data, field_mapping)
        
        # 业务洞察
        analysis_result['business_insights'] = self._extract_business_insights(data, field_mapping)
        
        return analysis_result
    
    def _recommend_models_by_field_mapping(self, data: pd.DataFrame, field_mapping: Dict[str, Any]) -> List[Dict[str, Any]]:
        """基于字段说明推荐模型"""
        recommendations = []
        
        # 统计字段类型
        numeric_fields = []
        categorical_fields = []
        time_fields = []
        id_fields = []
        
        for col in data.columns:
            field_info = field_mapping.get(col, {})
            comment = field_info.get('comment', '').lower()
            field_type = field_info.get('data_type', '').lower()
            col_lower = col.lower()
            
            # 检查是否是ID字段（优先检查）
            if (any(id_word in comment for id_word in ['id', '编号', '标识', '用户', '客户', '线索']) or
                any(id_word in col_lower for id_word in ['id', 'number', 'code', '编号', '编码'])):
                id_fields.append(col)
            
            elif pd.api.types.is_numeric_dtype(data[col]):
                numeric_fields.append(col)
            
            elif pd.api.types.is_datetime64_any_dtype(data[col]) or any(time_word in comment for time_word in ['时间', '日期', 'time', 'date']):
                time_fields.append(col)
            
            else:
                categorical_fields.append(col)
        
        # 基于字段类型推荐模型
        if len(numeric_fields) > 1:
            recommendations.append({
                'name': 'correlation_analysis',
                'confidence': 0.9,
                'reason': f'检测到{len(numeric_fields)}个数值型字段，适合进行相关性分析',
                'params': {
                    'columns': numeric_fields[:5]  # 限制为前5个数值字段
                }
            })
        
        if len(categorical_fields) > 0:
            recommendations.append({
                'name': 'categorical_analysis',
                'confidence': 0.8,
                'reason': f'检测到{len(categorical_fields)}个分类字段，适合进行分类分析',
                'params': {
                    'categorical_columns': categorical_fields[:3]  # 限制为前3个分类字段
                }
            })
        
        if len(time_fields) > 0 and len(numeric_fields) > 0:
            recommendations.append({
                'name': 'time_series_analysis',
                'confidence': 0.85,
                'reason': f'检测到时间字段和数值字段，适合进行时间序列分析',
                'params': {
                    'datetime_columns': time_fields[:2],  # 限制为前2个时间字段
                    'numeric_columns': numeric_fields[:3]  # 限制为前3个数值字段
                }
            })
        
        # 归因分析系列 - 需要ID字段和渠道字段
        channel_fields = []
        for col in categorical_fields:
            field_info = field_mapping.get(col, {})
            comment = field_info.get('comment', '').lower()
            if any(channel_word in comment for channel_word in ['渠道', 'channel', '来源', 'source']):
                channel_fields.append(col)
        
        if len(id_fields) > 0 and (len(channel_fields) > 0 or len(categorical_fields) > 0):
            # 选择最佳的渠道字段
            best_channel = channel_fields[0] if channel_fields else categorical_fields[0]
            
            # 首次点击归因
            if len(time_fields) > 0:
                recommendations.append({
                    'name': 'first_click_attribution',
                    'confidence': 0.95,
                    'reason': f'检测到用户ID({id_fields[0]})、渠道({best_channel})和时间字段，适合首次点击归因分析',
                    'params': {
                        'user_id_column': id_fields[0],
                        'channel_column': best_channel,
                        'event_time_column': time_fields[0] if time_fields else None
                    }
                })
                
                # 最后点击归因
                recommendations.append({
                    'name': 'last_click_attribution',
                    'confidence': 0.95,
                    'reason': f'检测到用户ID({id_fields[0]})、渠道({best_channel})和时间字段，适合最后点击归因分析',
                    'params': {
                        'user_id_column': id_fields[0],
                        'channel_column': best_channel,
                        'event_time_column': time_fields[0] if time_fields else None
                    }
                })
            
            # Markov渠道模型
            recommendations.append({
                'name': 'markov_channel_model',
                'confidence': 0.85,
                'reason': f'检测到用户ID({id_fields[0]})和渠道({best_channel})，适合Markov渠道模型分析',
                'params': {
                    'user_id_column': id_fields[0],
                    'channel_column': best_channel,
                    'datetime_columns': time_fields
                }
            })
            
            # Markov吸收链模型
            recommendations.append({
                'name': 'markov_absorption_model',
                'confidence': 0.85,
                'reason': f'检测到用户ID({id_fields[0]})和渠道({best_channel})，适合Markov吸收链模型分析',
                'params': {
                    'user_id_column': id_fields[0],
                    'channel_column': best_channel,
                    'datetime_columns': time_fields
                }
            })
            
            # 多维度归因分析 - 需要时间维度
            if len(time_fields) > 0:
                recommendations.append({
                    'name': 'multi_dimension_attribution',
                    'confidence': 0.85,
                    'reason': f'检测到用户ID({id_fields[0]})、渠道({best_channel})和时间维度，适合多维度归因分析',
                    'params': {
                        'user_id_column': id_fields[0],
                        'channel_column': best_channel,
                        'datetime_columns': time_fields
                    }
                })
        
        # 默认推荐描述性统计
        recommendations.append({
            'name': 'descriptive_statistics',
            'confidence': 0.95,
            'reason': '基础描述性统计，适用于所有数据集',
            'params': {}
        })
        
        return recommendations
    
    def _assess_data_quality(self, data: pd.DataFrame, field_mapping: Dict[str, Any]) -> Dict[str, Any]:
        """评估数据质量"""
        quality_assessment = {
            'completeness': {},
            'consistency': {},
            'accuracy': {}
        }
        
        total_fields = len(data.columns)
        total_records = len(data)
        
        # 完整性评估
        missing_stats = data.isnull().sum()
        completeness_scores = {}
        
        for col in data.columns:
            missing_ratio = missing_stats[col] / total_records
            completeness_scores[col] = {
                'missing_ratio': missing_ratio,
                'completeness_score': 1 - missing_ratio
            }
        
        quality_assessment['completeness'] = completeness_scores
        
        # 一致性评估
        for col in data.columns:
            field_info = field_mapping.get(col, {})
            field_type = field_info.get('data_type', '')
            
            if field_type and pd.api.types.is_numeric_dtype(data[col]):
                # 检查数值范围是否合理
                min_val = data[col].min()
                max_val = data[col].min()
                quality_assessment['consistency'][col] = {
                    'range_check': f'[{min_val}, {max_val}]',
                    'outlier_detection': '需要进一步分析'
                }
        
        return quality_assessment
    
    def _extract_business_insights(self, data: pd.DataFrame, field_mapping: Dict[str, Any]) -> List[str]:
        """提取业务洞察"""
        insights = []
        
        # 分析字段的业务含义
        for col in data.columns:
            field_info = field_mapping.get(col, {})
            comment = field_info.get('comment', '')
            
            if comment:
                insights.append(f"字段 '{col}': {comment}")
        
        # 数据规模洞察
        insights.append(f"数据集包含 {len(data)} 条记录，{len(data.columns)} 个字段")
        
        # 缺失值洞察
        missing_fields = data.columns[data.isnull().any()].tolist()
        if missing_fields:
            insights.append(f"发现缺失值的字段: {', '.join(missing_fields)}")
        
        return insights
    
    def _build_enhanced_analysis_prompt(self, data: pd.DataFrame, sample_data: List[Dict[str, Any]], 
                                      field_mapping: Dict[str, Any], base_analysis: Dict[str, Any]) -> str:
        """构建基于字段说明的增强分析提示词"""
        
        # 1. 字段说明信息
        field_mapping_info = ""
        if field_mapping:
            field_mapping_info = "\n字段说明文档信息:\n"
            for col in data.columns:
                field_info = field_mapping.get(col, {})
                comment = field_info.get('comment', '')
                field_type = field_info.get('data_type', '')
                if comment or field_type:
                    field_mapping_info += f"- {col}: {field_type} | {comment}\n"
        
        # 2. 基于字段说明的基础分析结果
        base_analysis_info = ""
        if base_analysis:
            recommendations = base_analysis.get('model_recommendations', [])
            if recommendations:
                base_analysis_info = "\n基于字段说明的初步推荐:\n"
                for rec in recommendations:
                    base_analysis_info += f"- {rec['name']}: {rec['reason']} (置信度: {rec['confidence']})\n"
        
        # 3. 数据基本信息
        data_info = f"""
数据基本信息:
- 行数: {len(data):,}
- 列数: {len(data.columns)}
- 列名: {list(data.columns)}
- 数据类型: {dict(data.dtypes)}
- 缺失值统计: {data.isnull().sum().to_dict()}
"""
        
        # 4. 字段统计信息
        field_stats = []
        for col in data.columns:
            col_data = data[col]
            dtype = str(col_data.dtype)
            non_null_count = col_data.count()
            null_count = col_data.isnull().sum()
            unique_count = col_data.nunique()
            
            # 添加字段说明信息
            field_info = field_mapping.get(col, {})
            comment = field_info.get('comment', '')
            field_type = field_info.get('data_type', '')
            
            if pd.api.types.is_numeric_dtype(col_data):
                min_val = col_data.min()
                max_val = col_data.max()
                mean_val = col_data.mean()
                field_stats.append(
                    f"- {col}: {dtype} | 业务含义: {comment} | 字段类型: {field_type} | "
                    f"非空: {non_null_count:,} | 唯一值: {unique_count:,} | "
                    f"范围: [{min_val:.2f}, {max_val:.2f}] | 均值: {mean_val:.2f}"
                )
            else:
                field_stats.append(
                    f"- {col}: {dtype} | 业务含义: {comment} | 字段类型: {field_type} | "
                    f"非空: {non_null_count:,} | 唯一值: {unique_count:,}"
                )
        
        # 5. 样本数据
        sample_str = ""
        if sample_data:
            sample_str = "\n样本数据 (前5行):\n"
            for i, record in enumerate(sample_data[:5], 1):
                record_str = f"记录 {i}: " + ", ".join([f"{k}={v}" for k, v in record.items()])
                sample_str += record_str + "\n"
        
        # 6. 构建增强提示词
        prompt = f"""
请基于字段说明文档对以下数据进行智能分析。

{field_mapping_info}

{base_analysis_info}

{data_info}

字段详细统计（包含业务含义）:
{chr(10).join(field_stats)}

{sample_str}

请基于字段说明文档的业务含义，结合数据特征，推荐最适合的分析模型。
注意：
1. 优先考虑字段说明文档中的业务含义
2. 结合字段类型和数据分布特征
3. 考虑业务场景的适用性

推荐的分析模型类型包括:
1. 描述性统计 (descriptive_statistics)
2. 相关性分析 (correlation_analysis) 
3. 时间序列分析 (time_series_analysis)
4. 分类分析 (categorical_analysis)
5. 归因分析 (attribution_analysis)
6. 聚类分析 (clustering_analysis)
7. 异常检测 (anomaly_detection)

请以JSON格式返回结果，包含以下字段:
- model_recommendations: 推荐的模型列表，每个模型包含name、confidence、reason字段
- field_analyses: 字段分析结果
- data_quality_assessment: 数据质量评估
- business_insights: 业务洞察

确保JSON格式正确，避免语法错误。
"""
        return prompt
    
    def _merge_field_and_llm_analysis(self, field_analysis: Dict[str, Any], llm_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """
        合并字段说明分析和LLM分析结果
        
        Args:
            field_analysis: 基于字段说明的分析结果
            llm_analysis: LLM分析结果
            
        Returns:
            Dict: 合并后的分析结果
        """
        merged_result = {
            'field_analyses': field_analysis.get('field_analyses', []),
            'model_recommendations': [],
            'data_quality_assessment': field_analysis.get('data_quality_assessment', {}),
            'business_insights': field_analysis.get('business_insights', []),
            'analysis_method': 'field_mapping_enhanced_by_llm'
        }
        
        # 合并模型推荐
        field_recommendations = field_analysis.get('model_recommendations', [])
        llm_recommendations = llm_analysis.get('model_recommendations', [])
        
        # 创建推荐映射
        recommendation_map = {}
        
        # 首先添加字段说明的推荐
        for rec in field_recommendations:
            model_name = rec['name']
            recommendation_map[model_name] = {
                'name': model_name,
                'confidence': rec['confidence'],
                'reason': rec['reason'],
                'source': 'field_mapping',
                'params': rec.get('params', {})  # 保留参数信息
            }
        
        # 然后添加或更新LLM的推荐
        for rec in llm_recommendations:
            model_name = rec['name']
            if model_name in recommendation_map:
                # 如果字段说明和LLM都推荐了同一个模型，取更高的置信度
                field_conf = recommendation_map[model_name]['confidence']
                llm_conf = rec['confidence']
                
                if llm_conf > field_conf:
                    recommendation_map[model_name] = {
                        'name': model_name,
                        'confidence': llm_conf,
                        'reason': rec['reason'],
                        'source': 'llm_enhanced',
                        'params': rec.get('params', {})  # 保留LLM参数信息
                    }
                else:
                    recommendation_map[model_name]['source'] = 'field_mapping_llm_agreed'
                    # 如果LLM有参数，更新参数
                    if 'params' in rec:
                        recommendation_map[model_name]['params'] = rec['params']
            else:
                # 新增LLM推荐的模型
                recommendation_map[model_name] = {
                    'name': model_name,
                    'confidence': rec['confidence'],
                    'reason': rec['reason'],
                    'source': 'llm_only',
                    'params': rec.get('params', {})  # 保留LLM参数信息
                }
        
        # 转换为列表并按置信度排序
        merged_recommendations = list(recommendation_map.values())
        merged_recommendations.sort(key=lambda x: x['confidence'], reverse=True)
        
        merged_result['model_recommendations'] = merged_recommendations
        
        # 合并业务洞察
        llm_insights = llm_analysis.get('business_insights', [])
        if llm_insights:
            merged_result['business_insights'].extend(llm_insights)
        
        return merged_result
    
    def _parse_analysis_response(self, response: str, data: pd.DataFrame) -> Dict[str, Any]:
        """解析分析响应"""
        try:
            # 提取JSON部分
            if '{' in response and '}' in response:
                start = response.find('{')
                end = response.rfind('}') + 1
                json_str = response[start:end]
                
                # 尝试修复JSON格式
                json_str = self._fix_json_format(json_str)
                
                result = json.loads(json_str)
                return result
            else:
                self.logger.warning("LLM响应格式异常")
                return self._get_fallback_recommendations(data)
                
        except Exception as e:
            self.logger.error(f"解析响应失败: {str(e)}")
            return self._extract_partial_info(response, data)
    
    def _fix_json_format(self, json_str: str) -> str:
        """修复JSON格式问题"""
        # 移除末尾的未完成字段
        lines = json_str.split('\n')
        fixed_lines = []
        for line in lines:
            if line.strip().startswith('"') and not line.strip().endswith(','):
                # 跳过未完成的字段行
                continue
            fixed_lines.append(line)
        
        # 确保JSON结构完整
        fixed_json = '\n'.join(fixed_lines)
        
        # 如果末尾缺少闭合括号，尝试补充
        open_braces = fixed_json.count('{')
        close_braces = fixed_json.count('}')
        if open_braces > close_braces:
            fixed_json += '}' * (open_braces - close_braces)
        
        return fixed_json
    
    def _extract_partial_info(self, response: str, data: pd.DataFrame) -> Dict[str, Any]:
        """从响应中提取部分信息"""
        result = {
            'model_recommendations': [],
            'field_analyses': [],
            'data_quality_assessment': {},
            'business_insights': []
        }
        
        # 尝试提取模型推荐
        if 'descriptive' in response.lower():
            result['model_recommendations'].append({
                'name': 'descriptive_statistics',
                'confidence': 0.8,
                'reason': '基于数据特征推荐描述性统计'
            })
        
        if 'correlation' in response.lower():
            result['model_recommendations'].append({
                'name': 'correlation_analysis',
                'confidence': 0.7,
                'reason': '检测到数值型字段，推荐相关性分析'
            })
        
        if 'time' in response.lower() or 'date' in response.lower():
            result['model_recommendations'].append({
                'name': 'time_series_analysis',
                'confidence': 0.8,
                'reason': '检测到时间相关字段，推荐时间序列分析'
            })
        
        return result
    
    def _get_fallback_recommendations(self, data: pd.DataFrame) -> Dict[str, Any]:
        """获取备用推荐"""
        recommendations = []
        
        # 基于数据特征的基础推荐
        numeric_cols = data.select_dtypes(include=['number']).columns
        categorical_cols = data.select_dtypes(include=['object', 'category']).columns
        
        if len(numeric_cols) > 1:
            recommendations.append({
                'name': 'correlation_analysis',
                'confidence': 0.8,
                'reason': '检测到多个数值型字段，适合进行相关性分析'
            })
        
        if len(categorical_cols) > 0:
            recommendations.append({
                'name': 'categorical_analysis',
                'confidence': 0.7,
                'reason': '检测到分类字段，适合进行分类分析'
            })
        
        # 默认推荐描述性统计
        recommendations.append({
            'name': 'descriptive_statistics',
            'confidence': 0.9,
            'reason': '基础描述性统计，适用于所有数据集'
        })
        
        return {
            'model_recommendations': recommendations,
            'field_analyses': [],
            'data_quality_assessment': {},
            'business_insights': []
        }
