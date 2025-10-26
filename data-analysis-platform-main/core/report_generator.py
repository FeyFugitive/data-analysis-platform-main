#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Report Generation Module
Generate standardized analysis reports (supports Markdown and HTML formats)
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional
import logging
import os
from datetime import datetime
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False

class ReportGenerator:
    """Report Generator"""
    
    def __init__(self, output_dir: str = "./output"):
        """
        Initialize report generator
        
        Args:
            output_dir: Output directory
        """
        self.output_dir = output_dir
        self.charts_dir = os.path.join(output_dir, "charts")
        self.logger = logging.getLogger(__name__)
        
        # Create output directories
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.charts_dir, exist_ok=True)
        
    def generate_algorithm_judgment_table(self, recommendations: List[Dict[str, Any]], 
                                        should_execute: bool, format_type: str = 'markdown') -> str:
        """
        Generate algorithm determination results table
        
        Args:
            recommendations: Algorithm recommendation list
            should_execute: Whether to execute algorithms
            format_type: Output format ('markdown' or 'html')
            
        Returns:
            str: Formatted table
        """
        if format_type == 'html':
            return self._generate_algorithm_judgment_table_html(recommendations, should_execute)
        else:
            return self._generate_algorithm_judgment_table_markdown(recommendations, should_execute)
    
    def _generate_algorithm_judgment_table_markdown(self, recommendations: List[Dict[str, Any]], 
                                                  should_execute: bool) -> str:
        """生成Markdown格式的算法判定结果表格"""
        table = "## 《算法判定结果》\n\n"
        table += "| 算法 | 适用理由 | 置信度 | 是否执行 | 备注 |\n"
        table += "|------|----------|--------|----------|------|\n"
        
        for rec in recommendations:
            algorithm = rec.get('algorithm', '')
            reason = rec.get('reason', '')
            confidence = rec.get('confidence', 0.0)
            executable = rec.get('executable', False)
            
            if should_execute and executable and confidence >= 0.8:
                execute_status = "✅ 执行"
                remark = "条件满足，已执行"
            else:
                execute_status = "❌ 未执行"
                if not executable:
                    remark = "条件不满足"
                elif confidence < 0.8:
                    remark = "置信度不足"
                else:
                    remark = "信息不足"
            
            table += f"| {algorithm} | {reason} | {confidence:.2f} | {execute_status} | {remark} |\n"
        
        return table
    
    def _generate_algorithm_judgment_table_html(self, recommendations: List[Dict[str, Any]], 
                                               should_execute: bool) -> str:
        """Generate HTML format algorithm determination results table"""
        html = """
        <div class="algorithm-judgment">
            <h2>《Algorithm Determination Results》</h2>
            <table class="judgment-table">
                <thead>
                    <tr>
                        <th>Algorithm</th>
                        <th>Applicable Reason</th>
                        <th>Confidence</th>
                        <th>Executed</th>
                        <th>Notes</th>
                    </tr>
                </thead>
                <tbody>
        """
        
        for rec in recommendations:
            algorithm = rec.get('algorithm', '')
            reason = rec.get('reason', '')
            confidence = rec.get('confidence', 0.0)
            executable = rec.get('executable', False)
            
            if should_execute and executable and confidence >= 0.8:
                execute_status = "✅ 执行"
                status_class = "status-success"
                remark = "条件满足，已执行"
            else:
                execute_status = "❌ 未执行"
                status_class = "status-failed"
                if not executable:
                    remark = "条件不满足"
                elif confidence < 0.8:
                    remark = "置信度不足"
                else:
                    remark = "信息不足"
            
            html += f"""
                    <tr>
                        <td>{algorithm}</td>
                        <td>{reason}</td>
                        <td>{confidence:.2f}</td>
                        <td class="{status_class}">{execute_status}</td>
                        <td>{remark}</td>
                    </tr>
            """
        
        html += """
                </tbody>
            </table>
        </div>
        """
        
        return html
        
    def generate_analysis_report(self, data: pd.DataFrame, examination_result: Dict[str, Any],
                               algorithm_results: Dict[str, Any], 
                               recommendations: List[Dict[str, Any]], 
                               format_type: str = 'markdown') -> str:
        """
        生成分析报告
        
        Args:
            data: 原始数据
            examination_result: 体检结果
            algorithm_results: 算法执行结果
            recommendations: 算法推荐列表
            format_type: 输出格式 ('markdown' 或 'html')
            
        Returns:
            str: 格式化的报告
        """
        try:
            self.logger.info("开始生成分析报告")
            
            if format_type == 'html':
                return self._generate_analysis_report_html(data, examination_result, algorithm_results, recommendations)
            else:
                return self._generate_analysis_report_markdown(data, examination_result, algorithm_results, recommendations)
                
        except Exception as e:
            self.logger.error(f"生成分析报告失败: {str(e)}")
            import traceback
            self.logger.error(f"错误详情: {traceback.format_exc()}")
            return f"生成报告失败: {str(e)}"
    
    def _generate_analysis_report_markdown(self, data: pd.DataFrame, examination_result: Dict[str, Any],
                                         algorithm_results: Dict[str, Any], 
                                         recommendations: List[Dict[str, Any]]) -> str:
        """生成Markdown格式的分析报告"""
        report = "# 数据分析报告\n\n"
        report += f"**生成时间**: {datetime.now().strftime('%Y年%m月%d日 %H:%M:%S')}\n\n"
        
        # 数据概览
        self.logger.info("生成数据概览")
        report += self._generate_data_overview(data, examination_result)
        
        # 各算法结果
        self.logger.info("生成算法结果")
        report += self._generate_algorithm_results(algorithm_results)
        
        # 关键洞察
        self.logger.info("生成关键洞察")
        report += self._generate_key_insights(algorithm_results)
        
        # 局限与下一步建议
        self.logger.info("生成局限与建议")
        report += self._generate_limitations_and_suggestions(data, examination_result)
        
        self.logger.info("分析报告生成完成")
        return report
    
    def _generate_analysis_report_html(self, data: pd.DataFrame, examination_result: Dict[str, Any],
                                     algorithm_results: Dict[str, Any], 
                                     recommendations: List[Dict[str, Any]]) -> str:
        """生成HTML格式的分析报告"""
        # 生成图表
        chart_paths = self.generate_charts(data, algorithm_results)
        
        # 生成各部分内容
        data_overview = self._generate_data_overview_html(data, examination_result)
        algorithm_judgment = self._generate_algorithm_judgment_table_html(recommendations, True)
        algorithm_results_html = self._generate_algorithm_results_html(algorithm_results, chart_paths)
        key_insights = self._generate_key_insights_html(algorithm_results)
        limitations_suggestions = self._generate_limitations_and_suggestions_html(data, examination_result)
        
        # HTML模板
        html_content = f"""
        <!DOCTYPE html>
        <html lang="zh-CN">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>智能归因分析报告</title>
            <style>
                body {{
                    font-family: 'Microsoft YaHei', 'SimHei', Arial, sans-serif;
                    line-height: 1.6;
                    margin: 0;
                    padding: 20px;
                    background-color: #f5f5f5;
                }}
                .container {{
                    max-width: 1200px;
                    margin: 0 auto;
                    background-color: white;
                    padding: 30px;
                    border-radius: 10px;
                    box-shadow: 0 2px 10px rgba(0,0,0,0.1);
                }}
                h1 {{
                    color: #2c3e50;
                    text-align: center;
                    border-bottom: 3px solid #3498db;
                    padding-bottom: 10px;
                    margin-bottom: 30px;
                }}
                h2 {{
                    color: #34495e;
                    border-left: 4px solid #3498db;
                    padding-left: 15px;
                    margin-top: 30px;
                }}
                h3 {{
                    color: #2c3e50;
                    margin-top: 25px;
                }}
                .info-box {{
                    background-color: #ecf0f1;
                    border-left: 4px solid #3498db;
                    padding: 15px;
                    margin: 15px 0;
                    border-radius: 5px;
                }}
                .success-box {{
                    background-color: #d5f4e6;
                    border-left: 4px solid #27ae60;
                    padding: 15px;
                    margin: 15px 0;
                    border-radius: 5px;
                }}
                .warning-box {{
                    background-color: #fef9e7;
                    border-left: 4px solid #f39c12;
                    padding: 15px;
                    margin: 15px 0;
                    border-radius: 5px;
                }}
                .judgment-table {{
                    width: 100%;
                    border-collapse: collapse;
                    margin: 20px 0;
                    box-shadow: 0 2px 5px rgba(0,0,0,0.1);
                }}
                .judgment-table th,
                .judgment-table td {{
                    border: 1px solid #ddd;
                    padding: 12px;
                    text-align: left;
                }}
                .judgment-table th {{
                    background-color: #3498db;
                    color: white;
                    font-weight: bold;
                }}
                .judgment-table tr:nth-child(even) {{
                    background-color: #f8f9fa;
                }}
                .judgment-table tr:hover {{
                    background-color: #e8f4fd;
                }}
                .status-success {{
                    color: #27ae60;
                    font-weight: bold;
                }}
                .status-failed {{
                    color: #e74c3c;
                    font-weight: bold;
                }}
                .chart-container {{
                    text-align: center;
                    margin: 20px 0;
                }}
                .chart-container img {{
                    max-width: 100%;
                    height: auto;
                    border-radius: 8px;
                    box-shadow: 0 4px 8px rgba(0,0,0,0.1);
                }}
                .insights-list {{
                    background-color: #f8f9fa;
                    padding: 20px;
                    border-radius: 8px;
                    margin: 20px 0;
                }}
                .insights-list li {{
                    margin: 10px 0;
                    padding: 8px 0;
                    border-bottom: 1px solid #e9ecef;
                }}
                .insights-list li:last-child {{
                    border-bottom: none;
                }}
                .limitations {{
                    background-color: #fff3cd;
                    border: 1px solid #ffeaa7;
                    border-radius: 8px;
                    padding: 20px;
                    margin: 20px 0;
                }}
                .suggestions {{
                    background-color: #d1ecf1;
                    border: 1px solid #bee5eb;
                    border-radius: 8px;
                    padding: 20px;
                    margin: 20px 0;
                }}
                .metric-grid {{
                    display: grid;
                    grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
                    gap: 15px;
                    margin: 20px 0;
                }}
                .metric-card {{
                    background-color: #f8f9fa;
                    padding: 15px;
                    border-radius: 8px;
                    text-align: center;
                    border-left: 4px solid #3498db;
                }}
                .metric-value {{
                    font-size: 24px;
                    font-weight: bold;
                    color: #2c3e50;
                }}
                .metric-label {{
                    color: #7f8c8d;
                    margin-top: 5px;
                }}
            </style>
        </head>
        <body>
            <div class="container">
                <h1>智能归因分析报告</h1>
                
                <div class="info-box">
                    <strong>生成时间</strong>: {datetime.now().strftime('%Y年%m月%d日 %H:%M:%S')}
                </div>
                
                {data_overview}
                
                {algorithm_judgment}
                
                {algorithm_results_html}
                
                {key_insights}
                
                {limitations_suggestions}
            </div>
        </body>
        </html>
        """
        
        self.logger.info("HTML分析报告生成完成")
        return html_content
    
    def _generate_data_overview(self, data: pd.DataFrame, examination_result: Dict[str, Any]) -> str:
        """生成数据概览"""
        overview = "## 数据概览\n\n"
        
        # 基础信息
        overview += f"- **数据行数**: {len(data):,}\n"
        overview += f"- **数据列数**: {len(data.columns)}\n"
        overview += f"- **内存使用**: {data.memory_usage(deep=True).sum() / 1024 / 1024:.2f} MB\n"
        overview += f"- **重复记录**: {data.duplicated().sum():,}\n"
        overview += f"- **总缺失值**: {data.isnull().sum().sum():,}\n"
        
        # 列类型统计
        column_types = examination_result.get('column_types', {})
        type_counts = {}
        for col_type in column_types.values():
            type_counts[col_type] = type_counts.get(col_type, 0) + 1
        
        overview += "\n### 列类型分布\n"
        for col_type, count in type_counts.items():
            overview += f"- **{col_type}**: {count}列\n"
        
        # 缺失值统计
        missing_rates = examination_result.get('missing_rates', {})
        high_missing_cols = [col for col, rate in missing_rates.items() if rate > 0.5]
        if high_missing_cols:
            overview += f"\n### 高缺失值列（>50%）\n"
            for col in high_missing_cols[:5]:  # 只显示前5个
                overview += f"- {col}: {missing_rates[col]:.1%}\n"
        
        # 日期解析率
        date_parse_rates = examination_result.get('date_parse_rates', {})
        datetime_cols = [col for col, rate in date_parse_rates.items() if rate > 0.8]
        if datetime_cols:
            overview += f"\n### 可解析日期列（>80%）\n"
            for col in datetime_cols:
                overview += f"- {col}: {date_parse_rates[col]:.1%}\n"
        
        overview += "\n"
        return overview
    
    def _generate_data_overview_html(self, data: pd.DataFrame, examination_result: Dict[str, Any]) -> str:
        """生成HTML格式的数据概览"""
        html = """
        <div class="info-box">
            <h3>数据概览</h3>
            <div class="metric-grid">
                <div class="metric-card">
                    <div class="metric-value">{total_rows:,}</div>
                    <div class="metric-label">数据行数</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value">{total_columns}</div>
                    <div class="metric-label">数据列数</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value">{memory_usage:.2f} MB</div>
                    <div class="metric-label">内存使用</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value">{duplicate_records:,}</div>
                    <div class="metric-label">重复记录</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value">{total_missing:,}</div>
                    <div class="metric-label">总缺失值</div>
                </div>
            </div>
        </div>
        """
        
        total_rows = len(data)
        total_columns = len(data.columns)
        memory_usage = data.memory_usage(deep=True).sum() / 1024 / 1024
        duplicate_records = data.duplicated().sum()
        total_missing = data.isnull().sum().sum()
        
        return html.format(
            total_rows=total_rows,
            total_columns=total_columns,
            memory_usage=memory_usage,
            duplicate_records=duplicate_records,
            total_missing=total_missing
        )
    
    def _generate_algorithm_results(self, algorithm_results: Dict[str, Any]) -> str:
        """生成各算法结果"""
        results = "## 各算法结果\n\n"
        
        for algorithm_name, result in algorithm_results.items():
            if 'error' in result:
                results += f"### {algorithm_name}\n\n"
                results += f"**执行状态**: ❌ 执行失败\n"
                results += f"**错误信息**: {result['error']}\n\n"
                continue
            
            results += f"### {algorithm_name}\n\n"
            
            if algorithm_name == '描述统计':
                results += self._format_descriptive_stats(result)
            elif algorithm_name == '最后点击归因':
                results += self._format_last_click_attribution(result)
            elif algorithm_name == '首次点击归因':
                results += self._format_first_click_attribution(result)
            elif algorithm_name == 'Markov渠道模型':
                results += self._format_markov_channel_model(result)
            elif algorithm_name == 'Markov吸收链模型':
                results += self._format_markov_absorption_model(result)
            elif algorithm_name == '多维度归因分析':
                results += self._format_multi_dimension_attribution(result)
            elif algorithm_name == '相关性分析':
                results += self._format_correlation_analysis(result)
            elif algorithm_name == '时间趋势分析':
                results += self._format_trend_analysis(result)
            elif algorithm_name == '分类分析':
                results += self._format_classification_analysis(result)
            else:
                results += f"**结果**: {str(result)}\n\n"
        
        return results
    
    def _generate_algorithm_results_html(self, algorithm_results: Dict[str, Any], chart_paths: List[str]) -> str:
        """生成HTML格式的各算法结果"""
        html = """
        <div class="info-box">
            <h3>各算法结果</h3>
        """
        
        for algorithm_name, result in algorithm_results.items():
            if 'error' in result:
                html += f"""
                <div class="warning-box">
                    <h4>{algorithm_name}</h4>
                    <p><strong>执行状态</strong>: ❌ 执行失败</p>
                    <p><strong>错误信息</strong>: {result['error']}</p>
                </div>
                """
                continue
            
            html += f"""
                <div class="success-box">
                    <h4>{algorithm_name}</h4>
            """
            
            if algorithm_name == '描述统计':
                html += self._format_descriptive_stats_html(result)
            elif algorithm_name == '最后点击归因':
                html += self._format_last_click_attribution_html(result)
            elif algorithm_name == '首次点击归因':
                html += self._format_first_click_attribution_html(result)
            elif algorithm_name == 'Markov渠道模型':
                html += self._format_markov_channel_model_html(result)
            elif algorithm_name == 'Markov吸收链模型':
                html += self._format_markov_absorption_model_html(result)
            elif algorithm_name == '多维度归因分析':
                html += self._format_multi_dimension_attribution_html(result)
            elif algorithm_name == '相关性分析':
                html += self._format_correlation_analysis_html(result)
            elif algorithm_name == '时间趋势分析':
                html += self._format_trend_analysis_html(result, chart_paths)
            elif algorithm_name == '分类分析':
                html += self._format_classification_analysis_html(result)
            else:
                html += f"**结果**: {str(result)}\n\n"
            html += """
                </div>
            """
        
        html += """
        </div>
        """
        return html
    
    def _format_descriptive_stats(self, result: Dict[str, Any]) -> str:
        """格式化描述统计结果"""
        formatted = ""
        
        basic_stats = result.get('basic_stats', {})
        formatted += f"**基础统计**:\n"
        formatted += f"- 总记录数: {basic_stats.get('total_records', 0):,}\n"
        formatted += f"- 总列数: {basic_stats.get('total_columns', 0)}\n"
        formatted += f"- 重复记录: {basic_stats.get('duplicate_records', 0):,}\n"
        formatted += f"- 总缺失值: {basic_stats.get('missing_values_total', 0):,}\n\n"
        
        numerical_stats = result.get('numerical_stats', {})
        if numerical_stats:
            formatted += f"**数值列统计** (共{len(numerical_stats)}列):\n"
            for col, stats in list(numerical_stats.items())[:5]:  # 只显示前5个
                formatted += f"- {col}: 均值={stats['mean']:.2f}, 标准差={stats['std']:.2f}, 范围=[{stats['min']:.2f}, {stats['max']:.2f}]\n"
            if len(numerical_stats) > 5:
                formatted += f"- ... 还有{len(numerical_stats)-5}个数值列\n"
            formatted += "\n"
        
        categorical_stats = result.get('categorical_stats', {})
        if categorical_stats:
            formatted += f"**分类列统计** (共{len(categorical_stats)}列):\n"
            for col, stats in list(categorical_stats.items())[:3]:  # 只显示前3个
                formatted += f"- {col}: {stats['unique_count']}个唯一值, 最常见='{stats['most_common']}' ({stats['most_common_count']}次)\n"
            if len(categorical_stats) > 3:
                formatted += f"- ... 还有{len(categorical_stats)-3}个分类列\n"
            formatted += "\n"
        
        return formatted
    
    def _format_correlation_analysis(self, result: Dict[str, Any]) -> str:
        """格式化相关性分析结果"""
        formatted = ""
        
        strong_correlations = result.get('strong_correlations', [])
        formatted += f"**强相关关系** (共{len(strong_correlations)}对):\n"
        
        for i, corr in enumerate(strong_correlations[:10]):  # 只显示前10个
            strength_emoji = "🔥" if corr['strength'] == 'strong' else "⚡"
            formatted += f"{i+1}. {strength_emoji} {corr['variable1']} ↔ {corr['variable2']}: r={corr['correlation']:.3f}\n"
        
        if len(strong_correlations) > 10:
            formatted += f"- ... 还有{len(strong_correlations)-10}对强相关关系\n"
        
        formatted += "\n"
        return formatted
    
    def _format_correlation_analysis_html(self, result: Dict[str, Any]) -> str:
        """格式化HTML格式的相关性分析结果"""
        html = """
        <div class="info-box">
            <h3>相关性分析</h3>
        """
        
        strong_correlations = result.get('strong_correlations', [])
        if strong_correlations:
            html += "<p>**强相关关系** (共{}对):</p>".format(len(strong_correlations))
            html += "<ul>"
            for i, corr in enumerate(strong_correlations[:10]):  # 只显示前10个
                strength_emoji = "🔥" if corr['strength'] == 'strong' else "⚡"
                html += f"<li>{i+1}. {strength_emoji} {corr['variable1']} ↔ {corr['variable2']}: r={corr['correlation']:.3f}</li>"
            if len(strong_correlations) > 10:
                html += f"<li>- ... 还有{len(strong_correlations)-10}对强相关关系</li>"
            html += "</ul>"
        else:
            html += "<p>未发现强相关关系。</p>"
        html += "</div>"
        return html
    
    def _format_trend_analysis(self, result: Dict[str, Any]) -> str:
        """格式化时间趋势分析结果"""
        formatted = ""
        
        # 过滤掉非时间列的字段
        time_columns = [col for col in result.keys() if col not in ['data_cleaning_info', 'data_warnings', 'algorithm_name', 'execution_timestamp']]
        
        for datetime_col in time_columns:
            col_result = result[datetime_col]
            # 添加类型检查
            if not isinstance(col_result, dict):
                formatted += f"**{datetime_col}**: ❌ 结果格式错误\n\n"
                continue
                
            if 'error' in col_result:
                formatted += f"**{datetime_col}**: ❌ {col_result['error']}\n\n"
                continue
            
            time_stats = col_result.get('time_stats', {})
            formatted += f"**{datetime_col}**:\n"
            formatted += f"- 时间范围: {time_stats.get('start_date', 'N/A')} 至 {time_stats.get('end_date', 'N/A')}\n"
            formatted += f"- 总天数: {time_stats.get('total_days', 0)}天\n"
            formatted += f"- 日均记录: {time_stats.get('records_per_day', 0):.1f}条\n\n"
            
            numeric_trends = col_result.get('numeric_trends', {})
            if numeric_trends:
                formatted += f"**数值列趋势**:\n"
                for col, trend in list(numeric_trends.items())[:3]:  # 只显示前3个
                    if isinstance(trend, dict) and 'error' not in trend:
                        direction_emoji = "📈" if trend.get('trend_direction') == 'increasing' else "📉"
                        strength_text = "强" if trend.get('trend_strength') == 'strong' else "中等" if trend.get('trend_strength') == 'moderate' else "弱"
                        formatted += f"- {direction_emoji} {col}: {strength_text}{trend.get('trend_direction', '')}趋势 (R²={trend.get('r_squared', 0):.3f})\n"
                if len(numeric_trends) > 3:
                    formatted += f"- ... 还有{len(numeric_trends)-3}个数值列\n"
                formatted += "\n"
        
        return formatted
    
    def _format_trend_analysis_html(self, result: Dict[str, Any], chart_paths: List[str]) -> str:
        """格式化HTML格式的时间趋势分析结果"""
        html = """
        <div class="info-box">
            <h3>时间趋势分析</h3>
        """
        
        # 过滤掉非时间列的字段
        time_columns = [col for col in result.keys() if col not in ['data_cleaning_info', 'data_warnings', 'algorithm_name', 'execution_timestamp']]
        
        for datetime_col in time_columns:
            col_result = result[datetime_col]
            # 添加类型检查
            if not isinstance(col_result, dict):
                html += f"""
                <div class="warning-box">
                    <h4>{datetime_col}</h4>
                    <p><strong>结果格式错误</strong></p>
                </div>
                """
                continue
                
            if 'error' in col_result:
                html += f"""
                <div class="warning-box">
                    <h4>{datetime_col}</h4>
                    <p><strong>错误信息</strong>: {col_result['error']}</p>
                </div>
                """
                continue
            
            time_stats = col_result.get('time_stats', {})
            html += f"""
                <div class="success-box">
                    <h4>{datetime_col}</h4>
                    <p><strong>时间范围</strong>: {time_stats.get('start_date', 'N/A')} 至 {time_stats.get('end_date', 'N/A')}</p>
                    <p><strong>总天数</strong>: {time_stats.get('total_days', 0)}天</p>
                    <p><strong>日均记录</strong>: {time_stats.get('records_per_day', 0):.1f}条</p>
                """
            
            numeric_trends = col_result.get('numeric_trends', {})
            if numeric_trends:
                html += "<p>**数值列趋势**:</p>"
                html += "<ul>"
                for col, trend in list(numeric_trends.items())[:3]:  # 只显示前3个
                    if isinstance(trend, dict) and 'error' not in trend:
                        direction_emoji = "📈" if trend.get('trend_direction') == 'increasing' else "📉"
                        strength_text = "强" if trend.get('trend_strength') == 'strong' else "中等" if trend.get('trend_strength') == 'moderate' else "弱"
                        html += f"<li>{direction_emoji} {col}: {strength_text}{trend.get('trend_direction', '')}趋势 (R²={trend.get('r_squared', 0):.3f})</li>"
                if len(numeric_trends) > 3:
                    html += f"<li>- ... 还有{len(numeric_trends)-3}个数值列</li>"
                html += "</ul>"
            html += "</div>"
        
        html += "</div>"
        return html
    
    def _format_last_click_attribution(self, result: Dict[str, Any]) -> str:
        """格式化最后点击归因结果"""
        formatted = ""
        
        if 'error' in result:
            formatted += f"❌ {result['error']}\n\n"
            return formatted
        
        attribution_method = result.get('attribution_method', '')
        total_users = result.get('total_users', 0)
        channel_attribution = result.get('channel_attribution', {})
        conversion_rates = result.get('conversion_rates', {})
        top_channels = result.get('top_channels', [])
        
        formatted += f"**归因方法**: {attribution_method}\n"
        formatted += f"**总用户数**: {total_users:,}\n"
        formatted += f"**渠道数量**: {len(channel_attribution)}\n\n"
        
        # 渠道归因结果
        formatted += f"**渠道归因结果** (前5个):\n"
        for i, (channel, count) in enumerate(top_channels):
            conversion_rate = conversion_rates.get(channel, 0)
            formatted += f"{i+1}. {channel}: {count:,}次 ({conversion_rate:.1%})\n"
        formatted += "\n"
        
        return formatted
    
    def _format_last_click_attribution_html(self, result: Dict[str, Any]) -> str:
        """格式化HTML格式的最后点击归因结果"""
        html = """
        <div class="info-box">
            <h3>最后点击归因</h3>
        """
        
        if 'error' in result:
            html += f"""
            <div class="warning-box">
                <p>❌ {result['error']}</p>
            </div>
            """
            return html
        
        attribution_method = result.get('attribution_method', '')
        total_users = result.get('total_users', 0)
        channel_attribution = result.get('channel_attribution', {})
        conversion_rates = result.get('conversion_rates', {})
        top_channels = result.get('top_channels', [])
        
        html += f"""
            <p><strong>归因方法</strong>: {attribution_method}</p>
            <p><strong>总用户数</strong>: {total_users:,}</p>
            <p><strong>渠道数量</strong>: {len(channel_attribution)}</p>
        """
        
        # 渠道归因结果
        html += "<p>**渠道归因结果** (前5个):</p>"
        html += "<ul>"
        for i, (channel, count) in enumerate(top_channels):
            conversion_rate = conversion_rates.get(channel, 0)
            html += f"<li>{i+1}. {channel}: {count:,}次 ({conversion_rate:.1%})</li>"
        html += "</ul>"
        html += "</div>"
        return html
    
    def _format_first_click_attribution(self, result: Dict[str, Any]) -> str:
        """格式化首次点击归因结果"""
        formatted = ""
        
        if 'error' in result:
            formatted += f"❌ {result['error']}\n\n"
            return formatted
        
        attribution_method = result.get('attribution_method', '')
        total_users = result.get('total_users', 0)
        channel_attribution = result.get('channel_attribution', {})
        conversion_rates = result.get('conversion_rates', {})
        top_channels = result.get('top_channels', [])
        
        formatted += f"**归因方法**: {attribution_method}\n"
        formatted += f"**总用户数**: {total_users:,}\n"
        formatted += f"**渠道数量**: {len(channel_attribution)}\n\n"
        
        # 渠道归因结果
        formatted += f"**渠道归因结果** (前5个):\n"
        for i, (channel, count) in enumerate(top_channels):
            conversion_rate = conversion_rates.get(channel, 0)
            formatted += f"{i+1}. {channel}: {count:,}次 ({conversion_rate:.1%})\n"
        formatted += "\n"
        
        return formatted
    
    def _format_first_click_attribution_html(self, result: Dict[str, Any]) -> str:
        """格式化HTML格式的首次点击归因结果"""
        html = """
        <div class="info-box">
            <h3>首次点击归因</h3>
        """
        
        if 'error' in result:
            html += f"""
            <div class="warning-box">
                <p>❌ {result['error']}</p>
            </div>
            """
            return html
        
        attribution_method = result.get('attribution_method', '')
        total_users = result.get('total_users', 0)
        channel_attribution = result.get('channel_attribution', {})
        conversion_rates = result.get('conversion_rates', {})
        top_channels = result.get('top_channels', [])
        
        html += f"""
            <p><strong>归因方法</strong>: {attribution_method}</p>
            <p><strong>总用户数</strong>: {total_users:,}</p>
            <p><strong>渠道数量</strong>: {len(channel_attribution)}</p>
        """
        
        # 渠道归因结果
        html += "<p>**渠道归因结果** (前5个):</p>"
        html += "<ul>"
        for i, (channel, count) in enumerate(top_channels):
            conversion_rate = conversion_rates.get(channel, 0)
            html += f"<li>{i+1}. {channel}: {count:,}次 ({conversion_rate:.1%})</li>"
        html += "</ul>"
        html += "</div>"
        return html
    
    def _format_markov_channel_model(self, result: Dict[str, Any]) -> str:
        """格式化Markov渠道模型结果"""
        formatted = ""
        
        if 'error' in result:
            formatted += f"❌ {result['error']}\n\n"
            return formatted
        
        attribution_method = result.get('attribution_method', '')
        total_users = result.get('total_users', 0)
        unique_channels = result.get('unique_channels', 0)
        transition_matrix_shape = result.get('transition_matrix_shape', (0, 0))
        removal_effects = result.get('removal_effects', {})
        top_removal_effects = result.get('top_removal_effects', [])
        
        formatted += f"**归因方法**: {attribution_method}\n"
        formatted += f"**总用户数**: {total_users:,}\n"
        formatted += f"**渠道数量**: {unique_channels}\n"
        formatted += f"**转移矩阵**: {transition_matrix_shape[0]}×{transition_matrix_shape[1]}\n\n"
        
        # Removal Effect结果
        formatted += f"**Removal Effect** (前5个):\n"
        for i, (state, effect) in enumerate(top_removal_effects):
            formatted += f"{i+1}. {state}: {effect:.2f}%\n"
        formatted += "\n"
        
        return formatted
    
    def _format_markov_channel_model_html(self, result: Dict[str, Any]) -> str:
        """格式化HTML格式的Markov渠道模型结果"""
        html = """
        <div class="info-box">
            <h3>Markov渠道模型</h3>
        """
        
        if 'error' in result:
            html += f"""
            <div class="warning-box">
                <p>❌ {result['error']}</p>
            </div>
            """
            return html
        
        attribution_method = result.get('attribution_method', '')
        total_users = result.get('total_users', 0)
        unique_channels = result.get('unique_channels', 0)
        transition_matrix_shape = result.get('transition_matrix_shape', (0, 0))
        removal_effects = result.get('removal_effects', {})
        top_removal_effects = result.get('top_removal_effects', [])
        
        html += f"""
            <p><strong>归因方法</strong>: {attribution_method}</p>
            <p><strong>总用户数</strong>: {total_users:,}</p>
            <p><strong>渠道数量</strong>: {unique_channels}</p>
            <p><strong>转移矩阵</strong>: {transition_matrix_shape[0]}×{transition_matrix_shape[1]}</p>
        """
        
        # Removal Effect结果
        html += "<p>**Removal Effect** (前5个):</p>"
        html += "<ul>"
        for i, (state, effect) in enumerate(top_removal_effects):
            html += f"<li>{i+1}. {state}: {effect:.2f}%</li>"
        html += "</ul>"
        html += "</div>"
        return html
    
    def _format_markov_absorption_model(self, result: Dict[str, Any]) -> str:
        """格式化Markov吸收链模型结果"""
        formatted = ""
        
        if 'error' in result:
            formatted += f"❌ {result['error']}\n\n"
            return formatted
        
        attribution_method = result.get('attribution_method', '')
        total_users = result.get('total_users', 0)
        conversion_rate = result.get('conversion_rate', 0)
        transition_matrix_shape = result.get('transition_matrix_shape', (0, 0))
        absorption_probabilities = result.get('absorption_probabilities', {})
        
        formatted += f"**归因方法**: {attribution_method}\n"
        formatted += f"**总用户数**: {total_users:,}\n"
        formatted += f"**转化率**: {conversion_rate:.1%}\n"
        formatted += f"**转移矩阵**: {transition_matrix_shape[0]}×{transition_matrix_shape[1]}\n\n"
        
        # 吸收概率
        formatted += f"**吸收概率**:\n"
        for state, probability in absorption_probabilities.items():
            formatted += f"- {state}: {probability:.1%}\n"
        formatted += "\n"
        
        return formatted
    
    def _format_markov_absorption_model_html(self, result: Dict[str, Any]) -> str:
        """格式化HTML格式的Markov吸收链模型结果"""
        html = """
        <div class="info-box">
            <h3>Markov吸收链模型</h3>
        """
        
        if 'error' in result:
            html += f"""
            <div class="warning-box">
                <p>❌ {result['error']}</p>
            </div>
            """
            return html
        
        attribution_method = result.get('attribution_method', '')
        total_users = result.get('total_users', 0)
        conversion_rate = result.get('conversion_rate', 0)
        transition_matrix_shape = result.get('transition_matrix_shape', (0, 0))
        absorption_probabilities = result.get('absorption_probabilities', {})
        
        html += f"""
            <p><strong>归因方法</strong>: {attribution_method}</p>
            <p><strong>总用户数</strong>: {total_users:,}</p>
            <p><strong>转化率</strong>: {conversion_rate:.1%}</p>
            <p><strong>转移矩阵</strong>: {transition_matrix_shape[0]}×{transition_matrix_shape[1]}</p>
        """
        
        # 吸收概率
        html += "<p>**吸收概率**:</p>"
        html += "<ul>"
        for state, probability in absorption_probabilities.items():
            html += f"<li>{state}: {probability:.1%}</li>"
        html += "</ul>"
        html += "</div>"
        return html
    
    def _format_multi_dimension_attribution(self, result: Dict[str, Any]) -> str:
        """格式化多维度归因分析结果"""
        formatted = ""
        
        if 'error' in result:
            formatted += f"❌ {result['error']}\n\n"
            return formatted
        
        attribution_method = result.get('attribution_method', '')
        total_users = result.get('total_users', 0)
        dimensions_analyzed = result.get('dimensions_analyzed', 0)
        dimension_results = result.get('dimension_results', {})
        
        formatted += f"**归因方法**: {attribution_method}\n"
        formatted += f"**总用户数**: {total_users:,}\n"
        formatted += f"**分析维度**: {dimensions_analyzed}个\n\n"
        
        # 各维度结果
        formatted += f"**各维度分析结果**:\n"
        for dimension, dim_result in dimension_results.items():
            unique_channels = dim_result.get('unique_channels', 0)
            channel_distribution = dim_result.get('channel_distribution', {})
            formatted += f"- {dimension}: {unique_channels}个渠道\n"
            
            # 显示前3个渠道
            top_channels = sorted(channel_distribution.items(), key=lambda x: x[1], reverse=True)[:3]
            for channel, count in top_channels:
                formatted += f"  - {channel}: {count}次\n"
        formatted += "\n"
        
        return formatted
    
    def _format_multi_dimension_attribution_html(self, result: Dict[str, Any]) -> str:
        """格式化HTML格式的多维度归因分析结果"""
        html = """
        <div class="info-box">
            <h3>多维度归因分析</h3>
        """
        
        if 'error' in result:
            html += f"""
            <div class="warning-box">
                <p>❌ {result['error']}</p>
            </div>
            """
            return html
        
        attribution_method = result.get('attribution_method', '')
        total_users = result.get('total_users', 0)
        dimensions_analyzed = result.get('dimensions_analyzed', 0)
        dimension_results = result.get('dimension_results', {})
        
        html += f"""
            <p><strong>归因方法</strong>: {attribution_method}</p>
            <p><strong>总用户数</strong>: {total_users:,}</p>
            <p><strong>分析维度</strong>: {dimensions_analyzed}个</p>
        """
        
        # 各维度结果
        html += "<p>**各维度分析结果**:</p>"
        html += "<ul>"
        for dimension, dim_result in dimension_results.items():
            unique_channels = dim_result.get('unique_channels', 0)
            channel_distribution = dim_result.get('channel_distribution', {})
            html += f"<li>{dimension}: {unique_channels}个渠道</li>"
            
            # 显示前3个渠道
            top_channels = sorted(channel_distribution.items(), key=lambda x: x[1], reverse=True)[:3]
            html += "<ul>"
            for channel, count in top_channels:
                html += f"<li>  - {channel}: {count}次</li>"
            html += "</ul>"
        html += "</ul>"
        html += "</div>"
        return html
    
    def _format_classification_analysis(self, result: Dict[str, Any]) -> str:
        """格式化分类分析结果"""
        formatted = ""
        
        if 'error' in result:
            formatted += f"❌ {result['error']}\n\n"
            return formatted
        
        # 处理分类统计结果（非机器学习分类）
        for col_name, col_result in result.items():
            if col_name in ['algorithm_name', 'execution_timestamp']:
                continue
                
            if not isinstance(col_result, dict):
                continue
                
            formatted += f"**{col_name}**:\n"
            formatted += f"- 总记录数: {col_result.get('total_records', 0):,}\n"
            formatted += f"- 唯一类别数: {col_result.get('unique_categories', 0)}\n"
            formatted += f"- 缺失值: {col_result.get('missing_count', 0)} ({col_result.get('missing_rate', 0):.1%})\n"
            formatted += f"- 多样性指数: {col_result.get('diversity_index', 0):.3f}\n\n"
            
            # 显示主要类别
            major_categories = col_result.get('major_categories', {})
            if major_categories:
                formatted += f"**主要类别** (占比>5%):\n"
                for category, proportion in major_categories.items():
                    formatted += f"- {category}: {proportion:.1f}%\n"
                formatted += "\n"
            
            # 显示前5个类别
            top_categories = col_result.get('top_5_categories', {})
            if top_categories:
                formatted += f"**前5个类别**:\n"
                for i, (category, count) in enumerate(top_categories.items(), 1):
                    formatted += f"{i}. {category}: {count:,}次\n"
                formatted += "\n"
        
        return formatted
    
    def _format_classification_analysis_html(self, result: Dict[str, Any]) -> str:
        """格式化HTML格式的分类分析结果"""
        html = ""
        
        if 'error' in result:
            html += f"<p><strong>❌ {result['error']}</strong></p>"
            return html
        
        # 处理分类统计结果（非机器学习分类）
        for col_name, col_result in result.items():
            if col_name in ['algorithm_name', 'execution_timestamp']:
                continue
                
            if not isinstance(col_result, dict):
                continue
                
            html += f"<h4>{col_name}</h4>"
            html += f"<ul>"
            html += f"<li>总记录数: {col_result.get('total_records', 0):,}</li>"
            html += f"<li>唯一类别数: {col_result.get('unique_categories', 0)}</li>"
            html += f"<li>缺失值: {col_result.get('missing_count', 0)} ({col_result.get('missing_rate', 0):.1%})</li>"
            html += f"<li>多样性指数: {col_result.get('diversity_index', 0):.3f}</li>"
            html += f"</ul>"
            
            # 显示主要类别
            major_categories = col_result.get('major_categories', {})
            if major_categories:
                html += f"<h5>主要类别 (占比>5%)</h5><ul>"
                for category, proportion in major_categories.items():
                    html += f"<li>{category}: {proportion:.1f}%</li>"
                html += f"</ul>"
            
            # 显示前5个类别
            top_categories = col_result.get('top_5_categories', {})
            if top_categories:
                html += f"<h5>前5个类别</h5><ol>"
                for category, count in top_categories.items():
                    html += f"<li>{category}: {count:,}次</li>"
                html += f"</ol>"
        
        return html
    
    def _format_group_difference_analysis(self, result: Dict[str, Any]) -> str:
        """格式化组间差异分析结果"""
        formatted = ""
        
        for cat_col, cat_result in result.items():
            formatted += f"**分组变量: {cat_col}**\n"
            
            significant_tests = 0
            total_tests = 0
            
            for num_col, num_result in cat_result.items():
                if 'error' in num_result:
                    continue
                
                total_tests += 1
                if num_result.get('significant', False):
                    significant_tests += 1
                
                if significant_tests <= 3:  # 只显示前3个显著结果
                    test_type = num_result.get('test_type', '')
                    p_value = num_result.get('p_value', 1)
                    significance_emoji = "✅" if p_value < 0.05 else "❌"
                    formatted += f"- {significance_emoji} {num_col}: {test_type}, p={p_value:.3f}\n"
            
            if total_tests > 0:
                formatted += f"- 显著差异比例: {significant_tests}/{total_tests} ({significant_tests/total_tests:.1%})\n"
            formatted += "\n"
        
        return formatted
    
    def _format_pca_analysis(self, result: Dict[str, Any]) -> str:
        """格式化PCA分析结果"""
        formatted = ""
        
        if 'error' in result:
            formatted += f"❌ {result['error']}\n\n"
            return formatted
        
        original_dims = result.get('original_dimensions', 0)
        n_components_90 = result.get('n_components_90', 0)
        total_variance = result.get('total_variance_explained', 0)
        
        formatted += f"**降维效果**:\n"
        formatted += f"- 原始维度: {original_dims}\n"
        formatted += f"- 解释90%方差所需主成分: {n_components_90}\n"
        formatted += f"- 总解释方差: {total_variance:.1%}\n"
        formatted += f"- 降维比例: {(1-n_components_90/original_dims)*100:.1f}%\n\n"
        
        # 特征重要性
        feature_importance = result.get('feature_importance_pc1', {})
        if feature_importance:
            formatted += f"**第一主成分特征重要性** (前5个):\n"
            for i, (feature, importance) in enumerate(list(feature_importance.items())[:5]):
                formatted += f"{i+1}. {feature}: {importance:.3f}\n"
            formatted += "\n"
        
        return formatted
    
    def _format_kmeans_analysis(self, result: Dict[str, Any]) -> str:
        """格式化KMeans分析结果"""
        formatted = ""
        
        if 'error' in result:
            formatted += f"❌ {result['error']}\n\n"
            return formatted
        
        optimal_k = result.get('optimal_k', 0)
        total_samples = result.get('total_samples', 0)
        silhouette_score = result.get('silhouette_score', 0)
        
        formatted += f"**聚类结果**:\n"
        formatted += f"- 最优聚类数: {optimal_k}\n"
        formatted += f"- 样本总数: {total_samples:,}\n"
        formatted += f"- 轮廓系数: {silhouette_score:.3f}\n\n"
        
        # 聚类统计
        cluster_stats = result.get('cluster_stats', {})
        if cluster_stats:
            formatted += f"**聚类分布**:\n"
            for cluster_name, stats in cluster_stats.items():
                size = stats.get('size', 0)
                percentage = stats.get('percentage', 0)
                formatted += f"- {cluster_name}: {size:,}个样本 ({percentage:.1f}%)\n"
            formatted += "\n"
        
        return formatted
    
    def _generate_key_insights(self, algorithm_results: Dict[str, Any]) -> str:
        """生成关键洞察"""
        insights = "## 关键洞察\n\n"
        
        insight_list = []
        
        # 从描述统计中提取洞察
        if '描述统计' in algorithm_results and 'error' not in algorithm_results['描述统计']:
            desc_result = algorithm_results['描述统计']
            basic_stats = desc_result.get('basic_stats', {})
            
            if basic_stats.get('duplicate_records', 0) > 0:
                insight_list.append(f"数据中存在{basic_stats['duplicate_records']}条重复记录，建议进行数据去重")
            
            if basic_stats.get('missing_values_total', 0) > 0:
                missing_ratio = basic_stats['missing_values_total'] / (basic_stats['total_records'] * basic_stats['total_columns'])
                if missing_ratio > 0.1:
                    insight_list.append(f"数据缺失率较高({missing_ratio:.1%})，建议进行缺失值处理")
        
        # 从归因分析中提取洞察
        for alg_name, result in algorithm_results.items():
            # 添加调试信息
            if not isinstance(result, dict):
                self.logger.warning(f"算法 {alg_name} 的结果不是字典类型: {type(result)}")
                continue
                
            if '归因' in alg_name and 'error' not in result:
                if 'last_click' in result.get('attribution_method', ''):
                    top_channels = result.get('top_channels', [])
                    if top_channels:
                        top_channel = top_channels[0]
                        insight_list.append(f"最后点击归因显示{top_channel[0]}是最有效的转化渠道({top_channel[1]}次)")
                
                elif 'first_click' in result.get('attribution_method', ''):
                    top_channels = result.get('top_channels', [])
                    if top_channels:
                        top_channel = top_channels[0]
                        insight_list.append(f"首次点击归因显示{top_channel[0]}是最有效的获客渠道({top_channel[1]}次)")
                
                elif 'markov_channel' in result.get('attribution_method', ''):
                    top_effects = result.get('top_removal_effects', [])
                    if top_effects:
                        top_effect = top_effects[0]
                        insight_list.append(f"Markov渠道模型显示{top_effect[0]}的移除效果最大({top_effect[1]:.2f}%)")
                
                elif 'markov_absorption' in result.get('attribution_method', ''):
                    conversion_rate = result.get('conversion_rate', 0)
                    insight_list.append(f"Markov吸收链模型显示整体转化率为{conversion_rate:.1%}")
        
        # 从相关性分析中提取洞察
        if '相关性分析' in algorithm_results and 'error' not in algorithm_results['相关性分析']:
            corr_result = algorithm_results['相关性分析']
            strong_corrs = corr_result.get('strong_correlations', [])
            
            if strong_corrs:
                top_corr = strong_corrs[0]
                insight_list.append(f"发现强相关关系: {top_corr['variable1']}与{top_corr['variable2']}的相关系数为{top_corr['correlation']:.3f}")
        
        # 从时间趋势分析中提取洞察
        if '时间趋势分析' in algorithm_results and 'error' not in algorithm_results['时间趋势分析']:
            trend_result = algorithm_results['时间趋势分析']
            if isinstance(trend_result, dict):
                for datetime_col, col_result in trend_result.items():
                    if isinstance(col_result, dict) and 'error' not in col_result:
                        numeric_trends = col_result.get('numeric_trends', {})
                        for col, trend in numeric_trends.items():
                            if isinstance(trend, dict) and 'error' not in trend:
                                direction = trend.get('trend_direction', '')
                                strength = trend.get('trend_strength', '')
                                if direction and strength:
                                    insight_list.append(f"{col}在{datetime_col}维度上呈现{strength}{direction}趋势")
                                    break
        
        # 如果洞察不足，添加通用洞察
        if len(insight_list) < 3:
            insight_list.append("建议进一步探索数据特征，可能需要更多的特征工程")
            insight_list.append("考虑收集更多相关数据以提高分析深度")
        
        # 确保洞察数量在3-5条之间
        if len(insight_list) > 5:
            insight_list = insight_list[:5]
        
        for i, insight in enumerate(insight_list, 1):
            insights += f"{i}. {insight}\n"
        
        insights += "\n"
        return insights
    
    def _generate_key_insights_html(self, algorithm_results: Dict[str, Any]) -> str:
        """生成HTML格式的关键洞察"""
        html = """
        <div class="info-box">
            <h3>关键洞察</h3>
        """
        
        insight_list = []
        
        # 确保algorithm_results是字典类型
        if not isinstance(algorithm_results, dict):
            html += "<p>无法生成洞察：算法结果格式错误</p>"
            html += "</div>"
            return html
        
        # 从描述统计中提取洞察
        if '描述统计' in algorithm_results and isinstance(algorithm_results['描述统计'], dict) and 'error' not in algorithm_results['描述统计']:
            desc_result = algorithm_results['描述统计']
            basic_stats = desc_result.get('basic_stats', {})
            
            if basic_stats.get('duplicate_records', 0) > 0:
                insight_list.append(f"数据中存在{basic_stats['duplicate_records']}条重复记录，建议进行数据去重")
            
            if basic_stats.get('missing_values_total', 0) > 0:
                missing_ratio = basic_stats['missing_values_total'] / (basic_stats['total_records'] * basic_stats['total_columns'])
                if missing_ratio > 0.1:
                    insight_list.append(f"数据缺失率较高({missing_ratio:.1%})，建议进行缺失值处理")
        
        # 从归因分析中提取洞察
        for alg_name, result in algorithm_results.items():
            # 添加调试信息
            if not isinstance(result, dict):
                self.logger.warning(f"算法 {alg_name} 的结果不是字典类型: {type(result)}")
                continue
                
            if '归因' in alg_name and 'error' not in result:
                if 'last_click' in result.get('attribution_method', ''):
                    top_channels = result.get('top_channels', [])
                    if top_channels:
                        top_channel = top_channels[0]
                        insight_list.append(f"最后点击归因显示{top_channel[0]}是最有效的转化渠道({top_channel[1]}次)")
                
                elif 'first_click' in result.get('attribution_method', ''):
                    top_channels = result.get('top_channels', [])
                    if top_channels:
                        top_channel = top_channels[0]
                        insight_list.append(f"首次点击归因显示{top_channel[0]}是最有效的获客渠道({top_channel[1]}次)")
                
                elif 'markov_channel' in result.get('attribution_method', ''):
                    top_effects = result.get('top_removal_effects', [])
                    if top_effects:
                        top_effect = top_effects[0]
                        insight_list.append(f"Markov渠道模型显示{top_effect[0]}的移除效果最大({top_effect[1]:.2f}%)")
                
                elif 'markov_absorption' in result.get('attribution_method', ''):
                    conversion_rate = result.get('conversion_rate', 0)
                    insight_list.append(f"Markov吸收链模型显示整体转化率为{conversion_rate:.1%}")
        
        # 从相关性分析中提取洞察
        if '相关性分析' in algorithm_results and isinstance(algorithm_results['相关性分析'], dict) and 'error' not in algorithm_results['相关性分析']:
            corr_result = algorithm_results['相关性分析']
            strong_corrs = corr_result.get('strong_correlations', [])
            
            if strong_corrs:
                top_corr = strong_corrs[0]
                insight_list.append(f"发现强相关关系: {top_corr['variable1']}与{top_corr['variable2']}的相关系数为{top_corr['correlation']:.3f}")
        
        # 从时间趋势分析中提取洞察
        if '时间趋势分析' in algorithm_results and isinstance(algorithm_results['时间趋势分析'], dict) and 'error' not in algorithm_results['时间趋势分析']:
            trend_result = algorithm_results['时间趋势分析']
            if isinstance(trend_result, dict):
                for datetime_col, col_result in trend_result.items():
                    if isinstance(col_result, dict) and 'error' not in col_result:
                        numeric_trends = col_result.get('numeric_trends', {})
                        for col, trend in numeric_trends.items():
                            if isinstance(trend, dict) and 'error' not in trend:
                                direction = trend.get('trend_direction', '')
                                strength = trend.get('trend_strength', '')
                                if direction and strength:
                                    insight_list.append(f"{col}在{datetime_col}维度上呈现{strength}{direction}趋势")
                                    break
        
        # 如果洞察不足，添加通用洞察
        if len(insight_list) < 3:
            insight_list.append("建议进一步探索数据特征，可能需要更多的特征工程")
            insight_list.append("考虑收集更多相关数据以提高分析深度")
        
        # 确保洞察数量在3-5条之间
        if len(insight_list) > 5:
            insight_list = insight_list[:5]
        
        html += "<ul>"
        for i, insight in enumerate(insight_list, 1):
            html += f"<li>{i}. {insight}</li>"
        html += "</ul>"
        html += "</div>"
        return html
    
    def _generate_limitations_and_suggestions(self, data: pd.DataFrame, examination_result: Dict[str, Any]) -> str:
        """生成局限与下一步建议"""
        limitations = "## 局限与下一步建议\n\n"
        
        # 数据质量局限
        missing_rates = examination_result.get('missing_rates', {})
        high_missing_cols = [col for col, rate in missing_rates.items() if rate > 0.3]
        if high_missing_cols:
            limitations += f"### 数据质量局限\n"
            limitations += f"- 部分列缺失值较多，可能影响分析结果的准确性\n"
            limitations += f"- 建议进行缺失值填充或删除高缺失值列\n\n"
        
        # 样本量局限
        if len(data) < 100:
            limitations += f"### 样本量局限\n"
            limitations += f"- 样本量较小({len(data)}条)，统计结果可能不够稳定\n"
            limitations += f"- 建议收集更多数据以提高分析可靠性\n\n"
        
        # 特征局限
        column_types = examination_result.get('column_types', {})
        numeric_cols = [col for col, col_type in column_types.items() if col_type == 'number']
        if len(numeric_cols) < 3:
            limitations += f"### 特征局限\n"
            limitations += f"- 数值特征较少({len(numeric_cols)}个)，限制了某些高级分析方法的应用\n"
            limitations += f"- 建议增加数值型特征或进行特征工程\n\n"
        
        # 下一步建议
        limitations += f"### 下一步建议\n"
        limitations += f"1. **数据清洗**: 处理缺失值、异常值和重复数据\n"
        limitations += f"2. **特征工程**: 创建新的特征变量，提高模型性能\n"
        limitations += f"3. **模型优化**: 尝试不同的算法和参数组合\n"
        limitations += f"4. **验证分析**: 使用交叉验证等方法验证模型稳定性\n"
        limitations += f"5. **业务解释**: 结合业务背景深入解释分析结果\n\n"
        
        return limitations
    
    def _generate_limitations_and_suggestions_html(self, data: pd.DataFrame, examination_result: Dict[str, Any]) -> str:
        """生成HTML格式的局限与下一步建议"""
        html = """
        <div class="limitations">
            <h3>局限与下一步建议</h3>
        """
        
        # 数据质量局限
        missing_rates = examination_result.get('missing_rates', {})
        high_missing_cols = [col for col, rate in missing_rates.items() if rate > 0.3]
        if high_missing_cols:
            html += "<p>### 数据质量局限</p>"
            html += "<ul>"
            html += "<li>部分列缺失值较多，可能影响分析结果的准确性</li>"
            html += "<li>建议进行缺失值填充或删除高缺失值列</li>"
            html += "</ul>"
        
        # 样本量局限
        if len(data) < 100:
            html += "<p>### 样本量局限</p>"
            html += "<ul>"
            html += f"<li>样本量较小({len(data)}条)，统计结果可能不够稳定</li>"
            html += "<li>建议收集更多数据以提高分析可靠性</li>"
            html += "</ul>"
        
        # 特征局限
        column_types = examination_result.get('column_types', {})
        numeric_cols = [col for col, col_type in column_types.items() if col_type == 'number']
        if len(numeric_cols) < 3:
            html += "<p>### 特征局限</p>"
            html += "<ul>"
            html += f"<li>数值特征较少({len(numeric_cols)}个)，限制了某些高级分析方法的应用</li>"
            html += "<li>建议增加数值型特征或进行特征工程</li>"
            html += "</ul>"
        
        # 下一步建议
        html += "<p>### 下一步建议</p>"
        html += "<ul>"
        html += "<li>1. **数据清洗**: 处理缺失值、异常值和重复数据</li>"
        html += "<li>2. **特征工程**: 创建新的特征变量，提高模型性能</li>"
        html += "<li>3. **模型优化**: 尝试不同的算法和参数组合</li>"
        html += "<li>4. **验证分析**: 使用交叉验证等方法验证模型稳定性</li>"
        html += "<li>5. **业务解释**: 结合业务背景深入解释分析结果</li>"
        html += "</ul>"
        html += "</div>"
        return html
    
    def save_report(self, report_content: str, filename: str = None, format_type: str = 'markdown') -> str:
        """
        保存报告到文件
        
        Args:
            report_content: 报告内容
            filename: 文件名（可选）
            format_type: 报告格式 ('markdown' 或 'html')
            
        Returns:
            str: 保存的文件路径
        """
        if filename is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            if format_type == 'html':
                filename = f"analysis_report_{timestamp}.html"
            else:
                filename = f"analysis_report_{timestamp}.md"
        
        file_path = os.path.join(self.output_dir, filename)
        
        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(report_content)
            
            self.logger.info(f"报告已保存到: {file_path}")
            return file_path
            
        except Exception as e:
            self.logger.error(f"保存报告失败: {str(e)}")
            return ""
    
    def generate_charts(self, data: pd.DataFrame, algorithm_results: Dict[str, Any]) -> List[str]:
        """
        生成图表
        
        Args:
            data: 原始数据
            algorithm_results: 算法结果
            
        Returns:
            List[str]: 生成的图表文件路径列表
        """
        chart_paths = []
        
        try:
            # 添加调试信息
            self.logger.info(f"开始生成图表，算法结果类型: {type(algorithm_results)}")
            self.logger.info(f"算法结果键: {list(algorithm_results.keys()) if isinstance(algorithm_results, dict) else 'Not a dict'}")
            
            # 确保algorithm_results是字典类型
            if not isinstance(algorithm_results, dict):
                self.logger.error(f"algorithm_results不是字典类型: {type(algorithm_results)}")
                return chart_paths
            
            # 相关性热力图
            if '相关性分析' in algorithm_results and 'error' not in algorithm_results['相关性分析']:
                chart_path = self._generate_correlation_heatmap(data, algorithm_results['相关性分析'])
                if chart_path:
                    chart_paths.append(chart_path)
            
            # 时间趋势图
            if '时间趋势分析' in algorithm_results and 'error' not in algorithm_results['时间趋势分析']:
                chart_path = self._generate_trend_chart(data, algorithm_results['时间趋势分析'])
                if chart_path:
                    chart_paths.append(chart_path)
            
            # 分布图
            if '描述统计' in algorithm_results and 'error' not in algorithm_results['描述统计']:
                chart_path = self._generate_distribution_charts(data, algorithm_results['描述统计'])
                if chart_path:
                    chart_paths.append(chart_path)
            
        except Exception as e:
            self.logger.error(f"生成图表失败: {str(e)}")
            import traceback
            self.logger.error(f"错误详情: {traceback.format_exc()}")
        
        return chart_paths
    
    def _generate_correlation_heatmap(self, data: pd.DataFrame, corr_result: Dict[str, Any]) -> str:
        """生成相关性热力图"""
        try:
            # 获取数值列
            numeric_cols = data.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) < 2:
                return ""
            
            # 计算相关系数矩阵
            corr_matrix = data[numeric_cols].corr()
            
            # 创建热力图
            plt.figure(figsize=(10, 8))
            plt.imshow(corr_matrix, cmap='coolwarm', aspect='auto')
            plt.colorbar()
            plt.xticks(range(len(corr_matrix.columns)), corr_matrix.columns, rotation=45, ha='right')
            plt.yticks(range(len(corr_matrix.columns)), corr_matrix.columns)
            plt.title('相关性热力图')
            plt.tight_layout()
            
            # 保存图表
            chart_path = os.path.join(self.charts_dir, 'correlation_heatmap.png')
            plt.savefig(chart_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            return chart_path
            
        except Exception as e:
            self.logger.error(f"生成相关性热力图失败: {str(e)}")
            return ""
    
    def _generate_trend_chart(self, data: pd.DataFrame, trend_result: Dict[str, Any]) -> str:
        """生成时间趋势图"""
        try:
            # 找到第一个可用的时间列
            datetime_cols = []
            for col in data.columns:
                try:
                    pd.to_datetime(data[col].head(10))
                    datetime_cols.append(col)
                except:
                    continue
            
            if not datetime_cols:
                return ""
            
            datetime_col = datetime_cols[0]
            numeric_cols = data.select_dtypes(include=[np.number]).columns
            
            if len(numeric_cols) == 0:
                return ""
            
            # 准备数据
            plot_data = data[[datetime_col, numeric_cols[0]]].dropna()
            plot_data[datetime_col] = pd.to_datetime(plot_data[datetime_col])
            plot_data = plot_data.sort_values(datetime_col)
            
            # 创建趋势图
            plt.figure(figsize=(12, 6))
            plt.plot(plot_data[datetime_col], plot_data[numeric_cols[0]], marker='o', markersize=2)
            plt.title(f'{numeric_cols[0]} 时间趋势图')
            plt.xlabel('时间')
            plt.ylabel(numeric_cols[0])
            plt.xticks(rotation=45)
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            
            # 保存图表
            chart_path = os.path.join(self.charts_dir, 'trend_chart.png')
            plt.savefig(chart_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            return chart_path
            
        except Exception as e:
            self.logger.error(f"生成时间趋势图失败: {str(e)}")
            return ""
    
    def _generate_distribution_charts(self, data: pd.DataFrame, desc_result: Dict[str, Any]) -> str:
        """生成分布图"""
        try:
            numeric_cols = data.select_dtypes(include=[np.number]).columns
            
            if len(numeric_cols) == 0:
                return ""
            
            # 选择第一个数值列进行分布分析
            col = numeric_cols[0]
            series = data[col].dropna()
            
            if len(series) < 10:
                return ""
            
            # 创建分布图
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
            
            # 直方图
            ax1.hist(series, bins=30, alpha=0.7, edgecolor='black')
            ax1.set_title(f'{col} 分布直方图')
            ax1.set_xlabel(col)
            ax1.set_ylabel('频数')
            ax1.grid(True, alpha=0.3)
            
            # 箱线图
            ax2.boxplot(series)
            ax2.set_title(f'{col} 箱线图')
            ax2.set_ylabel(col)
            ax2.grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            # 保存图表
            chart_path = os.path.join(self.charts_dir, 'distribution_charts.png')
            plt.savefig(chart_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            return chart_path
            
        except Exception as e:
            self.logger.error(f"生成分布图失败: {str(e)}")
            return ""
    
    def _format_descriptive_stats_html(self, result: Dict[str, Any]) -> str:
        """格式化HTML格式的描述统计结果"""
        html = """
        <div class="info-box">
            <h3>描述统计</h3>
        """
        
        basic_stats = result.get('basic_stats', {})
        html += f"""
            <p><strong>基础统计</strong>:</p>
            <ul>
                <li>总记录数: {basic_stats.get('total_records', 0):,}</li>
                <li>总列数: {basic_stats.get('total_columns', 0)}</li>
                <li>重复记录: {basic_stats.get('duplicate_records', 0):,}</li>
                <li>总缺失值: {basic_stats.get('missing_values_total', 0):,}</li>
            </ul>
        """
        
        numerical_stats = result.get('numerical_stats', {})
        if numerical_stats:
            html += f"<p><strong>数值列统计</strong> (共{len(numerical_stats)}列):</p>"
            html += "<ul>"
            for col, stats in list(numerical_stats.items())[:5]:  # 只显示前5个
                html += f"<li>{col}: 均值={stats['mean']:.2f}, 标准差={stats['std']:.2f}, 范围=[{stats['min']:.2f}, {stats['max']:.2f}]</li>"
            if len(numerical_stats) > 5:
                html += f"<li>... 还有{len(numerical_stats)-5}个数值列</li>"
            html += "</ul>"
        
        categorical_stats = result.get('categorical_stats', {})
        if categorical_stats:
            html += f"<p><strong>分类列统计</strong> (共{len(categorical_stats)}列):</p>"
            html += "<ul>"
            for col, stats in list(categorical_stats.items())[:3]:  # 只显示前3个
                html += f"<li>{col}: {stats['unique_count']}个唯一值, 最常见='{stats['most_common']}' ({stats['most_common_count']}次)</li>"
            if len(categorical_stats) > 3:
                html += f"<li>... 还有{len(categorical_stats)-3}个分类列</li>"
            html += "</ul>"
        
        html += "</div>"
        return html