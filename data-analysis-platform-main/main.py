#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Intelligent Data Analysis System Main Program
Senior Data Analysis Assistant - LLM-based Intelligent Data Analysis Platform
"""

import argparse
import sys
import os
import logging
from datetime import datetime
from typing import Dict, Any, List

# Add project path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from core.llm_analyzer import LLMAnalyzer
from core.algorithm_executor import AlgorithmExecutor
from core.report_generator import ReportGenerator
from utils.data_loader import DataLoader

class IntelligentDataAnalysisSystem:
    """Intelligent Data Analysis System"""
    
    def __init__(self, output_dir: str = "./output", enable_llm: bool = False, llm_config: Dict[str, Any] = None):
        """
        Initialize the system
        
        Args:
            output_dir: Output directory
            enable_llm: Whether to enable LLM functionality
            llm_config: LLM configuration
        """
        self.output_dir = output_dir
        self.enable_llm = enable_llm
        
        # Initialize LLM analyzer
        if enable_llm:
            self.llm_analyzer = LLMAnalyzer(enable_llm=True, llm_config=llm_config)
            print("🤖 LLM intelligent analysis functionality enabled")
        else:
            self.llm_analyzer = LLMAnalyzer()
            print("🔧 Using rule engine analysis mode")
        
        self.algorithm_executor = AlgorithmExecutor()
        self.report_generator = ReportGenerator(output_dir)
        self.data_loader = DataLoader()
        
        # Setup logging
        self._setup_logging()
        self.logger = logging.getLogger(__name__)
        
    def _setup_logging(self):
        """Setup logging"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(os.path.join(self.output_dir, 'analysis.log'), encoding='utf-8'),
                logging.StreamHandler(sys.stdout)
            ]
        )
    
    def run_analysis(self, data_path: str, schema_path: str, sheet_name: str = None, 
                    confidence_threshold: float = 0.8, format_type: str = 'markdown') -> Dict[str, Any]:
        """
        Run complete analysis workflow
        
        Args:
            data_path: Data file path
            sheet_name: Worksheet name (Excel files)
            confidence_threshold: Confidence threshold
            format_type: Report format ('markdown' or 'html')
            
        Returns:
            Dict[str, Any]: Analysis results
        """
        try:
            self.logger.info("Starting intelligent data analysis")
            
            # 1. Load data
            data = self._load_data(data_path, sheet_name)
            if data is None:
                return {'success': False, 'error': 'Data loading failed'}
            
            # 2. Data analysis and algorithm recommendation
            if self.enable_llm:
                # Use LLM for intelligent analysis
                self.logger.info("Using LLM for intelligent analysis")
                analysis_result = self.llm_analyzer.analyze_data_structure_with_llm(data, schema_path)
                
                if analysis_result.get('success', False):
                    # LLM analysis successful
                    llm_analysis = analysis_result.get('analysis', {})
                    self.logger.info(f"LLM analysis successful, analysis result: {llm_analysis}")
                    recommendations = self._convert_llm_recommendations(llm_analysis)
                    examination_result = {
                        'data_shape': data.shape,
                        'columns': list(data.columns),
                        'llm_analysis': llm_analysis
                    }
                else:
                    # LLM analysis failed, use fallback analysis
                    self.logger.warning(f"LLM analysis failed: {analysis_result.get('error', 'Unknown error')}")
                    
                    # Use fallback analysis based on field descriptions
                    fallback_analysis = analysis_result.get('fallback_analysis', {})
                    if fallback_analysis:
                        # If fallback analysis results exist, use them
                        self.logger.info(f"Using fallback analysis results: {fallback_analysis}")
                        recommendations = self._convert_llm_recommendations(fallback_analysis)
                        examination_result = {
                            'data_shape': data.shape,
                            'columns': list(data.columns),
                            'fallback_analysis': fallback_analysis
                        }
                    else:
                        # If no fallback analysis results, use original rule engine analysis
                        examination_result = self.llm_analyzer.quick_examination(data, schema_path=schema_path)
                        if 'error' in examination_result:
                            return {'success': False, 'error': f'Quick examination failed: {examination_result["error"]}'}
                        
                        recommendations = self.llm_analyzer.recommend_algorithms(examination_result, data, schema_path)
                        if not recommendations:
                            return {'success': False, 'error': 'Algorithm recommendation failed'}
            else:
                # Use original rule engine analysis
                examination_result = self.llm_analyzer.quick_examination(data, schema_path=schema_path)
                if 'error' in examination_result:
                    return {'success': False, 'error': f'Quick examination failed: {examination_result["error"]}'}
                
                recommendations = self.llm_analyzer.recommend_algorithms(examination_result, data, schema_path)
                if not recommendations:
                    return {'success': False, 'error': 'Algorithm recommendation failed'}
            
            # 4. Determine whether to execute algorithms
            should_execute, executable_algorithms = self.llm_analyzer.should_execute_algorithms(
                recommendations, confidence_threshold
            )
            
            # 5. Generate algorithm determination results table
            judgment_table = self.report_generator.generate_algorithm_judgment_table(
                recommendations, should_execute, format_type
            )
            
            # 6. Execute algorithms (if conditions are met)
            algorithm_results = {}
            if should_execute:
                self.logger.info(f"Starting execution of {len(executable_algorithms)} algorithms")
                for algorithm in executable_algorithms:
                    algorithm_name = algorithm['algorithm']
                    params = algorithm.get('params', {})
                    
                    # Add debugging information
                    self.logger.info(f"Executing algorithm: {algorithm_name}, parameters: {params}")
                    
                    result = self.algorithm_executor.execute_algorithm(algorithm_name, data, params)
                    algorithm_results[algorithm_name] = result
                    
                    if 'error' in result:
                        self.logger.warning(f"Algorithm {algorithm_name} execution failed: {result['error']}")
                    else:
                        self.logger.info(f"Algorithm {algorithm_name} executed successfully")
            else:
                self.logger.info("Algorithm execution conditions not met, only outputting algorithm list")
            
            # 7. Generate charts
            chart_paths = self.report_generator.generate_charts(data, algorithm_results)
            
            # 8. Generate analysis report
            if should_execute:
                analysis_report = self.report_generator.generate_analysis_report(
                    data, examination_result, algorithm_results, recommendations, format_type
                )
                
                # Save report
                report_path = self.report_generator.save_report(analysis_report, format_type=format_type)
                
                # Console output
                executed_algorithms = [alg['algorithm'] for alg in executable_algorithms]
                print(f"✅ Executed the following algorithms: {', '.join(executed_algorithms)}")
                print(f"📄 Report path: {report_path}")
                if chart_paths:
                    print(f"📊 Chart paths: {', '.join(chart_paths)}")
            else:
                # Only output algorithm list
                print("📋 Algorithm list + non-execution reasons (insufficient information/conditions not met)")
                print(judgment_table)
                report_path = ""
            
            # 9. Return results
            result = {
                'success': True,
                'data_path': data_path,
                'data_shape': data.shape,
                'examination_result': examination_result,
                'recommendations': recommendations,
                'should_execute': should_execute,
                'algorithm_results': algorithm_results,
                'judgment_table': judgment_table,
                'report_path': report_path,
                'chart_paths': chart_paths,
                'format_type': format_type,
                'execution_timestamp': datetime.now().isoformat()
            }
            
            self.logger.info("Intelligent data analysis completed")
            return result
            
        except Exception as e:
            self.logger.error(f"Analysis workflow failed: {str(e)}")
            return {'success': False, 'error': str(e)}
    
    def _load_data(self, data_path: str, sheet_name: str = None) -> Any:
        """Load data"""
        try:
            # Handle sheet_name parameter
            if sheet_name is not None:
                # Try to convert to integer (if it's an index)
                try:
                    sheet_name = int(sheet_name)
                except ValueError:
                    # Keep as string
                    pass
            
            # Load data
            data = self.data_loader.load_data(data_path, sheet_name)
            
            if data is None:
                return None
            
            # Validate data
            if not self.data_loader.validate_data(data):
                return None
            
            return data
            
        except Exception as e:
            self.logger.error(f"Data loading failed: {str(e)}")
            return None
    
    def _convert_llm_recommendations(self, llm_analysis: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Convert LLM analysis results to algorithm recommendation format
        
        Args:
            llm_analysis: LLM analysis results
            
        Returns:
            List[Dict]: Algorithm recommendation list
        """
        recommendations = []
        
        # Algorithm name mapping (English to English)
        algorithm_name_map = {
            'descriptive_statistics': 'Descriptive Statistics',
            'correlation_analysis': 'Correlation Analysis',
            'time_series_analysis': 'Time Trend Analysis',
            'categorical_analysis': 'Categorical Analysis',
            'last_click_attribution': 'Last Click Attribution',
            'first_click_attribution': 'First Click Attribution',
            'markov_channel_model': 'Markov Channel Model',
            'markov_absorption_model': 'Markov Absorption Model',
            'multi_dimension_attribution': 'Multi-Dimension Attribution',
            'attribution_analysis': 'Multi-Dimension Attribution',  # General attribution analysis mapped to multi-dimension attribution
            'clustering_analysis': 'Clustering Analysis',
            'anomaly_detection': 'Anomaly Detection'
        }
        
        # Extract model recommendations from LLM analysis results
        model_recommendations = llm_analysis.get('model_recommendations', [])
        
        for rec in model_recommendations:
            model_name = rec.get('name', '')
            confidence = rec.get('confidence', 0.5)
            reason = rec.get('reason', '')
            params = rec.get('params', {})
            
            # Convert to algorithm name expected by algorithm executor
            algorithm_name = algorithm_name_map.get(model_name, model_name)
            
            # Convert to standard recommendation format
            recommendation = {
                'model_name': model_name,
                'algorithm': algorithm_name,  # Use English algorithm name
                'confidence': confidence,
                'reason': reason,
                'priority': int((1 - confidence) * 10),  # Higher confidence, lower priority
                'required_fields': [],
                'optional_fields': [],
                'executable': True,  # Add executable field
                'params': params  # Add algorithm parameters
            }
            
            recommendations.append(recommendation)
        
        # If no LLM recommendations, add default recommendation
        if not recommendations:
            recommendations.append({
                'model_name': 'descriptive_statistics',
                'algorithm': 'Descriptive Statistics',  # Use English name
                'confidence': 0.9,
                'reason': 'LLM did not provide recommendations, using default descriptive statistics',
                'priority': 1,
                'required_fields': [],
                'optional_fields': [],
                'executable': True
            })
        
        return recommendations

def main():
    """Main function"""
    parser = argparse.ArgumentParser(
        description='Intelligent Data Analysis Platform - LLM-based Intelligent Data Analysis System',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example usage:
  python main.py --data data/vehicle_order_status_indicators.xlsx --schema "data/field_description.xlsx"
  python main.py --data data.xlsx --schema schema.xlsx --sheet "Sheet1" --format html
  python main.py --data data.csv --schema schema.xlsx --out ./results --format markdown
        """
    )
    
    parser.add_argument('--data', required=True, help='Data file path (Excel/CSV)')
    parser.add_argument('--schema', required=True, help='Field description document path (Excel format, containing field names, data types, comments, etc.)')
    parser.add_argument('--sheet', help='Worksheet name or index (Excel files, defaults to first worksheet)')
    parser.add_argument('--out', default='./output', help='Output directory (default: ./output)')
    parser.add_argument('--confidence', type=float, default=0.8, 
                       help='Confidence threshold (default: 0.8)')
    parser.add_argument('--format', choices=['markdown', 'html'], default='markdown',
                       help='Report format (default: markdown)')
    parser.add_argument('--enable-llm', action='store_true', 
                       help='Enable LLM intelligent analysis functionality')
    parser.add_argument('--llm-provider', choices=['dashscope', 'openai', 'azure'], default='dashscope',
                       help='LLM provider (default: dashscope)')
    parser.add_argument('--llm-api-key', help='LLM API key')
    parser.add_argument('--llm-model', default='qwen-turbo', help='LLM model name (default: qwen-turbo)')
    
    args = parser.parse_args()
    
    # Check input files
    if not os.path.exists(args.data):
        print(f"❌ Error: Data file does not exist: {args.data}")
        sys.exit(1)
    
    if not os.path.exists(args.schema):
        print(f"❌ Error: Field description document does not exist: {args.schema}")
        sys.exit(1)
    
    # Create output directory
    os.makedirs(args.out, exist_ok=True)
    
    # Prepare LLM configuration
    llm_config = None
    if args.enable_llm:
        llm_config = {
            'provider': args.llm_provider,
            'model_name': args.llm_model,
            'timeout': 300,
            'max_tokens': 4000,
            'temperature': 0.7,
            'top_p': 0.8
        }
        
        # Set API key
        if args.llm_api_key:
            llm_config['api_key'] = args.llm_api_key
        else:
            # Try to get from environment variables
            env_key_map = {
                'dashscope': 'DASHSCOPE_API_KEY',
                'openai': 'OPENAI_API_KEY',
                'azure': 'AZURE_OPENAI_API_KEY'
            }
            env_key = env_key_map.get(args.llm_provider)
            if env_key and os.getenv(env_key):
                llm_config['api_key'] = os.getenv(env_key)
            else:
                print(f"❌ Error: API key must be provided when enabling LLM functionality")
                print(f"   Please use --llm-api-key parameter or set environment variable {env_key}")
                print(f"   Example: --llm-api-key 'your-api-key-here'")
                print(f"   Or set environment variable: export {env_key}='your-api-key-here'")
                sys.exit(1)
    
    # Create analysis system
    system = IntelligentDataAnalysisSystem(args.out, args.enable_llm, llm_config)
    
    # Run analysis
    print("🚀 Starting intelligent data analysis...")
    result = system.run_analysis(args.data, args.schema, args.sheet, args.confidence, args.format)
    
    if result['success']:
        print("✅ Analysis completed!")
        
        # Display algorithm determination results
        if 'judgment_table' in result:
            print("\n" + "="*50)
            print(result['judgment_table'])
        
        # Display execution statistics
        if result.get('should_execute', False):
            executed_count = len([r for r in result['algorithm_results'].values() if 'error' not in r])
            total_count = len(result['algorithm_results'])
            print(f"\n📊 Execution statistics: {executed_count}/{total_count} algorithms successfully executed")
            
        # Display format information
        format_type = result.get('format_type', 'markdown')
        print(f"📋 Report format: {format_type.upper()}")
    else:
        print(f"❌ Analysis failed: {result.get('error', 'Unknown error')}")
        sys.exit(1)

if __name__ == "__main__":
    main() 