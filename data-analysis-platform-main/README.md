# Intelligent Data Analysis Platform

## System Overview

This is an LLM-based intelligent data analysis platform specifically designed for data analysis and attribution analysis. The system uses a **Schema Comparison** approach to compare the current data schema with the required schemas of various analysis models, intelligently selecting the most suitable analysis algorithms.

### Core Features

- **Schema Comparison Mechanism**: Compares data schema (A) with model schemas (B, C, D) individually to select executable models
- **Intelligent Field Recognition**: Accurately identifies and maps data fields to standard types based on field description documents
- **LLM-Driven Algorithm Selection**: Intelligently recommends analysis algorithms based on schema matching results, only executing when confidence ≥ 0.8
- **Multi-dimensional Analysis**: Supports attribution analysis, correlation analysis, time trend analysis, classification analysis, and descriptive statistics
- **Standardized Reports**: Generates professional analysis reports and algorithm determination results
- **Data Quality Assurance**: Each model has dedicated data validation and cleaning mechanisms

## Supported Analysis Algorithms

### 1. Last Click Attribution
- **Use Case**: Identifies the last channel before conversion
- **Confidence Requirement**: ≥0.8
- **Output**: Conversion attribution results for each channel

### 2. First Click Attribution  
- **Use Case**: Identifies the first channel users encountered
- **Confidence Requirement**: ≥0.8
- **Output**: Customer acquisition attribution results for each channel

### 3. Markov Channel Model
- **Use Case**: Analyzes channel transition probabilities in user paths
- **Confidence Requirement**: ≥0.8
- **Output**: Transition matrix and Removal Effect analysis

### 4. Multi-Dimension Attribution
- **Use Case**: Attribution analysis based on multiple time dimensions
- **Confidence Requirement**: ≥0.8
- **Output**: Attribution results for each time dimension

### 5. Descriptive Statistics
- **Use Case**: Basic data feature analysis
- **Confidence Requirement**: 1.0 (always executed)
- **Output**: Data overview and distribution characteristics

### 6. Correlation Analysis
- **Use Case**: Correlation analysis between numerical variables
- **Confidence Requirement**: ≥0.8
- **Output**: Strong correlation relationship identification

### 7. Time Trend Analysis
- **Use Case**: Analysis of numerical variables' trends over time
- **Confidence Requirement**: ≥0.8
- **Output**: Trend direction and strength analysis

### 8. Categorical Analysis
- **Use Case**: Distribution and feature analysis of categorical variables
- **Confidence Requirement**: ≥0.8
- **Output**: Categorical statistics and diversity indicators

## Schema Comparison Mechanism

The system uses a **Schema Comparison** approach to compare the current data schema with the required schemas of various analysis models:

### Comparison Process
1. **Data Schema Extraction**: Extract the current data schema from data files and field description documents
2. **Model Schema Definition**: Each analysis model defines its required field types and patterns
3. **Schema Comparison**: Compare data schema (A) with model schemas (B, C, D) individually
4. **Model Selection**: Select executable models with the highest matching degree

### Field Mapping Rules
The system performs intelligent field recognition and mapping based on field description documents:

- **User ID Field**: `order_number` (Order Number)
- **Channel Field**: `big_channel_name` (Major Channel Name)  
- **Conversion Field**: `order_number` (Order Number, same as User ID)
- **Event Time Fields**: 21 datetime fields, including:
  - `order_create_time`, `wish_create_time`, `intention_payment_time`
  - `deposit_payment_time`, `lock_time`, `final_payment_time`
  - `delivery_date`, `blind_lock_time`, `approve_refund_time`
  - `order_cancel_time`, `hold_set_date`, `hold_release_date`
  - `clue_create_date`, `first_test_drive_time`, `clue_create_time`
  - `intention_refund_time`, `deposit_refund_time`, `clue_first_create_time`
  - `etl_load_time`, `launch_time`, `intent_pay_start_time`

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### Basic Usage (Rule Engine Mode)

```bash
# Analyze data (requires data file and field description file)
python main.py --data data/vehicle_order_status_indicators.xlsx --schema "data/field_description.xlsx"

# Specify worksheet
python main.py --data data.xlsx --schema schema.xlsx --sheet "Sheet1"

# Specify output directory
```

### LLM Intelligent Analysis Mode

The system supports intelligent analysis based on large language models, enabling more accurate understanding of data characteristics and recommendation of the most suitable analysis algorithms.

#### 1. Get API Keys

**Alibaba Cloud Dashscope (Recommended)**:
1. Visit [Alibaba Cloud Dashscope Console](https://dashscope.console.aliyun.com/)
2. Register/Login to Alibaba Cloud account
3. Create API Key
4. Copy API Key, format like: `sk-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx`

**OpenAI**:
1. Visit [OpenAI API Keys](https://platform.openai.com/api-keys)
2. Register/Login to OpenAI account
3. Create API Key
4. Copy API Key, format like: `sk-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx`

#### 2. Using LLM Analysis

**Method 1: Command Line Parameters**
```bash
# Use Alibaba Cloud Dashscope
python main.py --data data.xlsx --schema schema.xlsx --enable-llm --llm-provider dashscope --llm-api-key "sk-your-dashscope-key"

# Use OpenAI
python main.py --data data.xlsx --schema schema.xlsx --enable-llm --llm-provider openai --llm-api-key "sk-your-openai-key"

# Specify LLM model
python main.py --data data.xlsx --schema schema.xlsx --enable-llm --llm-provider dashscope --llm-api-key "sk-your-key" --llm-model "qwen-max"
python main.py --data data.xlsx --schema schema.xlsx --enable-llm --llm-provider openai --llm-api-key "sk-your-key" --llm-model "gpt-4"
```

**Method 2: Environment Variables**
```bash
# Use Alibaba Cloud Dashscope
export DASHSCOPE_API_KEY="sk-your-dashscope-key"
python main.py --data data.xlsx --schema schema.xlsx --enable-llm --llm-provider dashscope

# Use OpenAI
export OPENAI_API_KEY="sk-your-openai-key"
python main.py --data data.xlsx --schema schema.xlsx --enable-llm --llm-provider openai
```

#### 3. Advantages of LLM Analysis

- **Intelligent Field Recognition**: More accurately identifies key fields like user ID, channels, and time based on field description documents
- **Optimized Algorithm Recommendation**: Recommends the most suitable analysis algorithms by combining data characteristics and business scenarios
- **Automatic Parameter Configuration**: Automatically configures optimal parameters for each algorithm
- **Confidence Assessment**: Provides confidence scores for each recommended algorithm

#### 4. Security Considerations

⚠️ **Important Security Reminder**:
- Please keep your API keys secure and do not commit them to code repositories
- It is recommended to use environment variables to set API keys
- Regularly rotate API keys for security
- Monitor API usage to avoid exceeding quotas

#### 5. Troubleshooting

**API Key Error**:
```bash
❌ Error: API key must be provided when enabling LLM functionality
   Please use --llm-api-key parameter or set environment variable DASHSCOPE_API_KEY
   Example: --llm-api-key 'your-api-key-here'
   Or set environment variable: export DASHSCOPE_API_KEY='your-api-key-here'
```

**Solutions**:
1. Check if the API key is correct
2. Confirm the API key has sufficient balance
3. Check if network connection is normal
4. Confirm the correct LLM provider is selected (dashscope or openai)
5. If LLM analysis fails, the system will automatically fall back to rule engine mode
python main.py --data data.xlsx --schema schema.xlsx --out ./results
```

### Parameter Description

- `--data`: **Required**, data file path (supports Excel/CSV)
- `--schema`: **Required**, field description document path (Excel format, containing field names, data types, comments, etc.)
- `--sheet`: Optional, worksheet name or index (Excel files, defaults to first worksheet)
- `--out`: Optional, output directory (default: ./output)

### Field Description Document Format

The field description document should contain the following columns:
- **Field Name**: Column names in the data file
- **Data Type**: Field data types (such as varchar, bigint, datetime, etc.)
- **Comments**: Business meaning and usage description of the fields

## Output Results

### Algorithm Determination Results Table
```
| Algorithm | Applicable Reason | Confidence | Executed | Notes |
|-----------|-------------------|------------|----------|-------|
| Descriptive Statistics | Any data can be analyzed with descriptive statistics | 1.00 | ✅ Executed | Conditions met, executed |
| Last Click Attribution | Identified user ID, channel, event time | 0.95 | ✅ Executed | Conditions met, executed |
| First Click Attribution | Identified user ID, channel, event time | 0.95 | ✅ Executed | Conditions met, executed |
| Markov Channel Model | Identified user ID, channel, time fields | 0.90 | ✅ Executed | Conditions met, executed |
```

### Analysis Report Content

1. **Data Overview** - Data scale, missing rate, field identification results
2. **Algorithm Results** - Attribution results, channel effectiveness, key indicators
3. **Key Insights** - 3-5 attribution analysis insights
4. **Limitations and Suggestions** - Data quality limitations and improvement suggestions

### Output Files
- `analysis_report_YYYYMMDD_HHMMSS.md` - Data analysis report
- `charts/correlation_heatmap.png` - Correlation heatmap
- `charts/trend_chart.png` - Time trend chart
- `charts/distribution_charts.png` - Data distribution charts

## System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Command Line Interface Layer (CLI)       │
├─────────────────────────────────────────────────────────────┤
│                    Business Logic Layer                     │
│  ┌─────────────┐ ┌─────────────┐ ┌─────────────┐ ┌─────────┐ │
│  │LLM Analyzer │ │Algorithm    │ │Report       │ │Data     │ │
│  │             │ │Executor     │ │Generator    │ │Loader   │ │
│  └─────────────┘ └─────────────┘ └─────────────┘ └─────────┘ │
├─────────────────────────────────────────────────────────────┤
│                    Data Analysis Layer                      │
│  ┌─────────────┐ ┌─────────────┐ ┌─────────────┐ ┌─────────┐ │
│  │Field        │ │Analysis     │ │Confidence   │ │Chart    │ │
│  │Recognition  │ │Algorithms   │ │Calculation  │ │Generation│ │
│  └─────────────┘ └─────────────┘ └─────────────┘ └─────────┘ │
└─────────────────────────────────────────────────────────────┘
```

## Core Modules

### 1. LLM Analyzer (`core/llm_analyzer.py`)
- Schema comparison mechanism (data schema vs model schema)
- Intelligent field recognition based on field description documents
- Analysis algorithm recommendation and confidence calculation

### 2. Algorithm Executor (`core/algorithm_executor.py`)
- Implementation of 8 analysis algorithms (attribution, correlation, time trend, classification, descriptive statistics)
- Model-specific data validation and cleaning
- Failure degradation mechanism

### 3. Report Generator (`core/report_generator.py`)
- Professional data analysis report generation
- Automatic chart generation (correlation, trend, distribution)
- Key insights extraction

### 4. Data Loader (`utils/data_loader.py`)
- Supports Excel/CSV file loading
- Multiple encoding format support
- Data validation functionality

## Example Output

### Console Output
```
🚀 Starting intelligent data analysis...
✅ Executed the following algorithms: Last Click Attribution, Correlation Analysis, Time Trend Analysis, Categorical Analysis, Descriptive Statistics
📄 Report path: ./output/analysis_report_20250814_155328.md
📊 Chart paths: ./output/charts/correlation_heatmap.png, ./output/charts/trend_chart.png, ./output/charts/distribution_charts.png
✅ Analysis completed!

==================================================
## 《Algorithm Determination Results》

| Algorithm | Applicable Reason | Confidence | Executed | Notes |
|-----------|-------------------|------------|----------|-------|
| Last Click Attribution | Schema matching successful: user_id(order_number), channel(big_channel_name), timestamp(order_create_time), conversion(order_number) | 1.00 | ✅ Executed | Conditions met, executed |
| Correlation Analysis | Schema matching successful: numeric_field(['current_retained_locked_count', 'waiting_deposit_count', 'waiting_lock_count', 'current_period_refund_count', 'retained_locked2delivery_count']) | 1.00 | ✅ Executed | Conditions met, executed |
| Time Trend Analysis | Schema matching successful: timestamp(order_create_time), numeric_field(current_retained_locked_count) | 1.00 | ✅ Executed | Conditions met, executed |
| Categorical Analysis | Schema matching successful: categorical_field(blind_lock_status) | 1.00 | ✅ Executed | Conditions met, executed |
| Descriptive Statistics | Schema matching successful:  | 1.00 | ✅ Executed | Conditions met, executed |

📊 Execution statistics: 5/5 algorithms successfully executed
```

## Command Line Parameters

### Required Parameters
- `--data`: Data file path (supports Excel .xlsx/.xls or CSV .csv)
- `--schema`: Field description document path (Excel format, containing field names, data types, comments, etc.)

### Optional Parameters
- `--sheet`: Worksheet name or index (Excel files, defaults to first worksheet)
- `--out`: Output directory (default: ./output)
- `--confidence`: Confidence threshold (default: 0.8)
- `--format`: Report format (default: markdown, optional: html)

### LLM Related Parameters
- `--enable-llm`: Enable LLM intelligent analysis functionality
- `--llm-api-key`: LLM API key (required when enabling LLM)
- `--llm-model`: LLM model name (default: qwen-turbo)
- `--llm-provider`: LLM provider (default: dashscope)

### Examples
```bash
# Basic usage
python main.py --data data.xlsx --schema schema.xlsx

# Enable LLM analysis
python main.py --data data.xlsx --schema schema.xlsx --enable-llm --llm-api-key "sk-your-key"

# Complete parameter example
python main.py \
  --data data.xlsx \
  --schema schema.xlsx \
  --sheet "Sheet1" \
  --out ./results \
  --confidence 0.9 \
  --format html \
  --enable-llm \
  --llm-api-key "sk-your-key" \
  --llm-model "qwen-max"
```

## Technology Stack

- **Python 3.8+**
- **pandas** - Data processing
- **numpy** - Numerical computing
- **scikit-learn** - Machine learning algorithms
- **matplotlib** - Chart generation
- **scipy** - Statistical analysis
- **openpyxl** - Excel file processing
- **aiohttp** - Asynchronous HTTP requests (LLM API calls)

## Development Guide

### Adding New Analysis Algorithms
1. Add algorithm implementation in `core/algorithm_executor.py`
2. Register new ModelSchema in `core/llm_analyzer.py`
3. Add result formatting in `core/report_generator.py`

### Custom Field Mapping
Field pattern matching in `core/llm_analyzer.py` can be modified to adapt to different dataset field mappings.

## License

MIT License 