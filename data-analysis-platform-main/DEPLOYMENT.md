# 部署指南

## 安全部署到GitHub

### 1. 准备工作

在将项目上传到GitHub之前，请确保：

1. **检查敏感信息**：
   - 确认没有硬编码的API密钥
   - 确认没有包含真实数据的文件
   - 确认`.gitignore`文件已正确配置

2. **测试功能**：
   - 测试规则引擎模式：`python main.py --data data.xlsx --schema schema.xlsx`
   - 测试LLM模式（需要API密钥）：`python main.py --data data.xlsx --schema schema.xlsx --enable-llm --llm-api-key "your-key"`

### 2. 文件结构检查

确保以下文件存在且内容正确：

```
data-analysis-platform-main/
├── .gitignore                    # 忽略敏感文件和输出文件
├── config.example.env            # 环境配置示例
├── requirements.txt              # Python依赖
├── README.md                     # 项目说明和使用教程
├── MODEL_INTERFACES.md           # 算法接口文档
├── DEPLOYMENT.md                 # 本部署指南
├── main.py                       # 主程序
├── core/                         # 核心模块
├── utils/                        # 工具模块
└── data/                         # 示例数据（可选）
```

### 3. 上传到GitHub

```bash
# 初始化Git仓库
git init

# 添加文件
git add .

# 提交更改
git commit -m "Initial commit: 智能数据分析平台"

# 添加远程仓库
git remote add origin https://github.com/your-username/your-repo-name.git

# 推送到GitHub
git push -u origin main
```

### 4. 用户使用指南

用户克隆项目后，需要：

1. **安装依赖**：
   ```bash
   pip install -r requirements.txt
   ```

2. **配置API密钥**（如果使用LLM功能）：
   ```bash
   # 方式1：环境变量
   export DASHSCOPE_API_KEY="your-api-key"
   
   # 方式2：命令行参数
   python main.py --data data.xlsx --schema schema.xlsx --enable-llm --llm-api-key "your-api-key"
   ```

3. **运行分析**：
   ```bash
   # 规则引擎模式
   python main.py --data data.xlsx --schema schema.xlsx
   
   # LLM智能模式
   python main.py --data data.xlsx --schema schema.xlsx --enable-llm --llm-api-key "your-api-key"
   ```

### 5. 安全注意事项

✅ **已实现的安全措施**：
- 移除了硬编码的API密钥
- 添加了API密钥验证
- 配置了`.gitignore`文件
- 提供了环境变量配置示例

⚠️ **用户需要注意**：
- 不要将真实的API密钥提交到代码仓库
- 使用环境变量或命令行参数提供API密钥
- 定期轮换API密钥
- 监控API使用量

### 6. 故障排除

**常见问题**：

1. **API密钥错误**：
   - 检查API密钥是否正确
   - 确认API密钥有足够余额
   - 检查网络连接

2. **依赖安装失败**：
   - 确保Python版本为3.8+
   - 使用虚拟环境：`python -m venv venv && source venv/bin/activate`

3. **数据文件格式错误**：
   - 确保数据文件格式正确（Excel或CSV）
   - 检查字段说明文档格式

### 7. 更新和维护

**定期更新**：
- 更新依赖包：`pip install -r requirements.txt --upgrade`
- 检查API密钥有效期
- 更新文档和示例

**版本管理**：
- 使用语义化版本号
- 记录更新日志
- 保持向后兼容性
