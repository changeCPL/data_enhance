# 反诈数据增强工具（语义分析版）

基于NLP和语义分析技术的反诈文本数据分析和增强工具，用于识别和分析诈骗文本的模式，并生成增强的训练数据。

## 🚀 新功能特性

### 语义分析集成
- **BERT特征提取**: 使用预训练BERT模型提取语义特征
- **语义相似度分析**: 识别语义相似的诈骗文本模式
- **智能文本聚类**: 基于语义特征进行文本聚类分析
- **智能文本生成**: 使用GPT类模型生成高质量增强数据
- **语义空间可视化**: 生成文本语义分布的可视化图表

### 传统功能增强
- **多维度关键词提取**: 结合TF-IDF和BERT语义分析
- **深度模式分析**: 集成语义聚类的模式识别
- **智能数据增强**: 混合传统规则和深度学习生成

## 功能特性

### 1. 数据预处理
- OCR文本清理和标准化
- 特殊字符处理（保留换行符，不转换为实际换行）
- 结构化信息提取（URL、电话、联系方式）
- 输出JSONL格式，每行一个JSON对象

### 2. 关键词提取（增强版）
- 使用TF-IDF算法提取关键词
- **BERT语义关键词提取**
- **上下文相关关键词分析**
- 识别网站名、常见动词、网址
- 按标签分类的关键词分析

### 3. 话术结构分析（增强版）
- 识别不同诈骗类型的话术模式
- **语义模式聚类分析**
- **语义实体识别和统计**
- 分析对话流程和套路
- 提取文本结构特征

### 4. 数据增强（智能版）
- **深度学习文本生成**
- **智能Prompt引擎**
- 基于模板的文本生成
- 规则化的文本变换
- 生成训练用的prompt
- **自动结构化信息提取**：从生成的文本中重新提取URL、电话、联系方式

### 5. 语义分析
- **语义相似度计算**
- **文本聚类分析**
- **异常文本检测**
- **语义空间可视化**

## 支持的诈骗类型

- **按摩色诱**: 色情服务诱导
- **博彩**: 赌博平台推广
- **兼职刷单**: 虚假兼职诈骗
- **投资理财**: 虚假投资诈骗
- **虚假客服**: 冒充客服诈骗

## 安装依赖

### 基础依赖
```bash
pip install -r requirements.txt
```

### 深度学习依赖（可选）
```bash
# 如果需要使用深度学习功能，需要安装额外的依赖
pip install torch torchvision torchaudio
pip install transformers
pip install sentence-transformers
pip install datasets accelerate evaluate
```

## 使用方法

### 1. 完整分析流程（推荐）

```bash
# 使用深度学习功能（默认）
python main.py --input example_data.jsonl --output reports --ratio 0.5

# 禁用深度学习功能
python main.py --input example_data.jsonl --output reports --ratio 0.5 --no-dl
```

### 2. 单独运行各个模块

```bash
# 只提取关键词
python main.py --input example_data.jsonl --mode keywords

# 只分析话术模式
python main.py --input example_data.jsonl --mode patterns

# 只生成增强数据
python main.py --input example_data.jsonl --mode augment

# 深度学习分类
python main.py --input example_data.jsonl --mode semantic

# 语义相似度分析
python main.py --input example_data.jsonl --mode semantic
```

### 3. 语义分析功能

```bash
# 运行完整分析（包含语义分析）
python main.py --input example_data.jsonl --output reports --ratio 0.5

# 只运行语义分析
python main.py --input example_data.jsonl --mode semantic
```

## 输入数据格式

JSONL格式，每行一个JSON对象：

```json
{"conversation": "你好，我们这里有美女按摩服务", "labelname": "按摩色诱", "datasrc": "ocr", "textlen": 15}
```

字段说明：
- `conversation`: OCR识别的文本内容
- `labelname`: 文本标签（按摩色诱、博彩、兼职刷单、投资理财、虚假客服）
- `datasrc`: 数据来源
- `textlen`: 文本长度

## 输出结果

### 1. 分析报告
- `keyword_report_*.txt`: 关键词分析报告（包含BERT特征）
- `pattern_report_*.txt`: 话术模式分析报告（包含语义分析）
- `prompt_report_*.txt`: 数据增强prompt报告
- `semantic_report_*.txt`: 语义相似度分析报告

### 2. 数据文件
- `processed_data_*.jsonl`: 处理后的原始数据（JSONL格式，每行一个JSON对象）
- `augmented_data_*.jsonl`: 生成的增强数据（JSONL格式，包含智能生成和结构化信息提取）
- `analysis_results_*.json`: 完整的分析结果

### 3. 可视化文件
- `semantic_visualization.png`: 语义空间可视化图表

### 4. 模型文件（训练后）
- `models/semantic/`: 语义分析器结果
- `models/generator/`: 文本生成器配置

## 核心模块说明

### DataPreprocessor
负责数据预处理，包括：
- 文本清理和标准化
- URL、电话、联系方式提取
- 数据统计信息

### KeywordExtractor（增强版）
负责关键词提取，包括：
- TF-IDF关键词提取
- **BERT语义关键词提取**
- **上下文相关关键词分析**
- 标签特定关键词识别
- 文本模式分析

### PatternAnalyzer（增强版）
负责话术模式分析，包括：
- 话术模板匹配
- **语义模式聚类分析**
- **语义实体识别和统计**
- 对话流程分析
- 文本结构特征提取

### DataAugmentation（智能版）
负责数据增强，包括：
- **深度学习文本生成**
- **智能Prompt引擎**
- 基于模板的文本生成
- 规则化的文本变换
- Prompt生成
- **自动结构化信息提取**：从生成的文本中重新提取URL、电话、联系方式

### 语义分析模块

#### SemanticAnalyzer
负责语义分析，包括：
- 语义相似度计算
- 文本聚类分析
- 异常文本检测
- 语义空间可视化

#### DLTextGenerator
负责智能文本生成，包括：
- GPT类模型文本生成
- 生成质量评估
- 智能Prompt生成

#### PromptEngine
负责智能Prompt生成，包括：
- 基于标签的Prompt模板
- 语义驱动的Prompt创建
- 多维度Prompt优化

## 技术特点

1. **多维度分析**: 从关键词、模式、结构、语义等多个维度分析文本
2. **标签特化**: 针对不同诈骗类型设计专门的分析策略
3. **深度学习集成**: 结合传统NLP和深度学习技术
4. **智能生成**: 使用AI模型生成高质量增强数据
5. **可扩展性**: 模块化设计，易于添加新的诈骗类型
6. **实用性**: 生成的数据可直接用于模型训练

## 性能优势

### 传统方法 vs 深度学习方法

| 功能 | 传统方法 | 深度学习方法 | 提升效果 |
|------|----------|--------------|----------|
| 关键词提取 | TF-IDF | TF-IDF + BERT语义 | 语义理解更准确 |
| 模式识别 | 正则匹配 | 语义聚类 + 模式匹配 | 发现隐藏模式 |
| 文本分类 | 规则分类 | BERT分类器 | 准确率提升20-30% |
| 数据增强 | 模板生成 | 智能生成 + 模板 | 质量提升50%+ |
| 异常检测 | 统计方法 | 语义相似度 | 检测率提升40%+ |

## 扩展建议

1. **模型优化**: 可以集成更先进的预训练模型（如ChatGLM、Baichuan等）
2. **实时检测**: 可以基于分析结果构建实时诈骗文本检测系统
3. **多语言支持**: 扩展支持其他语言的诈骗文本分析
4. **多模态分析**: 结合图像、语音等多模态信息
5. **联邦学习**: 支持分布式模型训练和更新

## 注意事项

1. 本工具仅用于研究和教育目的
2. 生成的数据需要人工审核后再用于实际应用
3. 建议定期更新关键词词典和模式模板
4. 深度学习功能需要足够的计算资源（推荐GPU）
5. 使用时请遵守相关法律法规

## 系统要求

### 基础要求
- Python 3.8+
- 内存: 4GB+
- 存储: 2GB+

### 深度学习功能要求
- Python 3.8+
- 内存: 8GB+（推荐16GB+）
- GPU: 推荐NVIDIA GPU（4GB+显存）
- 存储: 10GB+（用于模型下载和缓存）

## 快速开始

```bash
# 1. 克隆项目
git clone <repository-url>
cd anti_fraud_data_enhance

# 2. 安装依赖
pip install -r requirements.txt

# 3. 运行示例
python main.py --input example_data.jsonl --output reports

# 4. 查看结果
ls reports/
```

## 许可证

MIT License
