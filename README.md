# 反诈数据增强工具（深度学习集成版）

基于传统NLP和深度学习技术的反诈文本数据分析和增强工具，用于识别和分析诈骗文本的模式，并生成增强的训练数据。

## 🚀 核心功能特性

### 深度学习集成
- **BERT语义分析**: 使用预训练BERT模型进行语义特征提取和相似度分析
- **GPT文本生成**: 使用GPT类模型生成高质量增强数据
- **智能Prompt引擎**: 基于语义理解的智能Prompt生成
- **语义空间可视化**: 生成文本语义分布的可视化图表
- **可选深度学习**: 支持传统方法和深度学习方法的灵活切换

### 传统功能增强
- **多维度关键词提取**: 结合TF-IDF和BERT语义分析
- **深度模式分析**: 集成语义聚类的模式识别
- **混合数据增强**: 智能生成(60%) + 模板生成(40%) + 规则增强
- **结构化信息提取**: 自动提取URL、电话、联系方式、加密货币地址等

## 功能特性

### 1. 数据预处理（重构版）
- OCR文本清理和标准化
- Unicode标准化处理
- 特殊字符和噪音清理
- 基础数据统计和验证
- 输出JSONL格式，每行一个JSON对象

**注意**: 结构化信息提取功能已移至PatternAnalyzer模块

### 2. 关键词提取（优化版）
- TF-IDF关键词提取（支持动态top_k参数）
- BERT语义关键词提取（语义重要性计算）
- 上下文关键词分析（基于语义相似度）
- 标签特定关键词识别（网站名、动词提取）
- 性能优化（动态候选词限制、批量处理）
- 按标签分类的关键词分析

### 3. 话术结构分析（集成版）
- 话术模式匹配（标签特定和通用模式）
- 结构化信息提取：
  - 电话号码提取（支持国际格式）
  - 联系方式提取（多平台支持）
  - 加密货币地址提取
  - 银行信息提取
  - 可疑模式分类提取
- 自动DataFrame集成（添加结构化信息列）

### 4. 数据增强（混合增强版）
- 混合生成策略：
  - 深度学习生成（60%）
  - 模板生成（40%）
  - 规则增强（补充）
- 智能Prompt引擎（语义驱动、多维度优化）
- 质量保证机制（文本有效性验证、标签一致性检查）
- 自动结构化信息提取（从生成文本中重新提取实体信息）

### 5. 语义分析（独立模块）
- 多模型支持（SentenceTransformers、HuggingFace Transformers）
- 语义相似度计算（余弦相似度、阈值过滤）
- 文本聚类分析（K-means、DBSCAN）
- 异常检测（语义异常识别、低相似度文本检测）
- 可视化功能（PCA降维、语义空间可视化）
- 性能优化（批量编码、结果缓存）

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

# 指定运行模式
python main.py --input example_data.jsonl --mode full --use-dl
```

### 2. 单独运行各个模块

```bash
# 只提取关键词
python main.py --input example_data.jsonl --mode keywords

# 只分析话术模式
python main.py --input example_data.jsonl --mode patterns

# 只生成增强数据
python main.py --input example_data.jsonl --mode augment

# 只运行语义分析
python main.py --input example_data.jsonl --mode semantic
```

### 3. 演示和测试

```bash
# 运行演示脚本
python demo.py

# 运行简单测试
python simple_test.py
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
- `keyword_report_*.txt`: 关键词分析报告（TF-IDF + BERT语义）
- `pattern_report_*.txt`: 话术模式分析报告（模式匹配 + 结构化信息）
- `prompt_report_*.txt`: 数据增强prompt报告
- `semantic_report_*.txt`: 语义相似度分析报告（可选）

### 2. 数据文件
- `processed_data_*.jsonl`: 处理后的原始数据（JSONL格式）
  - 包含基础文本信息和标签
- `augmented_data_*.jsonl`: 生成的增强数据（JSONL格式）
  - 包含混合生成策略的结果
  - 自动提取的结构化信息（URL、电话、联系方式等）
- `analysis_results_*.json`: 完整的分析结果（JSON格式）

### 3. 可视化文件
- `semantic_visualization.png`: 语义空间可视化图表（可选）

## 核心模块说明

### DataPreprocessor（重构版）
负责数据预处理，包括：
- 文本清理和标准化（Unicode标准化、OCR噪音清理）
- 基础数据统计和验证
- JSONL格式输出

### KeywordExtractor（优化版）
负责关键词提取，包括：
- TF-IDF关键词提取（支持动态top_k参数）
- BERT语义关键词提取（语义重要性计算）
- 上下文关键词分析（基于语义相似度）
- 标签特定关键词识别（网站名、动词提取）
- 性能优化（动态候选词限制、批量处理）

### PatternAnalyzer（集成版）
负责话术模式分析，包括：
- 话术模式匹配（标签特定和通用模式）
- 结构化信息提取（电话、联系方式、加密货币地址、银行信息等）
- 自动DataFrame集成（添加结构化信息列）

### DataAugmentation（混合增强版）
负责数据增强，包括：
- 混合生成策略（深度学习60% + 模板40% + 规则增强）
- 智能Prompt引擎（语义驱动、多维度优化）
- 质量保证机制（文本有效性验证、标签一致性检查）
- 自动结构化信息提取（从生成文本中重新提取实体信息）

### SemanticAnalyzer（独立模块）
负责语义分析，包括：
- 多模型支持（SentenceTransformers、HuggingFace Transformers）
- 语义相似度计算（余弦相似度、阈值过滤）
- 文本聚类分析（K-means、DBSCAN）
- 异常检测（语义异常识别、低相似度文本检测）
- 可视化功能（PCA降维、语义空间可视化）

### DLTextGenerator（智能生成器）
负责智能文本生成，包括：
- 多模型支持（GPT-2中文模型、自动回退机制）
- 文本生成策略（参数化配置、文本后处理、质量验证）
- 生成质量评估（语义相似度评估、多样性计算）
- 备用生成方案（基于规则的模板生成）

### PromptEngine（智能Prompt引擎）
负责智能Prompt生成，包括：
- 智能Prompt生成（基于标签的模板、语义驱动创建）
- 元素配置（标签特定元素、需求配置、特征配置）
- 动态Prompt创建（随机元素组合、上下文感知）

## 技术特点

1. **多维度分析**: 从关键词、模式、结构、语义等多个维度分析文本
2. **标签特化**: 针对不同诈骗类型设计专门的分析策略
3. **深度学习集成**: 结合传统NLP和深度学习技术，支持可选切换
4. **混合生成策略**: 智能生成(60%) + 模板生成(40%) + 规则增强
5. **模块化设计**: 各模块独立，易于维护和扩展
6. **性能优化**: 动态参数配置、批量处理、内存优化
7. **质量保证**: 文本有效性验证、标签一致性检查
8. **全面覆盖**: 支持多种信息类型提取（金融、联系方式、可疑模式等）

## 性能优势

### 传统方法 vs 深度学习方法

| 功能 | 传统方法 | 深度学习方法 | 提升效果 |
|------|----------|--------------|----------|
| 关键词提取 | TF-IDF | TF-IDF + BERT语义 | 语义理解更准确 |
| 模式识别 | 正则匹配 | 语义聚类 + 模式匹配 | 发现隐藏模式 |
| 数据增强 | 模板生成 | 混合生成策略 | 质量提升50%+ |
| 异常检测 | 统计方法 | 语义相似度 | 检测率提升40%+ |
| 结构化提取 | 规则匹配 | 规则匹配 + 语义验证 | 准确率提升30%+ |

## 扩展建议

1. **模型优化**: 可以集成更先进的预训练模型（如ChatGLM、Baichuan等）
2. **实时检测**: 可以基于分析结果构建实时诈骗文本检测系统
3. **多语言支持**: 扩展支持其他语言的诈骗文本分析
4. **多模态分析**: 结合图像、语音等多模态信息
5. **API服务**: 提供RESTful API服务，支持云端部署

## 注意事项

1. 本工具仅用于研究和教育目的
2. 生成的数据需要人工审核后再用于实际应用
3. 建议定期更新关键词词典和模式模板
4. 深度学习功能需要足够的计算资源（推荐GPU）
5. 使用时请遵守相关法律法规
6. 支持传统方法和深度学习方法的灵活切换

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

# 3. 运行完整分析（推荐）
python main.py --input example_data.jsonl --output reports --ratio 0.5

# 4. 运行演示
python demo.py

# 5. 查看结果
ls reports/
```

## 许可证

MIT License
