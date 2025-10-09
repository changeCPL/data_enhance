# 反诈数据增强工具

基于NLP/NER技术的反诈文本数据分析和增强工具，用于识别和分析诈骗文本的模式，并生成增强的训练数据。

## 功能特性

### 1. 数据预处理
- OCR文本清理和标准化
- 特殊字符处理
- 结构化信息提取（URL、电话、联系方式）

### 2. 关键词提取
- 使用TF-IDF算法提取关键词
- 识别网站名、常见动词、网址
- 按标签分类的关键词分析

### 3. 话术结构分析
- 识别不同诈骗类型的话术模式
- 分析对话流程和套路
- 提取文本结构特征

### 4. 数据增强
- 基于模板的文本生成
- 规则化的文本变换
- 生成训练用的prompt

## 支持的诈骗类型

- **按摩色诱**: 色情服务诱导
- **博彩**: 赌博平台推广
- **兼职刷单**: 虚假兼职诈骗
- **投资理财**: 虚假投资诈骗
- **虚假客服**: 冒充客服诈骗

## 安装依赖

```bash
pip install -r requirements.txt
```

## 使用方法

### 1. 完整分析流程

```bash
python main.py --input example_data.jsonl --output reports --ratio 0.5
```

### 2. 单独运行各个模块

```bash
# 只提取关键词
python main.py --input example_data.jsonl --mode keywords

# 只分析话术模式
python main.py --input example_data.jsonl --mode patterns

# 只生成增强数据
python main.py --input example_data.jsonl --mode augment
```

### 3. 使用示例数据

```bash
python main.py --input example_data.jsonl --output reports
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
- `keyword_report_*.txt`: 关键词分析报告
- `pattern_report_*.txt`: 话术模式分析报告
- `prompt_report_*.txt`: 数据增强prompt报告

### 2. 数据文件
- `processed_data_*.csv`: 处理后的原始数据
- `augmented_data_*.csv`: 生成的增强数据
- `analysis_results_*.json`: 完整的分析结果

## 核心模块说明

### DataPreprocessor
负责数据预处理，包括：
- 文本清理和标准化
- URL、电话、联系方式提取
- 数据统计信息

### KeywordExtractor
负责关键词提取，包括：
- TF-IDF关键词提取
- 标签特定关键词识别
- 文本模式分析

### PatternAnalyzer
负责话术模式分析，包括：
- 话术模板匹配
- 对话流程分析
- 文本结构特征提取

### DataAugmentation
负责数据增强，包括：
- 基于模板的文本生成
- 规则化的文本变换
- Prompt生成

## 技术特点

1. **多维度分析**: 从关键词、模式、结构等多个维度分析文本
2. **标签特化**: 针对不同诈骗类型设计专门的分析策略
3. **可扩展性**: 模块化设计，易于添加新的诈骗类型
4. **实用性**: 生成的数据可直接用于模型训练

## 扩展建议

1. **深度学习模型**: 可以集成BERT等预训练模型进行更深入的文本分析
2. **实时检测**: 可以基于分析结果构建实时诈骗文本检测系统
3. **多语言支持**: 扩展支持其他语言的诈骗文本分析
4. **可视化**: 添加数据可视化功能，更直观地展示分析结果

## 注意事项

1. 本工具仅用于研究和教育目的
2. 生成的数据需要人工审核后再用于实际应用
3. 建议定期更新关键词词典和模式模板
4. 使用时请遵守相关法律法规

## 许可证

MIT License
