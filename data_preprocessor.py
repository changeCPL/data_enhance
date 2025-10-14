"""
数据预处理模块
负责清理OCR文本，处理特殊字符，标准化数据格式
重构后只负责文本清理和标准化，不处理结构化信息提取
"""

import re
import json
import pandas as pd
from typing import List, Dict, Any
import unicodedata


class DataPreprocessor:
    def __init__(self):
        # 定义需要清理的OCR噪音字符
        self.ocr_noise_patterns = [
            r'[^\u4e00-\u9fff\u0020-\u007e\u3000-\u303f\uff00-\uffef]',  # 保留中文、英文、标点
            r'\s+',  # 多个空格合并为一个
            r'[\r\n\t]+',  # 换行符、制表符
        ]
        

    def clean_text(self, text: str) -> str:
        """
        清理OCR文本中的噪音字符
        """
        if not text:
            return ""
        
        # 标准化Unicode字符
        text = unicodedata.normalize('NFKC', text)
        
        # 移除OCR噪音字符
        for pattern in self.ocr_noise_patterns:
            text = re.sub(pattern, ' ', text)
        
        # 清理多余空格
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text

    def process_jsonl_file(self, file_path: str) -> pd.DataFrame:
        """
        处理JSONL文件，返回清理后的DataFrame
        重构后只负责文本清理，不提取结构化信息
        """
        data = []
        
        with open(file_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                try:
                    item = json.loads(line.strip())
                    
                    # 清理conversation文本
                    if 'conversation' in item:
                        original_text = item['conversation']
                        cleaned_text = self.clean_text(original_text)
                        
                        processed_item = {
                            'line_number': line_num,
                            'original_text': original_text,
                            'cleaned_text': cleaned_text,
                            'text_length': len(cleaned_text),
                            'label': item.get('labelname', ''),
                            'data_source': item.get('datasrc', ''),
                        }
                        
                        data.append(processed_item)
                        
                except json.JSONDecodeError as e:
                    print(f"第{line_num}行JSON解析错误: {e}")
                    continue
        
        return pd.DataFrame(data)
    
    def get_text_statistics(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        获取文本统计信息
        重构后只统计基础文本信息
        """
        # 转换numpy类型为Python原生类型
        def convert_numpy_types(obj):
            if isinstance(obj, dict):
                return {key: convert_numpy_types(value) for key, value in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy_types(item) for item in obj]
            elif isinstance(obj, tuple):
                return tuple(convert_numpy_types(item) for item in obj)
            elif hasattr(obj, 'item'):  # numpy scalar
                return obj.item()
            elif hasattr(obj, 'tolist'):  # numpy array
                return obj.tolist()
            else:
                return obj
        
        stats = {
            'total_samples': len(df),
            'label_distribution': df['label'].value_counts().to_dict(),
            'avg_text_length': df['text_length'].mean(),
            'text_length_stats': df['text_length'].describe().to_dict(),
        }
        
        return convert_numpy_types(stats)


if __name__ == "__main__":
    # 测试示例
    preprocessor = DataPreprocessor()
    
    # 测试文本清理
    test_text = "这是一个测试文本\n包含换行符\t和制表符   多个空格"
    cleaned = preprocessor.clean_text(test_text)
    print(f"原始文本: {repr(test_text)}")
    print(f"清理后: {repr(cleaned)}")
