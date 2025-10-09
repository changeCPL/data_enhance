"""
数据预处理模块
负责清理OCR文本，处理特殊字符，标准化数据格式
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
        
        # 常见的OCR错误映射
        self.ocr_corrections = {
            '0': 'O',  # 数字0误识别为字母O
            '1': 'l',  # 数字1误识别为字母l
            '5': 'S',  # 数字5误识别为字母S
            '8': 'B',  # 数字8误识别为字母B
        }
    
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
    
    def extract_urls(self, text: str) -> List[str]:
        """
        提取文本中的URL
        """
        url_pattern = r'https?://[^\s]+|www\.[^\s]+|[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}'
        urls = re.findall(url_pattern, text)
        return urls
    
    def extract_phone_numbers(self, text: str) -> List[str]:
        """
        提取文本中的电话号码
        """
        phone_patterns = [
            r'1[3-9]\d{9}',  # 中国手机号
            r'\d{3,4}-\d{7,8}',  # 固定电话
            r'\d{11}',  # 11位数字
        ]
        
        phones = []
        for pattern in phone_patterns:
            phones.extend(re.findall(pattern, text))
        
        return phones
    
    def extract_wechat_qq(self, text: str) -> List[str]:
        """
        提取微信、QQ等联系方式
        """
        contact_patterns = [
            r'微信[：:]\s*[a-zA-Z0-9_-]+',
            r'QQ[：:]\s*\d+',
            r'加微信[：:]\s*[a-zA-Z0-9_-]+',
            r'微信号[：:]\s*[a-zA-Z0-9_-]+',
        ]
        
        contacts = []
        for pattern in contact_patterns:
            contacts.extend(re.findall(pattern, text))
        
        return contacts
    
    def process_jsonl_file(self, file_path: str) -> pd.DataFrame:
        """
        处理JSONL文件，返回清理后的DataFrame
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
                        
                        # 提取结构化信息
                        urls = self.extract_urls(cleaned_text)
                        phones = self.extract_phone_numbers(cleaned_text)
                        contacts = self.extract_wechat_qq(cleaned_text)
                        
                        processed_item = {
                            'line_number': line_num,
                            'original_text': original_text,
                            'cleaned_text': cleaned_text,
                            'text_length': len(cleaned_text),
                            'label': item.get('labelname', ''),
                            'data_source': item.get('datasrc', ''),
                            'urls': urls,
                            'phone_numbers': phones,
                            'contacts': contacts,
                            'has_url': len(urls) > 0,
                            'has_phone': len(phones) > 0,
                            'has_contact': len(contacts) > 0,
                        }
                        
                        data.append(processed_item)
                        
                except json.JSONDecodeError as e:
                    print(f"第{line_num}行JSON解析错误: {e}")
                    continue
        
        return pd.DataFrame(data)
    
    def get_text_statistics(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        获取文本统计信息
        """
        stats = {
            'total_samples': len(df),
            'label_distribution': df['label'].value_counts().to_dict(),
            'avg_text_length': df['text_length'].mean(),
            'text_length_stats': df['text_length'].describe().to_dict(),
            'url_coverage': df['has_url'].mean(),
            'phone_coverage': df['has_phone'].mean(),
            'contact_coverage': df['has_contact'].mean(),
        }
        
        return stats


if __name__ == "__main__":
    # 测试示例
    preprocessor = DataPreprocessor()
    
    # 测试文本清理
    test_text = "这是一个测试文本\n包含换行符\t和制表符   多个空格"
    cleaned = preprocessor.clean_text(test_text)
    print(f"原始文本: {repr(test_text)}")
    print(f"清理后: {repr(cleaned)}")
    
    # 测试URL提取
    test_url_text = "访问网站 https://example.com 或 www.test.com"
    urls = preprocessor.extract_urls(test_url_text)
    print(f"提取的URL: {urls}")
