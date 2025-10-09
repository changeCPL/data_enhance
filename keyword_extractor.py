"""
关键词提取模块
负责从文本中提取网站名、常见动词、网址等关键信息
"""

import jieba
import jieba.posseg as pseg
from sklearn.feature_extraction.text import TfidfVectorizer
from collections import Counter, defaultdict
import re
import pandas as pd
from typing import List, Dict, Set, Tuple
import numpy as np


class KeywordExtractor:
    def __init__(self):
        # 初始化jieba分词
        jieba.initialize()
        
        # 定义不同标签的关键词词典
        self.label_keywords = {
            '按摩色诱': {
                'websites': ['同城', '交友', '约会', '美女', '按摩', 'spa', '会所'],
                'verbs': ['约', '聊', '见面', '服务', '按摩', '放松', '享受'],
                'patterns': [r'[0-9]+岁', r'[0-9]+元', r'[0-9]+小时']
            },
            '博彩': {
                'websites': ['彩票', '博彩', '赌场', '游戏', '娱乐', '平台'],
                'verbs': ['投注', '下注', '中奖', '赚钱', '充值', '提现'],
                'patterns': [r'[0-9]+倍', r'[0-9]+%', r'稳赚', r'包中']
            },
            '兼职刷单': {
                'websites': ['兼职', '刷单', '任务', '佣金', '返利'],
                'verbs': ['刷单', '兼职', '赚钱', '佣金', '返利', '垫付'],
                'patterns': [r'[0-9]+元/单', r'日赚[0-9]+', r'佣金[0-9]+%']
            },
            '投资理财': {
                'websites': ['投资', '理财', '基金', '股票', '外汇', '数字货币'],
                'verbs': ['投资', '理财', '收益', '回报', '分红', '增值'],
                'patterns': [r'年化[0-9]+%', r'收益[0-9]+%', r'保本']
            },
            '虚假客服': {
                'websites': ['客服', '售后', '退款', '理赔', '银行'],
                'verbs': ['退款', '理赔', '解冻', '验证', '确认', '操作'],
                'patterns': [r'验证码', r'银行卡', r'身份证', r'密码']
            }
        }
        
        # 通用网站名模式
        self.website_patterns = [
            r'[a-zA-Z0-9.-]+\.(com|cn|net|org|cc|me)',
            r'[a-zA-Z0-9]+\.(com|cn|net|org)',
            r'[a-zA-Z0-9]+平台',
            r'[a-zA-Z0-9]+网站',
            r'[a-zA-Z0-9]+网',
        ]
        
        # 通用动词词典
        self.common_verbs = {
            '约', '聊', '见面', '服务', '按摩', '放松', '享受',
            '投注', '下注', '中奖', '赚钱', '充值', '提现',
            '刷单', '兼职', '佣金', '返利', '垫付',
            '投资', '理财', '收益', '回报', '分红', '增值',
            '退款', '理赔', '解冻', '验证', '确认', '操作',
            '加', '联系', '咨询', '了解', '体验', '尝试'
        }
    
    def extract_websites(self, text: str) -> List[str]:
        """
        提取网站名
        """
        websites = []
        
        # 使用正则表达式提取URL格式的网站
        for pattern in self.website_patterns:
            matches = re.findall(pattern, text)
            websites.extend(matches)
        
        # 使用jieba分词提取可能的网站名
        words = jieba.lcut(text)
        for word in words:
            if any(keyword in word for keyword in ['网', '平台', '网站', 'app', '软件']):
                websites.append(word)
        
        return list(set(websites))
    
    def extract_verbs(self, text: str) -> List[Tuple[str, str]]:
        """
        提取动词及其词性
        """
        words = pseg.cut(text)
        verbs = []
        
        for word, flag in words:
            if flag.startswith('v') or word in self.common_verbs:
                verbs.append((word, flag))
        
        return verbs
    
    def extract_keywords_by_tfidf(self, texts: List[str], top_k: int = 20) -> List[Tuple[str, float]]:
        """
        使用TF-IDF提取关键词
        """
        # 自定义停用词
        stop_words = {'的', '了', '在', '是', '我', '有', '和', '就', '不', '人', '都', '一', '一个', '上', '也', '很', '到', '说', '要', '去', '你', '会', '着', '没有', '看', '好', '自己', '这'}
        
        # 预处理文本
        processed_texts = []
        for text in texts:
            words = jieba.lcut(text)
            words = [word for word in words if word not in stop_words and len(word) > 1]
            processed_texts.append(' '.join(words))
        
        # 计算TF-IDF
        vectorizer = TfidfVectorizer(max_features=1000, ngram_range=(1, 2))
        tfidf_matrix = vectorizer.fit_transform(processed_texts)
        
        # 获取特征名和分数
        feature_names = vectorizer.get_feature_names_out()
        scores = np.mean(tfidf_matrix.toarray(), axis=0)
        
        # 排序并返回top_k
        keyword_scores = list(zip(feature_names, scores))
        keyword_scores.sort(key=lambda x: x[1], reverse=True)
        
        return keyword_scores[:top_k]
    
    def extract_label_specific_keywords(self, texts: List[str], label: str) -> Dict[str, List[str]]:
        """
        提取特定标签的关键词
        """
        if label not in self.label_keywords:
            return {}
        
        label_info = self.label_keywords[label]
        results = {
            'websites': [],
            'verbs': [],
            'patterns': []
        }
        
        # 统计关键词出现频次
        website_counter = Counter()
        verb_counter = Counter()
        pattern_matches = []
        
        for text in texts:
            # 提取网站名
            websites = self.extract_websites(text)
            for website in websites:
                if any(keyword in website for keyword in label_info['websites']):
                    website_counter[website] += 1
            
            # 提取动词
            verbs = self.extract_verbs(text)
            for verb, _ in verbs:
                if verb in label_info['verbs']:
                    verb_counter[verb] += 1
            
            # 匹配模式
            for pattern in label_info['patterns']:
                matches = re.findall(pattern, text)
                pattern_matches.extend(matches)
        
        results['websites'] = [website for website, count in website_counter.most_common(10)]
        results['verbs'] = [verb for verb, count in verb_counter.most_common(10)]
        results['patterns'] = list(set(pattern_matches))
        
        return results
    
    def analyze_text_patterns(self, text: str) -> Dict[str, List[str]]:
        """
        分析文本中的模式
        """
        patterns = {
            'numbers': re.findall(r'[0-9]+', text),
            'money': re.findall(r'[0-9]+元|[0-9]+块|[0-9]+万|[0-9]+千', text),
            'time': re.findall(r'[0-9]+点|[0-9]+小时|[0-9]+分钟', text),
            'percentages': re.findall(r'[0-9]+%', text),
            'phone_numbers': re.findall(r'1[3-9]\d{9}', text),
            'urls': re.findall(r'https?://[^\s]+|www\.[^\s]+', text),
        }
        
        return patterns
    
    def extract_all_keywords(self, df: pd.DataFrame) -> Dict[str, Dict]:
        """
        从DataFrame中提取所有关键词
        """
        results = {}
        
        # 按标签分组
        for label in df['label'].unique():
            if pd.isna(label) or label == '':
                continue
                
            label_texts = df[df['label'] == label]['cleaned_text'].tolist()
            
            # 提取TF-IDF关键词
            tfidf_keywords = self.extract_keywords_by_tfidf(label_texts)
            
            # 提取标签特定关键词
            label_keywords = self.extract_label_specific_keywords(label_texts, label)
            
            # 分析文本模式
            all_patterns = defaultdict(list)
            for text in label_texts:
                patterns = self.analyze_text_patterns(text)
                for pattern_type, matches in patterns.items():
                    all_patterns[pattern_type].extend(matches)
            
            # 统计模式频次
            pattern_stats = {}
            for pattern_type, matches in all_patterns.items():
                pattern_stats[pattern_type] = dict(Counter(matches).most_common(10))
            
            results[label] = {
                'tfidf_keywords': tfidf_keywords,
                'label_keywords': label_keywords,
                'patterns': pattern_stats,
                'sample_count': len(label_texts)
            }
        
        return results
    
    def generate_keyword_report(self, keyword_results: Dict[str, Dict]) -> str:
        """
        生成关键词分析报告
        """
        report = "关键词提取分析报告\n"
        report += "=" * 50 + "\n\n"
        
        for label, data in keyword_results.items():
            report += f"标签: {label}\n"
            report += f"样本数量: {data['sample_count']}\n"
            report += "-" * 30 + "\n"
            
            # TF-IDF关键词
            report += "TF-IDF关键词 (Top 10):\n"
            for keyword, score in data['tfidf_keywords'][:10]:
                report += f"  {keyword}: {score:.4f}\n"
            
            # 标签特定关键词
            if data['label_keywords']:
                report += "\n标签特定关键词:\n"
                for category, keywords in data['label_keywords'].items():
                    if keywords:
                        report += f"  {category}: {', '.join(keywords[:5])}\n"
            
            # 模式统计
            report += "\n文本模式统计:\n"
            for pattern_type, stats in data['patterns'].items():
                if stats:
                    report += f"  {pattern_type}: {dict(list(stats.items())[:3])}\n"
            
            report += "\n" + "=" * 50 + "\n\n"
        
        return report


if __name__ == "__main__":
    # 测试示例
    extractor = KeywordExtractor()
    
    # 测试文本
    test_text = "欢迎来到同城交友平台，这里有美女按摩服务，价格优惠，每小时200元"
    
    # 提取网站名
    websites = extractor.extract_websites(test_text)
    print(f"提取的网站名: {websites}")
    
    # 提取动词
    verbs = extractor.extract_verbs(test_text)
    print(f"提取的动词: {verbs}")
    
    # 分析模式
    patterns = extractor.analyze_text_patterns(test_text)
    print(f"文本模式: {patterns}")
