"""
优化版关键词提取模块 - 合并版本
主要优化：
1. 合并冗余的正则表达式模式
2. 优化关键词匹配逻辑，减少冗余检查
3. 使用更高效的数据结构
4. 针对性能问题优化语义和上下文关键词提取
"""

import jieba
import jieba.posseg as pseg
from sklearn.feature_extraction.text import TfidfVectorizer
from collections import Counter, defaultdict
import re
import pandas as pd
from typing import List, Dict, Set, Tuple
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics.pairwise import cosine_similarity
import warnings
warnings.filterwarnings('ignore')


class KeywordExtractor:
    def __init__(self, use_bert: bool = True, bert_model: str = "bert-base-chinese", semantic_analyzer=None):
        # 初始化jieba分词
        jieba.initialize()
        
        # 深度学习模型配置
        self.use_bert = use_bert
        self.bert_model_name = bert_model
        self.bert_tokenizer = None
        self.bert_model = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # 语义分析器引用
        self.semantic_analyzer = semantic_analyzer
        
        # 初始化BERT模型（仅用于传统关键词提取）
        if self.use_bert and self.semantic_analyzer is None:
            self._initialize_bert_model()
        
        # 优化后的标签关键词词典 - 使用最小覆盖集
        self.label_keywords = {
            '按摩色诱': {
                # 核心关键词，能覆盖所有变体
                'websites': ['同城', '交友', '约会', '美女', '按摩', 'spa', '会所'],
                'verbs': ['约', '聊', '见面', '服务', '按摩', '放松', '享受']
            },
            '博彩': {
                'websites': ['彩票', '博彩', '赌场', '游戏', '娱乐', '平台'],
                'verbs': ['投注', '下注', '中奖', '赚钱', '充值', '提现']
            },
            '兼职刷单': {
                'websites': ['兼职', '刷单', '任务', '佣金', '返利'],
                'verbs': ['刷单', '兼职', '赚钱', '佣金', '返利', '垫付']
            },
            '投资理财': {
                'websites': ['投资', '理财', '基金', '股票', '外汇', '数字货币'],
                'verbs': ['投资', '理财', '收益', '回报', '分红', '增值']
            },
            '虚假客服': {
                'websites': ['客服', '售后', '退款', '理赔', '银行'],
                'verbs': ['退款', '理赔', '解冻', '验证', '确认', '操作']
            }
        }
        
        # 优化后的网站模式 - 专门用于关键词提取
        self.website_patterns = [
            # 标准域名模式
            r'[a-zA-Z0-9.-]+\.(com|cn|net|org|cc|me|info|biz|co|tv|top|xyz|site|online|tech|app|io|ai|ml)(\.cn)?',
            # 中文网站模式
            r'[a-zA-Z0-9\u4e00-\u9fff]+(网|站|平台)([+._\-]?[a-zA-Z0-9\u4e00-\u9fff]*)?',
            # 特殊变体模式
            r'[a-zA-Z0-9\u4e00-\u9fff]*[+._\-*#@&%$][a-zA-Z0-9\u4e00-\u9fff]*(网|站|平台)?',
        ]
        
        # 通用动词词典 - 专门用于关键词提取
        self.common_verbs = {
            '约', '聊', '见面', '服务', '按摩', '放松', '享受',
            '投注', '下注', '中奖', '赚钱', '充值', '提现',
            '刷单', '兼职', '佣金', '返利', '垫付',
            '投资', '理财', '收益', '回报', '分红', '增值',
            '退款', '理赔', '解冻', '验证', '确认', '操作',
            '加', '联系', '咨询', '了解', '体验', '尝试'
        }
    
    def _initialize_bert_model(self):
        """初始化BERT模型"""
        try:
            self.bert_tokenizer = AutoTokenizer.from_pretrained(self.bert_model_name)
            self.bert_model = AutoModel.from_pretrained(self.bert_model_name)
            self.bert_model.to(self.device)
            self.bert_model.eval()
            print(f"BERT模型初始化成功: {self.bert_model_name}")
        except Exception as e:
            print(f"BERT模型初始化失败: {e}")
            self.use_bert = False
    
    def get_optimal_candidate_limit(self, text_count: int, top_k: int) -> int:
        """
        根据文本数量和top_k动态调整候选词数量限制
        """
        if top_k == -1:
            # 根据文本数量动态调整
            if text_count < 100:
                return min(500, text_count * 5)
            elif text_count < 500:
                return min(1000, text_count * 3)
            else:
                return min(1500, text_count * 2)
        else:
            # 候选词数量为top_k的合理倍数
            return max(50, min(top_k * 4, 1000))
    
    def get_optimal_similarity_config(self, text_count: int) -> dict:
        """
        根据文本数量动态调整相似度配置
        """
        if text_count < 100:
            return {'threshold': 0.8, 'max_pairs': 1000}
        elif text_count < 500:
            return {'threshold': 0.75, 'max_pairs': 5000}
        else:
            return {'threshold': 0.7, 'max_pairs': 10000}
    
    def extract_websites(self, text: str) -> List[str]:
        """
        提取网站名 - 使用优化的模式匹配和jieba分词
        """
        websites = []
        
        # 使用优化的正则表达式提取URL格式的网站
        for pattern in self.website_patterns:
            matches = re.findall(pattern, text)
            # 处理findall返回的元组
            for match in matches:
                if isinstance(match, tuple):
                    websites.append(''.join(match))
                else:
                    websites.append(match)
        
        # 使用jieba分词提取可能的网站名
        words = jieba.lcut(text)
        for word in words:
            if any(suffix in word for suffix in ['网', '平台', '网站', 'app', '软件']):
                websites.append(word)
        
        return list(set(websites))  # 去重
    
    def extract_verbs(self, text: str) -> List[str]:
        """
        提取动词 - 使用优化的匹配逻辑
        """
        verbs = []
        words = jieba.lcut(text)
        
        # 使用集合操作提高效率
        word_set = set(words)
        found_verbs = word_set.intersection(self.common_verbs)
        verbs.extend(found_verbs)
        
        # 使用词性标注提取动词
        pos_words = pseg.lcut(text)
        for word, flag in pos_words:
            if flag.startswith('v') and len(word) > 1:  # 动词且长度大于1
                verbs.append(word)
        
        return list(set(verbs))  # 去重
    
    def extract_tfidf_keywords(self, texts: List[str], top_k: int = 20) -> List[Tuple[str, float]]:
        """
        使用TF-IDF提取关键词
        """
        if not texts:
            return []
        
        # 预处理文本
        processed_texts = []
        for text in texts:
            # 使用jieba分词
            words = jieba.lcut(text)
            # 过滤停用词和短词
            filtered_words = [word for word in words if len(word) > 1 and word not in {'的', '了', '在', '是', '我', '有', '和', '就', '不', '人', '都', '一', '一个', '上', '也', '很', '到', '说', '要', '去', '你', '会', '着', '没有', '看', '好', '自己', '这'}]
            processed_texts.append(' '.join(filtered_words))
        
        # 使用TF-IDF
        vectorizer = TfidfVectorizer(max_features=1000, ngram_range=(1, 2))
        tfidf_matrix = vectorizer.fit_transform(processed_texts)
        
        # 获取特征名和重要性分数
        feature_names = vectorizer.get_feature_names_out()
        mean_scores = np.mean(tfidf_matrix.toarray(), axis=0)
        
        # 创建关键词-分数对
        keywords = list(zip(feature_names, mean_scores))
        keywords.sort(key=lambda x: x[1], reverse=True)
        
        # 如果top_k为-1，返回所有关键词；否则返回前top_k个
        if top_k == -1:
            return keywords
        else:
            return keywords[:top_k]
    
    def extract_bert_features(self, texts: List[str]) -> np.ndarray:
        """使用BERT提取文本特征"""
        if not self.use_bert or self.bert_model is None:
            return np.array([])
        
        features = []
        batch_size = 8  # 批处理大小
        
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            
            # 编码文本
            inputs = self.bert_tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors='pt'
            )
            
            # 移动到设备
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # 获取特征
            with torch.no_grad():
                outputs = self.bert_model(**inputs)
                # 使用[CLS]标记的特征
                batch_features = outputs.last_hidden_state[:, 0, :].cpu().numpy()
                features.append(batch_features)
        
        if features:
            return np.vstack(features)
        else:
            return np.array([])
    
    def extract_semantic_keywords(self, texts: List[str], top_k: int = 20) -> List[Tuple[str, float]]:
        """
        提取语义相关的关键词 - 委托给语义分析模块
        """
        if self.semantic_analyzer is not None:
            return self.semantic_analyzer.extract_semantic_keywords(texts, top_k)
        else:
            print("警告: 语义分析器未初始化，无法提取语义关键词")
            return []
    
    def extract_contextual_keywords(self, texts: List[str], top_k: int = 20) -> List[Tuple[str, float]]:
        """
        提取上下文相关的关键词 - 委托给语义分析模块
        """
        if self.semantic_analyzer is not None:
            return self.semantic_analyzer.extract_contextual_keywords(texts, top_k)
        else:
            print("警告: 语义分析器未初始化，无法提取上下文关键词")
            return []
    
    def extract_advanced_keywords(self, texts: List[str], top_k: int = 20) -> Dict[str, List[Tuple[str, float]]]:
        """
        提取高级关键词（结合语义和上下文）- 委托给语义分析模块
        """
        if self.semantic_analyzer is not None:
            return self.semantic_analyzer.extract_advanced_keywords(texts, top_k)
        else:
            print("警告: 语义分析器未初始化，无法提取高级关键词")
            return {
                'semantic_keywords': [],
                'contextual_keywords': [],
                'combined_keywords': []
            }
    
    def extract_label_specific_keywords(self, texts: List[str], label: str, top_k: int = 20) -> Dict[str, List[Tuple[str, int]]]:
        """
        提取特定标签的关键词
        """
        if label not in self.label_keywords:
            return {}
        
        label_info = self.label_keywords[label]
        results = {
            'websites': [],
            'verbs': []
        }
        
        # 统计关键词出现频次
        website_counter = Counter()
        verb_counter = Counter()
        
        for text in texts:
            # 统计网站名
            websites = self.extract_websites(text)
            for website in websites:
                # 检查是否匹配标签特定的网站关键词
                for keyword in label_info['websites']:
                    if keyword in website:
                        website_counter[keyword] += 1
                        break
            
            # 统计动词
            verbs = self.extract_verbs(text)
            for verb in verbs:
                if verb in label_info['verbs']:
                    verb_counter[verb] += 1
 
        # 获取最频繁的关键词（保存关键词和出现次数）
        if top_k == -1:
            results['websites'] = [(word, count) for word, count in website_counter.most_common()]
            results['verbs'] = [(word, count) for word, count in verb_counter.most_common()]
        else:
            results['websites'] = [(word, count) for word, count in website_counter.most_common(top_k)]
            results['verbs'] = [(word, count) for word, count in verb_counter.most_common(top_k)]
         
        # 打印最终统计
        print(f"    最终统计:")
        print(f"      网站关键词匹配: {len(results['websites'])} 个")
        print(f"      动词关键词匹配: {len(results['verbs'])} 个")

        return results
    
    def extract_all_keywords(self, df: pd.DataFrame, top_k: int = -1) -> Dict[str, Dict]:
        """
        从DataFrame中提取所有关键词（按标签分组）
        """
        results = {}
        
        # 按标签分组
        for label in df['label'].unique():
            if pd.isna(label) or label == '':
                continue
                
            label_texts = df[df['label'] == label]['cleaned_text'].tolist()
            
            print(f"\n\n开始提取关键词，标签: {label}, 文本数量: {len(label_texts)}")
            
            # 动态调整top_k值以避免性能问题（仅用于语义和上下文关键词）
            if top_k == -1:
                # 根据文本数量动态调整top_k
                if len(label_texts) < 100:
                    effective_top_k = 50
                elif len(label_texts) < 500:
                    effective_top_k = 30
                else:
                    effective_top_k = 20
                print(f"动态调整top_k: {top_k} -> {effective_top_k}")
            else:
                effective_top_k = top_k
            
            # 提取TF-IDF关键词
            tfidf_keywords = self.extract_tfidf_keywords(label_texts, top_k)
            
            # 提取高级关键词（语义+上下文）
            advanced_keywords = {}
            semantic_keywords = []
            contextual_keywords = []
            combined_keywords = []
            
            if self.semantic_analyzer is not None:
                try:
                    advanced_keywords = self.extract_advanced_keywords(label_texts, effective_top_k)
                    semantic_keywords = advanced_keywords.get('semantic_keywords', [])
                    contextual_keywords = advanced_keywords.get('contextual_keywords', [])
                    combined_keywords = advanced_keywords.get('combined_keywords', [])
                except Exception as e:
                    print(f"高级关键词提取失败: {e}")
            elif self.use_bert:
                # 回退到传统方法
                try:
                    semantic_keywords = self.extract_semantic_keywords(label_texts, effective_top_k)
                    contextual_keywords = self.extract_contextual_keywords(label_texts, effective_top_k)
                except Exception as e:
                    print(f"传统关键词提取失败: {e}")
            
            # 提取标签特定关键词
            label_keywords = self.extract_label_specific_keywords(label_texts, label, top_k)
            
            # 提取BERT特征（用于兼容性）
            bert_features = None
            if self.use_bert:
                try:
                    bert_features = self.extract_bert_features(label_texts)
                except Exception as e:
                    print(f"BERT特征提取失败: {e}")
            
            results[label] = {
                'tfidf_keywords': tfidf_keywords,
                'semantic_keywords': semantic_keywords,
                'contextual_keywords': contextual_keywords,
                'combined_keywords': combined_keywords,
                'label_keywords': label_keywords,
                'bert_features_shape': bert_features.shape if bert_features is not None else None,
                'sample_count': len(label_texts)
            }
            
            # 打印每一项结果的数量用于调试
            print(f"\n\n------标签 {label} 结果统计------:")
            print(f"  TF-IDF关键词数量: {len(tfidf_keywords)}")
            print(f"  语义关键词数量: {len(semantic_keywords)}")
            print(f"  上下文关键词数量: {len(contextual_keywords)}")
            print(f"  综合关键词数量: {len(combined_keywords)}")
            print(f"  标签特定关键词数量: {len(label_keywords.get('websites', [])) + len(label_keywords.get('verbs', []))}")
            print(f"    网站关键词: {len(label_keywords.get('websites', []))}")
            print(f"    动词关键词: {len(label_keywords.get('verbs', []))}")
            print("--------------------------------")
        
        return results
    
    def generate_keyword_report(self, keyword_results: Dict[str, Dict]) -> str:
        """
        生成关键词分析报告
        """
        report = "关键词分析报告\n"
        report += "=" * 50 + "\n\n"
        
        for label, data in keyword_results.items():
            report += f"标签: {label}\n"
            report += "-" * 30 + "\n"
            
            # TF-IDF关键词
            if data['tfidf_keywords']:
                report += f"\nTF-IDF关键词 (共{len(data['tfidf_keywords'])}个):\n"
                for word, score in data['tfidf_keywords']:
                    report += f"  {word}: {score:.4f}\n"
            
            # 语义关键词
            if data['semantic_keywords']:
                report += f"\n语义关键词 (共{len(data['semantic_keywords'])}个):\n"
                for word, score in data['semantic_keywords']:
                    report += f"  {word}: {score:.4f}\n"
            
            # 上下文关键词
            if data['contextual_keywords']:
                report += f"\n上下文关键词 (共{len(data['contextual_keywords'])}个):\n"
                for word, score in data['contextual_keywords']:
                    report += f"  {word}: {score:.4f}\n"
            
            # 综合关键词
            if data.get('combined_keywords'):
                report += f"\n综合关键词 (共{len(data['combined_keywords'])}个):\n"
                for word, score in data['combined_keywords']:
                    report += f"  {word}: {score:.4f}\n"
            
            # 标签特定关键词
            if data['label_keywords']:
                report += "\n标签特定关键词:\n"
                for category, keywords in data['label_keywords'].items():
                    if keywords:
                        if isinstance(keywords[0], tuple):  # 新的格式：(关键词, 次数)
                            report += f"  {category} (共{len(keywords)}个):\n"
                            for keyword, count in keywords:
                                report += f"    {keyword}: {count}次\n"
                        else:  # 旧的格式：只有关键词
                            report += f"  {category} (共{len(keywords)}个): {', '.join(keywords)}\n"
            
            report += f"\n样本数量: {data['sample_count']}\n"
            if data['bert_features_shape']:
                report += f"BERT特征维度: {data['bert_features_shape']}\n"
            
            report += "\n" + "=" * 50 + "\n\n"
        
        return report
