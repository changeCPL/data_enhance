"""
语义相似度分析模块
基于深度学习的语义理解和相似度计算
包含优化的关键词提取功能
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional, Union
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans, DBSCAN
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import TfidfVectorizer
import matplotlib.pyplot as plt
from collections import defaultdict, Counter
from dataclasses import dataclass
from tqdm import tqdm
import jieba
import jieba.posseg as pseg
import re
import random
import warnings
warnings.filterwarnings('ignore')

try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    print("警告: sentence-transformers未安装，将使用基础BERT模型")

from transformers import AutoTokenizer, AutoModel
import torch


@dataclass
class SemanticConfig:
    """语义分析配置"""
    model_name: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    batch_size: int = 32
    max_length: int = 128
    similarity_threshold: float = 0.7
    clustering_method: str = "kmeans"  # kmeans, dbscan
    n_clusters: int = 5
    min_samples: int = 2  # for DBSCAN
    eps: float = 0.5  # for DBSCAN
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    
    # 关键词提取配置
    keyword_batch_size: int = 100
    max_candidate_words: int = 1000
    semantic_threshold: float = 0.3
    contextual_threshold: float = 0.7
    max_similarity_pairs: int = 5000
    enable_caching: bool = True
    cache_size: int = 10000


class SemanticAnalyzer:
    """语义相似度分析器"""
    
    def __init__(self, config: SemanticConfig = None):
        self.config = config or SemanticConfig()
        self.model = None
        self.tokenizer = None
        self.embeddings_cache = {}
        
        # 关键词提取相关缓存
        self.keyword_cache = {}
        self.word_embeddings_cache = {}
        
        # 初始化jieba分词
        jieba.initialize()
        
        # 停用词集合
        self.stop_words = {
            '的', '了', '在', '是', '我', '有', '和', '就', '不', '人', '都', '一', '一个', 
            '上', '也', '很', '到', '说', '要', '去', '你', '会', '着', '没有', '看', '好', 
            '自己', '这', '那', '他', '她', '它', '们', '什么', '怎么', '为什么', '哪里', 
            '时候', '可以', '应该', '需要', '想要', '希望', '觉得', '认为', '知道', '了解'
        }
        
        # 支持的模型列表
        self.supported_models = {
            'multilingual': 'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2',
            'chinese': 'sentence-transformers/distiluse-base-multilingual-cased',
            'bert-base': 'bert-base-chinese',
            'roberta': 'hfl/chinese-roberta-wwm-ext',
            'macbert': 'hfl/chinese-macbert-base'
        }
        
        self._initialize_model()
    
    def _initialize_model(self):
        """初始化语义模型"""
        try:
            if SENTENCE_TRANSFORMERS_AVAILABLE and 'sentence-transformers' in self.config.model_name:
                # 使用SentenceTransformers
                model_name = self.supported_models.get(
                    self.config.model_name.split('/')[-1], 
                    self.config.model_name
                )
                self.model = SentenceTransformer(model_name)
                print(f"SentenceTransformers模型加载成功: {model_name}")
            else:
                # 使用HuggingFace Transformers
                model_name = self.supported_models.get(self.config.model_name, self.config.model_name)
                self.tokenizer = AutoTokenizer.from_pretrained(model_name)
                self.model = AutoModel.from_pretrained(model_name)
                self.model.to(self.config.device)
                self.model.eval()
                print(f"Transformers模型加载成功: {model_name}")
        except Exception as e:
            print(f"模型初始化失败: {e}")
            # 回退到基础模型
            self._fallback_model()
    
    def _fallback_model(self):
        """回退到基础BERT模型"""
        try:
            self.tokenizer = AutoTokenizer.from_pretrained('bert-base-chinese')
            self.model = AutoModel.from_pretrained('bert-base-chinese')
            self.model.to(self.config.device)
            self.model.eval()
            print("使用回退模型: bert-base-chinese")
        except Exception as e:
            print(f"回退模型也失败: {e}")
            self.model = None
    
    def encode_texts(self, texts: List[str], use_cache: bool = True) -> np.ndarray:
        """编码文本为向量"""
        if self.model is None:
            raise ValueError("模型未初始化")
        
        # 检查缓存
        if use_cache:
            cached_embeddings = []
            uncached_texts = []
            uncached_indices = []
            
            for i, text in enumerate(texts):
                if text in self.embeddings_cache:
                    cached_embeddings.append(self.embeddings_cache[text])
                else:
                    uncached_texts.append(text)
                    uncached_indices.append(i)
            
            if uncached_texts:
                new_embeddings = self._encode_batch(uncached_texts)
                for text, embedding in zip(uncached_texts, new_embeddings):
                    self.embeddings_cache[text] = embedding
                
                # 修复合并逻辑
                all_embeddings = [None] * len(texts)
                
                # 先填充缓存的嵌入
                cached_idx = 0
                for i, text in enumerate(texts):
                    if text in self.embeddings_cache and i not in uncached_indices:
                        all_embeddings[i] = cached_embeddings[cached_idx]
                        cached_idx += 1
                
                # 再填充未缓存的嵌入
                uncached_idx = 0
                for i in uncached_indices:
                    all_embeddings[i] = new_embeddings[uncached_idx]
                    uncached_idx += 1
                
                result = np.array(all_embeddings)
                
                # 验证结果
                if result is None or len(result) == 0:
                    raise ValueError("编码结果为空")
                
                if np.any(np.isnan(result)) or np.any(np.isinf(result)):
                    print("警告: 编码结果包含异常值，将进行清理")
                    result = np.nan_to_num(result, nan=0.0, posinf=1.0, neginf=-1.0)
                
                return result
            else:
                result = np.array(cached_embeddings)
                
                # 验证结果
                if result is None or len(result) == 0:
                    raise ValueError("编码结果为空")
                
                if np.any(np.isnan(result)) or np.any(np.isinf(result)):
                    print("警告: 编码结果包含异常值，将进行清理")
                    result = np.nan_to_num(result, nan=0.0, posinf=1.0, neginf=-1.0)
                
                return result
        else:
            return self._encode_batch(texts)
    
    def _encode_batch(self, texts: List[str]) -> np.ndarray:
        """批量编码文本"""
        if hasattr(self.model, 'encode'):
            # SentenceTransformers
            embeddings = self.model.encode(
                texts, 
                batch_size=self.config.batch_size,
                show_progress_bar=False,
                convert_to_numpy=True
            )
            
            # 验证编码结果
            if embeddings is None or len(embeddings) == 0:
                raise ValueError("编码结果为空")
            
            # 检查是否有异常值
            if np.any(np.isnan(embeddings)) or np.any(np.isinf(embeddings)):
                print("警告: 编码结果包含异常值，将进行清理")
                embeddings = np.nan_to_num(embeddings, nan=0.0, posinf=1.0, neginf=-1.0)
            
            return embeddings
        else:
            # HuggingFace Transformers
            embeddings = []
            
            for i in range(0, len(texts), self.config.batch_size):
                batch_texts = texts[i:i + self.config.batch_size]
                
                # 分词
                inputs = self.tokenizer(
                    batch_texts,
                    truncation=True,
                    padding=True,
                    max_length=self.config.max_length,
                    return_tensors='pt'
                )
                
                # 移动到设备
                inputs = {k: v.to(self.config.device) for k, v in inputs.items()}
                
                # 编码
                with torch.no_grad():
                    outputs = self.model(**inputs)
                    # 使用[CLS]标记的表示
                    batch_embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
                    
                    # 检查批次编码结果
                    if np.any(np.isnan(batch_embeddings)) or np.any(np.isinf(batch_embeddings)):
                        print("警告: 批次编码结果包含异常值，将进行清理")
                        batch_embeddings = np.nan_to_num(batch_embeddings, nan=0.0, posinf=1.0, neginf=-1.0)
                    
                    embeddings.extend(batch_embeddings)
            
            result = np.array(embeddings)
            
            # 最终验证
            if result is None or len(result) == 0:
                raise ValueError("编码结果为空")
            
            if np.any(np.isnan(result)) or np.any(np.isinf(result)):
                print("警告: 最终编码结果包含异常值，将进行清理")
                result = np.nan_to_num(result, nan=0.0, posinf=1.0, neginf=-1.0)
            
            return result
    
    def compute_similarity_matrix(self, texts: List[str]) -> np.ndarray:
        """计算文本相似度矩阵"""
        embeddings = self.encode_texts(texts)
        similarity_matrix = cosine_similarity(embeddings)
        return similarity_matrix
    
    def find_similar_texts(self, query_text: str, candidate_texts: List[str], 
                          top_k: int = 5, threshold: float = None) -> List[Tuple[str, float, int]]:
        """找到与查询文本相似的文本"""
        if threshold is None:
            threshold = self.config.similarity_threshold
        
        all_texts = [query_text] + candidate_texts
        embeddings = self.encode_texts(all_texts)
        
        query_embedding = embeddings[0:1]
        candidate_embeddings = embeddings[1:]
        
        similarities = cosine_similarity(query_embedding, candidate_embeddings)[0]
        
        # 找到相似度高于阈值的文本
        similar_indices = np.where(similarities >= threshold)[0]
        similar_results = [
            (candidate_texts[i], similarities[i], i) 
            for i in similar_indices
        ]
        
        # 按相似度排序
        similar_results.sort(key=lambda x: x[1], reverse=True)
        
        return similar_results[:top_k]
    
    def cluster_texts(self, texts: List[str], labels: List[str] = None) -> Dict:
        """对文本进行聚类分析"""
        embeddings = self.encode_texts(texts)
        
        if self.config.clustering_method == "kmeans":
            clusterer = KMeans(n_clusters=self.config.n_clusters, random_state=42)
            cluster_labels = clusterer.fit_predict(embeddings)
        elif self.config.clustering_method == "dbscan":
            clusterer = DBSCAN(min_samples=self.config.min_samples, eps=self.config.eps)
            cluster_labels = clusterer.fit_predict(embeddings)
        else:
            raise ValueError(f"不支持的聚类方法: {self.config.clustering_method}")
        
        # 分析聚类结果
        cluster_analysis = self._analyze_clusters(texts, labels, cluster_labels)
        
        return {
            'cluster_labels': cluster_labels.tolist(),
            'cluster_centers': clusterer.cluster_centers_ if hasattr(clusterer, 'cluster_centers_') else None,
            'n_clusters': len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0),
            'analysis': cluster_analysis
        }
    
    def _analyze_clusters(self, texts: List[str], labels: List[str], cluster_labels: List[int]) -> Dict:
        """分析聚类结果"""
        analysis = {}
        
        for cluster_id in set(cluster_labels):
            if cluster_id == -1:  # 噪声点
                continue
            
            cluster_mask = np.array(cluster_labels) == cluster_id
            cluster_texts = [texts[i] for i in range(len(texts)) if cluster_mask[i]]
            cluster_labels_subset = [labels[i] for i in range(len(labels)) if cluster_mask[i]] if labels else None
            
            # 计算聚类统计
            cluster_stats = {
                'size': len(cluster_texts),
                'texts': cluster_texts[:5],  # 只保存前5个文本作为示例
                'label_distribution': dict(Counter(cluster_labels_subset)) if cluster_labels_subset else None
            }
            
            # 计算聚类内相似度
            if len(cluster_texts) > 1:
                cluster_embeddings = self.encode_texts(cluster_texts)
                cluster_similarity_matrix = cosine_similarity(cluster_embeddings)
                # 计算平均相似度（排除对角线）
                mask = np.ones_like(cluster_similarity_matrix, dtype=bool)
                np.fill_diagonal(mask, False)
                cluster_stats['avg_similarity'] = np.mean(cluster_similarity_matrix[mask])
            else:
                cluster_stats['avg_similarity'] = 1.0
            
            analysis[f'cluster_{cluster_id}'] = cluster_stats
        
        return analysis
    
    def analyze_semantic_patterns(self, df: pd.DataFrame) -> Dict:
        """分析语义模式"""
        print("开始语义模式分析...")
        
        # 按标签分组分析
        results = {}
        
        for label in df['label'].unique():
            if pd.isna(label) or label == '':
                continue
            
            label_data = df[df['label'] == label]
            texts = label_data['cleaned_text'].tolist()
            
            print(f"分析标签: {label} ({len(texts)} 条文本)")
            
            # 计算相似度矩阵
            similarity_matrix = self.compute_similarity_matrix(texts)
            
            # 聚类分析
            cluster_results = self.cluster_texts(texts, [label] * len(texts))
            
            # 计算语义统计
            semantic_stats = self._compute_semantic_stats(similarity_matrix, texts)
            
            # 找到代表性文本
            representative_texts = self._find_representative_texts(texts, similarity_matrix)
            
            results[label] = {
                'sample_count': len(texts),
                'similarity_matrix': similarity_matrix.tolist(),
                'semantic_stats': semantic_stats,
                'clustering': cluster_results,
                'representative_texts': representative_texts
            }
        
        return results
    
    def _compute_semantic_stats(self, similarity_matrix: np.ndarray, texts: List[str]) -> Dict:
        """计算语义统计信息"""
        # 排除对角线元素
        mask = np.ones_like(similarity_matrix, dtype=bool)
        np.fill_diagonal(mask, False)
        
        similarities = similarity_matrix[mask]
        
        stats = {
            'mean_similarity': float(np.mean(similarities)),
            'std_similarity': float(np.std(similarities)),
            'min_similarity': float(np.min(similarities)),
            'max_similarity': float(np.max(similarities)),
            'high_similarity_ratio': float(np.mean(similarities > 0.8)),
            'low_similarity_ratio': float(np.mean(similarities < 0.3))
        }
        
        return stats
    
    def _find_representative_texts(self, texts: List[str], similarity_matrix: np.ndarray, 
                                 top_k: int = 3) -> List[Dict]:
        """找到代表性文本"""
        # 计算每个文本的平均相似度（与其他文本的相似度）
        avg_similarities = np.mean(similarity_matrix, axis=1)
        
        # 找到相似度最高的文本作为代表
        representative_indices = np.argsort(avg_similarities)[-top_k:][::-1]
        
        representatives = []
        for idx in representative_indices:
            representatives.append({
                'text': texts[idx],
                'avg_similarity': float(avg_similarities[idx]),
                'index': int(idx)
            })
        
        return representatives
    
    def detect_semantic_anomalies(self, df: pd.DataFrame, threshold: float = 0.3) -> Dict:
        """检测语义异常文本"""
        print("检测语义异常...")
        
        anomalies = {}
        
        for label in df['label'].unique():
            if pd.isna(label) or label == '':
                continue
            
            label_data = df[df['label'] == label]
            texts = label_data['cleaned_text'].tolist()
            
            if len(texts) < 3:  # 需要至少3个样本来检测异常
                continue
            
            # 计算相似度矩阵
            similarity_matrix = self.compute_similarity_matrix(texts)
            
            # 找到与其他文本相似度都很低的文本
            avg_similarities = np.mean(similarity_matrix, axis=1)
            anomaly_indices = np.where(avg_similarities < threshold)[0]
            
            if len(anomaly_indices) > 0:
                anomaly_texts = [
                    {
                        'text': texts[i],
                        'avg_similarity': float(avg_similarities[i]),
                        'index': int(i)
                    }
                    for i in anomaly_indices
                ]
                anomalies[label] = anomaly_texts
        
        return anomalies
    
    def generate_semantic_report(self, semantic_results: Dict) -> str:
        """生成语义分析报告"""
        report = "语义相似度分析报告\n"
        report += "=" * 50 + "\n\n"
        
        # 检查输入数据
        if not semantic_results:
            report += "无语义分析数据\n"
            return report
        
        for label, data in semantic_results.items():
            # 确保data是字典类型
            if not isinstance(data, dict):
                report += f"标签: {label}\n"
                report += f"数据格式错误: {type(data)}\n"
                report += "-" * 30 + "\n\n"
                continue
            report += f"标签: {label}\n"
            # 安全地获取样本数量
            sample_count = data.get('sample_count', '未知')
            report += f"样本数量: {sample_count}\n"
            report += "-" * 30 + "\n"
            
            # 语义统计
            stats = data.get('semantic_stats', {})
            if stats:
                report += "语义相似度统计:\n"
                report += f"  平均相似度: {stats.get('mean_similarity', 0):.4f}\n"
                report += f"  相似度标准差: {stats.get('std_similarity', 0):.4f}\n"
                report += f"  高相似度比例 (>0.8): {stats.get('high_similarity_ratio', 0):.2%}\n"
                report += f"  低相似度比例 (<0.3): {stats.get('low_similarity_ratio', 0):.2%}\n"
            else:
                report += "语义相似度统计: 无数据\n"
            
            # 聚类信息
            clustering = data.get('clustering', {})
            if clustering:
                report += f"\n聚类信息:\n"
                report += f"  聚类数量: {clustering.get('n_clusters', 0)}\n"
            else:
                report += f"\n聚类信息: 无数据\n"
            
            # 代表性文本
            representatives = data.get('representative_texts', [])
            if representatives:
                report += f"\n代表性文本:\n"
                for i, rep in enumerate(representatives, 1):
                    text = rep.get('text', '')[:50]
                    similarity = rep.get('avg_similarity', 0)
                    report += f"  {i}. {text}... (相似度: {similarity:.4f})\n"
            else:
                report += f"\n代表性文本: 无数据\n"
            
            report += "\n" + "=" * 50 + "\n\n"
        
        return report
    
    def visualize_semantic_space(self, df: pd.DataFrame, save_path: str = "semantic_visualization.png"):
        """可视化语义空间"""
        print("生成语义空间可视化...")
        
        # 获取所有文本和标签
        texts = df['cleaned_text'].tolist()
        labels = df['label'].tolist()
        
        # 编码文本
        embeddings = self.encode_texts(texts)
        
        # 降维到2D
        pca = PCA(n_components=2, random_state=42)
        embeddings_2d = pca.fit_transform(embeddings)
        
        # 创建可视化
        plt.figure(figsize=(12, 8))
        
        # 为每个标签使用不同颜色
        unique_labels = list(set(labels))
        colors = plt.cm.Set3(np.linspace(0, 1, len(unique_labels)))
        
        for i, label in enumerate(unique_labels):
            mask = np.array(labels) == label
            plt.scatter(
                embeddings_2d[mask, 0], 
                embeddings_2d[mask, 1],
                c=[colors[i]], 
                label=label, 
                alpha=0.7,
                s=50
            )
        
        plt.xlabel(f'PC1 (解释方差: {pca.explained_variance_ratio_[0]:.2%})')
        plt.ylabel(f'PC2 (解释方差: {pca.explained_variance_ratio_[1]:.2%})')
        plt.title('诈骗文本语义空间可视化')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        # 保存图片
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"语义空间可视化已保存: {save_path}")
        
        return {
            'pca_components': embeddings_2d.tolist(),
            'explained_variance_ratio': pca.explained_variance_ratio_.tolist(),
            'labels': labels
        }
    
    # ==================== 优化的关键词提取功能 ====================
    
    def _preprocess_text_for_keywords(self, text: str) -> List[str]:
        """预处理文本用于关键词提取"""
        # 使用jieba分词
        words = jieba.lcut(text)
        # 过滤停用词和短词
        filtered_words = [
            word for word in words 
            if len(word) > 1 and word not in self.stop_words
        ]
        return filtered_words
    
    def _get_candidate_words(self, texts: List[str], max_candidates: int = None) -> List[str]:
        """获取候选关键词，改进版本"""
        if max_candidates is None:
            max_candidates = self.config.max_candidate_words
        
        # 统计词频
        word_counts = Counter()
        total_texts = len(texts)
        
        for text in texts:
            words = self._preprocess_text_for_keywords(text)
            word_counts.update(words)
        
        # 定义重要诈骗关键词（不会被频率过滤）
        important_fraud_keywords = {
            '免费', '赚钱', '投资', '理财', '博彩', '刷单', '兼职', '充值', '提现',
            '按摩', '美女', '约会', '客服', '验证', '解冻', '退款', '中奖', '平台',
            '网站', 'app', '软件', '服务', '联系', '咨询', '体验', '尝试', '操作'
        }
        
        # 智能过滤候选词
        filtered_words = []
        for word, count in word_counts.most_common():
            frequency = count / total_texts
            
            # 保护重要诈骗关键词
            if word in important_fraud_keywords:
                filtered_words.append(word)
                continue
                
            # 过滤过于高频（>95%）和过于低频（<2%）的词
            if 0.02 <= frequency <= 0.95:
                filtered_words.append(word)
            
            if len(filtered_words) >= max_candidates:
                break
        
        return filtered_words
    
    def _get_balanced_text_groups(self, texts: List[str], word: str) -> Tuple[List[str], List[str]]:
        """获取平衡的文本分组"""
        import random
        
        texts_with_word = [text for text in texts if word in text]
        texts_without_word = [text for text in texts if word not in text]
        
        # 如果分组不平衡，进行采样平衡
        if len(texts_with_word) > 0 and len(texts_without_word) > 0:
            min_size = min(len(texts_with_word), len(texts_without_word))
            
            # 确保每组至少有2个文本
            if min_size < 2:
                return [], []
            
            # 如果差异过大，进行采样
            if abs(len(texts_with_word) - len(texts_without_word)) > min_size * 3:
                if len(texts_with_word) > min_size * 3:
                    texts_with_word = random.sample(texts_with_word, min_size * 3)
                if len(texts_without_word) > min_size * 3:
                    texts_without_word = random.sample(texts_without_word, min_size * 3)
        
        return texts_with_word, texts_without_word
    
    def _safe_semantic_importance_calculation(self, features_with, features_without):
        """安全的语义重要性计算"""
        try:
            # 验证输入
            if features_with is None or features_without is None:
                return 0.0
            
            if len(features_with) == 0 or len(features_without) == 0:
                return 0.0
            
            # 清理异常值
            features_with = np.nan_to_num(features_with, nan=0.0, posinf=1.0, neginf=-1.0)
            features_without = np.nan_to_num(features_without, nan=0.0, posinf=1.0, neginf=-1.0)
            
            # 计算均值
            mean_with = np.mean(features_with, axis=0)
            mean_without = np.mean(features_without, axis=0)
            
            # 验证均值
            if np.any(np.isnan(mean_with)) or np.any(np.isinf(mean_with)):
                return 0.0
            if np.any(np.isnan(mean_without)) or np.any(np.isinf(mean_without)):
                return 0.0
            
            # 计算相似度
            similarity = cosine_similarity([mean_with], [mean_without])[0][0]
            
            # 验证相似度
            if np.isnan(similarity) or np.isinf(similarity):
                return 0.0
            
            return 1 - similarity
            
        except Exception as e:
            print(f"语义重要性计算失败: {e}")
            return 0.0
    
    def _get_word_embedding(self, word: str) -> np.ndarray:
        """获取单词的嵌入向量（带缓存）"""
        if self.config.enable_caching and word in self.word_embeddings_cache:
            return self.word_embeddings_cache[word]
        
        # 使用模型编码单词
        if hasattr(self.model, 'encode'):
            # SentenceTransformers
            embedding = self.model.encode([word], convert_to_numpy=True)[0]
        else:
            # HuggingFace Transformers
            inputs = self.tokenizer(
                [word], 
                return_tensors='pt', 
                padding=True, 
                truncation=True,
                max_length=128
            )
            inputs = {k: v.to(self.config.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = self.model(**inputs)
                embedding = outputs.last_hidden_state[:, 0, :].cpu().numpy()[0]
        
        # 缓存结果
        if self.config.enable_caching:
            if len(self.word_embeddings_cache) < self.config.cache_size:
                self.word_embeddings_cache[word] = embedding
        
        return embedding
    
    def extract_semantic_keywords(self, texts: List[str], top_k: int = 20) -> List[Tuple[str, float]]:
        """
        提取语义相关的关键词 - 优化版本
        
        算法改进：
        1. 使用注意力机制计算词的重要性
        2. 多维度评分：语义重要性 + 词频 + 位置权重
        3. 批量处理和缓存优化
        4. 自适应候选词数量
        """
        if not texts or len(texts) < 2:
            return []
        
        # 检查缓存
        cache_key = f"semantic_{hash(str(texts))}_{top_k}"
        if self.config.enable_caching and cache_key in self.keyword_cache:
            return self.keyword_cache[cache_key]
        
        print(f"开始语义关键词分析，文本数量: {len(texts)}")
        
        # 获取候选词
        candidate_words = self._get_candidate_words(texts)
        if not candidate_words:
            return []
        
        print(f"候选词数量: {len(candidate_words)}")
        
        # 计算每个词的语义重要性
        word_importance = []
        
        # 批量处理候选词
        for i in range(0, len(candidate_words), self.config.keyword_batch_size):
            batch_words = candidate_words[i:i + self.config.keyword_batch_size]
            batch_importance = []
            
            for word in batch_words:
                try:
                    # 使用改进的文本分组策略
                    texts_with_word, texts_without_word = self._get_balanced_text_groups(texts, word)
                    
                    if len(texts_with_word) == 0 or len(texts_without_word) == 0:
                        continue
                    
                    # 提取特征
                    try:
                        features_with = self.encode_texts(texts_with_word)
                        features_without = self.encode_texts(texts_without_word)
                    except Exception as e:
                        print(f"编码文本时出错: {e}")
                        continue
                    
                    if len(features_with) == 0 or len(features_without) == 0:
                        continue
                    
                    # 使用安全的语义重要性计算
                    semantic_importance = self._safe_semantic_importance_calculation(features_with, features_without)
                    
                    # 计算词频权重
                    word_freq = sum(1 for text in texts if word in text) / len(texts)
                    
                    # 计算位置权重（词在文本中的平均位置）
                    position_weights = []
                    for text in texts:
                        if word in text:
                            words_in_text = self._preprocess_text_for_keywords(text)
                            if word in words_in_text:
                                position = words_in_text.index(word) / len(words_in_text)
                                position_weights.append(position)
                    
                    position_weight = np.mean(position_weights) if position_weights else 0.5
                    
                    # 确保所有值都是有效数值
                    if np.isnan(word_freq) or np.isinf(word_freq):
                        word_freq = 0.0
                    if np.isnan(position_weight) or np.isinf(position_weight):
                        position_weight = 0.5
                    
                    # 综合评分：语义重要性 + 词频 + 位置权重
                    final_score = (
                        semantic_importance * 0.6 +  # 语义重要性权重最高
                        word_freq * 0.3 +           # 词频权重
                        (1 - position_weight) * 0.1  # 位置权重（越靠前越重要）
                    )
                    
                    # 确保最终评分是有效数值
                    if np.isnan(final_score) or np.isinf(final_score):
                        final_score = 0.0
                    
                    batch_importance.append((word, final_score))
                    
                except Exception as e:
                    print(f"处理词 '{word}' 时出错: {e}")
                    continue
            
            word_importance.extend(batch_importance)
            print(f"已处理 {min(i + self.config.keyword_batch_size, len(candidate_words))}/{len(candidate_words)} 个候选词")
        
        # 按重要性排序
        word_importance.sort(key=lambda x: x[1], reverse=True)
        
        # 过滤低重要性词汇
        filtered_importance = [
            (word, score) for word, score in word_importance 
            if score > self.config.semantic_threshold
        ]
        
        result = filtered_importance[:top_k] if top_k > 0 else filtered_importance
        
        # 缓存结果
        if self.config.enable_caching:
            self.keyword_cache[cache_key] = result
        
        print(f"语义关键词提取完成，返回 {len(result)} 个关键词")
        return result
    
    def extract_contextual_keywords(self, texts: List[str], top_k: int = 20) -> List[Tuple[str, float]]:
        """
        提取上下文相关的关键词 - 优化版本
        
        算法改进：
        1. 使用聚类预筛选减少计算量
        2. 引入图论方法分析复杂语义关系
        3. 自适应阈值调整
        4. 批量处理和内存优化
        """
        if not texts or len(texts) < 2:
            return []
        
        # 检查缓存
        cache_key = f"contextual_{hash(str(texts))}_{top_k}"
        if self.config.enable_caching and cache_key in self.keyword_cache:
            return self.keyword_cache[cache_key]
        
        print(f"开始上下文关键词分析，文本数量: {len(texts)}")
        
        # 提取BERT特征
        features = self.encode_texts(texts)
        if len(features) == 0:
            return []
        
        # 计算相似度矩阵
        similarity_matrix = cosine_similarity(features)
        
        # 使用聚类预筛选高相似度文本对
        high_similarity_pairs = self._find_high_similarity_pairs_clustered(
            texts, similarity_matrix
        )
        
        print(f"找到高相似度文本对: {len(high_similarity_pairs)}")
        
        if len(high_similarity_pairs) == 0:
            return []
        
        # 分析高相似度文本对中的共同关键词
        common_keywords = self._analyze_common_keywords_graph(
            texts, high_similarity_pairs
        )
        
        # 转换为列表并排序
        keyword_importance = [(word, score) for word, score in common_keywords.items()]
        keyword_importance.sort(key=lambda x: x[1], reverse=True)
        
        result = keyword_importance[:top_k] if top_k > 0 else keyword_importance
        
        # 缓存结果
        if self.config.enable_caching:
            self.keyword_cache[cache_key] = result
        
        print(f"上下文关键词提取完成，返回 {len(result)} 个关键词")
        return result
    
    def _find_high_similarity_pairs_clustered(self, texts: List[str], similarity_matrix: np.ndarray) -> List[Tuple[int, int, float]]:
        """使用聚类方法找到高相似度文本对"""
        high_similarity_pairs = []
        
        # 自适应阈值
        similarities = similarity_matrix[np.triu_indices_from(similarity_matrix, k=1)]
        threshold = max(
            self.config.contextual_threshold,
            np.percentile(similarities, 80)  # 使用80分位数作为阈值
        )
        
        print(f"使用相似度阈值: {threshold:.3f}")
        
        # 限制搜索的对数
        max_pairs = min(self.config.max_similarity_pairs, len(texts) * (len(texts) - 1) // 2)
        pair_count = 0
        
        for i in range(len(texts)):
            for j in range(i + 1, len(texts)):
                if similarity_matrix[i][j] > threshold:
                    high_similarity_pairs.append((i, j, similarity_matrix[i][j]))
                    pair_count += 1
                    
                    if pair_count >= max_pairs:
                        break
            if pair_count >= max_pairs:
                break
        
        return high_similarity_pairs
    
    def _analyze_common_keywords_graph(self, texts: List[str], high_similarity_pairs: List[Tuple[int, int, float]]) -> Dict[str, float]:
        """使用图论方法分析共同关键词"""
        # 构建文本词汇集合
        text_words = []
        for text in texts:
            words = set(self._preprocess_text_for_keywords(text))
            text_words.append(words)
        
        # 构建词汇-文本关系图
        word_text_graph = defaultdict(list)
        for text_idx, words in enumerate(text_words):
            for word in words:
                word_text_graph[word].append(text_idx)
        
        # 分析共同关键词
        common_keywords = {}
        
        for i, j, similarity in high_similarity_pairs:
            # 找出共同词汇
            common_words = text_words[i].intersection(text_words[j])
            
            # 计算关键词重要性（基于相似度和图论权重）
            for word in common_words:
                if word not in common_keywords:
                    common_keywords[word] = 0
                
                # 基础相似度权重
                base_weight = similarity
                
                # 确保base_weight是有效数值
                if np.isnan(base_weight) or np.isinf(base_weight):
                    base_weight = 0.0
                
                # 图论权重：词汇在文本集合中的分布
                text_connections = len(word_text_graph[word])
                graph_weight = min(1.0, text_connections / len(texts))
                
                # 确保graph_weight是有效数值
                if np.isnan(graph_weight) or np.isinf(graph_weight):
                    graph_weight = 0.0
                
                # 综合权重
                final_weight = base_weight * (1 + graph_weight * 0.5)
                
                # 确保final_weight是有效数值
                if np.isnan(final_weight) or np.isinf(final_weight):
                    final_weight = 0.0
                
                common_keywords[word] += final_weight
        
        return common_keywords
    
    def extract_advanced_keywords(self, texts: List[str], top_k: int = 20) -> Dict[str, List[Tuple[str, float]]]:
        """
        提取高级关键词（结合语义和上下文）
        
        返回：
        - semantic_keywords: 语义关键词
        - contextual_keywords: 上下文关键词
        - combined_keywords: 综合关键词
        """
        print("开始高级关键词提取...")
        
        # 提取语义关键词
        semantic_keywords = self.extract_semantic_keywords(texts, top_k)
        
        # 提取上下文关键词
        contextual_keywords = self.extract_contextual_keywords(texts, top_k)
        
        # 综合关键词（加权合并）
        combined_keywords = self._combine_keywords(semantic_keywords, contextual_keywords, top_k)
        
        return {
            'semantic_keywords': semantic_keywords,
            'contextual_keywords': contextual_keywords,
            'combined_keywords': combined_keywords
        }
    
    def _combine_keywords(self, semantic_keywords: List[Tuple[str, float]], 
                         contextual_keywords: List[Tuple[str, float]], 
                         top_k: int) -> List[Tuple[str, float]]:
        """合并语义和上下文关键词"""
        # 创建词汇到分数的映射
        word_scores = {}
        
        # 添加语义关键词分数
        for word, score in semantic_keywords:
            # 确保score是有效数值
            if np.isnan(score) or np.isinf(score):
                score = 0.0
            word_scores[word] = score * 0.6  # 语义关键词权重
        
        # 添加上下文关键词分数
        for word, score in contextual_keywords:
            # 确保score是有效数值
            if np.isnan(score) or np.isinf(score):
                score = 0.0
            if word in word_scores:
                word_scores[word] += score * 0.4  # 上下文关键词权重
            else:
                word_scores[word] = score * 0.4
        
        # 排序并返回
        combined = [(word, score) for word, score in word_scores.items()]
        combined.sort(key=lambda x: x[1], reverse=True)
        
        return combined[:top_k] if top_k > 0 else combined


if __name__ == "__main__":
    # 测试示例
    print("语义分析器测试")
    
    # 创建测试数据
    test_data = [
        {"cleaned_text": "你好，我们这里有20岁美女按摩服务，200元2小时，包你满意", "label": "按摩色诱"},
        {"cleaned_text": "欢迎来到正规博彩平台，稳赚不赔，充值送50%", "label": "博彩"},
        {"cleaned_text": "兼职刷单，50元一单，简单操作，需要垫付", "label": "兼职刷单"},
        {"cleaned_text": "理财产品，年化20%，保本保息，专业团队", "label": "投资理财"},
        {"cleaned_text": "你好，我是银行客服，您的账户异常，需要验证", "label": "虚假客服"},
        {"cleaned_text": "同城约会，18岁妹妹，300元3小时，保证服务质量", "label": "按摩色诱"},
        {"cleaned_text": "信誉平台，日赚1000，包中10倍，快速提现", "label": "博彩"},
    ]
    
    df = pd.DataFrame(test_data)
    
    # 测试语义分析器
    config = SemanticConfig(n_clusters=3)
    analyzer = SemanticAnalyzer(config)
    
    # 测试相似度计算
    query = "美女按摩服务，价格优惠"
    candidates = ["按摩spa", "投资理财", "博彩平台", "同城约会"]
    similar = analyzer.find_similar_texts(query, candidates, top_k=2)
    print(f"相似文本: {similar}")
    
    # 测试聚类
    cluster_results = analyzer.cluster_texts(df['cleaned_text'].tolist(), df['label'].tolist())
    print(f"聚类结果: {cluster_results['n_clusters']} 个聚类")
    
    # 测试语义模式分析
    semantic_results = analyzer.analyze_semantic_patterns(df)
    print(f"语义分析完成，分析了 {len(semantic_results)} 个标签")
    
    # 生成报告
    report = analyzer.generate_semantic_report(semantic_results)
    print("语义分析报告:")
    print(report[:500] + "...")
