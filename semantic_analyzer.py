"""
语义相似度分析模块
基于深度学习的语义理解和相似度计算
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional, Union
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans, DBSCAN
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from collections import defaultdict, Counter
from dataclasses import dataclass
from tqdm import tqdm
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


class SemanticAnalyzer:
    """语义相似度分析器"""
    
    def __init__(self, config: SemanticConfig = None):
        self.config = config or SemanticConfig()
        self.model = None
        self.tokenizer = None
        self.embeddings_cache = {}
        
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
                
                # 合并结果
                all_embeddings = [None] * len(texts)
                for i, emb in enumerate(cached_embeddings):
                    all_embeddings[i] = emb
                for i, emb in zip(uncached_indices, new_embeddings):
                    all_embeddings[i] = emb
                
                return np.array(all_embeddings)
            else:
                return np.array(cached_embeddings)
        else:
            return self._encode_batch(texts)
    
    def _encode_batch(self, texts: List[str]) -> np.ndarray:
        """批量编码文本"""
        if hasattr(self.model, 'encode'):
            # SentenceTransformers
            embeddings = self.model.encode(
                texts, 
                batch_size=self.config.batch_size,
                show_progress_bar=True,
                convert_to_numpy=True
            )
            return embeddings
        else:
            # HuggingFace Transformers
            embeddings = []
            
            for i in tqdm(range(0, len(texts), self.config.batch_size), desc="编码文本"):
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
                    embeddings.extend(batch_embeddings)
            
            return np.array(embeddings)
    
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
        
        for label, data in semantic_results.items():
            report += f"标签: {label}\n"
            report += f"样本数量: {data['sample_count']}\n"
            report += "-" * 30 + "\n"
            
            # 语义统计
            stats = data['semantic_stats']
            report += "语义相似度统计:\n"
            report += f"  平均相似度: {stats['mean_similarity']:.4f}\n"
            report += f"  相似度标准差: {stats['std_similarity']:.4f}\n"
            report += f"  高相似度比例 (>0.8): {stats['high_similarity_ratio']:.2%}\n"
            report += f"  低相似度比例 (<0.3): {stats['low_similarity_ratio']:.2%}\n"
            
            # 聚类信息
            clustering = data['clustering']
            report += f"\n聚类信息:\n"
            report += f"  聚类数量: {clustering['n_clusters']}\n"
            
            # 代表性文本
            representatives = data['representative_texts']
            report += f"\n代表性文本:\n"
            for i, rep in enumerate(representatives, 1):
                report += f"  {i}. {rep['text'][:50]}... (相似度: {rep['avg_similarity']:.4f})\n"
            
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
