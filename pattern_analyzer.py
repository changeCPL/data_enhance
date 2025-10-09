"""
话术结构分析模块
负责识别和分析不同标签的套路模式、话术结构
集成深度学习模型进行语义模式分析
"""

import re
import pandas as pd
from collections import Counter, defaultdict
from typing import List, Dict, Tuple, Set
import numpy as np
from dataclasses import dataclass
import torch
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
import warnings
warnings.filterwarnings('ignore')


@dataclass
class PatternTemplate:
    """话术模式模板"""
    name: str
    pattern: str
    description: str
    examples: List[str]
    frequency: int = 0


class PatternAnalyzer:
    def __init__(self, use_bert: bool = True, bert_model: str = "bert-base-chinese"):
        # 深度学习模型配置
        self.use_bert = use_bert
        self.bert_model_name = bert_model
        self.bert_tokenizer = None
        self.bert_model = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # 初始化BERT模型
        if self.use_bert:
            self._initialize_bert_model()
        # 定义不同标签的话术模式模板
        self.pattern_templates = {
            '按摩色诱': [
                PatternTemplate(
                    name="价格诱惑",
                    pattern=r"[0-9]+元.*[0-9]+小时|[0-9]+小时.*[0-9]+元",
                    description="通过价格和时间组合吸引用户",
                    examples=["200元2小时", "300元3小时包夜"]
                ),
                PatternTemplate(
                    name="年龄描述",
                    pattern=r"[0-9]+岁.*美女|[0-9]+岁.*妹妹",
                    description="通过年龄描述吸引用户",
                    examples=["20岁美女", "18岁妹妹"]
                ),
                PatternTemplate(
                    name="服务承诺",
                    pattern=r"包.*满意|保证.*服务|100%.*服务",
                    description="承诺服务质量",
                    examples=["包你满意", "保证服务质量"]
                ),
                PatternTemplate(
                    name="联系方式",
                    pattern=r"加.*微信|联系.*电话|QQ.*[0-9]+",
                    description="提供联系方式",
                    examples=["加微信详聊", "联系我电话"]
                )
            ],
            '博彩': [
                PatternTemplate(
                    name="收益承诺",
                    pattern=r"稳赚.*[0-9]+%|日赚.*[0-9]+|包中.*[0-9]+倍",
                    description="承诺高收益",
                    examples=["稳赚50%", "日赚1000", "包中10倍"]
                ),
                PatternTemplate(
                    name="平台推广",
                    pattern=r"正规.*平台|官方.*平台|信誉.*平台",
                    description="强调平台正规性",
                    examples=["正规平台", "官方平台"]
                ),
                PatternTemplate(
                    name="充值优惠",
                    pattern=r"充值.*送.*[0-9]+%|首充.*[0-9]+%",
                    description="充值优惠活动",
                    examples=["充值送50%", "首充100%"]
                ),
                PatternTemplate(
                    name="提现保证",
                    pattern=r"秒到账|快速.*提现|24小时.*到账",
                    description="保证提现速度",
                    examples=["秒到账", "快速提现"]
                )
            ],
            '兼职刷单': [
                PatternTemplate(
                    name="佣金承诺",
                    pattern=r"[0-9]+元.*单|[0-9]+%.*佣金|日赚.*[0-9]+",
                    description="承诺佣金收益",
                    examples=["50元一单", "20%佣金", "日赚500"]
                ),
                PatternTemplate(
                    name="操作简单",
                    pattern=r"简单.*操作|轻松.*赚钱|在家.*兼职",
                    description="强调操作简单",
                    examples=["简单操作", "轻松赚钱"]
                ),
                PatternTemplate(
                    name="垫付要求",
                    pattern=r"先垫付.*[0-9]+元|需要.*本金|准备.*资金",
                    description="要求垫付资金",
                    examples=["先垫付100元", "需要本金"]
                ),
                PatternTemplate(
                    name="时间灵活",
                    pattern=r"时间.*自由|随时.*兼职|空闲.*时间",
                    description="强调时间灵活",
                    examples=["时间自由", "随时兼职"]
                )
            ],
            '投资理财': [
                PatternTemplate(
                    name="高收益",
                    pattern=r"年化.*[0-9]+%|收益.*[0-9]+%|回报.*[0-9]+倍",
                    description="承诺高收益",
                    examples=["年化20%", "收益30%"]
                ),
                PatternTemplate(
                    name="保本承诺",
                    pattern=r"保本.*保息|零风险|稳赚.*不赔",
                    description="承诺保本",
                    examples=["保本保息", "零风险"]
                ),
                PatternTemplate(
                    name="专业团队",
                    pattern=r"专业.*团队|资深.*分析师|金牌.*导师",
                    description="强调专业性",
                    examples=["专业团队", "资深分析师"]
                ),
                PatternTemplate(
                    name="限时优惠",
                    pattern=r"限时.*优惠|仅限.*[0-9]+天|错过.*后悔",
                    description="制造紧迫感",
                    examples=["限时优惠", "仅限3天"]
                )
            ],
            '虚假客服': [
                PatternTemplate(
                    name="身份伪装",
                    pattern=r"银行.*客服|官方.*客服|系统.*通知",
                    description="伪装官方身份",
                    examples=["银行客服", "官方客服"]
                ),
                PatternTemplate(
                    name="紧急情况",
                    pattern=r"账户.*异常|资金.*风险|立即.*处理",
                    description="制造紧急情况",
                    examples=["账户异常", "资金风险"]
                ),
                PatternTemplate(
                    name="验证要求",
                    pattern=r"验证.*身份|提供.*密码|确认.*信息",
                    description="要求验证信息",
                    examples=["验证身份", "提供密码"]
                ),
                PatternTemplate(
                    name="操作指导",
                    pattern=r"按.*操作|点击.*链接|输入.*验证码",
                    description="指导用户操作",
                    examples=["按提示操作", "点击链接"]
                )
            ]
        }
        
        # 通用话术结构模式
        self.common_patterns = [
            PatternTemplate(
                name="问候语",
                pattern=r"你好|您好|亲|亲爱的",
                description="开场问候",
                examples=["你好", "您好", "亲"]
            ),
            PatternTemplate(
                name="数字强调",
                pattern=r"[0-9]+[元%倍小时天]",
                description="使用数字强调",
                examples=["100元", "50%", "3倍"]
            ),
            PatternTemplate(
                name="情感词汇",
                pattern=r"免费|优惠|限时|独家|专属",
                description="情感诱导词汇",
                examples=["免费", "优惠", "限时"]
            ),
            PatternTemplate(
                name="行动号召",
                pattern=r"立即|马上|现在|抓紧|不要错过",
                description="催促行动",
                examples=["立即行动", "马上参与"]
            )
        ]
    
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
    
    def extract_bert_features(self, texts: List[str]) -> np.ndarray:
        """使用BERT提取文本特征"""
        if not self.use_bert or self.bert_model is None:
            return np.array([])
        
        features = []
        batch_size = 16
        
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            
            # 分词
            inputs = self.bert_tokenizer(
                batch_texts,
                truncation=True,
                padding=True,
                max_length=128,
                return_tensors='pt'
            )
            
            # 移动到设备
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # 提取特征
            with torch.no_grad():
                outputs = self.bert_model(**inputs)
                # 使用[CLS]标记的表示作为句子特征
                batch_features = outputs.last_hidden_state[:, 0, :].cpu().numpy()
                features.extend(batch_features)
        
        return np.array(features)
    
    def analyze_semantic_patterns(self, texts: List[str]) -> Dict:
        """分析语义模式"""
        if not self.use_bert or len(texts) == 0:
            return {}
        
        # 提取BERT特征
        features = self.extract_bert_features(texts)
        if len(features) == 0:
            return {}
        
        # 计算相似度矩阵
        similarity_matrix = cosine_similarity(features)
        
        # 聚类分析
        n_clusters = min(5, len(texts))
        if n_clusters > 1:
            kmeans = KMeans(n_clusters=n_clusters, random_state=42)
            cluster_labels = kmeans.fit_predict(features)
        else:
            cluster_labels = [0] * len(texts)
        
        # 分析每个聚类的特征
        cluster_analysis = {}
        for cluster_id in set(cluster_labels):
            cluster_texts = [texts[i] for i in range(len(texts)) if cluster_labels[i] == cluster_id]
            cluster_features = features[cluster_labels == cluster_id]
            
            # 计算聚类内相似度
            if len(cluster_features) > 1:
                cluster_similarity = cosine_similarity(cluster_features)
                avg_similarity = np.mean(cluster_similarity[np.triu_indices_from(cluster_similarity, k=1)])
            else:
                avg_similarity = 1.0
            
            cluster_analysis[f'cluster_{cluster_id}'] = {
                'texts': cluster_texts[:3],  # 保存前3个文本作为示例
                'size': len(cluster_texts),
                'avg_similarity': float(avg_similarity)
            }
        
        return {
            'similarity_matrix': similarity_matrix.tolist(),
            'cluster_labels': cluster_labels.tolist(),
            'cluster_analysis': cluster_analysis,
            'n_clusters': len(set(cluster_labels))
        }
    
    def extract_semantic_entities(self, text: str) -> Dict[str, List[str]]:
        """提取语义实体"""
        if not self.use_bert:
            return {}
        
        # 简单的实体识别（可以扩展为更复杂的NER）
        entities = {
            'numbers': re.findall(r'[0-9]+', text),
            'money': re.findall(r'[0-9]+[元块万]', text),
            'time': re.findall(r'[0-9]+[小时天分钟]', text),
            'percentages': re.findall(r'[0-9]+%', text),
            'contacts': re.findall(r'微信|QQ|电话|联系', text),
            'platforms': re.findall(r'平台|网站|app|软件', text),
            'services': re.findall(r'服务|按摩|spa|投资|理财|兼职|刷单', text)
        }
        
        return entities
    
    def match_patterns(self, text: str, templates: List[PatternTemplate]) -> List[Tuple[PatternTemplate, str]]:
        """
        匹配文本中的模式
        """
        matches = []
        
        for template in templates:
            pattern_matches = re.findall(template.pattern, text)
            for match in pattern_matches:
                matches.append((template, match))
        
        return matches
    
    def analyze_text_structure(self, text: str) -> Dict[str, any]:
        """
        分析文本结构特征
        """
        features = {
            'length': len(text),
            'sentence_count': len(re.split(r'[。！？]', text)),
            'has_numbers': bool(re.search(r'[0-9]+', text)),
            'has_money': bool(re.search(r'[0-9]+[元块万]', text)),
            'has_percentage': bool(re.search(r'[0-9]+%', text)),
            'has_contact': bool(re.search(r'微信|QQ|电话|联系', text)),
            'has_urgency': bool(re.search(r'立即|马上|现在|抓紧', text)),
            'has_promise': bool(re.search(r'保证|承诺|稳赚|包中', text)),
            'has_emotion': bool(re.search(r'免费|优惠|限时|独家', text)),
        }
        
        return features
    
    def extract_conversation_flow(self, text: str) -> List[str]:
        """
        提取对话流程
        """
        # 按标点符号分割句子
        sentences = re.split(r'[。！？；]', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        # 分析每个句子的功能
        flow = []
        for sentence in sentences:
            if re.search(r'你好|您好|亲', sentence):
                flow.append('问候')
            elif re.search(r'介绍|推荐|了解', sentence):
                flow.append('介绍')
            elif re.search(r'价格|费用|收费', sentence):
                flow.append('报价')
            elif re.search(r'联系|微信|电话', sentence):
                flow.append('联系方式')
            elif re.search(r'立即|马上|现在', sentence):
                flow.append('催促')
            else:
                flow.append('其他')
        
        return flow
    
    def analyze_label_patterns(self, df: pd.DataFrame, label: str) -> Dict[str, any]:
        """
        分析特定标签的模式（集成语义分析）
        """
        label_data = df[df['label'] == label]
        if len(label_data) == 0:
            return {}
        
        texts = label_data['cleaned_text'].tolist()
        
        # 获取该标签的模板
        templates = self.pattern_templates.get(label, []) + self.common_patterns
        
        # 统计模式匹配
        pattern_stats = defaultdict(int)
        pattern_examples = defaultdict(list)
        structure_features = []
        conversation_flows = []
        semantic_entities = []
        
        for _, row in label_data.iterrows():
            text = row['cleaned_text']
            
            # 匹配模式
            matches = self.match_patterns(text, templates)
            for template, match in matches:
                pattern_stats[template.name] += 1
                if len(pattern_examples[template.name]) < 5:  # 最多保存5个例子
                    pattern_examples[template.name].append(match)
            
            # 分析结构特征
            features = self.analyze_text_structure(text)
            structure_features.append(features)
            
            # 分析对话流程
            flow = self.extract_conversation_flow(text)
            conversation_flows.append(flow)
            
            # 提取语义实体
            entities = self.extract_semantic_entities(text)
            semantic_entities.append(entities)
        
        # 计算结构特征统计
        structure_stats = {}
        if structure_features:
            for key in structure_features[0].keys():
                if isinstance(structure_features[0][key], bool):
                    structure_stats[key] = sum(f[key] for f in structure_features) / len(structure_features)
                else:
                    values = [f[key] for f in structure_features]
                    structure_stats[key] = {
                        'mean': np.mean(values),
                        'std': np.std(values),
                        'min': np.min(values),
                        'max': np.max(values)
                    }
        
        # 分析对话流程模式
        flow_patterns = defaultdict(int)
        for flow in conversation_flows:
            flow_key = ' -> '.join(flow[:5])  # 只取前5个步骤
            flow_patterns[flow_key] += 1
        
        # 语义模式分析
        semantic_patterns = {}
        if self.use_bert and len(texts) > 1:
            try:
                semantic_patterns = self.analyze_semantic_patterns(texts)
            except Exception as e:
                print(f"语义模式分析失败: {e}")
        
        # 语义实体统计
        entity_stats = {}
        if semantic_entities:
            all_entities = defaultdict(list)
            for entities in semantic_entities:
                for entity_type, entity_list in entities.items():
                    all_entities[entity_type].extend(entity_list)
            
            for entity_type, entity_list in all_entities.items():
                entity_stats[entity_type] = dict(Counter(entity_list).most_common(10))
        
        return {
            'pattern_frequency': dict(pattern_stats),
            'pattern_examples': dict(pattern_examples),
            'structure_stats': structure_stats,
            'conversation_flows': dict(flow_patterns),
            'semantic_patterns': semantic_patterns,
            'entity_stats': entity_stats,
            'sample_count': len(label_data)
        }
    
    def generate_pattern_report(self, pattern_results: Dict[str, Dict]) -> str:
        """
        生成模式分析报告（包含语义分析）
        """
        report = "话术结构分析报告（集成深度学习）\n"
        report += "=" * 60 + "\n\n"
        
        for label, data in pattern_results.items():
            report += f"标签: {label}\n"
            report += f"样本数量: {data['sample_count']}\n"
            report += "-" * 40 + "\n"
            
            # 模式频率
            report += "话术模式频率 (Top 10):\n"
            sorted_patterns = sorted(data['pattern_frequency'].items(), 
                                   key=lambda x: x[1], reverse=True)
            for pattern, freq in sorted_patterns[:10]:
                report += f"  {pattern}: {freq}次\n"
            
            # 模式例子
            report += "\n模式例子:\n"
            for pattern, examples in data['pattern_examples'].items():
                if examples:
                    report += f"  {pattern}: {', '.join(examples[:3])}\n"
            
            # 结构特征
            report += "\n文本结构特征:\n"
            for feature, stats in data['structure_stats'].items():
                if isinstance(stats, dict):
                    report += f"  {feature}: 平均{stats['mean']:.2f}, 范围{stats['min']}-{stats['max']}\n"
                else:
                    report += f"  {feature}: {stats:.2%}\n"
            
            # 对话流程
            report += "\n常见对话流程 (Top 5):\n"
            sorted_flows = sorted(data['conversation_flows'].items(), 
                                key=lambda x: x[1], reverse=True)
            for flow, count in sorted_flows[:5]:
                report += f"  {flow}: {count}次\n"
            
            # 语义模式分析
            if data.get('semantic_patterns'):
                semantic = data['semantic_patterns']
                report += f"\n语义模式分析:\n"
                report += f"  聚类数量: {semantic.get('n_clusters', 0)}\n"
                
                if semantic.get('cluster_analysis'):
                    report += "  聚类详情:\n"
                    for cluster_id, cluster_info in semantic['cluster_analysis'].items():
                        report += f"    {cluster_id}: {cluster_info['size']}个样本, 平均相似度{cluster_info['avg_similarity']:.3f}\n"
                        for i, text in enumerate(cluster_info['texts'][:2], 1):
                            report += f"      示例{i}: {text[:30]}...\n"
            
            # 语义实体统计
            if data.get('entity_stats'):
                report += "\n语义实体统计:\n"
                for entity_type, entities in data['entity_stats'].items():
                    if entities:
                        top_entities = list(entities.items())[:3]
                        report += f"  {entity_type}: {', '.join([f'{k}({v})' for k, v in top_entities])}\n"
            
            report += "\n" + "=" * 60 + "\n\n"
        
        return report
    
    def find_similar_patterns(self, text: str, threshold: float = 0.7) -> List[Tuple[str, float]]:
        """
        找到与输入文本相似的模式
        """
        similarities = []
        
        for label, templates in self.pattern_templates.items():
            for template in templates:
                # 简单的相似度计算（可以改进为更复杂的算法）
                if re.search(template.pattern, text):
                    similarities.append((f"{label}_{template.name}", 1.0))
        
        return sorted(similarities, key=lambda x: x[1], reverse=True)


if __name__ == "__main__":
    # 测试示例
    analyzer = PatternAnalyzer()
    
    # 测试文本
    test_text = "你好，我是同城交友平台客服，我们这里有20岁美女按摩服务，200元2小时，包你满意，加微信详聊"
    
    # 分析文本结构
    features = analyzer.analyze_text_structure(test_text)
    print(f"文本结构特征: {features}")
    
    # 提取对话流程
    flow = analyzer.extract_conversation_flow(test_text)
    print(f"对话流程: {flow}")
    
    # 匹配模式
    templates = analyzer.pattern_templates['按摩色诱'] + analyzer.common_patterns
    matches = analyzer.match_patterns(test_text, templates)
    print(f"匹配的模式: {[(t.name, match) for t, match in matches]}")
