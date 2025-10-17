"""
数据增强策略模块
基于提取的模式和关键词生成新的训练样本
集成深度学习模型进行智能文本生成
"""

import random
import re
import pandas as pd
from typing import List, Dict, Tuple, Any
from dataclasses import dataclass
from collections import defaultdict
import numpy as np
import warnings
warnings.filterwarnings('ignore')
from enum import Enum
from typing import Optional

from data_preprocessor import DataPreprocessor
from pattern_analyzer import PatternAnalyzer

# 导入深度学习模块
try:
    from dl_text_generator import DLTextGenerator, GenerationConfig, PromptEngine
    DL_AVAILABLE = True
except ImportError:
    DL_AVAILABLE = False
    print("警告: 深度学习文本生成模块未找到，将使用传统方法")


class GenerationMethod(Enum):
    """数据生成方法枚举"""
    # 模板生成方法
    TEMPLATE_RULE = "template_rule"                      # 规则模板生成
    
    # 规则增强方法
    RULE_SYNONYM = "rule_synonym"                        # 同义词替换
    RULE_NUMBER_CHANGE = "rule_number_change"            # 数字变化
    RULE_WORD_ORDER = "rule_word_order"                  # 语序调整
    
    # 语义生成方法（深度学习）
    SEMANTIC_TFIDF_PROMPT = "semantic_tfidf_prompt"      # 基于TF-IDF关键词的prompt
    SEMANTIC_SEMANTIC_PROMPT = "semantic_semantic_prompt" # 基于语义关键词的prompt
    SEMANTIC_CONTEXTUAL_PROMPT = "semantic_contextual_prompt" # 基于上下文关键词的prompt
    SEMANTIC_COMBINED_PROMPT = "semantic_combined_prompt" # 基于综合关键词的prompt
    SEMANTIC_LABEL_SPECIFIC_PROMPT = "semantic_label_specific_prompt" # 基于标签特定关键词的prompt
    SEMANTIC_PATTERN_PROMPT = "semantic_pattern_prompt"  # 基于话术模式的prompt
    SEMANTIC_ENTITY_PROMPT = "semantic_entity_prompt"    # 基于语义实体的智能prompt
    
    # 混合策略生成
    HYBRID_MULTI_FEATURE = "hybrid_multi_feature"        # 多特征混合
    
    # 回退方法
    FALLBACK_TRADITIONAL = "fallback_traditional"        # 传统方法回退
    FALLBACK_DL_ERROR = "fallback_dl_error"              # 深度学习错误回退


@dataclass
class GenerationMetadata:
    """数据生成元数据"""
    method: GenerationMethod
    method_description: str
    prompt_type: Optional[str] = None
    prompt_content: Optional[str] = None
    original_text: Optional[str] = None
    feature_weights: Optional[Dict[str, float]] = None
    generation_params: Optional[Dict[str, Any]] = None
    timestamp: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return {
            'generation_method': self.method.value,
            'method_description': self.method_description,
            'prompt_type': self.prompt_type,
            'prompt_content': self.prompt_content,
            'original_text': self.original_text,
            'feature_weights': self.feature_weights,
            'generation_params': self.generation_params,
            'timestamp': self.timestamp
        }


class GenerationTracker:
    """数据生成方式跟踪器"""
    
    def __init__(self):
        self.generation_stats = defaultdict(int)
        self.method_descriptions = {
            # 模板生成方法
            GenerationMethod.TEMPLATE_RULE: "规则模板生成 - 基于特定规则的模板生成",
            
            # 规则增强方法
            GenerationMethod.RULE_SYNONYM: "同义词替换 - 替换原文本中的同义词",
            GenerationMethod.RULE_NUMBER_CHANGE: "数字变化 - 修改原文本中的数字",
            GenerationMethod.RULE_WORD_ORDER: "语序调整 - 重新排列句子顺序",
            
            # 语义生成方法
            GenerationMethod.SEMANTIC_TFIDF_PROMPT: "基于TF-IDF关键词的prompt生成",
            GenerationMethod.SEMANTIC_SEMANTIC_PROMPT: "基于语义关键词的prompt生成",
            GenerationMethod.SEMANTIC_CONTEXTUAL_PROMPT: "基于上下文关键词的prompt生成",
            GenerationMethod.SEMANTIC_COMBINED_PROMPT: "基于综合关键词的prompt生成",
            GenerationMethod.SEMANTIC_LABEL_SPECIFIC_PROMPT: "基于标签特定关键词的prompt生成",
            GenerationMethod.SEMANTIC_PATTERN_PROMPT: "基于话术模式的prompt生成",
            GenerationMethod.SEMANTIC_ENTITY_PROMPT: "基于语义实体的智能prompt生成",
            
            # 混合策略生成
            GenerationMethod.HYBRID_MULTI_FEATURE: "多特征混合生成",
            
            # 回退方法
            GenerationMethod.FALLBACK_TRADITIONAL: "传统方法回退",
            GenerationMethod.FALLBACK_DL_ERROR: "深度学习错误回退"
        }
    
    def create_metadata(self, method: GenerationMethod, 
                       prompt_type: str = None,
                       prompt_content: str = None,
                       original_text: str = None,
                       feature_weights: Dict[str, float] = None,
                       generation_params: Dict[str, Any] = None) -> GenerationMetadata:
        """创建生成元数据"""
        from datetime import datetime
        
        return GenerationMetadata(
            method=method,
            method_description=self.method_descriptions.get(method, "未知方法"),
            prompt_type=prompt_type,
            prompt_content=prompt_content,
            original_text=original_text,
            feature_weights=feature_weights,
            generation_params=generation_params,
            timestamp=datetime.now().isoformat()
        )
    
    def track_generation(self, method: GenerationMethod):
        """跟踪生成统计"""
        self.generation_stats[method] += 1
    
    def get_generation_stats(self) -> Dict[str, int]:
        """获取生成统计"""
        return {method.value: count for method, count in self.generation_stats.items()}
    
    def get_generation_report(self) -> str:
        """生成跟踪报告"""
        report = "数据生成方式跟踪报告\n"
        report += "=" * 50 + "\n\n"
        
        total_generations = sum(self.generation_stats.values())
        if total_generations == 0:
            report += "暂无生成数据\n"
            return report
        
        report += f"总生成数量: {total_generations}\n\n"
        
        # 按类别分组统计
        categories = {
            "模板生成": [GenerationMethod.TEMPLATE_RULE],
            "规则增强": [GenerationMethod.RULE_SYNONYM, GenerationMethod.RULE_NUMBER_CHANGE, GenerationMethod.RULE_WORD_ORDER],
            "语义生成": [GenerationMethod.SEMANTIC_TFIDF_PROMPT, GenerationMethod.SEMANTIC_SEMANTIC_PROMPT, 
                      GenerationMethod.SEMANTIC_CONTEXTUAL_PROMPT, GenerationMethod.SEMANTIC_COMBINED_PROMPT,
                      GenerationMethod.SEMANTIC_LABEL_SPECIFIC_PROMPT, GenerationMethod.SEMANTIC_PATTERN_PROMPT,
                      GenerationMethod.SEMANTIC_ENTITY_PROMPT],
            "混合策略": [GenerationMethod.HYBRID_MULTI_FEATURE]
        }
        
        for category, methods in categories.items():
            category_count = sum(self.generation_stats[method] for method in methods)
            if category_count > 0:
                report += f"{category} ({category_count} 条, {category_count/total_generations*100:.1f}%):\n"
                for method in methods:
                    count = self.generation_stats[method]
                    if count > 0:
                        description = self.method_descriptions.get(method, "未知方法")
                        report += f"  - {method.value}: {count} 条 ({count/total_generations*100:.1f}%)\n"
                        report += f"    描述: {description}\n"
                report += "\n"
        
        return report
    
    def get_fallback_analysis_report(self, augmented_data: pd.DataFrame = None) -> str:
        """生成回退分析报告"""
        report = "回退分析报告\n"
        report += "=" * 50 + "\n\n"
        
        if augmented_data is None or 'generation_params' not in augmented_data.columns:
            report += "无回退数据可分析\n"
            return report
        
        # 分析回退情况
        fallback_data = []
        for _, row in augmented_data.iterrows():
            params = row.get('generation_params', {})
            if isinstance(params, dict) and 'fallback_type' in params:
                fallback_data.append({
                    'method': row.get('generation_method', 'unknown'),
                    'fallback_type': params.get('fallback_type', 'unknown'),
                    'reason': params.get('reason', 'unknown'),
                    'original_intent': params.get('original_intent', 'unknown'),
                    'actual_method': params.get('actual_method', 'unknown')
                })
        
        if not fallback_data:
            report += "未发现回退情况\n"
            return report
        
        report += f"发现 {len(fallback_data)} 条回退数据\n\n"
        
        # 按回退类型分组
        fallback_types = {}
        for item in fallback_data:
            fallback_type = item['fallback_type']
            if fallback_type not in fallback_types:
                fallback_types[fallback_type] = []
            fallback_types[fallback_type].append(item)
        
        for fallback_type, items in fallback_types.items():
            report += f"{fallback_type} ({len(items)} 条):\n"
            
            # 按原因分组
            reasons = {}
            for item in items:
                reason = item['reason']
                if reason not in reasons:
                    reasons[reason] = []
                reasons[reason].append(item)
            
            for reason, reason_items in reasons.items():
                report += f"  - 原因: {reason} ({len(reason_items)} 条)\n"
                
                # 按实际使用的方法分组
                actual_methods = {}
                for item in reason_items:
                    actual_method = item['actual_method']
                    if actual_method not in actual_methods:
                        actual_methods[actual_method] = 0
                    actual_methods[actual_method] += 1
                
                for actual_method, count in actual_methods.items():
                    report += f"    实际使用: {actual_method} ({count} 条)\n"
            
            report += "\n"
        
        return report


class IntelligentFeatureFusion:
    """智能特征融合器 - 充分利用所有提取的特征"""
    
    def __init__(self):
        self.feature_weights = {
            'tfidf_keywords': 0.15,
            'semantic_keywords': 0.25,
            'contextual_keywords': 0.20,
            'combined_keywords': 0.30,
            'label_keywords': 0.10
        }
        
        self.pattern_weights = {
            'pattern_frequency': 0.50,  # 增加话术模式权重
            'entity_based': 0.50        # 合并后的实体权重（原structure_stats + entity_stats）
        }
    
    def create_comprehensive_prompts(self, label: str, keyword_results: Dict, pattern_results: Dict) -> List[Dict]:
        """创建综合prompt策略，充分利用所有特征"""
        if label not in keyword_results or label not in pattern_results:
            return []
        
        keyword_data = keyword_results[label]
        pattern_data = pattern_results[label]
        
        prompts = []
        
        # 1. 基于TF-IDF关键词的prompt（高频词汇）
        if keyword_data.get('tfidf_keywords'):
            tfidf_prompt = self._create_tfidf_prompt(label, keyword_data['tfidf_keywords'])
            prompts.append({
                'type': 'tfidf_based',
                'prompt': tfidf_prompt,
                'weight': self.feature_weights['tfidf_keywords'],
                'description': '基于TF-IDF高频关键词生成',
                'generation_method': GenerationMethod.SEMANTIC_TFIDF_PROMPT
            })
        
        # 2. 基于语义关键词的prompt（语义理解）
        if keyword_data.get('semantic_keywords'):
            semantic_prompt = self._create_semantic_prompt(label, keyword_data['semantic_keywords'])
            prompts.append({
                'type': 'semantic_based',
                'prompt': semantic_prompt,
                'weight': self.feature_weights['semantic_keywords'],
                'description': '基于语义关键词生成',
                'generation_method': GenerationMethod.SEMANTIC_SEMANTIC_PROMPT
            })
        
        # 3. 基于上下文关键词的prompt（上下文相关）
        if keyword_data.get('contextual_keywords'):
            contextual_prompt = self._create_contextual_prompt(label, keyword_data['contextual_keywords'])
            prompts.append({
                'type': 'contextual_based',
                'prompt': contextual_prompt,
                'weight': self.feature_weights['contextual_keywords'],
                'description': '基于上下文关键词生成',
                'generation_method': GenerationMethod.SEMANTIC_CONTEXTUAL_PROMPT
            })
        
        # 4. 基于综合关键词的prompt（最优质量）
        if keyword_data.get('combined_keywords'):
            combined_prompt = self._create_keywords_combined_prompt(label, keyword_data['combined_keywords'])
            prompts.append({
                'type': 'combined_based',
                'prompt': combined_prompt,
                'weight': self.feature_weights['combined_keywords'],
                'description': '基于综合关键词生成',
                'generation_method': GenerationMethod.SEMANTIC_COMBINED_PROMPT
            })
        
        # 5. 基于标签特定关键词的prompt（领域专业）
        if keyword_data.get('label_keywords'):
            label_prompt = self._create_label_specific_prompt(label, keyword_data['label_keywords'])
            prompts.append({
                'type': 'label_specific',
                'prompt': label_prompt,
                'weight': self.feature_weights['label_keywords'],
                'description': '基于标签特定关键词生成',
                'generation_method': GenerationMethod.SEMANTIC_LABEL_SPECIFIC_PROMPT
            })
        
        # 6. 基于话术模式的prompt（真实模式）
        if pattern_data.get('pattern_frequency'):
            pattern_prompt = self._create_pattern_prompt(label, pattern_data['pattern_frequency'])
            prompts.append({
                'type': 'pattern_based',
                'prompt': pattern_prompt,
                'weight': self.pattern_weights['pattern_frequency'],
                'description': '基于话术模式生成',
                'generation_method': GenerationMethod.SEMANTIC_PATTERN_PROMPT
            })
        
        # 7. 基于语义实体的智能prompt（合并原structure和entity功能）
        if pattern_data.get('entity_stats'):
            structure_stats = pattern_data.get('structure_stats', {})
            entity_prompt = self._create_entity_based_prompt(label, pattern_data['entity_stats'], structure_stats)
            prompts.append({
                'type': 'entity_based',
                'prompt': entity_prompt,
                'weight': self.pattern_weights['entity_based'],  # 使用合并后的权重
                'description': '基于语义实体的智能生成（使用structure_stats进行频率判断）',
                'generation_method': GenerationMethod.SEMANTIC_ENTITY_PROMPT
            })
        
        return prompts
    
    def _create_tfidf_prompt(self, label: str, tfidf_keywords: List[Tuple[str, float]]) -> str:
        """创建基于TF-IDF关键词的prompt"""
        top_keywords = [kw[0] for kw in tfidf_keywords[:5]]
        return f"生成一段{label}相关的网页内容，必须包含以下高频关键词：{', '.join(top_keywords)}，内容要自然真实，长度适中"
    
    def _create_semantic_prompt(self, label: str, semantic_keywords: List[Tuple[str, float]]) -> str:
        """创建基于语义关键词的prompt"""
        top_keywords = [kw[0] for kw in semantic_keywords[:4]]
        return f"生成一段{label}相关的网页内容，必须包含以下语义关键词：{', '.join(top_keywords)}，保持语义一致性，内容要自然真实，长度适中"
    
    def _create_contextual_prompt(self, label: str, contextual_keywords: List[Tuple[str, float]]) -> str:
        """创建基于上下文关键词的prompt"""
        top_keywords = [kw[0] for kw in contextual_keywords[:4]]
        return f"生成一段{label}相关的网页内容，必须包含以下上下文关键词：{', '.join(top_keywords)}，保持上下文相关性，内容要自然真实，长度适中"
    
    def _create_keywords_combined_prompt(self, label: str, combined_keywords: List[Tuple[str, float]]) -> str:
        """创建基于综合关键词的prompt"""
        top_keywords = [kw[0] for kw in combined_keywords[:3]]
        return f"生成一段高质量的{label}相关网页内容，必须包含以下核心关键词：{', '.join(top_keywords)}，确保语义和上下文的一致性，内容要自然真实，长度适中"
    
    def _create_label_specific_prompt(self, label: str, label_keywords: Dict) -> str:
        """创建基于标签特定关键词的prompt"""
        website_keywords = [kw[0] for kw in label_keywords.get('websites', [])[:2]]
        verb_keywords = [kw[0] for kw in label_keywords.get('verbs', [])[:2]]
        
        prompt_parts = []
        if website_keywords:
            prompt_parts.append(f"包含网站关键词：{', '.join(website_keywords)}")
        if verb_keywords:
            prompt_parts.append(f"包含动作关键词：{', '.join(verb_keywords)}")
        
        if prompt_parts:
            return f"生成一段{label}相关的网页内容，{', '.join(prompt_parts)}，内容要自然真实，长度适中"
        return f"生成一段{label}相关的网页内容，内容要自然真实，长度适中"
    
    def _create_pattern_prompt(self, label: str, pattern_frequency: Dict) -> str:
        """创建基于话术模式的prompt"""
        top_patterns = list(pattern_frequency.keys())[:3]
        return f"生成一段{label}相关的网页内容，必须体现以下话术模式：{', '.join(top_patterns)}，内容要自然真实，长度适中"
    
    def _create_entity_based_prompt(self, label: str, entity_stats: Dict, structure_stats: Dict = None) -> str:
        """创建基于语义实体的智能prompt（使用structure_stats进行频率判断）"""
        requirements = []
        
        # 定义重要实体类型（使用具体值）
        important_entities = {'money', 'ages', 'percentages'}
        
        # 定义不同实体类型的频率阈值
        frequency_thresholds = {
            'has_money': 0.3,           # 30%的样本包含金钱信息
            'has_age': 0.3,             # 30%的样本包含年龄信息
            'has_percentage': 0.2,      # 20%的样本包含百分比
            'has_time': 0.3,            # 30%的样本包含时间信息
            'has_contact': 0.4,         # 40%的样本包含联系方式
            'has_phone': 0.3,           # 30%的样本包含电话号码
            'has_url': 0.2,             # 20%的样本包含网址
            'has_crypto': 0.1,          # 10%的样本包含加密货币地址
            'has_bank_info': 0.2,       # 20%的样本包含银行信息
            'has_urgency': 0.4,         # 40%的样本体现紧迫感
            'has_promise': 0.3,         # 30%的样本包含收益承诺
            'has_suspicious_patterns': 0.4  # 40%的样本包含可疑模式
        }
        
        # 基于structure_stats进行频率判断
        if structure_stats:
            # 检查各种特征是否达到频率阈值
            if structure_stats.get('has_money', 0) >= frequency_thresholds.get('has_money', 0.3):
                # 使用entity_stats中的具体金钱信息
                if 'money' in entity_stats and entity_stats['money']:
                    top_entities = list(entity_stats['money'].keys())[:2]
                    requirements.append(f"包含金钱信息：{', '.join(top_entities)}")
            
            if structure_stats.get('has_age', 0) >= frequency_thresholds.get('has_age', 0.3):
                # 使用entity_stats中的具体年龄信息
                if 'ages' in entity_stats and entity_stats['ages']:
                    top_entities = list(entity_stats['ages'].keys())[:2]
                    requirements.append(f"包含年龄信息：{', '.join(top_entities)}")
            
            if structure_stats.get('has_percentage', 0) >= frequency_thresholds.get('has_percentage', 0.2):
                # 使用entity_stats中的具体百分比信息
                if 'percentages' in entity_stats and entity_stats['percentages']:
                    top_entities = list(entity_stats['percentages'].keys())[:2]
                    requirements.append(f"包含百分比：{', '.join(top_entities)}")
            
            if structure_stats.get('has_time', 0) >= frequency_thresholds.get('has_time', 0.3):
                requirements.append("包含时间信息")
            
            if structure_stats.get('has_contact', 0) >= frequency_thresholds.get('has_contact', 0.4):
                requirements.append("包含联系方式, 包括但不限于这些类型：微|QQ|qq|扣扣|企鹅|telegram|tg|电报|twitter|推特|x|instagram|ins|facebook|fb|line|whatsapp|wa|联系|加我|找我|薇信|威信|微星|扣扣号|企鹅号")
            
            if structure_stats.get('has_phone', 0) >= frequency_thresholds.get('has_phone', 0.3):
                requirements.append("包含电话号码")
            
            if structure_stats.get('has_url', 0) >= frequency_thresholds.get('has_url', 0.2):
                requirements.append("包含网址链接")
            
            if structure_stats.get('has_crypto', 0) >= frequency_thresholds.get('has_crypto', 0.1):
                requirements.append("包含加密货币地址")
            
            if structure_stats.get('has_bank_info', 0) >= frequency_thresholds.get('has_bank_info', 0.2):
                requirements.append("包含银行信息")
            
            if structure_stats.get('has_urgency', 0) >= frequency_thresholds.get('has_urgency', 0.4):
                requirements.append("体现紧迫感")
            
            if structure_stats.get('has_promise', 0) >= frequency_thresholds.get('has_promise', 0.3):
                requirements.append("包含收益承诺")
            
            if structure_stats.get('has_suspicious_patterns', 0) >= frequency_thresholds.get('has_suspicious_patterns', 0.4):
                requirements.append("包含诈骗特征，包括但不限于：赚钱|发财|投资|理财|收益|回报|分红|中奖|奖金|佣金|返利|稳赚|包中|紧急|急|快|立即|马上|现在|今天|限时|过期|最后|机会|官方|政府|银行|警察|法院|税务局|工商局|公安|检察院|冻结|封号|删除|注销|起诉|逮捕|通缉|违法|犯罪|验证码|密码|身份证|银行卡|账号|个人信息|刷单|兼职|垫付|先付|保证金|手续费|解冻费")
        
        else:
            # 如果没有structure_stats，回退到简单的存在性检查
            for entity_type, entities in entity_stats.items():
                if not entities or len(entities) < 2:
                    continue
                
                if entity_type in important_entities:
                    top_entities = list(entities.keys())[:2]
                    if entity_type == 'money':
                        requirements.append(f"包含金钱信息：{', '.join(top_entities)}")
                    elif entity_type == 'ages':
                        requirements.append(f"包含年龄信息：{', '.join(top_entities)}")
                    elif entity_type == 'percentages':
                        requirements.append(f"包含百分比：{', '.join(top_entities)}")
                else:
                    if entity_type == 'contacts':
                        requirements.append("包含联系方式, 包括但不限于这些类型：微|QQ|qq|扣扣|企鹅|telegram|tg|电报|twitter|推特|x|instagram|ins|facebook|fb|line|whatsapp|wa|联系|加我|找我|薇信|威信|微星|扣扣号|企鹅号")
                    elif entity_type == 'phone_numbers':
                        requirements.append("包含电话号码")
                    elif entity_type == 'time':
                        requirements.append("包含时间信息")
                    elif entity_type == 'urls':
                        requirements.append("包含网址链接")
                    elif entity_type == 'crypto_addresses':
                        requirements.append("包含加密货币地址")
                    elif entity_type == 'bank_cards':
                        requirements.append("包含银行卡信息")
                    elif entity_type == 'bank_names':
                        requirements.append("包含银行信息")
                    elif entity_type == 'suspicious_patterns':
                        for pattern_type, patterns in entities.items():
                            if patterns:
                                if pattern_type == 'urgency_related':
                                    requirements.append("体现紧迫感")
                                elif pattern_type == 'money_related':
                                    requirements.append("包含收益承诺")
                                elif pattern_type == 'authority_related':
                                    requirements.append("包含权威身份")
                                elif pattern_type == 'threat_related':
                                    requirements.append("包含威胁信息")
                                elif pattern_type == 'privacy_related':
                                    requirements.append("要求个人信息")
                                elif pattern_type == 'scam_indicators':
                                    requirements.append("包含诈骗特征")
        
        if requirements:
            return f"生成一段{label}相关的网页内容，{', '.join(requirements[:4])}，内容要自然真实，长度适中"  # 限制最多4个要求
        return f"生成一段{label}相关的网页内容，内容要自然真实，长度适中"


class MultiLevelDataGenerator:
    """多层次数据生成器 - 实现多样化生成策略"""
    
    def __init__(self):
        self.generation_strategies = {
            'template_based': 0.25,      # 模板生成
            'rule_based': 0.25,          # 规则增强
            'semantic_based': 0.30,      # 语义生成
            'hybrid_based': 0.20         # 混合策略
        }
    
    def generate_diverse_samples(self, label: str, prompts: List[Dict], 
                                num_samples: int, augmenter, original_texts: List[str] = None) -> List[Tuple[str, GenerationMetadata]]:
        """生成多样化样本"""
        generated_data = []
        
        # 按权重分配样本数量
        strategy_counts = {}
        for strategy, weight in self.generation_strategies.items():
            strategy_counts[strategy] = int(num_samples * weight)
        
        # 1. 模板生成
        if strategy_counts['template_based'] > 0:
            template_results = augmenter.generate_label_augmentations(
                label, strategy_counts['template_based']
            )
            generated_data.extend(template_results)
        
        # 2. 规则增强（基于原始样本）
        if strategy_counts['rule_based'] > 0 and original_texts:
            rule_results = self._generate_rule_based_samples(
                original_texts, label, strategy_counts['rule_based'], augmenter
            )
            generated_data.extend(rule_results)
        
        # 3. 语义生成（使用所有prompt类型）
        if strategy_counts['semantic_based'] > 0 and augmenter.use_dl:
            semantic_results = self._generate_semantic_samples(
                prompts, strategy_counts['semantic_based'], augmenter
            )
            generated_data.extend(semantic_results)
        
        # 4. 混合策略生成
        if strategy_counts['hybrid_based'] > 0:
            hybrid_results = self._generate_hybrid_samples(
                label, prompts, strategy_counts['hybrid_based'], augmenter
            )
            generated_data.extend(hybrid_results)
        
        return generated_data[:num_samples]
    
    def _generate_semantic_samples(self, prompts: List[Dict], num_samples: int, augmenter) -> List[Tuple[str, GenerationMetadata]]:
        """基于语义prompt生成样本"""
        generated_data = []
        samples_per_prompt = max(1, num_samples // len(prompts)) if prompts else 0
        
        for prompt_info in prompts:
            if len(generated_data) >= num_samples:
                break
            
            try:
                dl_generated = augmenter.dl_generator.generate_text(
                    prompt_info['prompt'], 
                    num_return_sequences=min(3, samples_per_prompt)
                )
                
                for text in dl_generated:
                    # 创建生成元数据
                    metadata = augmenter.generation_tracker.create_metadata(
                        method=prompt_info.get('generation_method', GenerationMethod.SEMANTIC_SEMANTIC_PROMPT),
                        prompt_type=prompt_info.get('type', 'unknown'),
                        prompt_content=prompt_info['prompt'],
                        feature_weights=prompt_info.get('weight'),
                        generation_params={
                            'description': prompt_info.get('description', ''),
                            'num_return_sequences': min(3, samples_per_prompt)
                        }
                    )
                    
                    # 不在这里跟踪生成统计，等过滤后再记录
                    generated_data.append((text, metadata))
                    
            except Exception as e:
                print(f"语义生成失败: {e}")
                continue
        
        return generated_data
    
    def _generate_rule_based_samples(self, original_texts: List[str], label: str, 
                                   num_samples: int, augmenter) -> List[Tuple[str, GenerationMetadata]]:
        """基于原始样本的规则增强生成"""
        generated_data = []
        
        # 随机选择原始样本进行规则增强
        import random
        selected_texts = random.sample(original_texts, min(len(original_texts), num_samples))
        
        for text in selected_texts:
            try:
                rule_augmented = augmenter.generate_rule_based_augmentations(text, label)
                # 每个原始样本最多生成2个增强样本
                generated_data.extend(rule_augmented[:2])
            except Exception as e:
                print(f"规则增强失败: {e}")
                continue
        
        return generated_data[:num_samples]
    
    def _generate_hybrid_samples(self, label: str, prompts: List[Dict], 
                                num_samples: int, augmenter) -> List[Tuple[str, GenerationMetadata]]:
        """混合策略生成样本 - 结合多种特征生成更丰富的样本"""
        generated_data = []
        
        if not prompts:
            # 如果没有prompt，回退到纯模板生成
            return augmenter.generate_label_augmentations(label, num_samples)
        
        # 混合策略：结合多种特征类型生成样本
        for i in range(num_samples):
            try:
                # 随机选择2-3个不同类型的prompt进行组合
                import random
                selected_prompts = random.sample(prompts, min(3, len(prompts)))
                
                # 创建组合prompt
                combined_prompt = self._create_combined_prompt(label, selected_prompts)
                
                # 使用深度学习生成
                if augmenter.use_dl and augmenter.dl_generator:
                    generated = augmenter.dl_generator.generate_text(
                        combined_prompt, num_return_sequences=1
                    )
                    
                    for text in generated:
                        # 创建生成元数据
                        metadata = augmenter.generation_tracker.create_metadata(
                            method=GenerationMethod.HYBRID_MULTI_FEATURE,
                            prompt_type="多特征混合",
                            prompt_content=combined_prompt,
                            generation_params={
                                'selected_prompts': [p.get('type', 'unknown') for p in selected_prompts],
                                'num_prompts_combined': len(selected_prompts),
                                'label': label
                            }
                        )
                        
                        # 不在这里跟踪生成统计，等过滤后再记录
                        generated_data.append((text, metadata))
                else:
                    # 回退到模板生成
                    template_results = augmenter.generate_label_augmentations(label, 1)
                    generated_data.extend(template_results)
                    
            except Exception as e:
                print(f"混合生成失败: {e}")
                # 回退到模板生成
                template_results = augmenter.generate_label_augmentations(label, 1)
                generated_data.extend(template_results)
                continue
        
        return generated_data[:num_samples]
    
    def _create_combined_prompt(self, label: str, selected_prompts: List[Dict]) -> str:
        """创建组合prompt，融合多种特征类型"""
        # 提取不同prompt类型的关键信息
        prompt_elements = []
        
        for prompt_info in selected_prompts:
            prompt_type = prompt_info['type']
            prompt_text = prompt_info['prompt']
            
            if prompt_type == 'tfidf_based':
                # 提取关键词
                if '关键词' in prompt_text:
                    keywords_part = prompt_text.split('关键词：')[1] if '关键词：' in prompt_text else ''
                    prompt_elements.append(f"高频关键词：{keywords_part}")
            
            elif prompt_type == 'semantic_based':
                # 提取语义关键词
                if '语义关键词' in prompt_text:
                    keywords_part = prompt_text.split('语义关键词：')[1].split('，')[0] if '语义关键词：' in prompt_text else ''
                    prompt_elements.append(f"语义关键词：{keywords_part}")
            
            elif prompt_type == 'contextual_based':
                # 提取上下文关键词
                if '上下文关键词' in prompt_text:
                    keywords_part = prompt_text.split('上下文关键词：')[1].split('，')[0] if '上下文关键词：' in prompt_text else ''
                    prompt_elements.append(f"上下文关键词：{keywords_part}")
            
            elif prompt_type == 'combined_based':
                # 提取综合关键词
                if '核心关键词' in prompt_text:
                    keywords_part = prompt_text.split('核心关键词：')[1].split('，')[0] if '核心关键词：' in prompt_text else ''
                    prompt_elements.append(f"核心关键词：{keywords_part}")
            
            elif prompt_type == 'label_specific':
                # 提取标签特定关键词
                if '包含' in prompt_text and ('网站关键词' in prompt_text or '动作关键词' in prompt_text):
                    # 提取标签特定的要求
                    if '网站关键词' in prompt_text:
                        website_part = prompt_text.split('网站关键词：')[1].split('，')[0] if '网站关键词：' in prompt_text else ''
                        prompt_elements.append(f"网站关键词：{website_part}")
                    if '动作关键词' in prompt_text:
                        action_part = prompt_text.split('动作关键词：')[1].split('，')[0] if '动作关键词：' in prompt_text else ''
                        prompt_elements.append(f"动作关键词：{action_part}")
            
            elif prompt_type == 'pattern_based':
                # 提取话术模式
                if '话术模式' in prompt_text:
                    patterns_part = prompt_text.split('话术模式：')[1] if '话术模式：' in prompt_text else ''
                    prompt_elements.append(f"话术模式：{patterns_part}")
            
            elif prompt_type == 'entity_based':
                # 提取实体要求
                if '包含' in prompt_text:
                    # 提取所有包含的要求
                    contains_parts = []
                    parts = prompt_text.split('，')
                    for part in parts:
                        if '包含' in part and part.strip():
                            contains_parts.append(part.strip())
                    if contains_parts:
                        prompt_elements.append(f"实体要求：{', '.join(contains_parts)}")
        
        # 组合成综合prompt
        if prompt_elements:
            combined_prompt = f"生成一段{label}相关的网页内容，要求：{', '.join(prompt_elements)}，确保内容真实自然，长度适中"
        else:
            combined_prompt = f"生成一段{label}相关的网页内容，要求内容真实自然，长度适中"
        
        return combined_prompt




@dataclass
class AugmentationRule:
    """数据增强规则"""
    name: str
    description: str
    template: str
    variables: List[str]
    examples: List[str]


class DataAugmentation:
    def __init__(self, use_dl: bool = True, dl_config: GenerationConfig = None):
        # 数据预处理器（重构后只用于文本清理）
        self.preprocessor = DataPreprocessor()
        
        # 模式分析器（用于结构化信息提取）
        self.pattern_analyzer = PatternAnalyzer()
        
        # 深度学习配置
        self.use_dl = use_dl and DL_AVAILABLE
        self.dl_generator = None
        self.prompt_engine = None
        
        if self.use_dl:
            try:
                self.dl_generator = DLTextGenerator(dl_config or GenerationConfig())
                self.prompt_engine = PromptEngine()
                print("深度学习文本生成器初始化成功")
            except Exception as e:
                print(f"深度学习文本生成器初始化失败: {e}")
                self.use_dl = False
        
        # 智能特征融合器
        self.feature_fusion = IntelligentFeatureFusion()
        
        # 多样化生成器
        self.multi_generator = MultiLevelDataGenerator()
        
        # 生成方式跟踪器
        self.generation_tracker = GenerationTracker()
        # 定义不同标签的数据增强规则
        self.augmentation_rules = {
            '按摩色诱': [
                AugmentationRule(
                    name="价格变化",
                    description="改变价格数字",
                    template="{age}岁美女{service}，{price}{time}，{promise}，{contact}",
                    variables=["age", "service", "price", "time", "promise", "contact"],
                    examples=["20岁美女按摩，¥200 2小时，包你满意，立即注册"]
                ),
                AugmentationRule(
                    name="服务变化",
                    description="改变服务类型",
                    template="{platform}提供{service}，{price}，{quality}，{contact}",
                    variables=["platform", "service", "price", "quality", "contact"],
                    examples=["同城平台提供spa服务，价格优惠，质量保证，点击访问"]
                )
            ],
            '博彩': [
                AugmentationRule(
                    name="收益变化",
                    description="改变收益数字",
                    template="{platform}，{promise}，{bonus}，{contact}",
                    variables=["platform", "promise", "bonus", "contact"],
                    examples=["正规平台，稳赚不赔，充值送50%，立即注册"]
                ),
                AugmentationRule(
                    name="平台变化",
                    description="改变平台描述",
                    template="{platform}，{feature}，{guarantee}，{contact}",
                    variables=["platform", "feature", "guarantee", "contact"],
                    examples=["信誉平台，专业团队，秒到账，点击访问"]
                )
            ],
            '兼职刷单': [
                AugmentationRule(
                    name="佣金变化",
                    description="改变佣金数字",
                    template="{job_type}，{commission}，{requirement}，{contact}",
                    variables=["job_type", "commission", "requirement", "contact"],
                    examples=["刷单兼职，50元一单，需要垫付，立即注册"]
                ),
                AugmentationRule(
                    name="工作变化",
                    description="改变工作描述",
                    template="{job_type}，{advantage}，{time}，{contact}",
                    variables=["job_type", "advantage", "time", "contact"],
                    examples=["网络兼职，轻松赚钱，时间自由，点击访问"]
                )
            ],
            '投资理财': [
                AugmentationRule(
                    name="收益变化",
                    description="改变收益率",
                    template="{product}，{return_rate}，{guarantee}，{contact}",
                    variables=["product", "return_rate", "guarantee", "contact"],
                    examples=["理财产品，年化20%，保本保息，立即注册"]
                ),
                AugmentationRule(
                    name="产品变化",
                    description="改变产品类型",
                    template="{product}，{feature}，{urgency}，{contact}",
                    variables=["product", "feature", "urgency", "contact"],
                    examples=["基金投资，专业团队，限时优惠，点击访问"]
                )
            ],
            '虚假客服': [
                AugmentationRule(
                    name="身份变化",
                    description="改变客服身份",
                    template="{identity}，{situation}，{action}，{contact}",
                    variables=["identity", "situation", "action", "contact"],
                    examples=["银行客服，账户异常，立即处理，按提示操作"]
                ),
                AugmentationRule(
                    name="情况变化",
                    description="改变紧急情况",
                    template="{identity}，{situation}，{requirement}，{contact}",
                    variables=["identity", "situation", "requirement", "contact"],
                    examples=["官方客服，资金风险，验证身份，点击链接"]
                )
            ]
        }
        
        # 变量替换词典（针对网页OCR文本优化）
        self.variable_dict = {
            'age': ['18', '20', '22', '25', '28', '30'],
            'service': ['按摩', 'spa', '服务', '放松', '享受'],
            'price': ['¥100', '¥200', '¥300', '¥500', '¥800', '100元', '200元', '300元'],
            'time': ['1小时', '2小时', '3小时', '半天', '包夜'],
            'promise': ['包你满意', '保证质量', '100%服务', '专业服务'],
            'contact': ['立即注册', '点击访问', '免费下载', '在线申请', '了解更多'],
            'platform': ['同城平台', '交友网站', '正规平台', '信誉平台'],
            'quality': ['质量保证', '服务一流', '专业团队', '口碑很好'],
            'bonus': ['充值送50%', '首充100%', '新用户优惠', '限时活动'],
            'guarantee': ['秒到账', '快速提现', '24小时到账', '安全可靠'],
            'job_type': ['刷单兼职', '网络兼职', '任务兼职', '佣金兼职'],
            'commission': ['50元一单', '20%佣金', '日赚500', '高佣金'],
            'requirement': ['需要垫付', '先垫付100元', '准备本金', '资金要求'],
            'advantage': ['轻松赚钱', '简单操作', '在家兼职', '时间自由'],
            'product': ['理财产品', '基金投资', '股票投资', '外汇投资'],
            'return_rate': ['年化20%', '收益30%', '回报50%', '高收益'],
            'feature': ['专业团队', '资深分析师', '金牌导师', '经验丰富'],
            'urgency': ['限时优惠', '仅限3天', '错过后悔', '机会难得'],
            'identity': ['银行客服', '官方客服', '系统通知', '客服代表'],
            'situation': ['账户异常', '资金风险', '系统升级', '安全检测'],
            'action': ['立即处理', '马上操作', '按提示', '点击链接'],
            'requirement': ['验证身份', '提供密码', '确认信息', '输入验证码']
        }
    
    def generate_augmented_text(self, rule: AugmentationRule, num_samples: int = 5) -> List[Tuple[str, GenerationMetadata]]:
        """
        根据规则生成增强文本
        """
        generated_data = []
        
        for _ in range(num_samples):
            # 替换模板中的变量
            text = rule.template
            for variable in rule.variables:
                if variable in self.variable_dict:
                    replacement = random.choice(self.variable_dict[variable])
                    text = text.replace(f"{{{variable}}}", replacement)
            
            # 创建生成元数据
            metadata = self.generation_tracker.create_metadata(
                method=GenerationMethod.TEMPLATE_RULE,
                generation_params={
                    'rule_name': rule.name,
                    'rule_description': rule.description,
                    'template': rule.template,
                    'variables': rule.variables
                }
            )
            
            # 不在这里跟踪生成统计，等过滤后再记录
            generated_data.append((text, metadata))
        
        return generated_data
    
    def generate_label_augmentations(self, label: str, num_samples: int = 20) -> List[Tuple[str, GenerationMetadata]]:
        """
        为特定标签生成增强样本
        """
        if label not in self.augmentation_rules:
            return []
        
        all_generated = []
        rules = self.augmentation_rules[label]
        
        # 为每个规则生成样本
        samples_per_rule = num_samples // len(rules)
        for rule in rules:
            generated = self.generate_augmented_text(rule, samples_per_rule)
            all_generated.extend(generated)
        
        # 如果还需要更多样本，随机选择规则继续生成
        remaining = num_samples - len(all_generated)
        if remaining > 0:
            for _ in range(remaining):
                rule = random.choice(rules)
                generated = self.generate_augmented_text(rule, 1)
                all_generated.extend(generated)
        
        return all_generated[:num_samples]
    
    def create_prompts_for_label(self, label: str, keyword_results: Dict, pattern_results: Dict) -> List[str]:
        """
        为特定标签创建prompt模板
        """
        prompts = []
        
        if label not in keyword_results or label not in pattern_results:
            return prompts
        
        keyword_data = keyword_results[label]
        pattern_data = pattern_results[label]
        
        # 基于关键词的prompt
        if keyword_data['tfidf_keywords']:
            top_keywords = [kw[0] for kw in keyword_data['tfidf_keywords'][:5]]
            prompt = f"生成一段{label}相关的网页内容文案，必须包含以下关键词：{', '.join(top_keywords)}，风格要像网页广告"
            prompts.append(prompt)
        
        # 基于模式的prompt
        if pattern_data['pattern_frequency']:
            top_patterns = list(pattern_data['pattern_frequency'].keys())[:3]
            prompt = f"生成一段{label}相关的网页内容文案，必须包含以下话术模式：{', '.join(top_patterns)}，风格要像网页广告"
            prompts.append(prompt)
        
        # 基于结构的prompt
        structure_features = pattern_data['structure_stats']
        if 'has_numbers' in structure_features and structure_features['has_numbers'] > 0.5:
            prompt = f"生成一段{label}相关的网页内容文案，必须包含数字（价格、时间、百分比等），风格要像网页广告"
            prompts.append(prompt)
        
        if 'has_contact' in structure_features and structure_features['has_contact'] > 0.5:
            prompt = f"生成一段{label}相关的网页内容文案，必须包含联系方式（微信、QQ、电话等），风格要像网页广告"
            prompts.append(prompt)
        
        
        return prompts
    
    def generate_rule_based_augmentations(self, original_text: str, label: str) -> List[Tuple[str, GenerationMetadata]]:
        """
        基于规则的文本增强
        """
        augmented_data = []
        
        # 同义词替换
        synonyms = {
            '你好': ['您好', 'Hi', 'Hello'],
            '美女': ['妹妹', '小姐姐', '姑娘'],
            '按摩': ['spa', '放松', '服务'],
            '赚钱': ['收益', '获利', '盈利'],
            '平台': ['网站', 'app', '软件'],
            '客服': ['服务', '售后', '支持']
        }
        
        for original, replacements in synonyms.items():
            if original in original_text:
                for replacement in replacements:
                    new_text = original_text.replace(original, replacement)
                    if new_text != original_text:
                        # 创建生成元数据
                        metadata = self.generation_tracker.create_metadata(
                            method=GenerationMethod.RULE_SYNONYM,
                            original_text=original_text,
                            generation_params={
                                'original_word': original,
                                'replacement_word': replacement,
                                'label': label
                            }
                        )
                        
                        # 不在这里跟踪生成统计，等过滤后再记录
                        augmented_data.append((new_text, metadata))
        
        # 数字变化
        numbers = re.findall(r'[0-9]+', original_text)
        for number in numbers:
            # 增加或减少10-50%
            change = random.uniform(0.1, 0.5)
            if random.random() > 0.5:
                new_number = str(int(int(number) * (1 + change)))
            else:
                new_number = str(int(int(number) * (1 - change)))
            
            new_text = original_text.replace(number, new_number)
            if new_text != original_text:
                # 创建生成元数据
                metadata = self.generation_tracker.create_metadata(
                    method=GenerationMethod.RULE_NUMBER_CHANGE,
                    original_text=original_text,
                    generation_params={
                        'original_number': number,
                        'new_number': new_number,
                        'change_ratio': change,
                        'label': label
                    }
                )
                
                # 不在这里跟踪生成统计，等过滤后再记录
                augmented_data.append((new_text, metadata))
        
        # 语序调整（简单的句子重排）
        sentences = re.split(r'[，。！？]', original_text)
        if len(sentences) > 1:
            random.shuffle(sentences)
            new_text = '，'.join(sentences).rstrip('，') + '。'
            if new_text != original_text:
                # 创建生成元数据
                metadata = self.generation_tracker.create_metadata(
                    method=GenerationMethod.RULE_WORD_ORDER,
                    original_text=original_text,
                    generation_params={
                        'original_sentences': sentences,
                        'shuffled_sentences': sentences,
                        'label': label
                    }
                )
                
                # 不在这里跟踪生成统计，等过滤后再记录
                augmented_data.append((new_text, metadata))
        
        return augmented_data
    
    def generate_dl_augmentations(self, label: str, num_samples: int = 20, 
                                 keyword_results: Dict = None, pattern_results: Dict = None) -> List[Tuple[str, GenerationMetadata]]:
        """使用深度学习模型生成增强文本"""
        if not self.use_dl or self.dl_generator is None:
            # 回退到传统方法
            traditional_results = self.generate_label_augmentations(label, num_samples)
            # 为传统方法添加回退标记，但保留实际使用的生成方法
            fallback_results = []
            for text, metadata in traditional_results:
                # 创建回退元数据，但记录实际使用的生成方法
                fallback_metadata = self.generation_tracker.create_metadata(
                    method=metadata.method,  # 使用实际的方法，而不是回退方法
                    generation_params={
                        'reason': '深度学习不可用',
                        'fallback_type': 'FALLBACK_TRADITIONAL',
                        'original_intent': '深度学习生成',
                        'actual_method': metadata.method.value,
                        'label': label
                    }
                )
                # 不在这里跟踪生成统计，等过滤后再记录
                fallback_results.append((text, fallback_metadata))
            return fallback_results
        
        generated_data = []
        
        try:
            # 使用智能Prompt引擎生成prompt
            prompts = self.prompt_engine.generate_prompts(label, min(5, num_samples // 4))
            
            for prompt in prompts:
                # 使用深度学习模型生成文本
                dl_generated = self.dl_generator.generate_text(prompt, num_return_sequences=3)
                
                for text in dl_generated:
                    if len(text) > 10 and self._is_valid_generated_text(text, label):
                        # 创建生成元数据
                        metadata = self.generation_tracker.create_metadata(
                            method=GenerationMethod.SEMANTIC_SEMANTIC_PROMPT,
                            prompt_type="智能Prompt引擎",
                            prompt_content=prompt,
                            generation_params={
                                'label': label,
                                'prompt_engine': True,
                                'num_return_sequences': 3
                            }
                        )
                        
                        # 不在这里跟踪生成统计，等过滤后再记录
                        generated_data.append((text, metadata))
                
                if len(generated_data) >= num_samples:
                    break
            
            # 如果还需要更多样本，使用基于关键词和模式的prompt
            if len(generated_data) < num_samples and keyword_results and pattern_results:
                additional_prompts = self._create_semantic_prompts(label, keyword_results, pattern_results)
                
                for prompt in additional_prompts:
                    if len(generated_data) >= num_samples:
                        break
                    
                    dl_generated = self.dl_generator.generate_text(prompt, num_return_sequences=2)
                    
                    for text in dl_generated:
                        if len(text) > 10 and self._is_valid_generated_text(text, label):
                            # 创建生成元数据
                            metadata = self.generation_tracker.create_metadata(
                                method=GenerationMethod.SEMANTIC_COMBINED_PROMPT,
                                prompt_type="关键词模式Prompt",
                                prompt_content=prompt,
                                generation_params={
                                    'label': label,
                                    'keyword_based': True,
                                    'pattern_based': True,
                                    'num_return_sequences': 2
                                }
                            )
                            
                            # 不在这里跟踪生成统计，等过滤后再记录
                            generated_data.append((text, metadata))
            
            return generated_data[:num_samples]
            
        except Exception as e:
            print(f"深度学习增强失败: {e}")
            # 回退到传统方法
            traditional_results = self.generate_label_augmentations(label, num_samples)
            fallback_results = []
            for text, metadata in traditional_results:
                # 创建回退元数据，但记录实际使用的生成方法
                fallback_metadata = self.generation_tracker.create_metadata(
                    method=metadata.method,  # 使用实际的方法，而不是回退方法
                    generation_params={
                        'reason': f'深度学习错误: {str(e)}',
                        'fallback_type': 'FALLBACK_DL_ERROR',
                        'original_intent': '深度学习生成',
                        'actual_method': metadata.method.value,
                        'label': label
                    }
                )
                # 不在这里跟踪生成统计，等过滤后再记录
                fallback_results.append((text, fallback_metadata))
            return fallback_results
    
    def _create_semantic_prompts(self, label: str, keyword_results: Dict, pattern_results: Dict) -> List[str]:
        """基于语义分析结果创建prompt"""
        prompts = []
        
        if label not in keyword_results or label not in pattern_results:
            return prompts
        
        keyword_data = keyword_results[label]
        pattern_data = pattern_results[label]
        
        # 基于关键词的prompt（优先使用综合关键词）
        keywords_to_use = None
        if keyword_data.get('combined_keywords'):
            keywords_to_use = keyword_data['combined_keywords']
        elif keyword_data.get('semantic_keywords'):
            keywords_to_use = keyword_data['semantic_keywords']
        
        if keywords_to_use:
            top_keywords = [kw[0] for kw in keywords_to_use[:3]]
            prompt = f"生成一段{label}相关的网页内容文案，必须包含以下关键词：{', '.join(top_keywords)}，风格要像网页广告"
            prompts.append(prompt)
        
        # 基于模式的prompt
        if pattern_data.get('pattern_frequency'):
            top_patterns = list(pattern_data['pattern_frequency'].keys())[:2]
            prompt = f"生成一段{label}相关的网页内容文案，必须包含以下话术模式：{', '.join(top_patterns)}，风格要像网页广告"
            prompts.append(prompt)
        
        # 基于语义实体的prompt
        if pattern_data.get('entity_stats'):
            entity_prompts = []
            for entity_type, entities in pattern_data['entity_stats'].items():
                if entities:
                    top_entity = list(entities.keys())[0]
                    entity_prompts.append(f"包含{entity_type}：{top_entity}")
            
            if entity_prompts:
                prompt = f"生成一段{label}相关的网页内容文案，{', '.join(entity_prompts)}，风格要像网页广告"
                prompts.append(prompt)
        
        return prompts
    
    def _is_valid_generated_text(self, text: str, label: str) -> bool:
        """验证生成的文本是否有效"""
        # 基本长度检查
        if len(text) < 10 or len(text) > 200:
            return False
        
        # 检查是否包含标签相关的关键词
        label_keywords = {
            '按摩色诱': ['按摩', '美女', '服务', 'spa'],
            '博彩': ['博彩', '平台', '投注', '游戏'],
            '兼职刷单': ['兼职', '刷单', '佣金', '任务'],
            '投资理财': ['投资', '理财', '收益', '基金'],
            '虚假客服': ['客服', '银行', '验证', '账户']
        }
        
        if label in label_keywords:
            keywords = label_keywords[label]
            if not any(keyword in text for keyword in keywords):
                return False
        
        # 检查是否包含过多重复字符
        if len(set(text)) < len(text) * 0.3:
            return False
        
        return True
    
    def create_augmentation_dataset(self, df: pd.DataFrame, keyword_results: Dict, 
                                  pattern_results: Dict, augmentation_ratio: float = 0.5) -> pd.DataFrame:
        """
        创建增强数据集（集成深度学习生成）
        """
        augmented_data = []
        
        # 为每个标签生成增强样本
        for label in df['label'].unique():
            if pd.isna(label) or label == '':
                continue
            
            label_data = df[df['label'] == label]
            original_count = len(label_data)
            target_count = int(original_count * augmentation_ratio)
            
            print(f"为标签 '{label}' 生成 {target_count} 条增强数据...")
            
            # 使用深度学习生成（如果可用）
            if self.use_dl:
                dl_count = int(target_count * 0.6)  # 60%使用深度学习生成
                template_count = target_count - dl_count
                
                # 深度学习生成
                dl_augmented = self.generate_dl_augmentations(
                    label, dl_count, keyword_results, pattern_results
                )
                
                for text in dl_augmented:
                    # 从生成的文本中提取结构化信息（使用PatternAnalyzer）
                    entities = self.pattern_analyzer.extract_semantic_entities(text)
                    urls = entities.get('urls', [])
                    phones = entities.get('phone_numbers', [])
                    contacts = entities.get('contacts', [])
                    crypto_addresses = entities.get('crypto_addresses', [])
                    bank_info = entities.get('bank_cards', []) + entities.get('bank_names', [])
                    suspicious_patterns = entities.get('suspicious_patterns', {})
                    
                    augmented_data.append({
                        'original_text': '',
                        'cleaned_text': text,
                        'text_length': len(text),
                        'label': label,
                        'data_source': 'dl_generated',
                        'urls': urls,
                        'phone_numbers': phones,
                        'contacts': contacts,
                        'crypto_addresses': crypto_addresses,
                        'bank_info': bank_info,
                        'suspicious_patterns': suspicious_patterns,
                        'has_url': len(urls) > 0,
                        'has_phone': len(phones) > 0,
                        'has_contact': len(contacts) > 0,
                        'has_crypto': len(crypto_addresses) > 0,
                        'has_bank_info': len(bank_info) > 0,
                        'has_suspicious_patterns': any(len(patterns) > 0 for patterns in suspicious_patterns.values()),
                        'augmentation_type': 'dl_generation'
                    })
                
                # 传统模板生成
                template_augmented = self.generate_label_augmentations(label, template_count)
                for text in template_augmented:
                    # 从生成的文本中提取结构化信息（使用PatternAnalyzer）
                    entities = self.pattern_analyzer.extract_semantic_entities(text)
                    urls = entities.get('urls', [])
                    phones = entities.get('phone_numbers', [])
                    contacts = entities.get('contacts', [])
                    crypto_addresses = entities.get('crypto_addresses', [])
                    bank_info = entities.get('bank_cards', []) + entities.get('bank_names', [])
                    suspicious_patterns = entities.get('suspicious_patterns', {})
                    
                    augmented_data.append({
                        'original_text': '',
                        'cleaned_text': text,
                        'text_length': len(text),
                        'label': label,
                        'data_source': 'augmented_template',
                        'urls': urls,
                        'phone_numbers': phones,
                        'contacts': contacts,
                        'crypto_addresses': crypto_addresses,
                        'bank_info': bank_info,
                        'suspicious_patterns': suspicious_patterns,
                        'has_url': len(urls) > 0,
                        'has_phone': len(phones) > 0,
                        'has_contact': len(contacts) > 0,
                        'has_crypto': len(crypto_addresses) > 0,
                        'has_bank_info': len(bank_info) > 0,
                        'has_suspicious_patterns': any(len(patterns) > 0 for patterns in suspicious_patterns.values()),
                        'augmentation_type': 'template'
                    })
            else:
                # 纯传统方法
                template_augmented = self.generate_label_augmentations(label, target_count)
                for text in template_augmented:
                    # 从生成的文本中提取结构化信息（使用PatternAnalyzer）
                    entities = self.pattern_analyzer.extract_semantic_entities(text)
                    urls = entities.get('urls', [])
                    phones = entities.get('phone_numbers', [])
                    contacts = entities.get('contacts', [])
                    crypto_addresses = entities.get('crypto_addresses', [])
                    bank_info = entities.get('bank_cards', []) + entities.get('bank_names', [])
                    suspicious_patterns = entities.get('suspicious_patterns', {})
                    
                    augmented_data.append({
                        'original_text': '',
                        'cleaned_text': text,
                        'text_length': len(text),
                        'label': label,
                        'data_source': 'augmented_template',
                        'urls': urls,
                        'phone_numbers': phones,
                        'contacts': contacts,
                        'crypto_addresses': crypto_addresses,
                        'bank_info': bank_info,
                        'suspicious_patterns': suspicious_patterns,
                        'has_url': len(urls) > 0,
                        'has_phone': len(phones) > 0,
                        'has_contact': len(contacts) > 0,
                        'has_crypto': len(crypto_addresses) > 0,
                        'has_bank_info': len(bank_info) > 0,
                        'has_suspicious_patterns': any(len(patterns) > 0 for patterns in suspicious_patterns.values()),
                        'augmentation_type': 'template'
                    })
            
            # 基于原始样本的规则增强
            rule_augmented_count = min(target_count // 3, len(label_data))  # 减少规则增强的比例
            if rule_augmented_count > 0:
                sample_indices = label_data.index.to_series().sample(rule_augmented_count)
                
                for idx in sample_indices:
                    original_text = label_data.loc[idx, 'cleaned_text']
                    rule_augmented = self.generate_rule_based_augmentations(original_text, label)
                    
                    for text in rule_augmented[:2]:  # 每个原始样本最多生成2个增强样本
                        # 从生成的文本中提取结构化信息（使用PatternAnalyzer）
                        entities = self.pattern_analyzer.extract_semantic_entities(text)
                        urls = entities.get('urls', [])
                        phones = entities.get('phone_numbers', [])
                        contacts = entities.get('contacts', [])
                        crypto_addresses = entities.get('crypto_addresses', [])
                        bank_info = entities.get('bank_cards', []) + entities.get('bank_names', [])
                        suspicious_patterns = entities.get('suspicious_patterns', {})
                        
                        augmented_data.append({
                            'original_text': original_text,
                            'cleaned_text': text,
                            'text_length': len(text),
                            'label': label,
                            'data_source': 'augmented_rule',
                            'urls': urls,
                            'phone_numbers': phones,
                            'contacts': contacts,
                            'crypto_addresses': crypto_addresses,
                            'bank_info': bank_info,
                            'suspicious_patterns': suspicious_patterns,
                            'has_url': len(urls) > 0,
                            'has_phone': len(phones) > 0,
                            'has_contact': len(contacts) > 0,
                            'has_crypto': len(crypto_addresses) > 0,
                            'has_bank_info': len(bank_info) > 0,
                            'has_suspicious_patterns': any(len(patterns) > 0 for patterns in suspicious_patterns.values()),
                            'augmentation_type': 'rule'
                        })
        
        return pd.DataFrame(augmented_data)
    
    def create_enhanced_augmentation_dataset(self, df: pd.DataFrame, keyword_results: Dict, 
                                           pattern_results: Dict, augmentation_ratio: float = 0.5) -> pd.DataFrame:
        """
        创建增强版数据增强数据集 - 充分利用所有特征
        """
        print("开始智能特征融合数据增强...")
        augmented_data = []
        
        # 为每个标签生成增强样本
        for label in df['label'].unique():
            if pd.isna(label) or label == '':
                continue
            
            label_data = df[df['label'] == label]
            original_count = len(label_data)
            target_count = int(original_count * augmentation_ratio)
            
            print(f"为标签 '{label}' 生成 {target_count} 条增强数据（智能融合）...")
            
            # 1. 智能特征融合 - 创建综合prompt策略
            comprehensive_prompts = self.feature_fusion.create_comprehensive_prompts(
                label, keyword_results, pattern_results
            )
            
            print(f"  生成了 {len(comprehensive_prompts)} 种prompt策略")
            
            # 2. 多样化生成 - 使用多层次生成策略
            original_texts = label_data['cleaned_text'].tolist()
            generated_data = self.multi_generator.generate_diverse_samples(
                label, comprehensive_prompts, target_count, self, original_texts
            )
            
            print(f"  生成了 {len(generated_data)} 条候选文本")
            
            # 3. 简单过滤（只保留长度合理的文本）
            filtered_data = []
            for text, metadata in generated_data:
                # 基本长度检查（只过滤过短的文本，不设置上限）
                if len(text) >= 10:
                    filtered_data.append((text, metadata))
                    # 只在成功通过过滤时记录跟踪统计
                    self.generation_tracker.track_generation(metadata.method)
            
            print(f"  过滤后保留 {len(filtered_data)} 条文本")
            
            # 4. 添加结构化信息
            for text, metadata in filtered_data:
                # 从生成的文本中提取结构化信息
                entities = self.pattern_analyzer.extract_semantic_entities(text)
                urls = entities.get('urls', [])
                phones = entities.get('phone_numbers', [])
                contacts = entities.get('contacts', [])
                crypto_addresses = entities.get('crypto_addresses', [])
                bank_info = entities.get('bank_cards', []) + entities.get('bank_names', [])
                suspicious_patterns = entities.get('suspicious_patterns', {})
                
                # 合并生成元数据
                data_row = {
                    'original_text': '',
                    'cleaned_text': text,
                    'text_length': len(text),
                    'label': label,
                    'data_source': 'enhanced_generated',
                    'urls': urls,
                    'phone_numbers': phones,
                    'contacts': contacts,
                    'crypto_addresses': crypto_addresses,
                    'bank_info': bank_info,
                    'suspicious_patterns': suspicious_patterns,
                    'has_url': len(urls) > 0,
                    'has_phone': len(phones) > 0,
                    'has_contact': len(contacts) > 0,
                    'has_crypto': len(crypto_addresses) > 0,
                    'has_bank_info': len(bank_info) > 0,
                    'has_suspicious_patterns': any(len(patterns) > 0 for patterns in suspicious_patterns.values()),
                    'augmentation_type': 'enhanced_fusion'
                }
                
                # 添加生成元数据
                data_row.update(metadata.to_dict())
                augmented_data.append(data_row)
            
            # 5. 如果过滤后文本不足，使用传统方法补充
            if len(filtered_data) < target_count * 0.8:  # 如果过滤后文本少于80%
                remaining_count = target_count - len(filtered_data)
                print(f"  过滤后文本不足，使用传统方法补充 {remaining_count} 条")
                
                # 使用传统方法生成补充文本
                traditional_results = self.generate_label_augmentations(label, remaining_count)
                for text, metadata in traditional_results:
                    entities = self.pattern_analyzer.extract_semantic_entities(text)
                    urls = entities.get('urls', [])
                    phones = entities.get('phone_numbers', [])
                    contacts = entities.get('contacts', [])
                    crypto_addresses = entities.get('crypto_addresses', [])
                    bank_info = entities.get('bank_cards', []) + entities.get('bank_names', [])
                    suspicious_patterns = entities.get('suspicious_patterns', {})
                    
                    # 创建回退元数据，但记录实际使用的生成方法
                    fallback_metadata = self.generation_tracker.create_metadata(
                        method=metadata.method,  # 使用实际的方法，而不是回退方法
                        generation_params={
                            'reason': '过滤后文本不足',
                            'fallback_type': 'FALLBACK_TRADITIONAL',
                            'original_intent': '智能特征融合生成',
                            'actual_method': metadata.method.value,
                            'label': label
                        }
                    )
                    
                    # 跟踪实际使用的生成方法
                    self.generation_tracker.track_generation(metadata.method)
                    
                    data_row = {
                        'original_text': '',
                        'cleaned_text': text,
                        'text_length': len(text),
                        'label': label,
                        'data_source': 'traditional_fallback',
                        'urls': urls,
                        'phone_numbers': phones,
                        'contacts': contacts,
                        'crypto_addresses': crypto_addresses,
                        'bank_info': bank_info,
                        'suspicious_patterns': suspicious_patterns,
                        'has_url': len(urls) > 0,
                        'has_phone': len(phones) > 0,
                        'has_contact': len(contacts) > 0,
                        'has_crypto': len(crypto_addresses) > 0,
                        'has_bank_info': len(bank_info) > 0,
                        'has_suspicious_patterns': any(len(patterns) > 0 for patterns in suspicious_patterns.values()),
                        'augmentation_type': 'traditional_fallback'
                    }
                    
                    # 添加生成元数据
                    data_row.update(fallback_metadata.to_dict())
                    augmented_data.append(data_row)
        
        print(f"智能特征融合数据增强完成，共生成 {len(augmented_data)} 条增强数据")
        return pd.DataFrame(augmented_data)
    
    def generate_prompts_report(self, keyword_results: Dict, pattern_results: Dict) -> str:
        """
        生成prompt报告
        """
        report = "数据增强Prompt报告\n"
        report += "=" * 50 + "\n\n"
        
        for label in keyword_results.keys():
            if label in pattern_results:
                prompts = self.create_prompts_for_label(label, keyword_results, pattern_results)
                
                report += f"标签: {label}\n"
                report += "-" * 30 + "\n"
                
                for i, prompt in enumerate(prompts, 1):
                    report += f"{i}. {prompt}\n"
                
                report += "\n"
        
        return report
    
    def generate_enhanced_prompts_report(self, keyword_results: Dict, pattern_results: Dict) -> str:
        """
        生成增强版prompt报告 - 展示智能特征融合的优势
        """
        report = "智能特征融合数据增强报告\n"
        report += "=" * 60 + "\n\n"
        
        report += "系统改进概述:\n"
        report += "-" * 30 + "\n"
        report += "1. 充分利用所有提取特征 (100% vs 30%)\n"
        report += "2. 智能特征融合策略\n"
        report += "3. 多层次数据生成\n"
        report += "4. 简单有效的过滤机制\n"
        report += "5. 多样性保证机制\n\n"
        
        for label in keyword_results.keys():
            if label in pattern_results:
                # 生成综合prompt策略
                comprehensive_prompts = self.feature_fusion.create_comprehensive_prompts(
                    label, keyword_results, pattern_results
                )
                
                report += f"标签: {label}\n"
                report += "=" * 40 + "\n"
                
                # 特征利用统计
                keyword_data = keyword_results[label]
                pattern_data = pattern_results[label]
                
                report += "特征利用统计:\n"
                report += f"  TF-IDF关键词: {'✓' if keyword_data.get('tfidf_keywords') else '✗'}\n"
                report += f"  语义关键词: {'✓' if keyword_data.get('semantic_keywords') else '✗'}\n"
                report += f"  上下文关键词: {'✓' if keyword_data.get('contextual_keywords') else '✗'}\n"
                report += f"  综合关键词: {'✓' if keyword_data.get('combined_keywords') else '✗'}\n"
                report += f"  标签特定关键词: {'✓' if keyword_data.get('label_keywords') else '✗'}\n"
                report += f"  话术模式: {'✓' if pattern_data.get('pattern_frequency') else '✗'}\n"
                report += f"  结构化特征: {'✓' if pattern_data.get('structure_stats') else '✗'}\n"
                report += f"  语义实体: {'✓' if pattern_data.get('entity_stats') else '✗'}\n\n"
                
                # 生成的prompt策略
                report += f"生成的Prompt策略 (共{len(comprehensive_prompts)}种):\n"
                for i, prompt_info in enumerate(comprehensive_prompts, 1):
                    report += f"  {i}. {prompt_info['description']} (权重: {prompt_info['weight']:.2f})\n"
                    report += f"     Prompt: {prompt_info['prompt']}\n\n"
                
                report += "\n" + "=" * 60 + "\n\n"
        
        return report
    
    def generate_generation_tracking_report(self) -> str:
        """
        生成数据生成方式跟踪报告
        """
        return self.generation_tracker.get_generation_report()
    
    def get_generation_statistics(self) -> Dict[str, int]:
        """
        获取生成统计信息
        """
        return self.generation_tracker.get_generation_stats()
    
    def generate_fallback_analysis_report(self, augmented_data: pd.DataFrame = None) -> str:
        """
        生成回退分析报告
        """
        return self.generation_tracker.get_fallback_analysis_report(augmented_data)


if __name__ == "__main__":
    # 测试示例
    augmenter = DataAugmentation()
    
    # 测试模板生成
    rule = augmenter.augmentation_rules['按摩色诱'][0]
    generated = augmenter.generate_augmented_text(rule, 3)
    print("生成的文本:")
    for text in generated:
        print(f"  {text}")
    
    # 测试标签增强
    label_augmented = augmenter.generate_label_augmentations('按摩色诱', 5)
    print(f"\n标签增强样本:")
    for text in label_augmented:
        print(f"  {text}")
    
    # 测试规则增强
    test_text = "你好，我们这里有20岁美女按摩，200元2小时，包你满意"
    rule_augmented = augmenter.generate_rule_based_augmentations(test_text, '按摩色诱')
    print(f"\n规则增强样本:")
    for text in rule_augmented:
        print(f"  {text}")
