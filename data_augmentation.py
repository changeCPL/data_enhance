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

# 导入深度学习模块
try:
    from dl_text_generator import DLTextGenerator, GenerationConfig, PromptEngine
    DL_AVAILABLE = True
except ImportError:
    DL_AVAILABLE = False
    print("警告: 深度学习文本生成模块未找到，将使用传统方法")


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
        # 定义不同标签的数据增强规则
        self.augmentation_rules = {
            '按摩色诱': [
                AugmentationRule(
                    name="价格变化",
                    description="改变价格数字",
                    template="{greeting}，我们这里有{age}岁美女{service}，{price}元{time}，{promise}，{contact}",
                    variables=["greeting", "age", "service", "price", "time", "promise", "contact"],
                    examples=["你好，我们这里有20岁美女按摩，200元2小时，包你满意，加微信详聊"]
                ),
                AugmentationRule(
                    name="服务变化",
                    description="改变服务类型",
                    template="{greeting}，{platform}提供{service}，{price}，{quality}，{contact}",
                    variables=["greeting", "platform", "service", "price", "quality", "contact"],
                    examples=["您好，同城平台提供spa服务，价格优惠，质量保证，联系我"]
                )
            ],
            '博彩': [
                AugmentationRule(
                    name="收益变化",
                    description="改变收益数字",
                    template="{greeting}，{platform}，{promise}，{bonus}，{contact}",
                    variables=["greeting", "platform", "promise", "bonus", "contact"],
                    examples=["你好，正规平台，稳赚不赔，充值送50%，加微信"]
                ),
                AugmentationRule(
                    name="平台变化",
                    description="改变平台描述",
                    template="{greeting}，{platform}，{feature}，{guarantee}，{contact}",
                    variables=["greeting", "platform", "feature", "guarantee", "contact"],
                    examples=["您好，信誉平台，专业团队，秒到账，联系我"]
                )
            ],
            '兼职刷单': [
                AugmentationRule(
                    name="佣金变化",
                    description="改变佣金数字",
                    template="{greeting}，{job_type}，{commission}，{requirement}，{contact}",
                    variables=["greeting", "job_type", "commission", "requirement", "contact"],
                    examples=["你好，刷单兼职，50元一单，需要垫付，加微信"]
                ),
                AugmentationRule(
                    name="工作变化",
                    description="改变工作描述",
                    template="{greeting}，{job_type}，{advantage}，{time}，{contact}",
                    variables=["greeting", "job_type", "advantage", "time", "contact"],
                    examples=["您好，网络兼职，轻松赚钱，时间自由，联系我"]
                )
            ],
            '投资理财': [
                AugmentationRule(
                    name="收益变化",
                    description="改变收益率",
                    template="{greeting}，{product}，{return_rate}，{guarantee}，{contact}",
                    variables=["greeting", "product", "return_rate", "guarantee", "contact"],
                    examples=["你好，理财产品，年化20%，保本保息，加微信"]
                ),
                AugmentationRule(
                    name="产品变化",
                    description="改变产品类型",
                    template="{greeting}，{product}，{feature}，{urgency}，{contact}",
                    variables=["greeting", "product", "feature", "urgency", "contact"],
                    examples=["您好，基金投资，专业团队，限时优惠，联系我"]
                )
            ],
            '虚假客服': [
                AugmentationRule(
                    name="身份变化",
                    description="改变客服身份",
                    template="{greeting}，我是{identity}，{situation}，{action}，{contact}",
                    variables=["greeting", "identity", "situation", "action", "contact"],
                    examples=["你好，我是银行客服，账户异常，立即处理，按提示操作"]
                ),
                AugmentationRule(
                    name="情况变化",
                    description="改变紧急情况",
                    template="{greeting}，{identity}，{situation}，{requirement}，{contact}",
                    variables=["greeting", "identity", "situation", "requirement", "contact"],
                    examples=["您好，官方客服，资金风险，验证身份，点击链接"]
                )
            ]
        }
        
        # 变量替换词典
        self.variable_dict = {
            'greeting': ['你好', '您好', '亲', '亲爱的', 'Hi', 'Hello'],
            'age': ['18', '20', '22', '25', '28', '30'],
            'service': ['按摩', 'spa', '服务', '放松', '享受'],
            'price': ['100元', '200元', '300元', '500元', '800元'],
            'time': ['1小时', '2小时', '3小时', '半天', '包夜'],
            'promise': ['包你满意', '保证质量', '100%服务', '专业服务'],
            'contact': ['加微信详聊', '联系我', '电话咨询', 'QQ联系'],
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
    
    def generate_augmented_text(self, rule: AugmentationRule, num_samples: int = 5) -> List[str]:
        """
        根据规则生成增强文本
        """
        generated_texts = []
        
        for _ in range(num_samples):
            # 替换模板中的变量
            text = rule.template
            for variable in rule.variables:
                if variable in self.variable_dict:
                    replacement = random.choice(self.variable_dict[variable])
                    text = text.replace(f"{{{variable}}}", replacement)
            
            generated_texts.append(text)
        
        return generated_texts
    
    def generate_label_augmentations(self, label: str, num_samples: int = 20) -> List[str]:
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
            prompt = f"生成一段{label}相关的文本，必须包含以下关键词：{', '.join(top_keywords)}"
            prompts.append(prompt)
        
        # 基于模式的prompt
        if pattern_data['pattern_frequency']:
            top_patterns = list(pattern_data['pattern_frequency'].keys())[:3]
            prompt = f"生成一段{label}相关的文本，必须包含以下话术模式：{', '.join(top_patterns)}"
            prompts.append(prompt)
        
        # 基于结构的prompt
        structure_features = pattern_data['structure_stats']
        if 'has_numbers' in structure_features and structure_features['has_numbers'] > 0.5:
            prompt = f"生成一段{label}相关的文本，必须包含数字（价格、时间、百分比等）"
            prompts.append(prompt)
        
        if 'has_contact' in structure_features and structure_features['has_contact'] > 0.5:
            prompt = f"生成一段{label}相关的文本，必须包含联系方式（微信、QQ、电话等）"
            prompts.append(prompt)
        
        # 基于对话流程的prompt
        if pattern_data['conversation_flows']:
            top_flow = list(pattern_data['conversation_flows'].keys())[0]
            prompt = f"生成一段{label}相关的文本，对话流程应该包含：{top_flow}"
            prompts.append(prompt)
        
        return prompts
    
    def generate_rule_based_augmentations(self, original_text: str, label: str) -> List[str]:
        """
        基于规则的文本增强
        """
        augmented_texts = []
        
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
                        augmented_texts.append(new_text)
        
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
                augmented_texts.append(new_text)
        
        # 语序调整（简单的句子重排）
        sentences = re.split(r'[，。！？]', original_text)
        if len(sentences) > 1:
            random.shuffle(sentences)
            new_text = '，'.join(sentences).rstrip('，') + '。'
            if new_text != original_text:
                augmented_texts.append(new_text)
        
        return augmented_texts
    
    def generate_dl_augmentations(self, label: str, num_samples: int = 20, 
                                 keyword_results: Dict = None, pattern_results: Dict = None) -> List[str]:
        """使用深度学习模型生成增强文本"""
        if not self.use_dl or self.dl_generator is None:
            return self.generate_label_augmentations(label, num_samples)
        
        generated_texts = []
        
        try:
            # 使用智能Prompt引擎生成prompt
            prompts = self.prompt_engine.generate_prompts(label, min(5, num_samples // 4))
            
            for prompt in prompts:
                # 使用深度学习模型生成文本
                dl_generated = self.dl_generator.generate_text(prompt, num_return_sequences=3)
                generated_texts.extend(dl_generated)
                
                if len(generated_texts) >= num_samples:
                    break
            
            # 如果还需要更多样本，使用基于关键词和模式的prompt
            if len(generated_texts) < num_samples and keyword_results and pattern_results:
                additional_prompts = self._create_semantic_prompts(label, keyword_results, pattern_results)
                
                for prompt in additional_prompts:
                    if len(generated_texts) >= num_samples:
                        break
                    
                    dl_generated = self.dl_generator.generate_text(prompt, num_return_sequences=2)
                    generated_texts.extend(dl_generated)
            
            # 过滤和清理生成的文本
            filtered_texts = []
            for text in generated_texts:
                if len(text) > 10 and self._is_valid_generated_text(text, label):
                    filtered_texts.append(text)
            
            return filtered_texts[:num_samples]
            
        except Exception as e:
            print(f"深度学习增强失败: {e}")
            return self.generate_label_augmentations(label, num_samples)
    
    def _create_semantic_prompts(self, label: str, keyword_results: Dict, pattern_results: Dict) -> List[str]:
        """基于语义分析结果创建prompt"""
        prompts = []
        
        if label not in keyword_results or label not in pattern_results:
            return prompts
        
        keyword_data = keyword_results[label]
        pattern_data = pattern_results[label]
        
        # 基于关键词的prompt
        if keyword_data.get('semantic_keywords'):
            top_keywords = [kw[0] for kw in keyword_data['semantic_keywords'][:3]]
            prompt = f"生成一段{label}相关的文本，必须包含以下关键词：{', '.join(top_keywords)}"
            prompts.append(prompt)
        
        # 基于模式的prompt
        if pattern_data.get('pattern_frequency'):
            top_patterns = list(pattern_data['pattern_frequency'].keys())[:2]
            prompt = f"生成一段{label}相关的文本，必须包含以下话术模式：{', '.join(top_patterns)}"
            prompts.append(prompt)
        
        # 基于语义实体的prompt
        if pattern_data.get('entity_stats'):
            entity_prompts = []
            for entity_type, entities in pattern_data['entity_stats'].items():
                if entities:
                    top_entity = list(entities.keys())[0]
                    entity_prompts.append(f"包含{entity_type}：{top_entity}")
            
            if entity_prompts:
                prompt = f"生成一段{label}相关的文本，{', '.join(entity_prompts)}"
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
                    augmented_data.append({
                        'original_text': '',
                        'cleaned_text': text,
                        'text_length': len(text),
                        'label': label,
                        'data_source': 'dl_generated',
                        'urls': [],
                        'phone_numbers': [],
                        'contacts': [],
                        'has_url': False,
                        'has_phone': False,
                        'has_contact': False,
                        'augmentation_type': 'dl_generation'
                    })
                
                # 传统模板生成
                template_augmented = self.generate_label_augmentations(label, template_count)
                for text in template_augmented:
                    augmented_data.append({
                        'original_text': '',
                        'cleaned_text': text,
                        'text_length': len(text),
                        'label': label,
                        'data_source': 'augmented_template',
                        'urls': [],
                        'phone_numbers': [],
                        'contacts': [],
                        'has_url': False,
                        'has_phone': False,
                        'has_contact': False,
                        'augmentation_type': 'template'
                    })
            else:
                # 纯传统方法
                template_augmented = self.generate_label_augmentations(label, target_count)
                for text in template_augmented:
                    augmented_data.append({
                        'original_text': '',
                        'cleaned_text': text,
                        'text_length': len(text),
                        'label': label,
                        'data_source': 'augmented_template',
                        'urls': [],
                        'phone_numbers': [],
                        'contacts': [],
                        'has_url': False,
                        'has_phone': False,
                        'has_contact': False,
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
                        augmented_data.append({
                            'original_text': original_text,
                            'cleaned_text': text,
                            'text_length': len(text),
                            'label': label,
                            'data_source': 'augmented_rule',
                            'urls': [],
                            'phone_numbers': [],
                            'contacts': [],
                            'has_url': False,
                            'has_phone': False,
                            'has_contact': False,
                            'augmentation_type': 'rule'
                        })
        
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
