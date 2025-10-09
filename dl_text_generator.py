"""
基于深度学习的文本生成模块
使用GPT类模型和文本生成技术进行数据增强
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import (
    AutoTokenizer, AutoModelForCausalLM, AutoModel,
    GPT2LMHeadModel, GPT2Tokenizer,
    TextGenerationPipeline, pipeline
)
import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Optional, Union
import json
import random
import re
from dataclasses import dataclass
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')


@dataclass
class GenerationConfig:
    """文本生成配置"""
    model_name: str = "uer/gpt2-chinese-cluecorpussmall"
    max_length: int = 50
    temperature: float = 0.8
    top_p: float = 0.9
    top_k: int = 50
    repetition_penalty: float = 1.1
    do_sample: bool = True
    num_return_sequences: int = 3
    batch_size: int = 8
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


class DLTextGenerator:
    """基于深度学习的文本生成器"""
    
    def __init__(self, config: GenerationConfig = None):
        self.config = config or GenerationConfig()
        self.tokenizer = None
        self.model = None
        self.generation_pipeline = None
        
        # 支持的模型列表
        self.supported_models = {
            'gpt2-chinese': 'uer/gpt2-chinese-cluecorpussmall',
            'gpt2-chinese-large': 'uer/gpt2-chinese-cluecorpussmall',
            'chinese-gpt': 'uer/gpt2-chinese-cluecorpussmall',
            'bert-base': 'bert-base-chinese',
            'roberta': 'hfl/chinese-roberta-wwm-ext'
        }
        
        self._initialize_model()
    
    def _initialize_model(self):
        """初始化生成模型"""
        try:
            model_name = self.supported_models.get(self.config.model_name, self.config.model_name)
            
            # 尝试加载GPT类模型
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(model_name)
                self.model = AutoModelForCausalLM.from_pretrained(model_name)
                self.model.to(self.config.device)
                self.model.eval()
                
                # 创建生成管道
                self.generation_pipeline = pipeline(
                    "text-generation",
                    model=self.model,
                    tokenizer=self.tokenizer,
                    device=0 if self.config.device == "cuda" else -1
                )
                print(f"GPT模型加载成功: {model_name}")
            except:
                # 回退到基础模型
                self._fallback_model()
                
        except Exception as e:
            print(f"模型初始化失败: {e}")
            self._fallback_model()
    
    def _fallback_model(self):
        """回退到基础模型"""
        try:
            self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
            self.model = GPT2LMHeadModel.from_pretrained('gpt2')
            self.model.to(self.config.device)
            self.model.eval()
            print("使用回退模型: gpt2")
        except Exception as e:
            print(f"回退模型也失败: {e}")
            self.model = None
    
    def generate_text(self, prompt: str, **kwargs) -> List[str]:
        """生成文本"""
        if self.model is None:
            return self._rule_based_generation(prompt)
        
        # 合并配置参数
        generation_kwargs = {
            'max_length': self.config.max_length,
            'temperature': self.config.temperature,
            'top_p': self.config.top_p,
            'top_k': self.config.top_k,
            'repetition_penalty': self.config.repetition_penalty,
            'do_sample': self.config.do_sample,
            'num_return_sequences': self.config.num_return_sequences,
            'pad_token_id': self.tokenizer.eos_token_id
        }
        generation_kwargs.update(kwargs)
        
        try:
            if self.generation_pipeline:
                # 使用管道生成
                results = self.generation_pipeline(
                    prompt,
                    **generation_kwargs
                )
                generated_texts = [result['generated_text'] for result in results]
            else:
                # 直接使用模型生成
                generated_texts = self._generate_with_model(prompt, generation_kwargs)
            
            # 后处理生成的文本
            processed_texts = [self._post_process_text(text, prompt) for text in generated_texts]
            
            return processed_texts
            
        except Exception as e:
            print(f"文本生成失败: {e}")
            return self._rule_based_generation(prompt)
    
    def _generate_with_model(self, prompt: str, generation_kwargs: Dict) -> List[str]:
        """使用模型直接生成文本"""
        inputs = self.tokenizer.encode(prompt, return_tensors='pt').to(self.config.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                inputs,
                **generation_kwargs
            )
        
        generated_texts = []
        for output in outputs:
            text = self.tokenizer.decode(output, skip_special_tokens=True)
            generated_texts.append(text)
        
        return generated_texts
    
    def _post_process_text(self, text: str, original_prompt: str) -> str:
        """后处理生成的文本"""
        # 移除原始prompt
        if text.startswith(original_prompt):
            text = text[len(original_prompt):].strip()
        
        # 清理文本
        text = re.sub(r'\s+', ' ', text)  # 合并多个空格
        text = text.strip()
        
        # 移除重复的句子
        sentences = re.split(r'[。！？]', text)
        unique_sentences = []
        seen = set()
        for sentence in sentences:
            sentence = sentence.strip()
            if sentence and sentence not in seen:
                unique_sentences.append(sentence)
                seen.add(sentence)
        
        text = '。'.join(unique_sentences)
        if text and not text.endswith(('。', '！', '？')):
            text += '。'
        
        return text
    
    def _rule_based_generation(self, prompt: str) -> List[str]:
        """基于规则的文本生成（备用方案）"""
        # 简单的模板替换
        templates = {
            '按摩': [
                "我们这里有{age}岁美女{service}，{price}元{time}，{promise}，{contact}",
                "{greeting}，{platform}提供{service}，{price}，{quality}，{contact}",
                "同城{service}，{age}岁妹妹，{price}，{promise}，{contact}"
            ],
            '博彩': [
                "{greeting}，{platform}，{promise}，{bonus}，{contact}",
                "欢迎来到{platform}，{feature}，{guarantee}，{contact}",
                "{platform}，{promise}，{bonus}，{guarantee}，{contact}"
            ],
            '兼职': [
                "{greeting}，{job_type}，{commission}，{requirement}，{contact}",
                "{job_type}，{advantage}，{time}，{contact}",
                "网络{job_type}，{commission}，{requirement}，{contact}"
            ],
            '投资': [
                "{greeting}，{product}，{return_rate}，{guarantee}，{contact}",
                "{product}，{feature}，{urgency}，{contact}",
                "专业{product}，{return_rate}，{guarantee}，{contact}"
            ],
            '客服': [
                "{greeting}，我是{identity}，{situation}，{action}，{contact}",
                "{identity}，{situation}，{requirement}，{contact}",
                "系统通知，{situation}，{action}，{contact}"
            ]
        }
        
        # 变量替换词典
        variables = {
            'greeting': ['你好', '您好', '亲', 'Hi'],
            'age': ['18', '20', '22', '25', '28'],
            'service': ['按摩', 'spa', '服务', '放松'],
            'price': ['100元', '200元', '300元', '500元'],
            'time': ['1小时', '2小时', '3小时', '半天'],
            'promise': ['包你满意', '保证质量', '100%服务'],
            'contact': ['加微信', '联系我', '电话咨询'],
            'platform': ['同城平台', '正规平台', '信誉平台'],
            'quality': ['质量保证', '服务一流', '专业团队'],
            'bonus': ['充值送50%', '首充100%', '新用户优惠'],
            'guarantee': ['秒到账', '快速提现', '24小时到账'],
            'job_type': ['刷单兼职', '网络兼职', '任务兼职'],
            'commission': ['50元一单', '20%佣金', '日赚500'],
            'requirement': ['需要垫付', '先垫付100元', '准备本金'],
            'advantage': ['轻松赚钱', '简单操作', '在家兼职'],
            'product': ['理财产品', '基金投资', '股票投资'],
            'return_rate': ['年化20%', '收益30%', '回报50%'],
            'feature': ['专业团队', '资深分析师', '金牌导师'],
            'urgency': ['限时优惠', '仅限3天', '错过后悔'],
            'identity': ['银行客服', '官方客服', '系统通知'],
            'situation': ['账户异常', '资金风险', '系统升级'],
            'action': ['立即处理', '马上操作', '按提示'],
        }
        
        # 根据prompt选择模板
        selected_templates = []
        for key, template_list in templates.items():
            if key in prompt:
                selected_templates.extend(template_list)
        
        if not selected_templates:
            selected_templates = list(templates.values())[0]  # 默认使用第一个
        
        generated_texts = []
        for _ in range(self.config.num_return_sequences):
            template = random.choice(selected_templates)
            text = template
            
            # 替换变量
            for var, values in variables.items():
                if f"{{{var}}}" in text:
                    text = text.replace(f"{{{var}}}", random.choice(values))
            
            generated_texts.append(text)
        
        return generated_texts
    
    def generate_augmented_dataset(self, df: pd.DataFrame, augmentation_ratio: float = 0.5) -> pd.DataFrame:
        """生成增强数据集"""
        print("开始生成深度学习增强数据...")
        
        augmented_data = []
        
        for label in df['label'].unique():
            if pd.isna(label) or label == '':
                continue
            
            label_data = df[df['label'] == label]
            original_count = len(label_data)
            target_count = int(original_count * augmentation_ratio)
            
            print(f"为标签 '{label}' 生成 {target_count} 条增强数据...")
            
            # 生成prompt
            prompts = self._create_prompts_for_label(label, label_data)
            
            generated_count = 0
            for prompt in prompts:
                if generated_count >= target_count:
                    break
                
                # 生成文本
                generated_texts = self.generate_text(prompt)
                
                for text in generated_texts:
                    if generated_count >= target_count:
                        break
                    
                    if len(text) > 10:  # 过滤太短的文本
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
                            'augmentation_type': 'dl_generation',
                            'generation_prompt': prompt
                        })
                        generated_count += 1
        
        return pd.DataFrame(augmented_data)
    
    def _create_prompts_for_label(self, label: str, label_data: pd.DataFrame) -> List[str]:
        """为特定标签创建生成prompt"""
        prompts = []
        
        # 基于标签的prompt模板
        label_prompts = {
            '按摩色诱': [
                "同城交友平台提供",
                "美女按摩服务，",
                "spa会所，",
                "约会服务，",
                "放松按摩，"
            ],
            '博彩': [
                "正规博彩平台，",
                "游戏平台，",
                "彩票网站，",
                "娱乐平台，",
                "投注网站，"
            ],
            '兼职刷单': [
                "网络兼职，",
                "刷单任务，",
                "佣金兼职，",
                "任务平台，",
                "返利兼职，"
            ],
            '投资理财': [
                "理财产品，",
                "基金投资，",
                "股票投资，",
                "外汇投资，",
                "数字货币，"
            ],
            '虚假客服': [
                "银行客服通知，",
                "官方客服，",
                "系统通知，",
                "客服代表，",
                "安全检测，"
            ]
        }
        
        # 获取标签特定的prompt
        base_prompts = label_prompts.get(label, ["这是一个"])
        
        # 从实际数据中提取关键词作为prompt
        texts = label_data['cleaned_text'].tolist()
        common_words = self._extract_common_words(texts)
        
        # 组合prompt
        for base_prompt in base_prompts:
            for word in common_words[:3]:  # 使用前3个常见词
                prompt = base_prompt + word + "，"
                prompts.append(prompt)
        
        return prompts[:10]  # 限制prompt数量
    
    def _extract_common_words(self, texts: List[str]) -> List[str]:
        """提取常见词汇"""
        import jieba
        from collections import Counter
        
        all_words = []
        for text in texts:
            words = jieba.lcut(text)
            # 过滤停用词和短词
            words = [w for w in words if len(w) > 1 and w not in ['的', '了', '在', '是', '我', '有', '和', '就', '不', '人', '都', '一', '一个', '上', '也', '很', '到', '说', '要', '去', '你', '会', '着', '没有', '看', '好', '自己', '这']]
            all_words.extend(words)
        
        word_counts = Counter(all_words)
        return [word for word, count in word_counts.most_common(10)]
    
    def generate_paraphrases(self, text: str, num_paraphrases: int = 3) -> List[str]:
        """生成文本的释义版本"""
        # 简单的释义策略
        paraphrases = []
        
        # 同义词替换
        synonyms = {
            '你好': ['您好', 'Hi', 'Hello'],
            '美女': ['妹妹', '小姐姐', '姑娘'],
            '按摩': ['spa', '放松', '服务'],
            '赚钱': ['收益', '获利', '盈利'],
            '平台': ['网站', 'app', '软件'],
            '客服': ['服务', '售后', '支持'],
            '投资': ['理财', '投资', '理财'],
            '兼职': ['工作', '任务', '项目'],
            '博彩': ['游戏', '娱乐', '投注']
        }
        
        for _ in range(num_paraphrases):
            paraphrase = text
            for original, replacements in synonyms.items():
                if original in paraphrase:
                    paraphrase = paraphrase.replace(original, random.choice(replacements))
            paraphrases.append(paraphrase)
        
        return paraphrases
    
    def generate_variations(self, text: str, num_variations: int = 3) -> List[str]:
        """生成文本变体"""
        variations = []
        
        # 数字变化
        numbers = re.findall(r'[0-9]+', text)
        for _ in range(num_variations):
            variation = text
            for number in numbers:
                # 随机变化数字
                change = random.uniform(0.1, 0.5)
                if random.random() > 0.5:
                    new_number = str(int(int(number) * (1 + change)))
                else:
                    new_number = str(int(int(number) * (1 - change)))
                variation = variation.replace(number, new_number)
            variations.append(variation)
        
        return variations
    
    def evaluate_generation_quality(self, original_texts: List[str], generated_texts: List[str]) -> Dict:
        """评估生成质量"""
        from deep_learning_classifier import FraudTextEmbedder
        
        # 使用语义相似度评估
        embedder = FraudTextEmbedder()
        
        # 计算语义相似度
        all_texts = original_texts + generated_texts
        embeddings = embedder.encode_texts(all_texts)
        
        original_embeddings = embeddings[:len(original_texts)]
        generated_embeddings = embeddings[len(original_texts):]
        
        # 计算平均相似度
        similarities = []
        for gen_emb in generated_embeddings:
            max_sim = 0
            for orig_emb in original_embeddings:
                sim = cosine_similarity([gen_emb], [orig_emb])[0][0]
                max_sim = max(max_sim, sim)
            similarities.append(max_sim)
        
        # 计算多样性（生成文本之间的相似度）
        diversity_scores = []
        for i in range(len(generated_embeddings)):
            for j in range(i + 1, len(generated_embeddings)):
                sim = cosine_similarity([generated_embeddings[i]], [generated_embeddings[j]])[0][0]
                diversity_scores.append(1 - sim)  # 多样性 = 1 - 相似度
        
        return {
            'avg_similarity': np.mean(similarities),
            'avg_diversity': np.mean(diversity_scores) if diversity_scores else 0,
            'similarity_std': np.std(similarities),
            'diversity_std': np.std(diversity_scores) if diversity_scores else 0
        }


class PromptEngine:
    """智能Prompt引擎"""
    
    def __init__(self):
        self.prompt_templates = {
            '按摩色诱': [
                "生成一段{label}相关的文本，包含以下元素：{elements}",
                "写一个{label}的广告文案，要求：{requirements}",
                "创作{label}相关的对话，包含：{features}"
            ],
            '博彩': [
                "生成{label}平台的推广文案，包含：{elements}",
                "写一个{label}的营销文本，要求：{requirements}",
                "创作{label}相关的宣传内容，包含：{features}"
            ],
            '兼职刷单': [
                "生成{label}的招聘信息，包含：{elements}",
                "写一个{label}的广告，要求：{requirements}",
                "创作{label}相关的宣传文案，包含：{features}"
            ],
            '投资理财': [
                "生成{label}产品的介绍，包含：{elements}",
                "写一个{label}的推广文案，要求：{requirements}",
                "创作{label}相关的营销内容，包含：{features}"
            ],
            '虚假客服': [
                "生成{label}的通知信息，包含：{elements}",
                "写一个{label}的警告，要求：{requirements}",
                "创作{label}相关的通知，包含：{features}"
            ]
        }
        
        self.elements = {
            '按摩色诱': ['年龄描述', '价格信息', '服务承诺', '联系方式'],
            '博彩': ['收益承诺', '平台推广', '充值优惠', '提现保证'],
            '兼职刷单': ['佣金承诺', '操作简单', '垫付要求', '时间灵活'],
            '投资理财': ['高收益', '保本承诺', '专业团队', '限时优惠'],
            '虚假客服': ['身份伪装', '紧急情况', '验证要求', '操作指导']
        }
        
        self.requirements = {
            '按摩色诱': ['吸引用户', '价格合理', '服务专业', '联系方便'],
            '博彩': ['收益诱人', '平台可信', '操作简单', '提现快速'],
            '兼职刷单': ['佣金丰厚', '操作简单', '时间自由', '门槛较低'],
            '投资理财': ['收益稳定', '风险较低', '专业指导', '机会难得'],
            '虚假客服': ['身份可信', '情况紧急', '操作简单', '信息真实']
        }
        
        self.features = {
            '按摩色诱': ['问候语', '服务介绍', '价格说明', '联系方式'],
            '博彩': ['平台介绍', '收益说明', '优惠活动', '联系方式'],
            '兼职刷单': ['工作介绍', '佣金说明', '要求说明', '联系方式'],
            '投资理财': ['产品介绍', '收益说明', '风险说明', '联系方式'],
            '虚假客服': ['身份说明', '情况说明', '操作要求', '联系方式']
        }
    
    def generate_prompts(self, label: str, num_prompts: int = 5) -> List[str]:
        """生成智能prompt"""
        if label not in self.prompt_templates:
            return [f"生成一段{label}相关的文本"]
        
        templates = self.prompt_templates[label]
        elements = self.elements[label]
        requirements = self.requirements[label]
        features = self.features[label]
        
        prompts = []
        for i in range(num_prompts):
            template = random.choice(templates)
            prompt = template.format(
                label=label,
                elements='、'.join(random.sample(elements, 2)),
                requirements='、'.join(random.sample(requirements, 2)),
                features='、'.join(random.sample(features, 2))
            )
            prompts.append(prompt)
        
        return prompts


if __name__ == "__main__":
    # 测试示例
    print("深度学习文本生成器测试")
    
    # 创建测试数据
    test_data = [
        {"cleaned_text": "你好，我们这里有20岁美女按摩服务，200元2小时，包你满意", "label": "按摩色诱"},
        {"cleaned_text": "欢迎来到正规博彩平台，稳赚不赔，充值送50%", "label": "博彩"},
        {"cleaned_text": "兼职刷单，50元一单，简单操作，需要垫付", "label": "兼职刷单"},
    ]
    
    df = pd.DataFrame(test_data)
    
    # 测试文本生成器
    config = GenerationConfig(num_return_sequences=2)
    generator = DLTextGenerator(config)
    
    # 测试单文本生成
    prompt = "同城交友平台提供"
    generated = generator.generate_text(prompt)
    print(f"生成文本: {generated}")
    
    # 测试释义生成
    test_text = "你好，我们这里有美女按摩服务"
    paraphrases = generator.generate_paraphrases(test_text, 2)
    print(f"释义文本: {paraphrases}")
    
    # 测试变体生成
    variations = generator.generate_variations(test_text, 2)
    print(f"变体文本: {variations}")
    
    # 测试增强数据集生成
    augmented_df = generator.generate_augmented_dataset(df, 0.5)
    print(f"增强数据集: {len(augmented_df)} 条")
    
    # 测试Prompt引擎
    prompt_engine = PromptEngine()
    prompts = prompt_engine.generate_prompts("按摩色诱", 3)
    print(f"智能Prompt: {prompts}")
