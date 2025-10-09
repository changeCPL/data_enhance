"""
话术结构分析模块
负责识别和分析不同标签的套路模式、话术结构
"""

import re
import pandas as pd
from collections import Counter, defaultdict
from typing import List, Dict, Tuple, Set
import numpy as np
from dataclasses import dataclass


@dataclass
class PatternTemplate:
    """话术模式模板"""
    name: str
    pattern: str
    description: str
    examples: List[str]
    frequency: int = 0


class PatternAnalyzer:
    def __init__(self):
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
        分析特定标签的模式
        """
        label_data = df[df['label'] == label]
        if len(label_data) == 0:
            return {}
        
        # 获取该标签的模板
        templates = self.pattern_templates.get(label, []) + self.common_patterns
        
        # 统计模式匹配
        pattern_stats = defaultdict(int)
        pattern_examples = defaultdict(list)
        structure_features = []
        conversation_flows = []
        
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
        
        return {
            'pattern_frequency': dict(pattern_stats),
            'pattern_examples': dict(pattern_examples),
            'structure_stats': structure_stats,
            'conversation_flows': dict(flow_patterns),
            'sample_count': len(label_data)
        }
    
    def generate_pattern_report(self, pattern_results: Dict[str, Dict]) -> str:
        """
        生成模式分析报告
        """
        report = "话术结构分析报告\n"
        report += "=" * 50 + "\n\n"
        
        for label, data in pattern_results.items():
            report += f"标签: {label}\n"
            report += f"样本数量: {data['sample_count']}\n"
            report += "-" * 30 + "\n"
            
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
            
            report += "\n" + "=" * 50 + "\n\n"
        
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
