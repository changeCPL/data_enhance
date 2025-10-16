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
    def __init__(self):
        # 定义不同标签的话术模式模板（合并优化版）
        self.pattern_templates = {
            '按摩色诱': [
                PatternTemplate(
                    name="价格时间模式",
                    pattern=r"[0-9]+元.*[0-9]+小时|[0-9]+小时.*[0-9]+元|[0-9]+元|[0-9]+小时",
                    description="价格和时间相关模式",
                    examples=["200元2小时", "300元3小时包夜", "200元", "2小时"]
                ),
                PatternTemplate(
                    name="年龄描述模式",
                    pattern=r"[0-9]+岁.*美女|[0-9]+岁.*妹妹|[0-9]+岁",
                    description="年龄描述相关模式",
                    examples=["20岁美女", "18岁妹妹", "20岁", "25岁"]
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
                    name="收益承诺模式",
                    pattern=r"稳赚.*[0-9]+%|日赚.*[0-9]+|包中.*[0-9]+倍|[0-9]+倍|[0-9]+%|稳赚|包中",
                    description="收益和保证相关模式",
                    examples=["稳赚50%", "日赚1000", "包中10倍", "10倍", "50%", "稳赚", "包中"]
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
                    name="佣金收益模式",
                    pattern=r"[0-9]+元.*单|[0-9]+%.*佣金|日赚.*[0-9]+|[0-9]+元/单|日赚[0-9]+|佣金[0-9]+%",
                    description="佣金和收益相关模式",
                    examples=["50元一单", "20%佣金", "日赚500", "50元/单", "日赚100", "佣金10%"]
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
                    name="收益承诺模式",
                    pattern=r"年化.*[0-9]+%|收益.*[0-9]+%|回报.*[0-9]+倍|年化[0-9]+%|收益[0-9]+%",
                    description="收益相关模式",
                    examples=["年化20%", "收益30%", "年化10%", "收益20%"]
                ),
                PatternTemplate(
                    name="保本承诺模式",
                    pattern=r"保本.*保息|零风险|稳赚.*不赔|保本",
                    description="保本相关模式",
                    examples=["保本保息", "零风险", "保本", "保本投资"]
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
                    name="验证要求模式",
                    pattern=r"验证.*身份|提供.*密码|确认.*信息|验证码|银行卡|身份证|密码",
                    description="验证和信息要求相关模式",
                    examples=["验证身份", "提供密码", "验证码", "银行卡", "身份证", "密码"]
                ),
                PatternTemplate(
                    name="操作指导",
                    pattern=r"按.*操作|点击.*链接|输入.*验证码",
                    description="指导用户操作",
                    examples=["按提示操作", "点击链接"]
                )
            ]
        }
        
        # 通用话术结构模式（针对网页OCR文本优化）
        self.common_patterns = [
            PatternTemplate(
                name="网页行动号召",
                pattern=r"立即|马上|现在|点击|访问|注册|下载|了解更多|立即参与|马上开始",
                description="网页行动号召词汇",
                examples=["立即参与", "点击访问", "马上注册", "了解更多"]
            ),
            PatternTemplate(
                name="数字强调",
                pattern=r"[0-9]+[元%倍小时天]|¥[0-9]+|\$[0-9]+|[0-9]+%|[0-9]+倍",
                description="使用数字和符号强调",
                examples=["¥100", "$50", "50%", "3倍", "100元"]
            ),
            PatternTemplate(
                name="网页营销词汇",
                pattern=r"免费|优惠|限时|独家|专属|特价|今日|限时优惠|今日特价|限时抢购",
                description="网页营销诱导词汇",
                examples=["免费", "限时优惠", "今日特价", "限时抢购"]
            ),
            PatternTemplate(
                name="网页操作词汇",
                pattern=r"点击|访问|注册|登录|下载|安装|申请|提交|确认|验证",
                description="网页操作相关词汇",
                examples=["点击注册", "立即访问", "免费下载", "在线申请"]
            ),
            PatternTemplate(
                name="网页标题格式",
                pattern=r"【.*】|《.*》|.*-.*|.*_.*|.*：.*",
                description="网页标题常见格式",
                examples=["【限时优惠】", "《投资理财》", "高收益-低风险", "专业团队：资深分析师"]
            )
        ]

    def _is_valid_phone(self, phone: str) -> bool:
        """验证电话号码是否有效"""
        # 清理格式
        clean_phone = re.sub(r'[-.\s+]', '', phone)
        
        # 长度检查
        if len(clean_phone) < 7 or len(clean_phone) > 15:
            return False
        
        # 排除常见的非电话号码
        invalid_patterns = [
            r'^0+$',  # 全零
            r'^1+$',  # 全一
            r'^123456',  # 连续数字
            r'^111111',  # 重复数字
        ]
        
        for pattern in invalid_patterns:
            if re.match(pattern, clean_phone):
                return False
        
        return True
    
    def _clean_contact(self, contact: str) -> str:
        """
        清理和标准化联系方式
        """
        # 移除多余的空格和标点
        contact = re.sub(r'[：:\s]+', ':', contact)
        contact = contact.strip()
        
        # 标准化格式
        if contact.startswith('@'):
            return contact
        elif ':' in contact:
            parts = contact.split(':', 1)
            if len(parts) == 2:
                platform, username = parts
                platform = platform.strip().lower()
                username = username.strip()
                
                # 标准化平台名称
                platform_map = {
                    'wx': '微信',
                    'wechat': '微信',
                    '微': '微信',
                    'qq': 'QQ',
                    '扣扣': 'QQ',
                    '企鹅': 'QQ',
                    'tg': 'Telegram',
                    'telegram': 'Telegram',
                    '电报': 'Telegram',
                    'twitter': 'Twitter',
                    '推特': 'Twitter',
                    'x': 'X',
                    'ins': 'Instagram',
                    'instagram': 'Instagram',
                    'fb': 'Facebook',
                    'facebook': 'Facebook',
                    'wa': 'WhatsApp',
                    'whatsapp': 'WhatsApp',
                }
                
                platform = platform_map.get(platform, platform)
                return f"{platform}:{username}"
        
        return contact
    
    def _categorize_suspicious_patterns(self, patterns: List[str]) -> Dict[str, List[str]]:
        """将可疑模式按类别分组"""
        category_keywords = {
            'money_related': ['赚钱', '发财', '投资', '理财', '收益', '回报', '分红', '中奖', '奖金', '佣金', '返利', '稳赚', '包中'],
            'urgency_related': ['紧急', '急', '快', '立即', '马上', '现在', '今天', '限时', '过期', '最后', '机会'],
            'authority_related': ['官方', '政府', '银行', '警察', '法院', '税务局', '工商局', '公安', '检察院'],
            'threat_related': ['冻结', '封号', '删除', '注销', '起诉', '逮捕', '通缉', '违法', '犯罪'],
            'privacy_related': ['验证码', '密码', '身份证', '银行卡', '账号', '个人信息'],
            'scam_indicators': ['刷单', '兼职', '垫付', '先付', '保证金', '手续费', '解冻费']
        }
        
        results = {}
        for category, keywords in category_keywords.items():
            matches = [pattern for pattern in patterns if pattern in keywords]
            results[category] = list(set(matches))
        
        return results
    
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
    
    def analyze_text_comprehensive(self, text: str) -> Dict[str, any]:
        """
        综合分析文本结构和实体信息（合并原 analyze_text_structure 和 extract_semantic_entities）
        避免重复的正则表达式匹配，提高效率
        """
        # 基础结构特征
        features = {
            'length': len(text),
            'sentence_count': len(re.split(r'[。！？]', text)),
        }
        
        # 定义所有正则表达式模式（复用 extract_semantic_entities 的模式）
        patterns = {
            # 电话号码模式
            'phone': r'1[3-9]\d{9}|\d{3,4}[-.\s]?\d{7,8}|\+?[1-9]\d{1,3}[-.\s]?\d{1,4}[-.\s]?\d{1,4}[-.\s]?\d{1,9}|\+86[-.\s]?1[3-9]\d{9}|\+1[-.\s]?\d{3}[-.\s]?\d{3}[-.\s]?\d{4}|\+44[-.\s]?\d{2,4}[-.\s]?\d{3,4}[-.\s]?\d{3,4}|\d{3,4}[-.\s]?\d{3,4}[-.\s]?\d{3,4}|(?<!\d)(?<![a-zA-Z])\d{7,11}(?!\d)(?![a-zA-Z])',
            
            # 联系方式模式
            'contact': r'(微信|加微信|微信号|wx|wechat|微|QQ|qq|扣扣|企鹅|telegram|tg|电报|twitter|推特|x|instagram|ins|facebook|fb|line|whatsapp|wa|联系|加我|找我|薇信|威信|微星|扣扣号|企鹅号)[：:\s]*([a-zA-Z0-9_@.-]+|\d+)|@[a-zA-Z0-9_]+',
            
            # 加密货币地址模式
            'crypto': r'[13][a-km-zA-HJ-NP-Z1-9]{25,34}|bc1[a-z0-9]{39,59}|0x[a-fA-F0-9]{40}|[LM3][a-km-zA-HJ-NP-Z1-9]{26,33}|X[1-9A-HJ-NP-Za-km-z]{25,34}|[A-Za-z0-9]{26,35}',
            
            # 银行信息模式
            'bank': r'\d{16,19}|工商银行|建设银行|农业银行|中国银行|交通银行|招商银行|浦发银行|中信银行|光大银行|华夏银行|民生银行|广发银行|平安银行|兴业银行|邮储银行|ICBC|CCB|ABC|BOC|BOCOM|CMB|SPDB|CITIC|CEB|HXB|CMBC|CGB|PAB|CIB|PSBC',
            
            # 可疑模式
            'suspicious': r'赚钱|发财|投资|理财|收益|回报|分红|中奖|奖金|佣金|返利|稳赚|包中|紧急|急|快|立即|马上|现在|今天|限时|过期|最后|机会|官方|政府|银行|警察|法院|税务局|工商局|公安|检察院|冻结|封号|删除|注销|起诉|逮捕|通缉|违法|犯罪|验证码|密码|身份证|银行卡|账号|个人信息|刷单|兼职|垫付|先付|保证金|手续费|解冻费',
            
            # 数字相关模式
            'numeric': r'[0-9]+岁|[0-9]+元|[0-9]+块|[0-9]+万|[0-9]+千|[0-9]+点|[0-9]+小时|[0-9]+分钟|[0-9]+%',
            
            # URL模式
            'url': r'https?://[^\s]+|www\.[^\s]+|[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}',
        }
        
        # 提取所有实体
        entities = {}
        for entity_type, pattern in patterns.items():
            matches = re.findall(pattern, text)
            if entity_type == 'contact':
                # 处理联系方式匹配结果
                contacts = []
                for match in matches:
                    if isinstance(match, tuple):
                        platform, username = match
                        if platform and username:
                            contact = f"{platform}:{username}"
                            cleaned_contact = self._clean_contact(contact)
                            if cleaned_contact and len(cleaned_contact) > 1:
                                contacts.append(cleaned_contact)
                    else:
                        cleaned_contact = self._clean_contact(match)
                        if cleaned_contact and len(cleaned_contact) > 1:
                            contacts.append(cleaned_contact)
                entities[entity_type] = list(set(contacts))  # 去重
            else:
                entities[entity_type] = matches
        
        # 分类整理结果
        structured_entities = {
            'ages': [m for m in entities.get('numeric', []) if '岁' in m],
            'money': [m for m in entities.get('numeric', []) if any(unit in m for unit in ['元', '块', '万', '千'])],
            'time': [m for m in entities.get('numeric', []) if any(unit in m for unit in ['点', '小时', '分钟'])],
            'percentages': [m for m in entities.get('numeric', []) if '%' in m],
            'phone_numbers': [m for m in entities.get('phone', []) if self._is_valid_phone(m)],
            'urls': entities.get('url', []),
            'crypto_addresses': entities.get('crypto', []),
            'bank_cards': [m for m in entities.get('bank', []) if re.match(r'\d{16,19}', m)],
            'bank_names': [m for m in entities.get('bank', []) if not re.match(r'\d{16,19}', m)],
            'contacts': entities.get('contact', []),
            'suspicious_patterns': self._categorize_suspicious_patterns(entities.get('suspicious', [])),
        }
        
        # 基于实体提取结果计算布尔特征（避免重复正则匹配）
        features.update({
            'has_numbers': len(entities.get('numeric', [])) > 0,
            'has_money': len(structured_entities['money']) > 0,
            'has_age': len(structured_entities['ages']) > 0,
            'has_percentage': len(structured_entities['percentages']) > 0,
            'has_time': len(structured_entities['time']) > 0,
            'has_contact': len(structured_entities['contacts']) > 0,
            'has_phone': len(structured_entities['phone_numbers']) > 0,
            'has_url': len(structured_entities['urls']) > 0,
            'has_crypto': len(structured_entities['crypto_addresses']) > 0,
            'has_bank_info': len(structured_entities['bank_cards']) > 0 or len(structured_entities['bank_names']) > 0,
            'has_suspicious_patterns': any(len(patterns) > 0 for patterns in structured_entities['suspicious_patterns'].values()),
        })
        
        # 添加更细粒度的可疑模式检测
        suspicious_cats = structured_entities['suspicious_patterns']
        features.update({
            'has_urgency': len(suspicious_cats.get('urgency_related', [])) > 0,
            'has_promise': len(suspicious_cats.get('money_related', [])) > 0,
        })
        
        return {
            'structure_features': features,
            'semantic_entities': structured_entities
        }
    
    def analyze_text_structure(self, text: str) -> Dict[str, any]:
        """
        分析文本结构特征（保持向后兼容，内部调用 analyze_text_comprehensive）
        """
        result = self.analyze_text_comprehensive(text)
        return result['structure_features']
    
    def extract_semantic_entities(self, text: str) -> Dict[str, List[str]]:
        """
        提取语义实体（保持向后兼容，内部调用 analyze_text_comprehensive）
        """
        result = self.analyze_text_comprehensive(text)
        return result['semantic_entities']
    
    def analyze_label_patterns(self, df: pd.DataFrame, label: str) -> Dict[str, any]:
        """
        分析特定标签的模式
        重构后同时提取结构化信息并更新DataFrame
        """
        label_data = df[df['label'] == label]
        if len(label_data) == 0:
            return {}
        
        texts = label_data['cleaned_text'].tolist()
        
        # 获取该标签的模板（已合并所有模式）
        templates = self.pattern_templates.get(label, []) + self.common_patterns
        
        # 统计模式匹配
        pattern_stats = defaultdict(int)
        pattern_examples = defaultdict(list)
        structure_features = []
        semantic_entities = []
        
        # 为DataFrame添加结构化信息列
        structured_data = []
        
        for _, row in label_data.iterrows():
            text = row['cleaned_text']
            
            # 匹配模式
            matches = self.match_patterns(text, templates)
            for template, match in matches:
                pattern_stats[template.name] += 1
                if len(pattern_examples[template.name]) < 5:  # 最多保存5个例子
                    pattern_examples[template.name].append(match)
            
            # 使用统一的分析方法（避免重复正则匹配）
            comprehensive_result = self.analyze_text_comprehensive(text)
            features = comprehensive_result['structure_features']
            entities = comprehensive_result['semantic_entities']
            
            structure_features.append(features)
            semantic_entities.append(entities)
            
            # 准备结构化数据（转换为字符串格式以兼容pandas）
            structured_row = {
                'urls': '|'.join(entities.get('urls', [])),
                'phone_numbers': '|'.join(entities.get('phone_numbers', [])),
                'contacts': '|'.join(entities.get('contacts', [])),
                'crypto_addresses': '|'.join(entities.get('crypto_addresses', [])),
                'bank_info': '|'.join(entities.get('bank_cards', []) + entities.get('bank_names', [])),
                'suspicious_patterns': str(entities.get('suspicious_patterns', {})),
                'has_url': features.get('has_url', False),
                'has_phone': features.get('has_phone', False),
                'has_contact': features.get('has_contact', False),
                'has_crypto': features.get('has_crypto', False),
                'has_bank_info': features.get('has_bank_info', False),
                'has_suspicious_patterns': features.get('has_suspicious_patterns', False),
            }
            structured_data.append(structured_row)
        
        # 确保DataFrame中有必要的列
        for key in structured_data[0].keys() if structured_data else []:
            if key not in df.columns:
                df[key] = None
        
        # 更新DataFrame
        for i, (idx, row) in enumerate(label_data.iterrows()):
            for key, value in structured_data[i].items():
                df.at[idx, key] = value
        
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
            'entity_stats': entity_stats,
            'sample_count': len(label_data)
        }
    
    def generate_pattern_report(self, pattern_results: Dict[str, Dict]) -> str:
        """
        生成模式分析报告
        """
        report = "话术结构分析报告\n"
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
    
    
    # 匹配模式
    templates = analyzer.pattern_templates['按摩色诱'] + analyzer.common_patterns
    matches = analyzer.match_patterns(test_text, templates)
    print(f"匹配的模式: {[(t.name, match) for t, match in matches]}")
