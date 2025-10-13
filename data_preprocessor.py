"""
数据预处理模块
负责清理OCR文本，处理特殊字符，标准化数据格式
"""

import re
import json
import pandas as pd
from typing import List, Dict, Any
import unicodedata


class DataPreprocessor:
    def __init__(self):
        # 定义需要清理的OCR噪音字符
        self.ocr_noise_patterns = [
            r'[^\u4e00-\u9fff\u0020-\u007e\u3000-\u303f\uff00-\uffef]',  # 保留中文、英文、标点
            r'\s+',  # 多个空格合并为一个
            r'[\r\n\t]+',  # 换行符、制表符
        ]
        
        # 常见的OCR错误映射
        self.ocr_corrections = {
            '0': 'O',  # 数字0误识别为字母O
            '1': 'l',  # 数字1误识别为字母l
            '5': 'S',  # 数字5误识别为字母S
            '8': 'B',  # 数字8误识别为字母B
        }
        
        # 定义公共的正则表达式模式
        self.patterns = self._init_patterns()
    
    def _init_patterns(self):
        """
        初始化所有正则表达式模式
        """
        return {
            # 电话号码模式：中国手机号(1[3-9]开头)、中国固定电话(区号-号码)、国际号码(带国家代码)、中国国际格式(+86)、美国格式(+1)、英国格式(+44)、通用格式、7-11位数字(排除QQ号等)
            'phone': r'1[3-9]\d{9}|\d{3,4}[-.\s]?\d{7,8}|\+?[1-9]\d{1,3}[-.\s]?\d{1,4}[-.\s]?\d{1,4}[-.\s]?\d{1,9}|\+86[-.\s]?1[3-9]\d{9}|\+1[-.\s]?\d{3}[-.\s]?\d{3}[-.\s]?\d{4}|\+44[-.\s]?\d{2,4}[-.\s]?\d{3,4}[-.\s]?\d{3,4}|\d{3,4}[-.\s]?\d{3,4}[-.\s]?\d{3,4}|(?<!\d)(?<![a-zA-Z])\d{7,11}(?!\d)(?![a-zA-Z])',
            
            # 联系方式模式：微信相关、QQ相关、Telegram相关、Twitter相关、其他平台(instagram、ins、facebook、fb、line、whatsapp、wa)、通用联系、@用户名
            'contact': r'(微信|加微信|微信号|wx|wechat|微|QQ|qq|扣扣|企鹅|telegram|tg|电报|twitter|推特|x|instagram|ins|facebook|fb|line|whatsapp|wa|联系|加我|找我|薇信|威信|微星|扣扣号|企鹅号)[：:\s]*([a-zA-Z0-9_@.-]+|\d+)|@[a-zA-Z0-9_]+',
            
            # 加密货币地址模式：Bitcoin Legacy地址(1或3开头)、Bitcoin Bech32地址(bc1开头)、Ethereum地址(0x开头)、Litecoin地址(L、M、3开头)、Dogecoin地址(X开头)、通用地址格式(26-35位字母数字)
            'crypto': r'[13][a-km-zA-HJ-NP-Z1-9]{25,34}|bc1[a-z0-9]{39,59}|0x[a-fA-F0-9]{40}|[LM3][a-km-zA-HJ-NP-Z1-9]{26,33}|X[1-9A-HJ-NP-Za-km-z]{25,34}|[A-Za-z0-9]{26,35}',
            
            # 银行信息模式：银行卡号(16-19位数字)、银行中文名称、银行英文缩写
            'bank': r'\d{16,19}|工商银行|建设银行|农业银行|中国银行|交通银行|招商银行|浦发银行|中信银行|光大银行|华夏银行|民生银行|广发银行|平安银行|兴业银行|邮储银行|ICBC|CCB|ABC|BOC|BOCOM|CMB|SPDB|CITIC|CEB|HXB|CMBC|CGB|PAB|CIB|PSBC',
            
            # 可疑模式：金钱相关、紧急相关、权威相关、威胁相关、隐私相关、诈骗指标
            'suspicious': r'赚钱|发财|投资|理财|收益|回报|分红|中奖|奖金|佣金|返利|稳赚|包中|紧急|急|快|立即|马上|现在|今天|限时|过期|最后|机会|官方|政府|银行|警察|法院|税务局|工商局|公安|检察院|冻结|封号|删除|注销|起诉|逮捕|通缉|违法|犯罪|验证码|密码|身份证|银行卡|账号|个人信息|刷单|兼职|垫付|先付|保证金|手续费|解冻费',
            
            # 数字相关模式：年龄(XX岁)、金额(XX元/块/万/千)、时间(XX点/小时/分钟)、百分比(XX%)
            'numeric': r'[0-9]+岁|[0-9]+元|[0-9]+块|[0-9]+万|[0-9]+千|[0-9]+点|[0-9]+小时|[0-9]+分钟|[0-9]+%',
            
            # URL模式
            'url': r'https?://[^\s]+|www\.[^\s]+|[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}',
        }
    
    def clean_text(self, text: str) -> str:
        """
        清理OCR文本中的噪音字符
        """
        if not text:
            return ""
        
        # 标准化Unicode字符
        text = unicodedata.normalize('NFKC', text)
        
        # 移除OCR噪音字符
        for pattern in self.ocr_noise_patterns:
            text = re.sub(pattern, ' ', text)
        
        # 清理多余空格
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def extract_urls(self, text: str) -> List[str]:
        """
        提取文本中的URL
        """
        urls = re.findall(self.patterns['url'], text)
        return urls
    
    def extract_phone_numbers(self, text: str) -> List[str]:
        """
        提取文本中的电话号码（支持国际号码，优化版）
        """
        matches = re.findall(self.patterns['phone'], text)
        
        # 过滤掉明显不是电话号码的数字
        phones = []
        for match in matches:
            if self._is_valid_phone(match):
                phones.append(match)
        
        return list(set(phones))  # 去重
    
    def _is_valid_phone(self, phone: str) -> bool:
        """
        验证是否为有效的电话号码
        """
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
    
    def extract_wechat_qq(self, text: str) -> List[str]:
        """
        提取各种联系方式（微信、QQ、Telegram、Twitter等，优化版）
        """
        matches = re.findall(self.patterns['contact'], text, re.IGNORECASE)
        
        # 处理匹配结果
        contacts = []
        for match in matches:
            if isinstance(match, tuple):
                platform, username = match
                if platform and username:
                    contact = f"{platform}:{username}"
                    contacts.append(contact)
            else:
                # 处理@用户名
                contacts.append(match)
        
        # 清理和标准化联系方式
        cleaned_contacts = []
        for contact in contacts:
            cleaned = self._clean_contact(contact)
            if cleaned and len(cleaned) > 1:  # 过滤太短的联系方式
                cleaned_contacts.append(cleaned)
        
        return list(set(cleaned_contacts))  # 去重
    
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
    
    def extract_crypto_addresses(self, text: str) -> List[str]:
        """
        提取加密货币地址（优化版）
        """
        addresses = re.findall(self.patterns['crypto'], text)
        return list(set(addresses))
    
    def extract_bank_info(self, text: str) -> List[str]:
        """
        提取银行相关信息（优化版）
        """
        bank_info = re.findall(self.patterns['bank'], text, re.IGNORECASE)
        return list(set(bank_info))
    
    def extract_suspicious_patterns(self, text: str) -> Dict[str, List[str]]:
        """
        提取可疑模式（按类别分组，优化版）
        """
        all_matches = re.findall(self.patterns['suspicious'], text)
        
        # 按类别分类
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
            matches = [match for match in all_matches if match in keywords]
            results[category] = list(set(matches))
        
        return results
    
    def process_jsonl_file(self, file_path: str) -> pd.DataFrame:
        """
        处理JSONL文件，返回清理后的DataFrame
        """
        data = []
        
        with open(file_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                try:
                    item = json.loads(line.strip())
                    
                    # 清理conversation文本
                    if 'conversation' in item:
                        original_text = item['conversation']
                        cleaned_text = self.clean_text(original_text)
                        
                        # 提取结构化信息
                        urls = self.extract_urls(cleaned_text)
                        phones = self.extract_phone_numbers(cleaned_text)
                        contacts = self.extract_wechat_qq(cleaned_text)
                        crypto_addresses = self.extract_crypto_addresses(cleaned_text)
                        bank_info = self.extract_bank_info(cleaned_text)
                        suspicious_patterns = self.extract_suspicious_patterns(cleaned_text)
                        
                        processed_item = {
                            'line_number': line_num,
                            'original_text': original_text,
                            'cleaned_text': cleaned_text,
                            'text_length': len(cleaned_text),
                            'label': item.get('labelname', ''),
                            'data_source': item.get('datasrc', ''),
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
                        }
                        
                        data.append(processed_item)
                        
                except json.JSONDecodeError as e:
                    print(f"第{line_num}行JSON解析错误: {e}")
                    continue
        
        return pd.DataFrame(data)
    
    def get_text_statistics(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        获取文本统计信息
        """
        # 转换numpy类型为Python原生类型
        def convert_numpy_types(obj):
            if isinstance(obj, dict):
                return {key: convert_numpy_types(value) for key, value in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy_types(item) for item in obj]
            elif isinstance(obj, tuple):
                return tuple(convert_numpy_types(item) for item in obj)
            elif hasattr(obj, 'item'):  # numpy scalar
                return obj.item()
            elif hasattr(obj, 'tolist'):  # numpy array
                return obj.tolist()
            else:
                return obj
        
        stats = {
            'total_samples': len(df),
            'label_distribution': df['label'].value_counts().to_dict(),
            'avg_text_length': df['text_length'].mean(),
            'text_length_stats': df['text_length'].describe().to_dict(),
            'url_coverage': df['has_url'].mean(),
            'phone_coverage': df['has_phone'].mean(),
            'contact_coverage': df['has_contact'].mean(),
            'crypto_coverage': df['has_crypto'].mean(),
            'bank_info_coverage': df['has_bank_info'].mean(),
            'suspicious_patterns_coverage': df['has_suspicious_patterns'].mean(),
        }
        
        return convert_numpy_types(stats)


if __name__ == "__main__":
    # 测试示例
    preprocessor = DataPreprocessor()
    
    # 测试文本清理
    test_text = "这是一个测试文本\n包含换行符\t和制表符   多个空格"
    cleaned = preprocessor.clean_text(test_text)
    print(f"原始文本: {repr(test_text)}")
    print(f"清理后: {repr(cleaned)}")
    
    # 测试URL提取
    test_url_text = "访问网站 https://example.com 或 www.test.com"
    urls = preprocessor.extract_urls(test_url_text)
    print(f"提取的URL: {urls}")
