"""
关键词提取模块
负责从文本中提取网站名、常见动词、网址等关键信息
集成深度学习模型进行更精确的关键词提取和语义分析
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
    def __init__(self, use_bert: bool = True, bert_model: str = "bert-base-chinese"):
        # 初始化jieba分词
        jieba.initialize()
        
        # 深度学习模型配置
        self.use_bert = use_bert
        self.bert_model_name = bert_model
        self.bert_tokenizer = None
        self.bert_model = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # 初始化BERT模型
        if self.use_bert:
            self._initialize_bert_model()
        
        # 定义不同标签的关键词词典（包含形似、谐音、网络符号变体）
        self.label_keywords = {
            '按摩色诱': {
                'websites': [
                    '同城', '交友', '约会', '美女', '按摩', 'spa', '会所',
                    # 形似变体
                    '冋城', '冋城交友', '美女按摸', '按摸', 'SPA', 'SPA会所',
                    '冋城网', '冋城站', '冋城平台', '冋城交友网', '冋城交友站',
                    '美女按摸网', '美女按摸站', '按摸网', '按摸站', '按摸平台',
                    # 谐音变体
                    '同诚', '交友网', '约会网', '美女网', '按摸网',
                    '同诚网', '同诚站', '同诚平台', '同诚交友', '同诚交友网',
                    '交友站', '约会站', '美女站', '按摸站', 'spa站',
                    # 网络符号变体
                    '同城+', '交友+', '美女+', '按摸+', 'spa+',
                    '同城.', '交友.', '美女.', '按摸.', 'spa.',
                    '同城_', '交友_', '美女_', '按摸_', 'spa_',
                    '同城-', '交友-', '美女-', '按摸-', 'spa-',
                    '同城*', '交友*', '美女*', '按摸*', 'spa*',
                    '同城#', '交友#', '美女#', '按摸#', 'spa#',
                    '同城@', '交友@', '美女@', '按摸@', 'spa@',
                    '同城&', '交友&', '美女&', '按摸&', 'spa&',
                    '同城%', '交友%', '美女%', '按摸%', 'spa%',
                    '同城$', '交友$', '美女$', '按摸$', 'spa$',
                    # 数字变体
                    '同城1', '同城2', '同城3', '同城01', '同城02', '同城03',
                    '交友1', '交友2', '交友3', '交友01', '交友02', '交友03',
                    '美女1', '美女2', '美女3', '美女01', '美女02', '美女03',
                    '按摸1', '按摸2', '按摸3', '按摸01', '按摸02', '按摸03',
                    # 组合变体
                    '冋城交友+', '冋城美女+', '冋城按摸+', '冋城spa+',
                    '同诚交友.', '同诚美女.', '同诚按摸.', '同诚spa.',
                    '美女按摸_', '美女spa_', '按摸spa_', '交友按摸_',
                    # 特殊字符组合
                    '同城++', '交友++', '美女++', '按摸++', 'spa++',
                    '同城..', '交友..', '美女..', '按摸..', 'spa..',
                    '同城__', '交友__', '美女__', '按摸__', 'spa__',
                    '同城--', '交友--', '美女--', '按摸--', 'spa--',
                    # 混合变体
                    '冋城+', '冋城.', '冋城_', '冋城-', '冋城*',
                    '同诚+', '同诚.', '同诚_', '同诚-', '同诚*',
                    '按摸+', '按摸.', '按摸_', '按摸-', '按摸*',
                    '美女+', '美女.', '美女_', '美女-', '美女*'
                ],
                'verbs': [
                    '约', '聊', '见面', '服务', '按摩', '放松', '享受',
                    # 形似变体
                    '冋', '冋聊', '冋面', '按摸', '放忪', '享爱',
                    '冋聊网', '冋面网', '按摸网', '放忪网', '享爱网',
                    '冋聊站', '冋面站', '按摸站', '放忪站', '享爱站',
                    # 谐音变体
                    '约聊', '见面聊', '按摸服务', '放松服务',
                    '约聊网', '见面聊网', '按摸服务网', '放松服务网',
                    '约聊站', '见面聊站', '按摸服务站', '放松服务站',
                    # 网络符号变体
                    '约+', '聊+', '见面+', '按摸+', '放松+',
                    '约.', '聊.', '见面.', '按摸.', '放松.',
                    '约_', '聊_', '见面_', '按摸_', '放松_',
                    '约-', '聊-', '见面-', '按摸-', '放松-',
                    '约*', '聊*', '见面*', '按摸*', '放松*',
                    '约#', '聊#', '见面#', '按摸#', '放松#',
                    '约@', '聊@', '见面@', '按摸@', '放松@',
                    '约&', '聊&', '见面&', '按摸&', '放松&',
                    '约%', '聊%', '见面%', '按摸%', '放松%',
                    '约$', '聊$', '见面$', '按摸$', '放松$',
                    # 数字变体
                    '约1', '约2', '约3', '约01', '约02', '约03',
                    '聊1', '聊2', '聊3', '聊01', '聊02', '聊03',
                    '见面1', '见面2', '见面3', '见面01', '见面02', '见面03',
                    '按摸1', '按摸2', '按摸3', '按摸01', '按摸02', '按摸03',
                    # 组合变体
                    '冋聊+', '冋面+', '按摸+', '放忪+', '享爱+',
                    '约聊.', '见面聊.', '按摸服务.', '放松服务.',
                    '冋聊_', '冋面_', '按摸_', '放忪_', '享爱_',
                    # 特殊字符组合
                    '约++', '聊++', '见面++', '按摸++', '放松++',
                    '约..', '聊..', '见面..', '按摸..', '放松..',
                    '约__', '聊__', '见面__', '按摸__', '放松__',
                    '约--', '聊--', '见面--', '按摸--', '放松--',
                    # 混合变体
                    '冋+', '冋.', '冋_', '冋-', '冋*',
                    '按摸+', '按摸.', '按摸_', '按摸-', '按摸*',
                    '放松+', '放松.', '放松_', '放松-', '放松*'
                ],
                'patterns': [
                    r'[0-9]+岁', r'[0-9]+元', r'[0-9]+小时',
                    r'[0-9]+岁美女', r'[0-9]+元[0-9]+小时',
                    r'[0-9]+岁妹妹', r'[0-9]+元包夜',
                    r'[0-9]+岁[0-9]+元', r'[0-9]+岁[0-9]+小时',
                    r'[0-9]+元[0-9]+小时[0-9]+岁', r'[0-9]+岁[0-9]+元[0-9]+小时',
                    r'[0-9]+岁美女[0-9]+元', r'[0-9]+元美女[0-9]+岁',
                    r'[0-9]+岁妹妹[0-9]+元', r'[0-9]+元妹妹[0-9]+岁',
                    r'[0-9]+元包夜[0-9]+岁', r'[0-9]+岁包夜[0-9]+元'
                ]
            },
            '博彩': {
                'websites': [
                    '彩票', '博彩', '赌场', '游戏', '娱乐', '平台',
                    # 形似变体
                    '彩票网', '博彩网', '赌场网', '游戏网', '娱乐网',
                    '彩票+', '博彩+', '赌场+', '游戏+', '娱乐+',
                    # 谐音变体
                    '彩票站', '博彩站', '赌场站', '游戏站', '娱乐站',
                    # 网络符号变体
                    '彩票.', '博彩.', '赌场.', '游戏.', '娱乐.'
                ],
                'verbs': [
                    '投注', '下注', '中奖', '赚钱', '充值', '提现',
                    # 形似变体
                    '投住', '下住', '中奖', '赚銭', '充值', '提现',
                    # 谐音变体
                    '投注网', '下注网', '中奖网', '赚钱网', '充值网',
                    # 网络符号变体
                    '投注+', '下注+', '中奖+', '赚钱+', '充值+'
                ],
                'patterns': [
                    r'[0-9]+倍', r'[0-9]+%', r'稳赚', r'包中',
                    r'[0-9]+倍回报', r'[0-9]+%收益', r'稳赚不赔', r'包中奖'
                ]
            },
            '兼职刷单': {
                'websites': [
                    '兼职', '刷单', '任务', '佣金', '返利',
                    # 形似变体
                    '兼职网', '刷单网', '任务网', '佣金网', '返利网',
                    '兼职+', '刷单+', '任务+', '佣金+', '返利+',
                    # 谐音变体
                    '兼职站', '刷单站', '任务站', '佣金站', '返利站',
                    # 网络符号变体
                    '兼职.', '刷单.', '任务.', '佣金.', '返利.'
                ],
                'verbs': [
                    '刷单', '兼职', '赚钱', '佣金', '返利', '垫付',
                    # 形似变体
                    '刷单网', '兼职网', '赚銭', '佣金网', '返利网', '垫付网',
                    # 谐音变体
                    '刷单站', '兼职站', '赚钱站', '佣金站', '返利站',
                    # 网络符号变体
                    '刷单+', '兼职+', '赚钱+', '佣金+', '返利+'
                ],
                'patterns': [
                    r'[0-9]+元/单', r'日赚[0-9]+', r'佣金[0-9]+%',
                    r'[0-9]+元一单', r'日赚[0-9]+元', r'佣金[0-9]+%返利'
                ]
            },
            '投资理财': {
                'websites': [
                    '投资', '理财', '基金', '股票', '外汇', '数字货币',
                    # 形似变体
                    '投资网', '理财网', '基金网', '股票网', '外汇网', '数字货币网',
                    '投资+', '理财+', '基金+', '股票+', '外汇+', '数字货币+',
                    # 谐音变体
                    '投资站', '理财站', '基金站', '股票站', '外汇站',
                    # 网络符号变体
                    '投资.', '理财.', '基金.', '股票.', '外汇.'
                ],
                'verbs': [
                    '投资', '理财', '收益', '回报', '分红', '增值',
                    # 形似变体
                    '投资网', '理财网', '收益网', '回报网', '分红网', '增值网',
                    # 谐音变体
                    '投资站', '理财站', '收益站', '回报站', '分红站',
                    # 网络符号变体
                    '投资+', '理财+', '收益+', '回报+', '分红+'
                ],
                'patterns': [
                    r'年化[0-9]+%', r'收益[0-9]+%', r'保本',
                    r'年化[0-9]+%收益', r'收益[0-9]+%保本', r'保本保息'
                ]
            },
            '虚假客服': {
                'websites': [
                    '客服', '售后', '退款', '理赔', '银行',
                    # 形似变体
                    '客服网', '售后网', '退款网', '理赔网', '银行网',
                    '客服+', '售后+', '退款+', '理赔+', '银行+',
                    # 谐音变体
                    '客服站', '售后站', '退款站', '理赔站', '银行站',
                    # 网络符号变体
                    '客服.', '售后.', '退款.', '理赔.', '银行.'
                ],
                'verbs': [
                    '退款', '理赔', '解冻', '验证', '确认', '操作',
                    # 形似变体
                    '退款网', '理赔网', '解冻网', '验证网', '确认网', '操作网',
                    # 谐音变体
                    '退款站', '理赔站', '解冻站', '验证站', '确认站',
                    # 网络符号变体
                    '退款+', '理赔+', '解冻+', '验证+', '确认+'
                ],
                'patterns': [
                    r'验证码', r'银行卡', r'身份证', r'密码',
                    r'验证码[0-9]+', r'银行卡[0-9]+', r'身份证[0-9]+', r'密码[0-9]+'
                ]
            }
        }
        
        # 通用网站名模式（包含各种变体）
        self.website_patterns = [
            # 标准域名模式
            r'[a-zA-Z0-9.-]+\.(com|cn|net|org|cc|me|info|biz|co|tv|top|xyz|site|online|tech|app|io|ai|ml)',
            r'[a-zA-Z0-9]+\.(com|cn|net|org|cc|me|info|biz|co|tv|top|xyz|site|online|tech|app|io|ai|ml)',
            r'[a-zA-Z0-9]+\.(com\.cn|net\.cn|org\.cn|gov\.cn|edu\.cn)',
            
            # 中文网站模式
            r'[a-zA-Z0-9\u4e00-\u9fff]+平台',
            r'[a-zA-Z0-9\u4e00-\u9fff]+网站',
            r'[a-zA-Z0-9\u4e00-\u9fff]+网',
            r'[a-zA-Z0-9\u4e00-\u9fff]+站',
            r'[a-zA-Z0-9\u4e00-\u9fff]+网\+',
            r'[a-zA-Z0-9\u4e00-\u9fff]+站\+',
            r'[a-zA-Z0-9\u4e00-\u9fff]+平台\+',
            
            # 形似变体模式
            r'[a-zA-Z0-9\u4e00-\u9fff]*冋城[a-zA-Z0-9\u4e00-\u9fff]*',
            r'[a-zA-Z0-9\u4e00-\u9fff]*冋城网[a-zA-Z0-9\u4e00-\u9fff]*',
            r'[a-zA-Z0-9\u4e00-\u9fff]*冋城站[a-zA-Z0-9\u4e00-\u9fff]*',
            r'[a-zA-Z0-9\u4e00-\u9fff]*冋城\+[a-zA-Z0-9\u4e00-\u9fff]*',
            r'[a-zA-Z0-9\u4e00-\u9fff]*冋城\.[a-zA-Z0-9\u4e00-\u9fff]*',
            r'[a-zA-Z0-9\u4e00-\u9fff]*冋城_[a-zA-Z0-9\u4e00-\u9fff]*',
            r'[a-zA-Z0-9\u4e00-\u9fff]*冋城-[a-zA-Z0-9\u4e00-\u9fff]*',
            
            # 谐音变体模式
            r'[a-zA-Z0-9\u4e00-\u9fff]*同诚[a-zA-Z0-9\u4e00-\u9fff]*',
            r'[a-zA-Z0-9\u4e00-\u9fff]*同诚网[a-zA-Z0-9\u4e00-\u9fff]*',
            r'[a-zA-Z0-9\u4e00-\u9fff]*同诚站[a-zA-Z0-9\u4e00-\u9fff]*',
            r'[a-zA-Z0-9\u4e00-\u9fff]*同诚\+[a-zA-Z0-9\u4e00-\u9fff]*',
            r'[a-zA-Z0-9\u4e00-\u9fff]*同诚\.[a-zA-Z0-9\u4e00-\u9fff]*',
            r'[a-zA-Z0-9\u4e00-\u9fff]*同诚_[a-zA-Z0-9\u4e00-\u9fff]*',
            r'[a-zA-Z0-9\u4e00-\u9fff]*同诚-[a-zA-Z0-9\u4e00-\u9fff]*',
            
            # 网络符号变体模式
            r'[a-zA-Z0-9\u4e00-\u9fff]*\+网[a-zA-Z0-9\u4e00-\u9fff]*',
            r'[a-zA-Z0-9\u4e00-\u9fff]*\+站[a-zA-Z0-9\u4e00-\u9fff]*',
            r'[a-zA-Z0-9\u4e00-\u9fff]*\+平台[a-zA-Z0-9\u4e00-\u9fff]*',
            r'[a-zA-Z0-9\u4e00-\u9fff]*\.网[a-zA-Z0-9\u4e00-\u9fff]*',
            r'[a-zA-Z0-9\u4e00-\u9fff]*\.站[a-zA-Z0-9\u4e00-\u9fff]*',
            r'[a-zA-Z0-9\u4e00-\u9fff]*\.平台[a-zA-Z0-9\u4e00-\u9fff]*',
            r'[a-zA-Z0-9\u4e00-\u9fff]*_网[a-zA-Z0-9\u4e00-\u9fff]*',
            r'[a-zA-Z0-9\u4e00-\u9fff]*_站[a-zA-Z0-9\u4e00-\u9fff]*',
            r'[a-zA-Z0-9\u4e00-\u9fff]*_平台[a-zA-Z0-9\u4e00-\u9fff]*',
            r'[a-zA-Z0-9\u4e00-\u9fff]*-网[a-zA-Z0-9\u4e00-\u9fff]*',
            r'[a-zA-Z0-9\u4e00-\u9fff]*-站[a-zA-Z0-9\u4e00-\u9fff]*',
            r'[a-zA-Z0-9\u4e00-\u9fff]*-平台[a-zA-Z0-9\u4e00-\u9fff]*',
            
            # 特殊符号变体模式
            r'[a-zA-Z0-9\u4e00-\u9fff]*\*网[a-zA-Z0-9\u4e00-\u9fff]*',
            r'[a-zA-Z0-9\u4e00-\u9fff]*\*站[a-zA-Z0-9\u4e00-\u9fff]*',
            r'[a-zA-Z0-9\u4e00-\u9fff]*\*平台[a-zA-Z0-9\u4e00-\u9fff]*',
            r'[a-zA-Z0-9\u4e00-\u9fff]*#网[a-zA-Z0-9\u4e00-\u9fff]*',
            r'[a-zA-Z0-9\u4e00-\u9fff]*#站[a-zA-Z0-9\u4e00-\u9fff]*',
            r'[a-zA-Z0-9\u4e00-\u9fff]*#平台[a-zA-Z0-9\u4e00-\u9fff]*',
            r'[a-zA-Z0-9\u4e00-\u9fff]*@网[a-zA-Z0-9\u4e00-\u9fff]*',
            r'[a-zA-Z0-9\u4e00-\u9fff]*@站[a-zA-Z0-9\u4e00-\u9fff]*',
            r'[a-zA-Z0-9\u4e00-\u9fff]*@平台[a-zA-Z0-9\u4e00-\u9fff]*',
            r'[a-zA-Z0-9\u4e00-\u9fff]*&网[a-zA-Z0-9\u4e00-\u9fff]*',
            r'[a-zA-Z0-9\u4e00-\u9fff]*&站[a-zA-Z0-9\u4e00-\u9fff]*',
            r'[a-zA-Z0-9\u4e00-\u9fff]*&平台[a-zA-Z0-9\u4e00-\u9fff]*',
            r'[a-zA-Z0-9\u4e00-\u9fff]*%网[a-zA-Z0-9\u4e00-\u9fff]*',
            r'[a-zA-Z0-9\u4e00-\u9fff]*%站[a-zA-Z0-9\u4e00-\u9fff]*',
            r'[a-zA-Z0-9\u4e00-\u9fff]*%平台[a-zA-Z0-9\u4e00-\u9fff]*',
            r'[a-zA-Z0-9\u4e00-\u9fff]*\$网[a-zA-Z0-9\u4e00-\u9fff]*',
            r'[a-zA-Z0-9\u4e00-\u9fff]*\$站[a-zA-Z0-9\u4e00-\u9fff]*',
            r'[a-zA-Z0-9\u4e00-\u9fff]*\$平台[a-zA-Z0-9\u4e00-\u9fff]*',
            r'[a-zA-Z0-9\u4e00-\u9fff]*!网[a-zA-Z0-9\u4e00-\u9fff]*',
            r'[a-zA-Z0-9\u4e00-\u9fff]*!站[a-zA-Z0-9\u4e00-\u9fff]*',
            r'[a-zA-Z0-9\u4e00-\u9fff]*!平台[a-zA-Z0-9\u4e00-\u9fff]*',
            r'[a-zA-Z0-9\u4e00-\u9fff]*\?网[a-zA-Z0-9\u4e00-\u9fff]*',
            r'[a-zA-Z0-9\u4e00-\u9fff]*\?站[a-zA-Z0-9\u4e00-\u9fff]*',
            r'[a-zA-Z0-9\u4e00-\u9fff]*\?平台[a-zA-Z0-9\u4e00-\u9fff]*',
            r'[a-zA-Z0-9\u4e00-\u9fff]*~网[a-zA-Z0-9\u4e00-\u9fff]*',
            r'[a-zA-Z0-9\u4e00-\u9fff]*~站[a-zA-Z0-9\u4e00-\u9fff]*',
            r'[a-zA-Z0-9\u4e00-\u9fff]*~平台[a-zA-Z0-9\u4e00-\u9fff]*',
            r'[a-zA-Z0-9\u4e00-\u9fff]*\^网[a-zA-Z0-9\u4e00-\u9fff]*',
            r'[a-zA-Z0-9\u4e00-\u9fff]*\^站[a-zA-Z0-9\u4e00-\u9fff]*',
            r'[a-zA-Z0-9\u4e00-\u9fff]*\^平台[a-zA-Z0-9\u4e00-\u9fff]*',
            r'[a-zA-Z0-9\u4e00-\u9fff]*\|网[a-zA-Z0-9\u4e00-\u9fff]*',
            r'[a-zA-Z0-9\u4e00-\u9fff]*\|站[a-zA-Z0-9\u4e00-\u9fff]*',
            r'[a-zA-Z0-9\u4e00-\u9fff]*\|平台[a-zA-Z0-9\u4e00-\u9fff]*',
            r'[a-zA-Z0-9\u4e00-\u9fff]*\\\\网[a-zA-Z0-9\u4e00-\u9fff]*',
            r'[a-zA-Z0-9\u4e00-\u9fff]*\\\\站[a-zA-Z0-9\u4e00-\u9fff]*',
            r'[a-zA-Z0-9\u4e00-\u9fff]*\\\\平台[a-zA-Z0-9\u4e00-\u9fff]*',
            r'[a-zA-Z0-9\u4e00-\u9fff]*/网[a-zA-Z0-9\u4e00-\u9fff]*',
            r'[a-zA-Z0-9\u4e00-\u9fff]*/站[a-zA-Z0-9\u4e00-\u9fff]*',
            r'[a-zA-Z0-9\u4e00-\u9fff]*/平台[a-zA-Z0-9\u4e00-\u9fff]*',
            
            # 数字变体
            r'[a-zA-Z0-9\u4e00-\u9fff]*[0-9]+网[a-zA-Z0-9\u4e00-\u9fff]*',
            r'[a-zA-Z0-9\u4e00-\u9fff]*[0-9]+站[a-zA-Z0-9\u4e00-\u9fff]*',
            r'[a-zA-Z0-9\u4e00-\u9fff]*[0-9]+平台[a-zA-Z0-9\u4e00-\u9fff]*',
            r'[0-9]+[a-zA-Z0-9\u4e00-\u9fff]*网[a-zA-Z0-9\u4e00-\u9fff]*',
            r'[0-9]+[a-zA-Z0-9\u4e00-\u9fff]*站[a-zA-Z0-9\u4e00-\u9fff]*',
            r'[0-9]+[a-zA-Z0-9\u4e00-\u9fff]*平台[a-zA-Z0-9\u4e00-\u9fff]*',
            
            # 组合符号变体
            r'[a-zA-Z0-9\u4e00-\u9fff]*\+\+网[a-zA-Z0-9\u4e00-\u9fff]*',
            r'[a-zA-Z0-9\u4e00-\u9fff]*\+\+站[a-zA-Z0-9\u4e00-\u9fff]*',
            r'[a-zA-Z0-9\u4e00-\u9fff]*\+\+平台[a-zA-Z0-9\u4e00-\u9fff]*',
            r'[a-zA-Z0-9\u4e00-\u9fff]*\.\.网[a-zA-Z0-9\u4e00-\u9fff]*',
            r'[a-zA-Z0-9\u4e00-\u9fff]*\.\.站[a-zA-Z0-9\u4e00-\u9fff]*',
            r'[a-zA-Z0-9\u4e00-\u9fff]*\.\.平台[a-zA-Z0-9\u4e00-\u9fff]*',
            r'[a-zA-Z0-9\u4e00-\u9fff]*__网[a-zA-Z0-9\u4e00-\u9fff]*',
            r'[a-zA-Z0-9\u4e00-\u9fff]*__站[a-zA-Z0-9\u4e00-\u9fff]*',
            r'[a-zA-Z0-9\u4e00-\u9fff]*__平台[a-zA-Z0-9\u4e00-\u9fff]*',
            r'[a-zA-Z0-9\u4e00-\u9fff]*--网[a-zA-Z0-9\u4e00-\u9fff]*',
            r'[a-zA-Z0-9\u4e00-\u9fff]*--站[a-zA-Z0-9\u4e00-\u9fff]*',
            r'[a-zA-Z0-9\u4e00-\u9fff]*--平台[a-zA-Z0-9\u4e00-\u9fff]*',
            
            # 特殊组合
            r'[a-zA-Z0-9\u4e00-\u9fff]*[0-9]\+网[a-zA-Z0-9\u4e00-\u9fff]*',
            r'[a-zA-Z0-9\u4e00-\u9fff]*[0-9]\.网[a-zA-Z0-9\u4e00-\u9fff]*',
            r'[a-zA-Z0-9\u4e00-\u9fff]*[0-9]_网[a-zA-Z0-9\u4e00-\u9fff]*',
            r'[a-zA-Z0-9\u4e00-\u9fff]*[0-9]-网[a-zA-Z0-9\u4e00-\u9fff]*',
            r'[a-zA-Z0-9\u4e00-\u9fff]*\+[0-9]网[a-zA-Z0-9\u4e00-\u9fff]*',
            r'[a-zA-Z0-9\u4e00-\u9fff]*\.[0-9]网[a-zA-Z0-9\u4e00-\u9fff]*',
            r'[a-zA-Z0-9\u4e00-\u9fff]*_[0-9]网[a-zA-Z0-9\u4e00-\u9fff]*',
            r'[a-zA-Z0-9\u4e00-\u9fff]*-[0-9]网[a-zA-Z0-9\u4e00-\u9fff]*'
        ]
        
        # 通用动词词典（包含形似、谐音、网络符号变体）
        self.common_verbs = {
            # 基础动词
            '约', '聊', '见面', '服务', '按摩', '放松', '享受',
            '投注', '下注', '中奖', '赚钱', '充值', '提现',
            '刷单', '兼职', '佣金', '返利', '垫付',
            '投资', '理财', '收益', '回报', '分红', '增值',
            '退款', '理赔', '解冻', '验证', '确认', '操作',
            '加', '联系', '咨询', '了解', '体验', '尝试',
            
            # 形似变体
            '冋', '冋聊', '冋面', '按摸', '放忪', '享爱',
            '投住', '下住', '赚銭', '垫付网', '投住网',
            '投住站', '下住站', '赚銭站', '垫付站',
            
            # 谐音变体
            '约聊', '见面聊', '按摸服务', '放松服务',
            '投注网', '下注网', '中奖网', '赚钱网', '充值网',
            '刷单网', '兼职网', '佣金网', '返利网',
            '投资网', '理财网', '收益网', '回报网', '分红网',
            '退款网', '理赔网', '解冻网', '验证网', '确认网',
            
            # 网络符号变体
            '约+', '聊+', '见面+', '按摸+', '放松+',
            '投注+', '下注+', '中奖+', '赚钱+', '充值+',
            '刷单+', '兼职+', '佣金+', '返利+',
            '投资+', '理财+', '收益+', '回报+', '分红+',
            '退款+', '理赔+', '解冻+', '验证+', '确认+',
            
            # 其他常见变体
            '约.', '聊.', '见面.', '按摸.', '放松.',
            '投注.', '下注.', '中奖.', '赚钱.', '充值.',
            '刷单.', '兼职.', '佣金.', '返利.',
            '投资.', '理财.', '收益.', '回报.', '分红.',
            '退款.', '理赔.', '解冻.', '验证.', '确认.'
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
    
    def extract_websites(self, text: str) -> List[str]:
        """
        提取网站名
        """
        websites = []
        
        # 使用正则表达式提取URL格式的网站
        for pattern in self.website_patterns:
            matches = re.findall(pattern, text)
            websites.extend(matches)
        
        # 使用jieba分词提取可能的网站名
        words = jieba.lcut(text)
        for word in words:
            if any(keyword in word for keyword in ['网', '平台', '网站', 'app', '软件']):
                websites.append(word)
        
        return list(set(websites))
    
    def extract_verbs(self, text: str) -> List[Tuple[str, str]]:
        """
        提取动词及其词性
        """
        words = pseg.cut(text)
        verbs = []
        
        for word, flag in words:
            if flag.startswith('v') or word in self.common_verbs:
                verbs.append((word, flag))
        
        return verbs
    
    def extract_keywords_by_tfidf(self, texts: List[str], top_k: int = -1) -> List[Tuple[str, float]]:
        """
        使用TF-IDF提取关键词
        """
        # 自定义停用词
        stop_words = {'的', '了', '在', '是', '我', '有', '和', '就', '不', '人', '都', '一', '一个', '上', '也', '很', '到', '说', '要', '去', '你', '会', '着', '没有', '看', '好', '自己', '这'}
        
        # 预处理文本
        processed_texts = []
        for text in texts:
            words = jieba.lcut(text)
            words = [word for word in words if word not in stop_words and len(word) > 1]
            processed_texts.append(' '.join(words))
        
        # 计算TF-IDF
        vectorizer = TfidfVectorizer(max_features=1000, ngram_range=(1, 2))
        tfidf_matrix = vectorizer.fit_transform(processed_texts)
        
        # 获取特征名和分数
        feature_names = vectorizer.get_feature_names_out()
        scores = np.mean(tfidf_matrix.toarray(), axis=0)
        
        # 排序并返回结果
        keyword_scores = list(zip(feature_names, scores))
        keyword_scores.sort(key=lambda x: x[1], reverse=True)
        
        # 如果top_k为-1，返回所有关键词；否则返回前top_k个
        if top_k == -1:
            return keyword_scores
        else:
            return keyword_scores[:top_k]
    
    def extract_label_specific_keywords(self, texts: List[str], label: str, top_k: int = -1) -> Dict[str, List[str]]:
        """
        提取特定标签的关键词
        """
        if label not in self.label_keywords:
            return {}
        
        label_info = self.label_keywords[label]
        results = {
            'websites': [],
            'verbs': [],
            'patterns': []
        }
        
        # 统计关键词出现频次
        website_counter = Counter()
        verb_counter = Counter()
        pattern_matches = []
        
        for text in texts:
            # 提取网站名
            websites = self.extract_websites(text)
            for website in websites:
                if any(keyword in website for keyword in label_info['websites']):
                    website_counter[website] += 1
            
            # 提取动词
            verbs = self.extract_verbs(text)
            for verb, _ in verbs:
                if verb in label_info['verbs']:
                    verb_counter[verb] += 1
            
            # 匹配模式
            for pattern in label_info['patterns']:
                matches = re.findall(pattern, text)
                pattern_matches.extend(matches)
        
        if top_k == -1:
            results['websites'] = [website for website, count in website_counter.most_common()]
            results['verbs'] = [verb for verb, count in verb_counter.most_common()]
        else:
            results['websites'] = [website for website, count in website_counter.most_common(top_k)]
            results['verbs'] = [verb for verb, count in verb_counter.most_common(top_k)]
        results['patterns'] = list(set(pattern_matches))
        
        return results
    
    def analyze_text_patterns(self, text: str) -> Dict[str, List[str]]:
        """
        分析文本中的模式
        """
        patterns = {
            'numbers': re.findall(r'[0-9]+', text),
            'money': re.findall(r'[0-9]+元|[0-9]+块|[0-9]+万|[0-9]+千', text),
            'time': re.findall(r'[0-9]+点|[0-9]+小时|[0-9]+分钟', text),
            'percentages': re.findall(r'[0-9]+%', text),
            'phone_numbers': re.findall(r'1[3-9]\d{9}', text),
            'urls': re.findall(r'https?://[^\s]+|www\.[^\s]+', text),
        }
        
        return patterns
    
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
    
    def extract_semantic_keywords(self, texts: List[str], top_k: int = -1) -> List[Tuple[str, float]]:
        """基于BERT语义相似度的关键词提取"""
        if not self.use_bert or self.bert_model is None:
            return self.extract_keywords_by_tfidf(texts, top_k)
        
        # 分词获取候选关键词
        all_words = []
        for text in texts:
            words = jieba.lcut(text)
            words = [w for w in words if len(w) > 1 and w not in ['的', '了', '在', '是', '我', '有', '和', '就', '不', '人', '都', '一', '一个', '上', '也', '很', '到', '说', '要', '去', '你', '会', '着', '没有', '看', '好', '自己', '这']]
            all_words.extend(words)
        
        # 统计词频
        word_counts = Counter(all_words)
        if top_k == -1:
            candidate_words = [word for word, count in word_counts.most_common()]
        else:
            # 候选词数量设为top_k的2倍，确保有足够的候选词
            candidate_limit = max(50, top_k * 2)
            candidate_words = [word for word, count in word_counts.most_common(candidate_limit)]
        
        if len(candidate_words) == 0:
            return []
        
        # 计算每个词的语义重要性
        word_importance = []
        for word in candidate_words:
            # 计算包含该词的文本与不包含该词的文本的语义差异
            texts_with_word = [text for text in texts if word in text]
            texts_without_word = [text for text in texts if word not in text]
            
            if len(texts_with_word) == 0 or len(texts_without_word) == 0:
                continue
            
            # 提取特征
            features_with = self.extract_bert_features(texts_with_word)
            features_without = self.extract_bert_features(texts_without_word)
            
            if len(features_with) == 0 or len(features_without) == 0:
                continue
            
            # 计算语义差异
            mean_with = np.mean(features_with, axis=0)
            mean_without = np.mean(features_without, axis=0)
            
            # 计算余弦相似度
            similarity = cosine_similarity([mean_with], [mean_without])[0][0]
            importance = 1 - similarity  # 差异越大，重要性越高
            
            word_importance.append((word, importance))
        
        # 按重要性排序
        word_importance.sort(key=lambda x: x[1], reverse=True)
        
        # 如果top_k为-1，返回所有关键词；否则返回前top_k个
        if top_k == -1:
            return word_importance
        else:
            return word_importance[:top_k]
    
    def extract_contextual_keywords(self, texts: List[str], top_k: int = -1) -> List[Tuple[str, float]]:
        """
        提取上下文相关的关键词
        分析同一标签下文本的语义相似性，找出在特定上下文中重要的关键词
        """
        if not self.use_bert or self.bert_model is None or len(texts) < 2:
            return []
        
        # 提取所有文本的BERT特征
        features = self.extract_bert_features(texts)
        if len(features) == 0:
            return []
        
        # 计算文本间的语义相似度矩阵
        similarity_matrix = cosine_similarity(features)
        
        # 找出语义相似度高的文本对
        high_similarity_pairs = []
        for i in range(len(texts)):
            for j in range(i + 1, len(texts)):
                similarity = similarity_matrix[i][j]
                if similarity > 0.7:  # 相似度阈值
                    high_similarity_pairs.append((i, j, similarity))
        
        if not high_similarity_pairs:
            return []
        
        # 分析高相似度文本对中的共同关键词
        common_keywords = {}
        for i, j, similarity in high_similarity_pairs:
            text1_words = set(jieba.lcut(texts[i]))
            text2_words = set(jieba.lcut(texts[j]))
            
            # 找出共同词汇
            common_words = text1_words.intersection(text2_words)
            
            # 过滤停用词和短词
            stop_words = {'的', '了', '在', '是', '我', '有', '和', '就', '不', '人', '都', '一', '一个', '上', '也', '很', '到', '说', '要', '去', '你', '会', '着', '没有', '看', '好', '自己', '这'}
            common_words = {word for word in common_words if len(word) > 1 and word not in stop_words}
            
            # 计算关键词重要性（基于相似度和词频）
            for word in common_words:
                if word not in common_keywords:
                    common_keywords[word] = 0
                common_keywords[word] += similarity
        
        # 转换为列表并排序
        keyword_importance = [(word, score) for word, score in common_keywords.items()]
        keyword_importance.sort(key=lambda x: x[1], reverse=True)
        
        # 如果top_k为-1，返回所有关键词；否则返回前top_k个
        if top_k == -1:
            return keyword_importance
        else:
            return keyword_importance[:top_k]
    
    def extract_all_keywords(self, df: pd.DataFrame, top_k: int = -1) -> Dict[str, Dict]:
        """
        从DataFrame中提取所有关键词（集成BERT特征）
        """
        results = {}
        
        # 按标签分组
        for label in df['label'].unique():
            if pd.isna(label) or label == '':
                continue
                
            label_texts = df[df['label'] == label]['cleaned_text'].tolist()
            
            # 提取TF-IDF关键词
            tfidf_keywords = self.extract_keywords_by_tfidf(label_texts, top_k)
            
            # 提取语义关键词（BERT）
            semantic_keywords = []
            if self.use_bert:
                try:
                    semantic_keywords = self.extract_semantic_keywords(label_texts, top_k)
                except Exception as e:
                    print(f"语义关键词提取失败: {e}")
            
            # 提取标签特定关键词
            label_keywords = self.extract_label_specific_keywords(label_texts, label, top_k)
            
            # 分析文本模式
            all_patterns = defaultdict(list)
            for text in label_texts:
                patterns = self.analyze_text_patterns(text)
                for pattern_type, matches in patterns.items():
                    all_patterns[pattern_type].extend(matches)
            
            # 统计模式频次
            pattern_stats = {}
            for pattern_type, matches in all_patterns.items():
                if top_k == -1:
                    pattern_stats[pattern_type] = dict(Counter(matches).most_common())
                else:
                    pattern_stats[pattern_type] = dict(Counter(matches).most_common(top_k))
            
            # 提取BERT特征
            bert_features = None
            if self.use_bert:
                try:
                    bert_features = self.extract_bert_features(label_texts)
                except Exception as e:
                    print(f"BERT特征提取失败: {e}")
            
            # 上下文关键词分析
            contextual_keywords = []
            if self.use_bert and len(label_texts) > 1:
                try:
                    # 分析同一标签下文本的语义相似性，找出共同关键词
                    contextual_keywords = self.extract_contextual_keywords(label_texts, top_k)
                except Exception as e:
                    print(f"上下文关键词分析失败: {e}")
            
            results[label] = {
                'tfidf_keywords': tfidf_keywords,
                'semantic_keywords': semantic_keywords,
                'contextual_keywords': contextual_keywords,
                'label_keywords': label_keywords,
                'patterns': pattern_stats,
                'bert_features_shape': bert_features.shape if bert_features is not None else None,
                'sample_count': len(label_texts)
            }
        
        return results
    
    def generate_keyword_report(self, keyword_results: Dict[str, Dict]) -> str:
        """
        生成关键词分析报告（包含BERT特征）
        """
        report = "关键词提取分析报告（集成深度学习）\n"
        report += "=" * 60 + "\n\n"
        
        for label, data in keyword_results.items():
            report += f"标签: {label}\n"
            report += f"样本数量: {data['sample_count']}\n"
            report += "-" * 40 + "\n"
            
            # TF-IDF关键词
            tfidf_count = len(data['tfidf_keywords'])
            report += f"TF-IDF关键词 (共{tfidf_count}个):\n"
            display_count = min(1000, tfidf_count)  # 最多显示1000个
            for keyword, score in data['tfidf_keywords'][:display_count]:
                report += f"  {keyword}: {score:.4f}\n"
            if tfidf_count > display_count:
                report += f"  ... 还有{tfidf_count - display_count}个关键词\n"
            
            # 语义关键词（BERT）
            if data.get('semantic_keywords'):
                semantic_count = len(data['semantic_keywords'])
                report += f"\n语义关键词 (BERT, 共{semantic_count}个):\n"
                display_count = min(1000, semantic_count)  # 最多显示1000个
                for keyword, score in data['semantic_keywords'][:display_count]:
                    report += f"  {keyword}: {score:.4f}\n"
                if semantic_count > display_count:
                    report += f"  ... 还有{semantic_count - display_count}个关键词\n"
            
            # 上下文关键词
            if data.get('contextual_keywords'):
                contextual_count = len(data['contextual_keywords'])
                report += f"\n上下文关键词 (共{contextual_count}个):\n"
                display_count = min(1000, contextual_count)  # 最多显示1000个
                for keyword, score in data['contextual_keywords'][:display_count]:
                    report += f"  {keyword}: {score:.4f}\n"
                if contextual_count > display_count:
                    report += f"  ... 还有{contextual_count - display_count}个关键词\n"
            
            # 标签特定关键词
            if data['label_keywords']:
                report += "\n标签特定关键词:\n"
                for category, keywords in data['label_keywords'].items():
                    if keywords:
                        report += f"  {category}: {', '.join(keywords[:5])}\n"
            
            # 模式统计
            report += "\n文本模式统计:\n"
            for pattern_type, stats in data['patterns'].items():
                if stats:
                    report += f"  {pattern_type}: {dict(list(stats.items())[:3])}\n"
            
            # BERT特征信息
            if data.get('bert_features_shape'):
                report += f"\nBERT特征维度: {data['bert_features_shape']}\n"
            
            report += "\n" + "=" * 60 + "\n\n"
        
        return report


if __name__ == "__main__":
    # 测试示例
    extractor = KeywordExtractor()
    
    # 测试文本
    test_text = "欢迎来到同城交友平台，这里有美女按摩服务，价格优惠，每小时200元"
    
    # 提取网站名
    websites = extractor.extract_websites(test_text)
    print(f"提取的网站名: {websites}")
    
    # 提取动词
    verbs = extractor.extract_verbs(test_text)
    print(f"提取的动词: {verbs}")
    
    # 分析模式
    patterns = extractor.analyze_text_patterns(test_text)
    print(f"文本模式: {patterns}")
