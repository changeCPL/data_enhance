"""
演示脚本
展示如何使用各个模块进行数据分析
"""

import pandas as pd
from data_preprocessor import DataPreprocessor
from keyword_extractor import KeywordExtractor
from pattern_analyzer import PatternAnalyzer
from data_augmentation import DataAugmentation


def demo_data_preprocessing():
    """演示数据预处理功能"""
    print("=" * 50)
    print("数据预处理演示")
    print("=" * 50)
    
    preprocessor = DataPreprocessor()
    
    # 测试文本清理
    test_texts = [
        "你好，我们这里有美女按摩服务\n包含换行符\t和制表符   多个空格",
        "访问网站 https://example.com 或 www.test.com",
        "联系我微信：abc123 或电话：13800138000"
    ]
    
    for i, text in enumerate(test_texts, 1):
        print(f"\n测试文本 {i}:")
        print(f"原始: {repr(text)}")
        
        cleaned = preprocessor.clean_text(text)
        print(f"清理后: {repr(cleaned)}")
        
        urls = preprocessor.extract_urls(text)
        phones = preprocessor.extract_phone_numbers(text)
        contacts = preprocessor.extract_wechat_qq(text)
        
        print(f"URL: {urls}")
        print(f"电话: {phones}")
        print(f"联系方式: {contacts}")


def demo_keyword_extraction():
    """演示关键词提取功能"""
    print("\n" + "=" * 50)
    print("关键词提取演示")
    print("=" * 50)
    
    extractor = KeywordExtractor()
    
    # 测试文本
    test_texts = [
        "你好，我们这里有20岁美女按摩服务，200元2小时，包你满意，加微信详聊",
        "欢迎来到正规博彩平台，稳赚不赔，充值送50%，秒到账，联系我",
        "兼职刷单，50元一单，简单操作，需要垫付100元，时间自由，加微信"
    ]
    
    for i, text in enumerate(test_texts, 1):
        print(f"\n测试文本 {i}: {text}")
        
        # 提取网站名
        websites = extractor.extract_websites(text)
        print(f"网站名: {websites}")
        
        # 提取动词
        verbs = extractor.extract_verbs(text)
        print(f"动词: {verbs}")
        
        # 分析模式
        patterns = extractor.analyze_text_patterns(text)
        print(f"模式: {patterns}")
    
    # 测试TF-IDF关键词提取
    print(f"\nTF-IDF关键词提取:")
    tfidf_keywords = extractor.extract_tfidf_keywords(test_texts, top_k=10)
    for keyword, score in tfidf_keywords:
        print(f"  {keyword}: {score:.4f}")


def demo_pattern_analysis():
    """演示话术模式分析功能"""
    print("\n" + "=" * 50)
    print("话术模式分析演示")
    print("=" * 50)
    
    analyzer = PatternAnalyzer()
    
    # 测试文本
    test_texts = [
        "你好，我是同城交友平台客服，我们这里有20岁美女按摩服务，200元2小时，包你满意，加微信详聊",
        "欢迎来到正规博彩平台，稳赚不赔，充值送50%，秒到账，联系我",
        "你好，我是银行客服，您的账户异常，需要验证身份，按提示操作"
    ]
    
    for i, text in enumerate(test_texts, 1):
        print(f"\n测试文本 {i}: {text}")
        
        # 分析文本结构
        features = analyzer.analyze_text_structure(text)
        print(f"结构特征: {features}")
        
        # 提取对话流程
        flow = analyzer.extract_conversation_flow(text)
        print(f"对话流程: {flow}")
        
        # 匹配模式
        if i == 1:
            templates = analyzer.pattern_templates['按摩色诱'] + analyzer.common_patterns
        elif i == 2:
            templates = analyzer.pattern_templates['博彩'] + analyzer.common_patterns
        else:
            templates = analyzer.pattern_templates['虚假客服'] + analyzer.common_patterns
        
        matches = analyzer.match_patterns(text, templates)
        print(f"匹配的模式: {[(t.name, match) for t, match in matches]}")


def demo_data_augmentation():
    """演示数据增强功能"""
    print("\n" + "=" * 50)
    print("数据增强演示")
    print("=" * 50)
    
    augmenter = DataAugmentation()
    
    # 测试模板生成
    print("1. 模板生成演示:")
    rule = augmenter.augmentation_rules['按摩色诱'][0]
    generated = augmenter.generate_augmented_text(rule, 3)
    for i, text in enumerate(generated, 1):
        print(f"  生成样本 {i}: {text}")
    
    # 测试标签增强
    print(f"\n2. 标签增强演示:")
    label_augmented = augmenter.generate_label_augmentations('按摩色诱', 5)
    for i, text in enumerate(label_augmented, 1):
        print(f"  增强样本 {i}: {text}")
    
    # 测试规则增强
    print(f"\n3. 规则增强演示:")
    test_text = "你好，我们这里有20岁美女按摩，200元2小时，包你满意"
    rule_augmented = augmenter.generate_rule_based_augmentations(test_text, '按摩色诱')
    print(f"  原始文本: {test_text}")
    for i, text in enumerate(rule_augmented, 1):
        print(f"  规则增强 {i}: {text}")


def demo_integrated_analysis():
    """演示集成分析功能"""
    print("\n" + "=" * 50)
    print("集成分析演示")
    print("=" * 50)
    
    # 创建示例数据
    sample_data = [
        {"conversation": "你好，我们这里有20岁美女按摩服务，200元2小时，包你满意，加微信详聊", "labelname": "按摩色诱", "datasrc": "demo"},
        {"conversation": "欢迎来到正规博彩平台，稳赚不赔，充值送50%，秒到账，联系我", "labelname": "博彩", "datasrc": "demo"},
        {"conversation": "兼职刷单，50元一单，简单操作，需要垫付100元，时间自由，加微信", "labelname": "兼职刷单", "datasrc": "demo"},
    ]
    
    # 转换为DataFrame
    df = pd.DataFrame(sample_data)
    
    # 数据预处理
    preprocessor = DataPreprocessor()
    processed_data = []
    
    for _, row in df.iterrows():
        cleaned_text = preprocessor.clean_text(row['conversation'])
        urls = preprocessor.extract_urls(cleaned_text)
        phones = preprocessor.extract_phone_numbers(cleaned_text)
        contacts = preprocessor.extract_wechat_qq(cleaned_text)
        
        processed_data.append({
            'cleaned_text': cleaned_text,
            'label': row['labelname'],
            'data_source': row['datasrc'],
            'urls': urls,
            'phone_numbers': phones,
            'contacts': contacts,
            'has_url': len(urls) > 0,
            'has_phone': len(phones) > 0,
            'has_contact': len(contacts) > 0,
        })
    
    processed_df = pd.DataFrame(processed_data)
    
    # 关键词提取
    extractor = KeywordExtractor()
    keyword_results = extractor.extract_all_keywords(processed_df)
    
    # 模式分析
    analyzer = PatternAnalyzer()
    pattern_results = {}
    for label in processed_df['label'].unique():
        pattern_results[label] = analyzer.analyze_label_patterns(processed_df, label)
    
    # 数据增强
    augmenter = DataAugmentation()
    augmented_data = augmenter.create_augmentation_dataset(processed_df, keyword_results, pattern_results, 0.5)
    
    print(f"原始数据: {len(processed_df)} 条")
    print(f"增强数据: {len(augmented_data)} 条")
    print(f"总数据: {len(processed_df) + len(augmented_data)} 条")
    
    # 显示增强数据示例
    print(f"\n增强数据示例:")
    for i, (_, row) in enumerate(augmented_data.head(3).iterrows(), 1):
        print(f"  {i}. {row['cleaned_text']} (标签: {row['label']}, 类型: {row['augmentation_type']})")


def main():
    """主演示函数"""
    print("反诈数据增强工具演示")
    print("本演示将展示各个模块的功能")
    
    # 运行各个演示
    demo_data_preprocessing()
    demo_keyword_extraction()
    demo_pattern_analysis()
    demo_data_augmentation()
    demo_integrated_analysis()
    
    print("\n" + "=" * 50)
    print("演示完成！")
    print("=" * 50)
    print("要运行完整分析，请使用:")
    print("python main.py --input example_data.jsonl --output reports")


if __name__ == "__main__":
    main()
