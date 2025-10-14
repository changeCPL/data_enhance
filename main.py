"""
主程序
整合所有模块，提供完整的数据分析和增强流程
"""

import os
import json
import pandas as pd
from typing import Dict, Any
import argparse
from datetime import datetime

from data_preprocessor import DataPreprocessor
from keyword_extractor import KeywordExtractor
from pattern_analyzer import PatternAnalyzer
from data_augmentation import DataAugmentation

# 导入深度学习模块
try:
    from semantic_analyzer import SemanticAnalyzer, SemanticConfig
    from dl_text_generator import DLTextGenerator, GenerationConfig
    DL_AVAILABLE = True
except ImportError:
    DL_AVAILABLE = False
    print("警告: 深度学习模块未找到，将使用传统方法")


class FraudDetectionAnalyzer:
    def __init__(self, use_dl: bool = True):
        self.use_dl = use_dl and DL_AVAILABLE
        
        # 传统模块
        self.preprocessor = DataPreprocessor()
        self.keyword_extractor = KeywordExtractor(use_bert=self.use_dl)
        self.pattern_analyzer = PatternAnalyzer()
        self.augmenter = DataAugmentation(use_dl=self.use_dl)
        
        # 深度学习模块
        self.semantic_analyzer = None
        
        if self.use_dl:
            try:
                self.semantic_analyzer = SemanticAnalyzer()
                print("语义分析模块初始化成功")
            except Exception as e:
                print(f"语义分析模块初始化失败: {e}")
                self.use_dl = False
        
        # 结果存储
        self.processed_data = None
        self.keyword_results = None
        self.pattern_results = None
        self.augmented_data = None
        self.semantic_results = None
    
    def load_and_process_data(self, jsonl_file: str) -> pd.DataFrame:
        """
        加载并处理JSONL数据
        """
        print(f"正在加载数据: {jsonl_file}")
        
        if not os.path.exists(jsonl_file):
            raise FileNotFoundError(f"文件不存在: {jsonl_file}")
        
        # 处理数据
        self.processed_data = self.preprocessor.process_jsonl_file(jsonl_file)
        
        print(f"数据加载完成，共 {len(self.processed_data)} 条记录")
        
        # 显示数据统计
        stats = self.preprocessor.get_text_statistics(self.processed_data)
        print(f"数据统计:")
        print(f"  总样本数: {stats['total_samples']}")
        print(f"  平均文本长度: {stats['avg_text_length']:.2f}")
        print(f"  标签分布: {stats['label_distribution']}")
        
        return self.processed_data
    
    def extract_keywords(self) -> Dict[str, Any]:
        """
        提取关键词
        """
        print("正在提取关键词...")
        
        if self.processed_data is None:
            raise ValueError("请先加载数据")
        
        self.keyword_results = self.keyword_extractor.extract_all_keywords(self.processed_data)
        
        print("关键词提取完成")
        return self.keyword_results
    
    def analyze_patterns(self) -> Dict[str, Any]:
        """
        分析话术模式
        """
        print("正在分析话术模式...")
        
        if self.processed_data is None:
            raise ValueError("请先加载数据")
        
        self.pattern_results = {}
        
        for label in self.processed_data['label'].unique():
            if pd.isna(label) or label == '':
                continue
            
            print(f"  分析标签: {label}")
            self.pattern_results[label] = self.pattern_analyzer.analyze_label_patterns(
                self.processed_data, label
            )
        
        print("话术模式分析完成")
        return self.pattern_results
    
    
    def run_semantic_analysis(self) -> Dict[str, Any]:
        """
        运行语义相似度分析
        """
        if not self.use_dl or self.semantic_analyzer is None:
            print("语义分析器不可用")
            return {}
        
        print("正在运行语义相似度分析...")
        
        if self.processed_data is None:
            raise ValueError("请先加载数据")
        
        try:
            self.semantic_results = self.semantic_analyzer.analyze_semantic_patterns(self.processed_data)
            
            # 检测语义异常
            anomalies = self.semantic_analyzer.detect_semantic_anomalies(self.processed_data)
            self.semantic_results['anomalies'] = anomalies
            
            # 生成语义空间可视化
            visualization_results = self.semantic_analyzer.visualize_semantic_space(
                self.processed_data, "reports/semantic_visualization.png"
            )
            self.semantic_results['visualization'] = visualization_results
            
            print("语义相似度分析完成")
            return self.semantic_results
            
        except Exception as e:
            print(f"语义分析失败: {e}")
            return {}
    
    def generate_augmented_data(self, augmentation_ratio: float = 0.5) -> pd.DataFrame:
        """
        生成增强数据
        """
        print("正在生成增强数据...")
        
        if self.processed_data is None or self.keyword_results is None or self.pattern_results is None:
            raise ValueError("请先完成关键词提取和模式分析")
        
        self.augmented_data = self.augmenter.create_augmentation_dataset(
            self.processed_data, self.keyword_results, self.pattern_results, augmentation_ratio
        )
        
        print(f"增强数据生成完成，共 {len(self.augmented_data)} 条新样本")
        return self.augmented_data
    
    def generate_reports(self, output_dir: str = "reports"):
        """
        生成分析报告
        """
        print("正在生成报告...")
        
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 生成关键词报告
        if self.keyword_results:
            keyword_report = self.keyword_extractor.generate_keyword_report(self.keyword_results)
            keyword_file = os.path.join(output_dir, f"keyword_report_{timestamp}.txt")
            with open(keyword_file, 'w', encoding='utf-8') as f:
                f.write(keyword_report)
            print(f"关键词报告已保存: {keyword_file}")
        
        # 生成模式分析报告
        if self.pattern_results:
            pattern_report = self.pattern_analyzer.generate_pattern_report(self.pattern_results)
            pattern_file = os.path.join(output_dir, f"pattern_report_{timestamp}.txt")
            with open(pattern_file, 'w', encoding='utf-8') as f:
                f.write(pattern_report)
            print(f"模式分析报告已保存: {pattern_file}")
        
        # 生成prompt报告
        if self.keyword_results and self.pattern_results:
            prompt_report = self.augmenter.generate_prompts_report(self.keyword_results, self.pattern_results)
            prompt_file = os.path.join(output_dir, f"prompt_report_{timestamp}.txt")
            with open(prompt_file, 'w', encoding='utf-8') as f:
                f.write(prompt_report)
            print(f"Prompt报告已保存: {prompt_file}")
        
        
        # 生成语义分析报告
        if self.semantic_results:
            semantic_report = self.semantic_analyzer.generate_semantic_report(self.semantic_results)
            semantic_file = os.path.join(output_dir, f"semantic_report_{timestamp}.txt")
            with open(semantic_file, 'w', encoding='utf-8') as f:
                f.write(semantic_report)
            print(f"语义分析报告已保存: {semantic_file}")
        
        # 保存增强数据
        if self.augmented_data is not None:
            augmented_file = os.path.join(output_dir, f"augmented_data_{timestamp}.jsonl")
            self.save_dataframe_to_jsonl(self.augmented_data, augmented_file)
            print(f"增强数据已保存: {augmented_file}")
        
        # 保存处理后的原始数据
        if self.processed_data is not None:
            processed_file = os.path.join(output_dir, f"processed_data_{timestamp}.jsonl")
            self.save_dataframe_to_jsonl(self.processed_data, processed_file)
            print(f"处理后数据已保存: {processed_file}")
    
    def save_results_json(self, output_dir: str = "reports"):
        """
        保存结果为JSON格式
        """
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
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
        
        results = {
            'timestamp': timestamp,
            'keyword_results': convert_numpy_types(self.keyword_results),
            'pattern_results': convert_numpy_types(self.pattern_results),
            'data_stats': convert_numpy_types(self.preprocessor.get_text_statistics(self.processed_data)) if self.processed_data is not None else None
        }
        
        results_file = os.path.join(output_dir, f"analysis_results_{timestamp}.json")
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        
        print(f"分析结果已保存: {results_file}")
    
    def save_dataframe_to_jsonl(self, df: pd.DataFrame, filepath: str):
        """
        将DataFrame保存为JSONL格式
        """
        with open(filepath, 'w', encoding='utf-8') as f:
            for _, row in df.iterrows():
                # 转换为字典并处理NaN值
                row_dict = row.to_dict()
                # 处理复杂数据类型和NaN值
                cleaned_dict = {}
                for k, v in row_dict.items():
                    try:
                        # 处理NaN值 - 使用try-except避免复杂类型的错误
                        if pd.isna(v):
                            continue
                    except (ValueError, TypeError):
                        # 如果pd.isna()无法处理，跳过NaN检查
                        pass
                    
                    # 处理列表类型
                    if isinstance(v, list):
                        # 如果列表为空，跳过
                        if len(v) == 0:
                            continue
                        # 如果列表不为空，保留
                        cleaned_dict[k] = v
                    # 处理numpy数组类型
                    elif hasattr(v, 'tolist'):
                        # 转换为Python列表
                        v_list = v.tolist()
                        if len(v_list) == 0:
                            continue
                        cleaned_dict[k] = v_list
                    # 处理其他类型
                    else:
                        cleaned_dict[k] = v
                
                f.write(json.dumps(cleaned_dict, ensure_ascii=False) + '\n')
    
    def run_full_analysis(self, jsonl_file: str, augmentation_ratio: float = 0.5, 
                         output_dir: str = "reports", use_dl: bool = True):
        """
        运行完整分析流程
        """
        print("开始完整分析流程...")
        print("=" * 50)
        
        try:
            # 1. 加载和处理数据
            self.load_and_process_data(jsonl_file)
            
            # 2. 提取关键词
            self.extract_keywords()
            
            # 3. 分析话术模式
            self.analyze_patterns()
            
            # 4. 深度学习分析（如果启用）
            if use_dl and self.use_dl:
                print("\n开始深度学习分析...")
                # 语义相似度分析
                self.run_semantic_analysis()
            
            # 5. 生成增强数据
            self.generate_augmented_data(augmentation_ratio)
            
            # 6. 生成报告
            self.generate_reports(output_dir)
            
            # 7. 保存JSON结果
            self.save_results_json(output_dir)
            
            print("=" * 50)
            print("分析完成！")
            
            # 显示分析摘要
            self._print_analysis_summary()
            
        except Exception as e:
            print(f"分析过程中出现错误: {e}")
            raise
    
    def _print_analysis_summary(self):
        """打印分析摘要"""
        print("\n分析摘要:")
        print("-" * 30)
        
        if self.processed_data is not None:
            print(f"原始数据: {len(self.processed_data)} 条")
        
        if self.augmented_data is not None:
            print(f"增强数据: {len(self.augmented_data)} 条")
            total = len(self.processed_data) + len(self.augmented_data)
            print(f"总数据: {total} 条")
        
        if self.use_dl:
            print("深度学习功能: 已启用")
            if self.semantic_results:
                print(f"语义分析: 完成 ({len(self.semantic_results)} 个标签)")
        else:
            print("深度学习功能: 未启用")
        
        print("-" * 30)


def main():
    parser = argparse.ArgumentParser(description='反诈数据分析和增强工具（集成深度学习）')
    parser.add_argument('--input', '-i', required=True, help='输入JSONL文件路径')
    parser.add_argument('--output', '-o', default='reports', help='输出目录')
    parser.add_argument('--ratio', '-r', type=float, default=0.5, help='数据增强比例')
    parser.add_argument('--mode', '-m', choices=['full', 'keywords', 'patterns', 'augment', 'semantic'], 
                       default='full', help='运行模式')
    parser.add_argument('--use-dl', action='store_true', default=True, help='启用深度学习功能')
    parser.add_argument('--no-dl', action='store_true', help='禁用深度学习功能')
    
    args = parser.parse_args()
    
    # 确定是否使用深度学习
    use_dl = args.use_dl and not args.no_dl
    
    analyzer = FraudDetectionAnalyzer(use_dl=use_dl)
    
    if args.mode == 'full':
        analyzer.run_full_analysis(args.input, args.ratio, args.output, use_dl)
    elif args.mode == 'keywords':
        analyzer.load_and_process_data(args.input)
        analyzer.extract_keywords()
        analyzer.generate_reports(args.output)
    elif args.mode == 'patterns':
        analyzer.load_and_process_data(args.input)
        analyzer.analyze_patterns()
        analyzer.generate_reports(args.output)
    elif args.mode == 'augment':
        analyzer.load_and_process_data(args.input)
        analyzer.extract_keywords()
        analyzer.analyze_patterns()
        analyzer.generate_augmented_data(args.ratio)
        analyzer.generate_reports(args.output)
    elif args.mode == 'semantic':
        analyzer.load_and_process_data(args.input)
        analyzer.run_semantic_analysis()
        analyzer.generate_reports(args.output)


if __name__ == "__main__":
    # 如果没有命令行参数，运行示例
    import sys
    if len(sys.argv) == 1:
        print("反诈数据分析和增强工具（集成深度学习）")
        print("使用方法:")
        print("  python main.py --input data.jsonl --output reports --ratio 0.5")
        print("  python main.py --input data.jsonl --mode keywords")
        print("  python main.py --input data.jsonl --mode patterns")
        print("  python main.py --input data.jsonl --mode augment")
        print("  python main.py --input data.jsonl --mode semantic")
        print("  python main.py --input data.jsonl --no-dl  # 禁用深度学习")
        print("\n深度学习功能:")
        print("  - 语义相似度分析")
        print("  - 智能文本生成")
        print("  - BERT特征提取")
        print("\n示例数据格式:")
        print('{"conversation": "你好，我们这里有美女按摩服务", "labelname": "按摩色诱", "datasrc": "ocr"}')
    else:
        main()
