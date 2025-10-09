"""
简单测试脚本
"""

import pandas as pd
import json
import numpy as np

# 测试JSON序列化修复
def test_json_serialization():
    print("测试JSON序列化...")
    
    # 创建包含numpy类型的数据
    test_data = {
        'numpy_int': np.int64(42),
        'numpy_float': np.float64(3.14),
        'normal_data': 'test'
    }
    
    # 转换函数
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
    
    try:
        converted_data = convert_numpy_types(test_data)
        json_str = json.dumps(converted_data, ensure_ascii=False, indent=2)
        print("✓ JSON序列化成功")
        print(f"转换后的数据: {converted_data}")
        return True
    except Exception as e:
        print(f"✗ JSON序列化失败: {e}")
        return False

# 测试DataFrame操作修复
def test_dataframe_operations():
    print("测试DataFrame操作...")
    
    # 创建测试DataFrame
    df = pd.DataFrame({
        'cleaned_text': [
            "你好，我们这里有20岁美女按摩服务，200元2小时，包你满意，加微信详聊",
            "欢迎来到正规博彩平台，稳赚不赔，充值送50%，秒到账，联系我"
        ],
        'label': ['按摩色诱', '博彩']
    })
    
    try:
        # 测试修复后的操作
        sample_indices = df.index.to_series().sample(min(2, len(df)))
        
        for idx in sample_indices:
            original_text = df.loc[idx, 'cleaned_text']
            print(f"索引 {idx}: {original_text}")
        
        print("✓ DataFrame操作成功")
        return True
    except Exception as e:
        print(f"✗ DataFrame操作失败: {e}")
        return False

if __name__ == "__main__":
    print("开始简单测试...")
    print("=" * 30)
    
    test1 = test_json_serialization()
    test2 = test_dataframe_operations()
    
    print("=" * 30)
    if test1 and test2:
        print("✓ 所有测试通过！")
    else:
        print("✗ 部分测试失败")
