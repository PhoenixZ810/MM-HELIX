from abc import ABC, abstractmethod
import os
import json


class BaseGenerator(ABC):
    """
    问题生成器的基类，定义了生成问题的通用接口。
    """
    
    def __init__(self, output_folder):
        """
        初始化基础生成器。
        
        Args:
            output_folder: 输出文件夹路径
        """
        self.output_folder = output_folder
    
    @abstractmethod
    def generate(self, num_cases, difficulty, output_folder=None):
        """
        生成问题的抽象方法，需要被子类实现。
        
        Args:
            num_cases: 要生成的问题数量
            difficulty: 问题难度级别
            seed: 随机种子
            output_folder: 输出文件夹路径，覆盖构造函数中设置的路径

        保存问题到output_folder中：
        output_folder/
            /images/
                /question_name.png
            /annotations.json
            
        Returns(Optional):
            生成的问题列表
        """
        pass

    @abstractmethod
    def _get_difficulty_params(self, difficulty):
        """
        根据难度级别获取相应的参数配置。
        
        Args:
            difficulty: 难度级别（1-5）
            
        Returns:
            dict: 包含难度参数的字典
        """ 
        pass

    def save_annotations(self, annotations, output_folder):
        """
        保存标注到annotations.json文件中

        Args:
            annotations: 标注列表
            output_folder: 输出文件夹路径
        """
        existing_annotations = []
        annotations_path = os.path.join(output_folder, "annotations.json")
        if os.path.exists(annotations_path):
            try:
                with open(annotations_path, 'r', encoding='utf-8') as f:
                    existing_annotations = json.load(f)
            except (json.JSONDecodeError, IOError):
                existing_annotations = []

        # Add new annotations, avoiding duplicates based on index
        if existing_annotations:
            existing_indices = {item.get('index') for item in existing_annotations if item.get('index')}
            new_annotations = [item for item in annotations if item.get('index') not in existing_indices]
        else:
            new_annotations = annotations

        all_annotations = existing_annotations + new_annotations

        with open(annotations_path, 'w', encoding='utf-8') as f:
            json.dump(all_annotations, f, ensure_ascii=False, indent=2)

        print(f"Saved {len(new_annotations)} new annotations to {annotations_path}")