import os
import random
import io
import base64
import time
from PIL import Image, ImageDraw, ImageFont
from collections import deque, defaultdict


from .base_generator import BaseGenerator

class WordLadderGenerator(BaseGenerator):
    """
    生成Word Ladder谜题的类
    Word Ladder是从起始单词通过每次修改一个字母最终变为目标单词的谜题
    """
    def __init__(self, output_folder):
        """
        初始化Word Ladder生成器
        Args:
            output_folder: 输出文件夹路径
        """
        super().__init__(output_folder)
        # 配置参数
        self.image_size = (800, 600)
        self.bg_color = "#f8f9fa"
        self.text_color = "#212529"
        self.accent_color = "#007bff"
        # 查找系统字体
        self.font_path = self._find_system_font()
        # 已使用的单词对，避免重复
        self.used_word_pairs = set()


    def _get_difficulty_params(self, difficulty):
        """
        根据难度级别获取相应的参数配置。
        Args:
            difficulty: 难度级别（1-5）
        Returns:
            dict: 包含难度参数的字典
        """
        if difficulty == 1:
            return {
                'min_steps': 3,
                'max_steps': 5,
                'word_length': 3
            }
        elif difficulty == 2:
            return {
                'min_steps': 4,
                'max_steps': 6,
                'word_length': 4
            }
        elif difficulty == 3:
            return {
                'min_steps': 5,
                'max_steps': 7,
                'word_length': 4
            }
        elif difficulty == 4:
            return {
                'min_steps': 6,
                'max_steps': 8,
                'word_length': 5
            }
        else:  # difficulty 5
            return {
                'min_steps': 7,
                'max_steps': 10,
                'word_length': 5
            }

    def generate(self, num_cases, difficulty, output_folder=None):
        """
        生成指定数量的Word Ladder谜题
        Args:
            num_cases: 要生成的问题数量
            difficulty: 问题难度级别
            output_folder: 输出文件夹路径，覆盖构造函数中设置的路径
        Returns:
            生成的问题列表
        """
        # 设置输出文件夹
        if output_folder is None:
            output_folder = self.output_folder
            
        # 创建输出目录结构
        os.makedirs(output_folder, exist_ok=True)
        image_folder = os.path.join(output_folder, "images")
        os.makedirs(image_folder, exist_ok=True)

        # 获取难度参数
        params = self._get_difficulty_params(difficulty)
        
        # 加载词典
        self.word_dict = self._load_word_dictionary()
        # 构建单词图 (相差一个字母的单词相连)
        self.word_graph = self._build_word_graph(params['word_length'])
        # 预计算可达路径
        self._analyze_word_graph(difficulty)
        
        # 生成谜题
        puzzles = []
        successful_cases = 0
        max_attempts = num_cases * 2  # 最多尝试生成的次数
        attempt = 0
        
        while successful_cases < num_cases and attempt < max_attempts:
            attempt += 1
            try:
                # 生成谜题
                case = self._generate_single_puzzle(successful_cases + 1, difficulty, output_folder, image_folder)
                puzzles.append(case)
                successful_cases += 1
            except ValueError as e:
                print(f"警告: {str(e)}")
                print(f"跳过当前尝试，继续生成下一个谜题...")
            except Exception as e:
                print(f"生成谜题时出错: {str(e)}")
                print(f"跳过当前尝试，继续生成下一个谜题...")
        
        if successful_cases < num_cases:
            print(f"警告: 只成功生成了 {successful_cases}/{num_cases} 个谜题")
            
        # 保存所有annotations到一个文件
        if puzzles:
            self.save_annotations(puzzles, output_folder)
            
        return puzzles

    def _find_system_font(self):
        """查找系统中可用的字体"""
        common_fonts = [
            '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', # Linux
            '/System/Library/Fonts/Helvetica.ttc', # macOS
            'C:/Windows/Fonts/arial.ttf', # Windows
            '/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf' # Ubuntu
        ]
        for font in common_fonts:
            if os.path.exists(font):
                return font
        return None

    def _load_word_dictionary(self):
        """初始化评估器，下载并加载英语词典"""
        english_words = set()
        try:
            import nltk
            import nltk.data
            # 检查 'words' 语料库是否已经下载
            try:
                nltk.data.find('corpora/words')
            except nltk.downloader.DownloadError:
                print("NLTK 'words' 语料库未找到，正在尝试下载...")
                nltk.download('words', quiet=True)
            
            from nltk.corpus import words
            english_words = set(words.words())
        except Exception as e:
            print(f"警告：无法加载NLTK词典 - {e}")
            # 可以选择加载备用词典或继续运行
            
        return english_words

    def _build_word_graph(self, word_length):
        """构建单词图，相差一个字母的单词相连，采用随机选择的方式提高效率"""
        graph = defaultdict(list)
        words = self.word_dict
        
        # 筛选特定长度的单词
        words_of_length = [w for w in words if len(w) == word_length]
        
        # 如果单词太少，还是使用全连接方式
        if len(words_of_length) <= 1000:
            for i, word1 in enumerate(words_of_length):
                for j, word2 in enumerate(words_of_length[i+1:], i+1):
                    if self._is_one_letter_apart(word1, word2):
                        graph[word1].append(word2)
                        graph[word2].append(word1)
            return graph
        
        # 基于字母位置构建索引以提高效率
        position_indices = self._build_position_indices(words_of_length)
        
        # 为每个单词找出潜在的相邻单词
        max_neighbors = 50  # 每个单词最多检查的潜在邻居数
        
        for word in words_of_length:
            # 获取潜在的邻居单词
            candidates = self._get_potential_neighbors(word, position_indices, max_neighbors)
            
            # 检查哪些单词真的只相差一个字母
            for candidate in candidates:
                if word != candidate and self._is_one_letter_apart(word, candidate):
                    graph[word].append(candidate)
        
        # 检查图的连通性，确保有足够的连接
        self._ensure_graph_connectivity(graph, words_of_length)
        return graph
        
    def _build_position_indices(self, words):
        """构建基于字母位置的索引，用于快速查找潜在邻居"""
        indices = defaultdict(set)
        for word in words:
            for i in range(len(word)):
                # 将单词的每个位置都替换为通配符，作为索引
                pattern = word[:i] + '*' + word[i+1:]
                indices[pattern].add(word)
        return indices
    
    def _get_potential_neighbors(self, word, position_indices, max_neighbors):
        """获取单词的潜在邻居"""
        candidates = set()
        
        # 通过位置索引查找可能的邻居
        for i in range(len(word)):
            pattern = word[:i] + '*' + word[i+1:]
            candidates.update(position_indices[pattern])
            
            # 如果候选数量已经足够，就停止添加
            if len(candidates) >= max_neighbors:
                break
        
        # 如果候选数量太多，随机采样
        if len(candidates) > max_neighbors:
            return random.sample(list(candidates), max_neighbors)
        
        return candidates
    
    def _ensure_graph_connectivity(self, graph, words_of_length):
        """确保图的连通性，添加额外的连接如有必要"""
        # 检查已在图中的单词
        words_in_graph = set(graph.keys())
        
        # 确保所有特定长度的单词都在图中
        for word in words_of_length:
            if word not in words_in_graph:
                # 为孤立的单词添加随机连接
                candidates = random.sample(list(words_in_graph), min(10, len(words_in_graph)))
                for candidate in candidates:
                    if self._is_one_letter_apart(word, candidate):
                        graph[word].append(candidate)
                        graph[candidate].append(word)
                        break

    def _analyze_word_graph(self, difficulty):
        """分析单词图并构建可达路径映射"""
        self.path_matrix = {}
        
        # 计算有多少单词可以作为起点（至少要有一个连接）
        # 仅考虑已在图中的单词（有连接的单词）
        connected_words = [word for word in self.word_graph.keys() if len(self.word_graph[word]) > 0]
        # 预计算一些单词对之间的路径长度（消耗资源较多，仅计算部分）
        sample_size = min(50, len(connected_words))
        sample_words = random.sample(connected_words, sample_size)

        # 获取难度参数
        params = self._get_difficulty_params(difficulty)
        min_steps = params['min_steps']
        max_steps = params['max_steps']
        word_length = params['word_length']

        min_steps = params['min_steps']
        max_steps = params['max_steps']
        
        # 为每个样本单词找出到其他单词的路径长度
        for start_word in sample_words:
            distances = self._bfs_distances(start_word, min_steps, max_steps)
            
            # 存储能在要求步数内到达的单词
            valid_targets = {}
            for target_word, distance in distances.items():
                if min_steps <= distance <= max_steps:
                    valid_targets[target_word] = distance
            
            if valid_targets:
                self.path_matrix[start_word] = valid_targets
        
        # 统计能形成有效路径的单词对数量
        total_pairs = 0
        for start in self.path_matrix:
            total_pairs += len(self.path_matrix[start])
        
    def _bfs_distances(self, start_word, min_steps, max_steps):
        """使用BFS计算从起始单词到所有可达单词的距离"""
        distances = {start_word: 0}
        queue = deque([start_word])
        
        while queue:
            current = queue.popleft()
            current_distance = distances[current]
            
            # 如果已经达到最大步数，不再继续探索
            if current_distance >= max_steps:
                continue
            
            for neighbor in self.word_graph[current]:
                if neighbor not in distances:
                    distances[neighbor] = current_distance + 1
                    queue.append(neighbor)
        
        # 移除起始单词
        distances.pop(start_word, None)
        return distances

    def _is_one_letter_apart(self, word1, word2):
        """检查两个单词是否只相差一个字母"""
        if len(word1) != len(word2):
            return False
        
        diff_count = 0
        for c1, c2 in zip(word1, word2):
            if c1 != c2:
                diff_count += 1
                if diff_count > 1:
                    return False
        
        return diff_count == 1

    def _find_word_ladder(self, start_word, target_word, difficulty=None):
        """使用BFS查找从起始单词到目标单词的路径"""
        if start_word == target_word:
            return [start_word]
        
        visited = {start_word: None}
        queue = deque([start_word])
        
        while queue:
            current = queue.popleft()
            
            for neighbor in self.word_graph[current]:
                if neighbor not in visited:
                    visited[neighbor] = current
                    if neighbor == target_word:
                        # 重建路径
                        path = [neighbor]
                        while path[-1] != start_word:
                            path.append(visited[path[-1]])
                        return list(reversed(path))
                    queue.append(neighbor)
        
        return None  # 没有找到路径

    def _find_valid_word_pair(self, difficulty):
        """查找有效的起始和目标单词对，确保路径长度在指定范围内"""
        # 获取难度参数
        params = self._get_difficulty_params(difficulty)
        min_steps = params['min_steps']
        max_steps = params['max_steps']
        word_length = params['word_length']
        
        # 如果已经预计算了路径，直接使用
        if hasattr(self, 'path_matrix') and self.path_matrix:
            # 首先尝试从预计算的矩阵中查找
            available_pairs = []
            for start in self.path_matrix:
                for target in self.path_matrix[start]:
                    if (start, target) not in self.used_word_pairs:
                        available_pairs.append((start, target))
            
            if available_pairs:
                start_word, target_word = random.choice(available_pairs)
                self.used_word_pairs.add((start_word, target_word))
                path = self._find_word_ladder(start_word, target_word, difficulty)
                return start_word, target_word, path
        
        # 如果预计算矩阵没有可用的词对，进行更广泛的搜索
        # 从已连接的单词中选择一些高连接度的单词
        word_connectivity = [(word, len(self.word_graph[word])) for word in self.word_graph.keys()]
        word_connectivity.sort(key=lambda x: x[1], reverse=True)
        
        # 选择连接度最高的前20%单词作为潜在起点
        top_words = [word for word, conn in word_connectivity[:max(20, len(word_connectivity)//5)]]
        
        # 设置随机种子
        random.seed(time.time())
        
        for _ in range(100):  # 尝试100次
            start_word = random.choice(top_words)
            
            # 使用预先计算的距离搜索适合目标
            distances = self._bfs_distances(start_word, min_steps, max_steps)
            valid_targets = [word for word, dist in distances.items()
                           if min_steps <= dist <= max_steps]
            
            if not valid_targets:
                continue
                
            target_word = random.choice(valid_targets)
            
            # 确保这个单词对未被使用过
            if (start_word, target_word) in self.used_word_pairs:
                continue
                
            # 查找路径
            path = self._find_word_ladder(start_word, target_word, difficulty)
            if path and len(path) - 1 >= min_steps and len(path) - 1 <= max_steps:
                self.used_word_pairs.add((start_word, target_word))
                return start_word, target_word, path
        
        # 再次尝试，但增加更多迭代次数
        for _ in range(300):  # 增加到300次尝试
            # 随机选择起始单词，但优先选择高连接度的单词
            start_word = random.choice(top_words)
            
            # 使用预先计算的距离搜索适合目标
            distances = self._bfs_distances(start_word, min_steps, max_steps)
            if not distances:
                continue
                
            valid_targets = [word for word, dist in distances.items()
                           if min_steps <= dist <= max_steps]
            
            if not valid_targets:
                continue
                
            target_word = random.choice(valid_targets)
            
            # 确保这个单词对未被使用过
            if (start_word, target_word) in self.used_word_pairs:
                continue
                
            # 查找路径
            path = self._find_word_ladder(start_word, target_word, difficulty)
            if path and len(path) - 1 >= min_steps and len(path) - 1 <= max_steps:
                self.used_word_pairs.add((start_word, target_word))
                return start_word, target_word, path
        
        # 如果还是失败，抛出异常
        raise ValueError(f"无法找到适合难度{difficulty}的单词对，请检查词典或调整难度设置。")

    def _generate_single_puzzle(self, case_id, difficulty, output_folder, image_folder):
        """
        生成单个Word Ladder谜题
        Args:
            case_id: 案例ID
            difficulty: 难度级别
            output_folder: 输出文件夹路径
            image_folder: 图像文件夹路径
        Returns:
            生成的数据点
        """
        # 获取难度参数
        params = self._get_difficulty_params(difficulty)
        min_steps = params['min_steps']
        max_steps = params['max_steps']
        word_length = params['word_length']
        
        # 步骤1: 查找有效的起始和目标单词对
        start_word, target_word, solution_path = self._find_valid_word_pair(difficulty)
        # 步骤2: 构建谜题数据
        puzzle_data = {
            'start_word': start_word,
            'target_word': target_word,
            'word_length': word_length,
            'min_steps': min_steps,
            'max_steps': max_steps,
            'difficulty': difficulty
        }
        # 步骤3: 生成图像
        puzzle_image_path, solution_image_path = self.visualize(
            {'puzzle_data': puzzle_data, 'solution_path': solution_path},
            case_id=case_id,
            difficulty=difficulty,
            output_folder=output_folder,
            image_folder=image_folder
        )
        
        # 步骤4: 生成数据点
        return self._create_datapoint(case_id, puzzle_data, solution_path, puzzle_image_path, difficulty)

    def visualize(self, puzzle, output_folder, image_folder, **kwargs):
        """生成谜题和解答的可视化图像"""
        case_id = kwargs.get('case_id', 0)
        difficulty = kwargs.get('difficulty', 1)
        
        puzzle_data = puzzle['puzzle_data']
        solution_path = puzzle['solution_path']
        
        # 生成谜题图像
        puzzle_image_data = self._generate_puzzle_image(puzzle_data)
        image_filename = f"wordladder_{difficulty}_{case_id}.png"
        # 保存文件路径
        full_puzzle_path = os.path.join(image_folder, image_filename)
        with open(full_puzzle_path, "wb") as f:
            f.write(base64.b64decode(puzzle_image_data))
        
        # 生成用于返回的相对路径
        puzzle_image_path = os.path.join("images", image_filename)
            
        # 生成解答图像
        solution_image_data = self._generate_solution_image(puzzle_data, solution_path)
        solution_image_filename = f"wordladder_{difficulty}_{case_id}_solution.png"
        # 保存文件路径
        full_solution_path = os.path.join(image_folder, solution_image_filename)
        with open(full_solution_path, "wb") as f:
            f.write(base64.b64decode(solution_image_data))
        
        # 生成用于返回的相对路径
        solution_image_path = os.path.join("images", solution_image_filename)
            
        return puzzle_image_path, solution_image_path

    def _generate_puzzle_image(self, puzzle_data):
        """为Word Ladder谜题生成图像"""
        width, height = self.image_size
        img = Image.new('RGB', (width, height), self.bg_color)
        draw = ImageDraw.Draw(img)
        
        # 设置字体
        title_font_size = 36
        word_font_size = 48
        instruction_font_size = 24
        
        try:
            if self.font_path:
                title_font = ImageFont.truetype(self.font_path, title_font_size)
                word_font = ImageFont.truetype(self.font_path, word_font_size)
                instruction_font = ImageFont.truetype(self.font_path, instruction_font_size)
            else:
                title_font = ImageFont.load_default()
                word_font = ImageFont.load_default()
                instruction_font = ImageFont.load_default()
        except:
            title_font = ImageFont.load_default()
            word_font = ImageFont.load_default()
            instruction_font = ImageFont.load_default()
        
        # 绘制标题
        title = "Word Ladder Puzzle"
        title_width = draw.textlength(title, font=title_font) if hasattr(draw, 'textlength') else title_font_size * len(title) * 0.6
        draw.text(((width - title_width) // 2, 40), title, fill=self.text_color, font=title_font)
        
        # 绘制起始和目标单词
        start_word = puzzle_data['start_word'].upper()
        target_word = puzzle_data['target_word'].upper()
        
        # 计算单词宽度
        start_width = draw.textlength(start_word, font=word_font) if hasattr(draw, 'textlength') else word_font_size * len(start_word) * 0.6
        target_width = draw.textlength(target_word, font=word_font) if hasattr(draw, 'textlength') else word_font_size * len(target_word) * 0.6
        
        # 绘制起始单词
        draw.text(((width - start_width) // 2, height // 3), start_word, fill=self.text_color, font=word_font)
        
        # 绘制向下箭头
        arrow_start_y = height // 3 + word_font_size + 20
        arrow_end_y = 2 * height // 3 - word_font_size - 20
        
        # 箭头主干
        draw.line([(width // 2, arrow_start_y), (width // 2, arrow_end_y)], fill=self.accent_color, width=3)
        
        # 箭头头部
        draw.line([(width // 2, arrow_end_y), (width // 2 - 10, arrow_end_y - 15)], fill=self.accent_color, width=3)
        draw.line([(width // 2, arrow_end_y), (width // 2 + 10, arrow_end_y - 15)], fill=self.accent_color, width=3)
        
        # 绘制目标单词
        draw.text(((width - target_width) // 2, 2 * height // 3), target_word, fill=self.text_color, font=word_font)
        
        # 转换为base64编码的图像
        buffer = io.BytesIO()
        img.save(buffer, format="PNG")
        image_data = base64.b64encode(buffer.getvalue()).decode()
        
        return image_data

    def _generate_solution_image(self, puzzle_data, solution_path):
        """为Word Ladder谜题的解答生成图像"""
        width, height = self.image_size
        
        # 调整高度以适应解决方案路径
        height = max(height, 150 + 80 * len(solution_path))
        img = Image.new('RGB', (width, height), self.bg_color)
        draw = ImageDraw.Draw(img)
        
        # 设置字体
        title_font_size = 36
        word_font_size = 36
        instruction_font_size = 24
        step_font_size = 20
        
        try:
            if self.font_path:
                title_font = ImageFont.truetype(self.font_path, title_font_size)
                word_font = ImageFont.truetype(self.font_path, word_font_size)
                instruction_font = ImageFont.truetype(self.font_path, instruction_font_size)
                step_font = ImageFont.truetype(self.font_path, step_font_size)
            else:
                title_font = ImageFont.load_default()
                word_font = ImageFont.load_default()
                instruction_font = ImageFont.load_default()
                step_font = ImageFont.load_default()
        except:
            title_font = ImageFont.load_default()
            word_font = ImageFont.load_default()
            instruction_font = ImageFont.load_default()
            step_font = ImageFont.load_default()
        
        # 绘制标题
        title = "Word Ladder Solution"
        title_width = draw.textlength(title, font=title_font) if hasattr(draw, 'textlength') else title_font_size * len(title) * 0.6
        draw.text(((width - title_width) // 2, 40), title, fill=self.text_color, font=title_font)
        
        # 绘制解决方案路径
        y_position = 120
        for i, word in enumerate(solution_path):
            word_upper = word.upper()
            
            # 计算单词宽度
            word_width = draw.textlength(word_upper, font=word_font) if hasattr(draw, 'textlength') else word_font_size * len(word_upper) * 0.6
            
            # 绘制步骤编号
            step_text = f"Step {i}:" if i > 0 else "Start:"
            draw.text((width // 4 - 50, y_position), step_text, fill=self.text_color, font=step_font)
            
            # 绘制单词
            draw.text((width // 2 - word_width // 2, y_position), word_upper, fill=self.text_color, font=word_font)
            
            # 如果不是最后一个单词，高亮显示变化的字母
            if i < len(solution_path) - 1:
                next_word = solution_path[i + 1]
                for j, (c1, c2) in enumerate(zip(word, next_word)):
                    if c1 != c2:
                        # 高亮显示将要变化的字母
                        highlight_char = word_upper[j]
                        char_width = draw.textlength(highlight_char, font=word_font) if hasattr(draw, 'textlength') else word_font_size * 0.6
                        char_x = width // 2 - word_width // 2 + draw.textlength(word_upper[:j], font=word_font) if hasattr(draw, 'textlength') else width // 2 - word_width // 2 + word_font_size * 0.6 * j
                        
                        # 绘制高亮背景
                        draw.rectangle([char_x - 2, y_position - 2, char_x + char_width + 2, y_position + word_font_size + 2],
                                    fill=self.accent_color)
                        
                        # 重新绘制字符（白色）
                        draw.text((char_x, y_position), highlight_char, fill="#ffffff", font=word_font)
                        break
            
            # 如果不是最后一个单词，绘制箭头
            if i < len(solution_path) - 1:
                arrow_y = y_position + word_font_size + 10
                
                # 箭头主干
                draw.line([(width // 2, arrow_y), (width // 2, arrow_y + 20)], fill=self.accent_color, width=2)
                
                # 箭头头部
                draw.line([(width // 2, arrow_y + 20), (width // 2 - 8, arrow_y + 10)], fill=self.accent_color, width=2)
                draw.line([(width // 2, arrow_y + 20), (width // 2 + 8, arrow_y + 10)], fill=self.accent_color, width=2)
            
            y_position += 80
        
        # 绘制总步数
        steps_text = f"Total steps: {len(solution_path) - 1}"
        steps_width = draw.textlength(steps_text, font=instruction_font) if hasattr(draw, 'textlength') else instruction_font_size * len(steps_text) * 0.6
        draw.text(((width - steps_width) // 2, y_position - 10), steps_text, fill=self.text_color, font=instruction_font)
        
        # 转换为base64编码的图像
        buffer = io.BytesIO()
        img.save(buffer, format="PNG")
        image_data = base64.b64encode(buffer.getvalue()).decode()
        
        return image_data

    def _create_datapoint(self, case_id, puzzle_data, solution_path, puzzle_image_path, difficulty):
        """创建数据点，并添加step字段评估解题步骤复杂度"""
        # 计算解题步骤复杂度
        step_complexity = self._calculate_step_complexity(puzzle_data, solution_path, difficulty)
        return {
            "index": f"WordLadder_difficulty{difficulty}_{case_id}",
            "category": "WordLadder",
            "question": self._generate_question_text(puzzle_data),
            "image": puzzle_image_path,
            "question_language": self._generate_detailed_question_text(puzzle_data),
            "answer": " -> ".join(solution_path),
#           "solution": solution_path,
#           "start_word": puzzle_data['start_word'],
            "difficulty": difficulty,
#           "target_word": puzzle_data['target_word'],
            "initial_state":{
                "solution": solution_path,
                "start_word": puzzle_data['start_word'],
                "target_word": puzzle_data['target_word'],
            },
#           "step": step_complexity # 新增字段：评估解题步骤的复杂程度
        }

    def _calculate_step_complexity(self, puzzle_data, solution_path, difficulty):
        """计算词梯谜题解题步骤的复杂度"""
        start_word = puzzle_data['start_word']
        target_word = puzzle_data['target_word']
        
        # 1. 基础复杂度 - 基于路径长度
        # 路径长度（包括起点和终点）
        path_length = len(solution_path)
        
        # 每一步的基础复杂度
        base_step_complexity = 10
        
        # 总体路径复杂度
        base_complexity = (path_length - 1) * base_step_complexity
        
        # 2. 单词复杂度 - 基于单词常见程度和长度
        # 单词长度影响
        word_length = len(start_word)
        length_factor = 1 + (word_length - 3) * 0.2  # 3个字母为基准，每增加1个字母复杂度增加20%
        
        # 3. 分支复杂度 - 基于每一步可能的选择数量
        total_branch_complexity = 0
        
        # 对于路径中的每个单词（除了目标词），计算其可能的下一步数量
        for i in range(len(solution_path) - 1):
            current_word = solution_path[i]
            
            # 获取当前单词的所有邻居
            neighbors = self.word_graph.get(current_word, [])
            
            # 计算未访问的邻居数量（分支数）
            unvisited_neighbors = [w for w in neighbors if w not in solution_path[:i]]
            
            # 分支复杂度基于可选择的路径数量
            branch_count = len(unvisited_neighbors)
            
            # 分支复杂度系数 - 更多选择意味着更难的决策
            branch_complexity = min(branch_count, 10) * 2  # 限制最大值，避免极端情况
            total_branch_complexity += branch_complexity
        
        # 平均每步的分支复杂度
        avg_branch_complexity = total_branch_complexity / (path_length - 1) if path_length > 1 else 1
        
        # 4. 路径不直接性 - 评估路径是否"弯曲"
        # 计算汉明距离（不同字母的数量）
        hamming_distance = sum(c1 != c2 for c1, c2 in zip(start_word, target_word))
        
        # 路径长度与汉明距离的比率，越高表示路径越不直接
        indirectness_ratio = (path_length - 1) / hamming_distance if hamming_distance > 0 else 1
        indirectness_factor = indirectness_ratio * 0.5  # 调整影响系数
        
        # 5. 中间词罕见程度 - 使用更罕见的单词增加复杂度
        # 这里简化为使用单词图中单词数量作为衡量标准
        connected_word_count = len(self.word_graph)
        rarity_factor = 1 + 5 / max(connected_word_count, 1)  # 避免除以零
        
        # 6. 级别调整
        level_factor = 1 + (puzzle_data['difficulty'] - 1) * 0.3  # 根据难度级别调整
        
        # 组合所有因子计算最终复杂度
        complexity_factors = [
            base_complexity,
            length_factor,
            avg_branch_complexity,
            indirectness_factor,
            rarity_factor
        ]
        
        # 乘积计算最终复杂度
        total_factor = 1
        for factor in complexity_factors[1:]:  # 跳过base_complexity
            total_factor *= factor
        
        # 最终复杂度计算
        step_complexity = int(base_complexity * total_factor * level_factor)
        
        # 设置最小值为10
        step_complexity = max(10, step_complexity)
        
        return step_complexity

    def _generate_question_text(self, puzzle_data):
        """生成带图像的问题描述"""
        start_word = puzzle_data['start_word']
        target_word = puzzle_data['target_word']
        min_steps = puzzle_data['min_steps']
        max_steps = puzzle_data['max_steps']
        
        return (
            f"This is a Word Ladder puzzle. Transform the left word into right word by changing one letter at a time, ensuring that each step forms a valid word. The rules are as follows\n"+
            f"1. Change exactly one letter at a time.\n"
            f"2. Each step must form a valid English word.\n"
            f"Please provide the complete solution path from '{start_word}' to '{target_word}' as a list of strings.\n"
            f"Example answer format: [\"hug\", \"bug\", \"beg\", \"bet\", \"set\"]."
        )

    def _generate_detailed_question_text(self, puzzle_data):
        """生成纯文本问题描述"""
        start_word = puzzle_data['start_word']
        target_word = puzzle_data['target_word']
        min_steps = puzzle_data['min_steps']
        max_steps = puzzle_data['max_steps']
        
        return (
            f"This is a Word Ladder puzzle. Transform the left word into right word by changing one letter at a time, ensuring that each step forms a valid word. The rules are as follows\n"+
            f"1. Change exactly one letter at a time.\n"
            f"2. Each step must form a valid English word.\n"
            f"3. Find a solution path with {min_steps} to {max_steps} steps.\n"
            f"Starting word: {start_word}.\n"
            f"Target word: {target_word}.\n"
            f"Please provide the complete solution path from '{start_word}' to '{target_word}' as a list of strings.\n"
            f"Example answer format: [\"hug\", \"bug\", \"beg\", \"bet\", \"set\"]."
        )

    def solve(self, puzzle):
        """求解Word Ladder谜题 - 返回最短路径"""
        start_word = puzzle['start_word']
        target_word = puzzle['target_word']
        difficulty = puzzle.get('difficulty', 1)  # 如果没有难度信息，默认为1
        
        # 使用BFS查找路径
        return self._find_word_ladder(start_word, target_word, difficulty)

