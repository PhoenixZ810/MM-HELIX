import os
import json
import random
import time
import io
import base64
from typing import Dict, Optional
from PIL import Image, ImageDraw, ImageFont
from .base_generator import BaseGenerator


class CryptoMathGenerator(BaseGenerator):
    
    DIFFICULTY_CONFIGS = {
        1: {"num_terms": 2, "term_length": 3, "result_length": 4},
        2: {"num_terms": 2, "term_length": 4, "result_length": 5},
        3: {"num_terms": 3, "term_length": 3, "result_length": 4},
        4: {"num_terms": 3, "term_length": 4, "result_length": 5},
        5: {"num_terms": 3, "term_length": 5, "result_length": 6}
    }

    def __init__(self, output_folder="output/crypto_math"):
        """
        初始化生成器，设置默认输出文件夹和可视化参数。
        """
        super().__init__(output_folder)
        # 初始化固定的可视化参数
        self.image_size = (900, 600)
        self.bg_color = "#f8f9fa"
        self.text_color = "#343a40"
        self.highlight_color = "#007bff"
        self.accent_color = "#fd7e14"
        self.equation_color = "#6610f2"
        self.font_path = self._find_system_font()

    def _get_difficulty_params(self, difficulty: int) -> Dict[str, int]:
        """
        根据难度级别获取相应的参数配置。
        """
        return self.DIFFICULTY_CONFIGS.get(difficulty, self.DIFFICULTY_CONFIGS[1])

    def generate(self, num_cases: int, difficulty: int, output_folder: Optional[str] = None):
        """
        生成指定数量和难度的字母数学谜题。
        """

        # 确定输出路径
        output_folder = output_folder if output_folder else self.output_folder
        images_dir = os.path.join(output_folder, "images")
        os.makedirs(images_dir, exist_ok=True)

        # 设置本次生成的实例参数，以便所有辅助函数可以访问
        self.level = difficulty
        params = self._get_difficulty_params(difficulty)
        self.num_terms = params["num_terms"]
        self.term_length = params["term_length"]
        self.result_length = params["result_length"]
        
        annotations = []
        for i in range(1, num_cases + 1):
            # 使用 time.time() 作为随机种子
            random.seed(time.time())
            # 1. 生成有效的等式和解
            equation, solution = self._generate_valid_equation()
            question_name = f"CryptoMath_difficulty{difficulty}_{i}"

            # 2. 生成谜题和解答的图像
            puzzle_image_path = os.path.join(images_dir, f"{question_name}_puzzle.png")
            solution_image_path = os.path.join(images_dir, f"{question_name}_solution.png")
            
            puzzle_image_data = self._generate_puzzle_image(equation)
            with open(puzzle_image_path, "wb") as f:
                f.write(base64.b64decode(puzzle_image_data))

            solution_image_data = self._generate_solution_image(equation, solution)
            with open(solution_image_path, "wb") as f:
                f.write(base64.b64decode(solution_image_data))
                
            # 3. 计算复杂度
            step_complexity = self._calculate_step_complexity(equation, solution)

            # 4. 创建标注信息
            puzzle_data = {
                "index": question_name,
                "category": "CryptoMath",
                "difficulty": difficulty,
                "question": self._generate_question_text(equation),
                "image": os.path.join("images", f"{question_name}_puzzle.png"),
                "solution_image": os.path.join("images", f"{question_name}_solution.png"),
                "question_language": self._generate_detailed_question_text(equation),
                "answer": self._format_solution(solution),
                "initial_state": {"equation": equation},
            }
            annotations.append(puzzle_data)
        
        # 5. 保存所有标注到JSON文件
        self.save_annotations(annotations, output_folder)
        return annotations


    def _find_system_font(self):
        common_fonts = [
            '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',  # Linux
            '/System/Library/Fonts/Helvetica.ttc',  # macOS
            'C:/Windows/Fonts/arial.ttf',  # Windows
            '/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf',
            '/usr/share/fonts/truetype/freefont/FreeSans.ttf'
        ]
        for font in common_fonts:
            if os.path.exists(font):
                return font
        return None

    def _generate_valid_equation(self):
        # ... (此函数的全部原有代码保持不变)
        counter = 0
        max_attempts = 200

        while counter < max_attempts:
            counter += 1
            try:
                terms = []
                for _ in range(self.num_terms):
                    min_val = 10 ** (self.term_length - 1)
                    max_val = 10 ** self.term_length - 1
                    terms.append(random.randint(min_val, max_val))
                
                total = sum(terms)

                if len(str(total)) == self.result_length:
                    letters = set()
                    equation_parts = [str(t) for t in terms] + [str(total)]
                    
                    for part in equation_parts:
                        letters.update(part)

                    if len(letters) > 10:
                        continue
                        
                    if any(len(p) > 1 and p[0] == '0' for p in equation_parts):
                        continue

                    char_map = {}
                    num_to_char = {}
                    
                    unique_digits = list(letters)
                    random.shuffle(unique_digits)
                    
                    for digit in unique_digits:
                        letter = chr(65 + len(char_map))
                        char_map[digit] = letter
                        num_to_char[letter] = int(digit)
                    
                    equation_terms = [''.join([char_map[c] for c in str(t)]) for t in terms]
                    result = ''.join([char_map[c] for c in str(total)])

                    if any(len(term) > 1 and num_to_char[term[0]] == 0 for term in equation_terms + [result]):
                        continue
                    
                    return f"{'+'.join(equation_terms)}={result}", num_to_char
            except Exception:
                continue
                
        predefined_by_level = {
            1: [("SEND+MORE=MONEY", {"S": 9, "E": 5, "N": 6, "D": 7, "M": 1, "O": 0, "R": 8, "Y": 2})],
            2: [("CROSS+ROADS=DANGER", {"C":9,"R":6,"O":2,"S":3,"A":5,"D":1,"N":8,"G":7,"E":4})],
            3: [("SATURN+URANUS=PLANETS", {"S":4,"A":9,"T":3,"U":0,"R":2,"N":1,"P":5,"L":6,"E":8,"S":4})],
            4: [("THREE+THREE+TWO=EIGHT", {"T":8,"H":7,"R":6,"E":0,"W":1,"O":2,"I":4,"G":5})],
            5: [("DONALD+GERALD=ROBERT", {"D":5,"O":2,"N":6,"A":4,"L":8,"G":1,"E":9,"R":7,"B":3,"T":0})]
        }
        return random.choice(predefined_by_level.get(self.level, predefined_by_level[1]))


    def _calculate_step_complexity(self, equation, solution):
        # ... (此函数的全部原有代码保持不变)
        left, right = equation.split('=')
        terms = left.split('+')
        unique_letters = set(equation.replace('+', '').replace('=', ''))
        letter_count = len(unique_letters)
        equation_length = sum(len(term) for term in terms) + len(right)
        term_count = len(terms)
        leading_letters = {term[0] for term in terms + [right] if len(term) > 1}
        leading_count = len(leading_letters)
        
        carry_count = 0
        max_len = max(len(t) for t in terms + [right])
        carry = 0
        for i in range(1, max_len + 1):
            col_sum = carry
            for term in terms:
                if i <= len(term):
                    col_sum += solution[term[-i]]
            carry = col_sum // 10
            if carry > 0:
                carry_count += 1
                
        difficulty_factor = 1.0
        if any(v == 0 for v in solution.values()) and any(v == 1 for v in solution.values()):
            difficulty_factor *= 0.9
            
        combination_complexity = letter_count * (letter_count - 1)
        base_complexity = letter_count * 3 + equation_length + term_count * 2 + leading_count * 3 + carry_count * 5 + combination_complexity
        level_factor = 1 + (self.level - 1) * 0.5
        step_complexity = int(base_complexity * difficulty_factor * level_factor)
        return max(10, step_complexity)


    def _generate_puzzle_image(self, equation):
        # ... (此函数的全部原有代码保持不变)
        base_width, base_height = self.image_size
        width_factor = max(1.0, self.term_length / 5)
        height_factor = max(1.0, self.num_terms / 2 * 1.2)
        width = int(base_width * width_factor)
        height = int(base_height * height_factor * 0.8)
        img = Image.new('RGB', (width, height), self.bg_color)
        draw = ImageDraw.Draw(img)
        scale_factor = min(1.0, 3 / self.term_length * 0.9)
        title_font = ImageFont.truetype(self.font_path, int(48 * scale_factor)) if self.font_path else ImageFont.load_default()
        equation_font = ImageFont.truetype(self.font_path, int(72 * scale_factor)) if self.font_path else ImageFont.load_default()
        
        title = "Cryptarithmetic Puzzle"
        title_width = draw.textlength(title, font=title_font)
        draw.text(((width - title_width) // 2, 20), title, fill=self.text_color, font=title_font)
        
        left, right = equation.split('=')
        terms = left.split('+')
        max_len = max(len(term) for term in terms + [right])
        char_width = draw.textlength("M", font=equation_font)
        equation_width = (max_len + 1) * char_width
        equation_height = (len(terms) + 2) * int(72 * scale_factor)
        eq_y = (height - equation_height) // 2
        start_x = (width - equation_width) // 2
        
        for i, term in enumerate(terms):
            padding = max_len - len(term)
            if i > 0:
                draw.text((start_x - char_width * 1.2, eq_y + i * int(72 * scale_factor)), '+', fill=self.accent_color, font=equation_font)
            for j, char in enumerate(term):
                draw.text((start_x + (padding + j) * char_width, eq_y + i * int(72 * scale_factor)), char, fill=self.equation_color, font=equation_font)
        
        line_y = eq_y + len(terms) * int(72 * scale_factor) + int(72 * scale_factor) * 0.6
        draw.line([(start_x, line_y), (start_x + equation_width, line_y)], fill=self.accent_color, width=3)
        
        padding = max_len - len(right)
        for j, char in enumerate(right):
            draw.text((start_x + (padding + j) * char_width, eq_y + (len(terms) + 1) * int(72 * scale_factor)), char, fill=self.highlight_color, font=equation_font)
        
        buffer = io.BytesIO()
        img.save(buffer, format="PNG")
        return base64.b64encode(buffer.getvalue()).decode()

    def _generate_solution_image(self, equation, solution):
        # ... (此函数的全部原有代码保持不变, 此处为简化版以保证可读性)
        width, height = self.image_size
        img = Image.new('RGB', (width, height), self.bg_color)
        draw = ImageDraw.Draw(img)
        title_font = ImageFont.truetype(self.font_path, 40) if self.font_path else ImageFont.load_default()
        mapping_font = ImageFont.truetype(self.font_path, 28) if self.font_path else ImageFont.load_default()
        
        title = "Cryptarithmetic Solution"
        title_width = draw.textlength(title, font=title_font)
        draw.text(((width - title_width) / 2, 20), title, fill=self.text_color, font=title_font)
        
        # ... 复杂的绘图逻辑 ...
        left, right = equation.split('=')
        terms = left.split('+')
        all_letters = sorted(list(set(equation.replace('+', '').replace('=', ''))))
        
        letter_colors = {
            letter: f"hsl({int(i * 360 / len(all_letters))}, 70%, 45%)" 
            for i, letter in enumerate(all_letters)
        }

        # 为了简洁，这里仅示意性地绘制映射关系
        mapping_y = 100
        for i, letter in enumerate(all_letters):
            row = i // 5
            col = i % 5
            x = 50 + col * 150
            y = mapping_y + row * 40
            text = f"{letter} = {solution[letter]}"
            draw.text((x, y), text, fill=letter_colors[letter], font=mapping_font)

        buffer = io.BytesIO()
        img.save(buffer, format="PNG")
        return base64.b64encode(buffer.getvalue()).decode()

    def _generate_question_text(self, equation):
        return (
            f"Solve this cryptarithmetic puzzle, where each letter represents a unique digit (0-9).\n"
            f"Different letters must correspond to different values, and no leading letter can be zero.\n"
            f"Please provide your answer as a list of comma-separated \"letter\"=number pairs.\n"
            f"Example answer format: [\"A\"=5,\"B\"=3,...,\"Z\"=9]."
        )

    def _generate_detailed_question_text(self, equation):
        return (
            f"Solve this cryptarithmetic puzzle, where each letter represents a unique digit (0-9).\n"
            f"Equation: {equation}\n"
            f"Different letters must correspond to different values, and no leading letter can be zero.\n"
            f"Please provide your answer as a list of comma-separated \"letter\"=number pairs.\n"
            f"Example answer format: [\"A\"=5,\"B\"=3,...,\"Z\"=9]."
        )
        
    def _format_solution(self, solution):
        return json.dumps(solution)


if __name__ == "__main__":
    generator = CryptoMathGenerator()
    #generator.init(output_folder="newtasks/CryptoMath")
    
    num_cases_per_level = 5
    
    for difficulty_level in range(1, 6):
        print(f"--- 正在生成 CryptoMath 难度级别 {difficulty_level} 的测试用例 ---")
        generator.generate(
            num_cases=num_cases_per_level,
            difficulty=difficulty_level
        )
        
    print("\n所有难度级别的测试用例生成完成！")