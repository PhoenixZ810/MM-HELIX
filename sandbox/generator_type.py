from generator.shortest_path_generator import ShortestPathGenerator
from generator.max_flow_generator import MaxFlowGenerator
from generator.topological_sort_generator import TopologicalSortGenerator
from generator.isomorphism_generator import IsomorphismGenerator
from generator.hamiltonian_path_generator import HamiltonianPathGenerator
from generator.hamiltonian_cycle_generator import HamiltonianCycleGenerator
from generator.eulerian_path_generator import EulerianPathGenerator
from generator.eulerian_cycle_generator import EulerianCycleGenerator

from generator.best_time_to_buy_and_sell_stock_generator import BestTimeToBuyAndSellStockGenerator
from generator.calcudoku_generator import CalcudokuGenerator
from generator.container_with_most_water_generator import ContainerWithMostWaterGenerator
from generator.count_hills_and_valleys_generator import CountHillsAndValleysInArrayGenerator
from generator.crypto_math_generator import CryptoMathGenerator

from generator.h_index_generator import HIndexGenerator
from generator.kukurasu_generator import KukurasuGenerator
from generator.largest_rectangle_in_histogram_generator import LargestRectangleInHistogramGenerator
from generator.longest_increasing_subsequence_generator import LongestIncreasingSubsequenceGenerator

from generator.skyscrapers_generator import SkyscrapersGenerator

from generator.trapping_rain_water_generator import TrappingRainWaterGenerator
from generator.twenty_four_points_generator import TwentyFourPointsGenerator
from generator.wordladder_generator import WordLadderGenerator

from generator.aquarium_generator import AquariumGenerator
from generator.binairo_generator import BinairoGenerator
from generator.futoshiki_generator import FutoshikiGenerator
from generator.bridges_generator import BridgesGenerator
from generator.eulero_generator import EuleroGenerator
from generator.campsite_generator import CampsiteGenerator
from generator.hanoi_generator import HanoiGenerator
from generator.kakuro_generator import KakuroGenerator
from generator.hitori_generator import HitoriGenerator
from generator.maze_generator import MazeGenerator
from generator.minesweeper_generator import MinesweeperGenerator
from generator.nibbles_generator import NibblesGenerator
from generator.sokoban_generator import SokobanGenerator
from generator.wordsearch_generator import WordSearchGenerator
from generator.numbrix_generator import NumbrixGenerator
from generator.tapa_generator import TapaGenerator
from generator.snake_generator import SnakeGenerator
from generator.shingoki_generator import ShingokiGenerator
from generator.nonograms_generator import NonogramsGenerator
from generator.slidingpuzzle_generator import SlidingPuzzleGenerator



generator_type_map = {
    "shortest_path": ShortestPathGenerator,
    "max_flow": MaxFlowGenerator,
    "topological_sort": TopologicalSortGenerator,
    "isomorphism": IsomorphismGenerator,
    "hamiltonian_path": HamiltonianPathGenerator,
    "hamiltonian_cycle": HamiltonianCycleGenerator,
    "eulerian_path": EulerianPathGenerator,
    "eulerian_cycle": EulerianCycleGenerator,
    "best_time_to_buy_and_sell_stock": BestTimeToBuyAndSellStockGenerator,
    "calcudoku": CalcudokuGenerator,
    "container_with_most_water": ContainerWithMostWaterGenerator,
    "count_hills_and_valleys": CountHillsAndValleysInArrayGenerator,
    "crypto_math": CryptoMathGenerator,
    "h_index": HIndexGenerator,
    "kukurasu": KukurasuGenerator,
    "longest_increasing_subsequence": LongestIncreasingSubsequenceGenerator,
    "largest_rectangle_in_histogram": LargestRectangleInHistogramGenerator,
    "skyscrapers": SkyscrapersGenerator,
    "trapping_rain_water": TrappingRainWaterGenerator,
    "24_points": TwentyFourPointsGenerator,
    "aquarium": AquariumGenerator,
    "binairo": BinairoGenerator,
    "futoshiki": FutoshikiGenerator,
    "bridges": BridgesGenerator,
    "eulero": EuleroGenerator,
    "campsite": CampsiteGenerator,
    "hanoi": HanoiGenerator,
    "kakuro": KakuroGenerator,
    "hitori": HitoriGenerator,
    "maze": MazeGenerator,
    "minesweeper": MinesweeperGenerator,
    "nibbles": NibblesGenerator,
    "sokoban": SokobanGenerator,
    "wordsearch": WordSearchGenerator,
    "numbrix": NumbrixGenerator,
    "tapa": TapaGenerator,
    "shingoki": ShingokiGenerator,
    "nonogram": NonogramsGenerator,
    "slidingpuzzle": SlidingPuzzleGenerator,
    "snake": SnakeGenerator,
    "wordladder": WordLadderGenerator,

}





