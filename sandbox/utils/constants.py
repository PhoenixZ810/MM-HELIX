PROMPT_SOKOBAN_IMAGE = """
Your task is to solve the Sokoban puzzle according to the rules and current state shown in the image:

### Game Rules:
1. You are the player and can move up, down, left, or right
2. You can push boxes one space at a time
3. You cannot pull boxes
4. Boxes can only be pushed if there's an empty space behind them
5. The goal is to push all boxes onto target positions
6. Walls cannot be moved through or pushed

### You will be given an image, in the image:

1. Red circles represent the player
2. Blue squares represent boxes
3. Yellow circles represent target positions
4. Brown blocks represent walls
5. Other blocks represent empty spaces

### Direction Definitions:
- "up": Move up
- "down": Move down
- "left": Move left
- "right": Move right

### Output Format Requirements:
Your final answer should be in the format of a space-separated sequence of moves like: up right down left
"""

PROMPT_SOKOBAN = """
Your task is to solve the Sokoban puzzle according to the rules and current state shown in the image:

### Game Rules:
1. You are the player and can move up, down, left, or right
2. You can push boxes one space at a time
3. You cannot pull boxes
4. Boxes can only be pushed if there's an empty space behind them
5. The goal is to push all boxes onto target positions
6. Walls cannot be moved through or pushed

### You will be given sokoban state

1. @ represent the player
2. $ represent boxes
3. . circles represent target positions
4. # blocks represent walls

### Direction Definitions:
- "up": Move up
- "down": Move down
- "left": Move left
- "right": Move right

### Current Sokoban State is shown below:
{}

### Output Format Requirements:
Your final answer should be in the format of a space-separated sequence of moves like: up right down left
"""



PROMPT_MAZE_IMAGE = """
Your task is to solve the maze game according to the rules and current state below:

### Game Rules:
1. The maze consists of a grid of cells
2. Walls are represented by **bold black line** between cells, not as cells themselves
3. You can move horizontally or vertically between adjacent cells if there is no wall between them
4. You can only move through one cell at a time in any direction
5. The goal is to find a path from the start cell (Green Circle) to the end cell (Red Cross)

### Direction Definitions:
- "up": Move to the cell above the current position
- "down": Move to the cell below the current position
- "left": Move to the cell to the left of the current position
- "right": Move to the cell to the right of the current position

### Current Maze State:
The maze is represented in the image shown below

In this representation:
- green circule marks the start position
- red cross marks the end position

### Output Format Requirements:
Your final answer should be in the format like: right down left up
"""


PROMPT_MAZE = """
Your task is to solve the maze game according to the rules and current state below:

### Game Rules:
1. The maze consists of a grid of cells
2. Walls are represented by the '+', '-', and '|' characters between cells, not as cells themselves
3. You can move horizontally or vertically between adjacent cells if there is no wall between them
4. You can only move through one cell at a time in any direction
5. The goal is to find a path from the start cell (S) to the end cell (E)

### Direction Definitions:
- "up": Move to the cell above the current position
- "down": Move to the cell below the current position
- "left": Move to the cell to the left of the current position
- "right": Move to the cell to the right of the current position

### Example:
Here's a simple maze:
+-+-+
|S 0|
+ + +
|0|E|
+-+-+

The solution to this maze would be: right down
Because:
1. From S, move right (to the cell with 0)
2. From that cell, move down (to the cell with 0)

### Current Maze State:
The maze is represented in text format as follows:
{}

In this representation:
- '+' characters are wall intersections
- '-' characters are horizontal walls
- '|' characters are vertical walls
- '0' represent the cells (paths)
- 'S' marks the start position
- 'E' marks the end position

### Output Format Requirements:
Your final answer should be in the format like: right down left up
"""

PROMPT_15PUZZLE_IMAGE = """
Your task is to solve the 15-puzzle game according to the rules and current state below:

### Game Rules:
1. The puzzle is played on a 4x4 grid with 15 numbered tiles and one empty space
2. You can only move tiles horizontally or vertically into the empty space
3. The goal is to arrange the tiles in numerical order with:
   - First row: 1, 2, 3, 4
   - Second row: 5, 6, 7, 8
   - Third row: 9, 10, 11, 12
   - Fourth row: 13, 14, 15, empty space

### Coordinate System:
- The grid positions are numbered from left to right and top to bottom
- Columns (horizontal): numbered 1, 2, 3, 4 from left to right
- Rows (vertical): numbered 1, 2, 3, 4 from top to bottom
- Each position can be identified by its row and column (row, column)

### Current Puzzle State:
The initial_state is represented in the image shown below

### Output Format Requirements:
"up" means the tile below the empty space moves up into the empty space
"down" means the tile above the empty space moves down into the empty space
"left" means the tile to the right of the empty space moves left into the empty space
"right" means the tile to the left of the empty space moves right into the empty space

Your final answer format should be given like: up down up left right
"""


PROMPT_15PUZZLE = """
Your task is to solve the 15-puzzle game according to the rules and current state below:

### Game Rules:
1. The puzzle is played on a 4x4 grid with 15 numbered tiles and one empty space
2. You can only move tiles horizontally or vertically into the empty space
3. The goal is to arrange the tiles in numerical order with:
   - First row: 1, 2, 3, 4
   - Second row: 5, 6, 7, 8
   - Third row: 9, 10, 11, 12
   - Fourth row: 13, 14, 15, empty space

### Coordinate System:
- The grid positions are numbered from left to right and top to bottom
- Columns (horizontal): numbered 1, 2, 3, 4 from left to right
- Rows (vertical): numbered 1, 2, 3, 4 from top to bottom
- Each position can be identified by its row and column (row, column)

### Current Puzzle State:
The initial_state {} represents a 4x4 grid reading from left to right, top to bottom, where 0 represents the empty space and numbers 1-15 represent the tiles.

### Output Format Requirements:
"up" means the tile below the empty space moves up into the empty space
"down" means the tile above the empty space moves down into the empty space
"left" means the tile to the right of the empty space moves left into the empty space
"right" means the tile to the left of the empty space moves right into the empty space

Your final answer format should be given like: up down up left right
"""

PROMPT_HANOI_IMAGE = """
Your task is to solve the hanoi game according to the rules and current state below:

### Game Rules:
1. The Tower of Hanoi consists of three pegs (numbered 1, 2, and 3) and n(maybe 3 or 4 or 5) disks of different sizes (from 1 to n)
2. Disks are stacked on pegs with larger disks always below smaller ones
3. Only one disk can be moved at a time, from the top of one peg to the top of another
4. A larger disk cannot be placed on top of a smaller disk

### Current Hanoi State:
The current state of the Tower of Hanoi is in the image shown below

### Goal State:
The goal is to move all disks to peg 3, maintaining the size order (largest at bottom, smallest at top).

For 3 disks: Peg 1: [], Peg 2: [], Peg 3: [3, 2, 1]
For 4 disks: Peg 1: [], Peg 2: [], Peg 3: [4, 3, 2, 1]  
For 5 disks: Peg 1: [], Peg 2: [], Peg 3: [5, 4, 3, 2, 1]

In this representation:
- Each peg is shown with its contents in array format
- Numbers represent disk sizes (higher numbers = larger disks)
- Disks are listed from bottom to top (first element = bottom disk, last element = top disk)

### Output Format Requirements:
Your final solution format should be given like:(x,y) (x,y) (x,y)..., where x is the disk number and y is the destination peg number
"""


PROMPT_HANOI = """
Your task is to solve the hanoi game according to the rules and current state below:

### Game Rules:
1. The Tower of Hanoi consists of three pegs (numbered 1, 2, and 3) and n(maybe 3 or 4 or 5) disks of different sizes (from 1 to n)
2. Disks are stacked on pegs with larger disks always below smaller ones
3. Only one disk can be moved at a time, from the top of one peg to the top of another
4. A larger disk cannot be placed on top of a smaller disk

### Current Hanoi State:
The current state of the Tower of Hanoi is represented as follows:
{}

### Goal State:
#### For 3 disks

[
   [],
   [],
   [3, 2, 1],
]

#### For 4 disks
[
   [],
   [],
   [4, 3, 2, 1],
]

#### For 5 disks
[
   [],
   [],
   [5, 4, 3, 2, 1],
]

In text representation:
- Each array [] represents a peg (from 1 to 3)
- Numbers inside the arrays represent disks (higher numbers = larger disks)
- The first/top elements in an array are at the bottom of the peg
- The last/bottom elements in an array are at the top of the peg

### Output Format Requirements:
Your final solution format should be given like:(x,y) (x,y) (x,y)..., where x is the disk number and y is the destination peg number
"""

PROMPT_WORDSEARCH_IMAGE = """
Your task is to solve the wordsearch game according to the rules and current state below:

### Task 
You are given a word search puzzle. Your task is to find the listed word hidden in the grid and provide their exact locations in the specified format.

### Game Rules
1. Words can be hidden horizontally, vertically, or diagonally.
2. Words can read forwards or backwards.
3. Words always follow a straight line (no zigzagging).
4. Each word's location should be identified by:
   - The starting position (coordinate where the first letter appears)
   - The direction in which the word extends

### Coordinate System
- The grid uses coordinates where (x, y) represents the position.
- x-axis: Numbers from 1 to width, running horizontally from left to right.
- y-axis: Numbers from 1 to height, running vertically from top to bottom.
- Example: Position (3, 4) means column 3 from left, row 4 from top.

### Direction Notation
- N: North (upward)
- S: South (downward)
- E: East (rightward)
- W: West (leftward)
- NE: Northeast (up and right)
- NW: Northwest (up and left)
- SE: Southeast (down and right)
- SW: Southwest (down and left)

### WordSearch State:
The current state of the WordSearch is shown in the image given below


### Output Format Requirements:
Your final answer format should be given like: WORD DIRECTION (x, y) @, where WORD is the word you found, DIRECTION is the direction in which the word extends, and (x, y) is the starting position of the word.
"""


PROMPT_WORDSEARCH = """
Your task is to solve the wordsearch game according to the rules and current state below:

### Task 
You are given a word search puzzle. Your task is to find the listed word hidden in the grid and provide their exact locations in the specified format.

### Game Rules
1. Words can be hidden horizontally, vertically, or diagonally.
2. Words can read forwards or backwards.
3. Words always follow a straight line (no zigzagging).
4. Each word's location should be identified by:
   - The starting position (coordinate where the first letter appears)
   - The direction in which the word extends

### Coordinate System
- The grid uses coordinates where (x, y) represents the position.
- x-axis: Numbers from 1 to width, running horizontally from left to right.
- y-axis: Numbers from 1 to height, running vertically from top to bottom.
- Example: Position (3, 4) means column 3 from left, row 4 from top.


### Direction Notation
- N: North (upward)
- S: South (downward)
- E: East (rightward)
- W: West (leftward)
- NE: Northeast (up and right)
- NW: Northwest (up and left)
- SE: Southeast (down and right)
- SW: Southwest (down and left)


### WordSearch State:
The current state of the WordSearch is shown:
{}


### Output Format Requirements:
Your final answer format should be given like: WORD DIRECTION (x, y) @, where WORD is the word you found, DIRECTION is the direction in which the word extends, and (x, y) is the starting position of the word.
"""


PROMPT_NUMBRIX_IMAGE = """
Your task is to solve the Numbrix puzzle based on the following rules and the current state:

### Game Rules:

1. Numbrix is played on a square grid, where some cells are already filled with numbers.
2. You must fill in the empty cells with numbers to create a continuous path starting from 1 up to the **maximum number in the sequence**, which is **not necessarily equal to the total number of cells (n²)**.
3. The numbers must be adjacent either horizontally or vertically (not diagonally).
4. Each number can only be used once.
5. The path must form a single continuous sequence where consecutive numbers are adjacent.
6. **Not every empty cell needs to be filled.** Depending on the puzzle configuration, some cells may remain empty.

### Important Notes:

* The highest number in the puzzle might be equal or less than the total number of grid cells (e.g., $n^2 - 1$, or even smaller).
* It is your job to determine what the highest number is, based on the filled numbers and the constraints of the puzzle.

### Current Numbrix State:
The current state of the Numbrix puzzle is shown in the image below.

### Output Format Requirements:

1. The final answer should be the completed grid with all numbers from 1 to the correct highest number, aligned clearly in rows and columns.

#### Example answer format for a 5x5 grid:

|11|10|9|2|3|
|12|13|8|1|4|
|15|14|7|6|5|
|16|19|20|23|24|
|17|18|21|22|25|
"""


PROMPT_NUMBRIX = """
Your task is to solve the Numbrix puzzle based on the following rules and the current state:

### Game Rules:

1. Numbrix is played on a square grid, where some cells are already filled with numbers.
2. You must fill in the empty cells with numbers to create a continuous path starting from 1 up to the **maximum number in the sequence**, which is **not necessarily equal to the total number of cells (n²)**.
3. The numbers must be adjacent either horizontally or vertically (not diagonally).
4. Each number can only be used once.
5. The path must form a single continuous sequence where consecutive numbers are adjacent.
6. **Not every empty cell needs to be filled.** Depending on the puzzle configuration, some cells may remain empty.

### Important Notes:
* The highest number in the puzzle might be equal or less than the total number of grid cells (e.g., $n^2 - 1$, or even smaller).
* It is your job to determine what the highest number is, based on the filled numbers and the constraints of the puzzle.

### Current Numbrix State:
The current state of the Numbrix puzzle is shown below:
{}

In this representation:

* Filled cells contain the given numbers.
* Empty cells are blank spaces.
* Your goal is to fill in the empty cells to complete a valid number sequence from 1 to the correct maximum number, following the rules above.

### Output Format Requirements:

1. The final answer should be the completed grid with all numbers from 1 to the correct highest number, aligned clearly in rows and columns.

#### Example answer format for a 5x5 grid:

|11|10|9|2|3|
|12|13|8|1|4|
|15|14|7|6|5|
|16|19|20|23|24|
|17|18|21|22|25|
"""


PROMPT_MINESWEEPER_IMAGE = """
Your task is to solve the Minesweeper puzzle according to the rules and the current state below:

### Game Rules:
1. Minesweeper is played on a grid where some cells contain hidden mines.
2. Numbers on the grid represent how many mines are adjacent to that cell (including diagonally).
3. A cell with no number means it has no adjacent mines (this is represented as a blank cell).
4. The goal is to identify the location of all mines without detonating any.
5. You can mark a cell as containing a mine if you're certain based on logical deduction.

### Current Minesweeper State:
The current state of the Minesweeper puzzle is shown in the image below.

### Output Format Requirements:
Your final answer should list all mine locations using 0-based coordinates in the format (row,col).

**Example answer format:**
(0,5),(0,7),(1,1),(1,2)
"""



PROMPT_MINESWEEPER = """

Your task is to solve the Minesweeper puzzle according to the rules and the current state below:

### Game Rules:
1. Minesweeper is played on a grid where some cells contain hidden mines.
2. Numbers on the grid represent how many mines are adjacent to that cell (including diagonally).
3. A cell with no number means it has no adjacent mines (this is represented as a blank cell).
4. The goal is to identify the location of all mines without detonating any.
5. You can mark a cell as containing a mine if you're certain based on logical deduction.

### Current Minesweeper State:
The current state of the Minesweeper puzzle is shown below:
{}

In this representation:
- Numbers indicate the count of adjacent mines.
- Empty cells (unrevealed cells) are represented by a space (` `).
- The goal is to identify the positions of all the mines (using `*`).

### Output Format Requirements:
Your final answer should list all mine locations using 0-based coordinates in the format (row,col).

**Example answer format:**
(0,5),(0,7),(1,1),(1,2)
"""



PROMPT_EULERO_IMAGE = """
Your task is to solve the Eulero puzzle, based on the rules and the current puzzle state shown below.

### Goal:
Fill all empty cells such that the following rules are satisfied:

### Global Rules:
1. Each cell contains a **letter-number pair** (like A1).
2. Each **letter** appears **exactly once** in every row and every column.
3. Each **number** appears **exactly once** in every row and every column.
4. Each **letter-number pair** is **unique across the entire grid** (i.e., no duplicate pairs anywhere).
5. For an N×N grid, the letters used are the first N letters of the alphabet (A=1, B=2, ..., up to the N-th letter), and the numbers used are from 1 to N.

### Current Puzzle State:
The puzzle is displayed in the image below:
- Some cells are pre-filled with letter-number pairs.
- Blank cells are empty and must be filled in.

### Output Format:
Each row should be represented as a single line of letter-number pairs, separated by `|` (without spaces).
Each row must be on a new line using `\n` to separate them.

**For example**:
A1|B2|C3\nB3|C1|A2\nC2|A3|B1  
"""


PROMPT_EULERO = """
Your task is to solve the Eulero puzzle, based on the rules and the current puzzle state shown below.

### Goal:
Fill all empty cells such that the following rules are satisfied:

### Global Rules:
1. Each cell contains a **letter-number pair** (like A1).
2. Each **letter** appears **exactly once** in every row and every column.
3. Each **number** appears **exactly once** in every row and every column.
4. Each **letter-number pair** is **unique across the entire grid** (i.e., no duplicate pairs anywhere).
5. For an N×N grid, the letters used are the first N letters of the alphabet (A=1, B=2, ..., up to the N-th letter), and the numbers used are from 1 to N.

### Current Puzzle State:
The puzzle is displayed below:
{}

### Output Format:
Each row should be represented as a single line of letter-number pairs, separated by `|` (without spaces).
Each row must be on a new line using `\n` to separate them.

**For example**:
A1|B2|C3\nB3|C1|A2\nC2|A3|B1  
"""

PROMPT_NIBBLES_IMAGE = """
You are a puzzle solver focusing on traditional Snake Game puzzles.

### Game Rules:
1. You control a snake that starts with an initial length (2 segments)
2. The snake moves in one direction at a time: up, down, left, or right
3. The goal is to eat all the apples on the grid
4. When the snake eats an apple, it grows by one segment
5. The snake cannot:
   - Move outside the grid boundaries
   - Collide with its own body
   - Move directly backwards (reverse direction)

### Current Puzzle State:
The image shows a Snake Game puzzle

### Movement Directions:
- up: Move one cell upward
- down: Move one cell downward  
- left: Move one cell leftward
- right: Move one cell rightward

### Output Format Requirements:
Your answer should be a sequence of movement directions to solve the puzzle.
Example: up down left right
"""

PROMPT_NIBBLES = """You are a puzzle solver focusing on traditional Snake Game puzzles.

### Game Rules:
1. You control a snake that starts with an initial length (2 segments)
2. The snake moves in one direction at a time: up, down, left, or right
3. The goal is to eat all the apples on the grid
4. When the snake eats an apple, it grows by one segment
5. The snake cannot:
   - Move outside the grid boundaries
   - Collide with its own body
   - Move directly backwards (reverse direction)

### Current Puzzle State:
{}

### Movement Directions:
- up: Move one cell upward
- down: Move one cell downward  
- left: Move one cell leftward
- right: Move one cell rightward

### Output Format Requirements:
Your answer should be a sequence of movement directions to solve the puzzle.
Example: up down left right
"""

PROMPT_SNAKE_IMAGE = """You are a puzzle solver focusing on Snake puzzles.

### Game Rules:
1. Control a snake to move around the grid using directional commands (up, down, left, right)
2. The snake must eat all apples on the grid to win
3. When the snake eats an apple, it grows longer by one segment
4. The snake cannot collide with walls or itself
5. The snake moves one cell at a time in the chosen direction

### Input:
- An image showing the initial state with the snake and apples

### Goal:
Find a sequence of directional moves to eat all apples without the snake colliding with walls or itself.

### Output Format Requirements:
Your answer should be a sequence of directional moves separated by spaces.
Valid moves are: up, down, left, right
Example: up right down left up
"""

PROMPT_SNAKE = """You are a puzzle solver focusing on Snake puzzles.

### Game Rules:
1. Control a snake to move around the grid using directional commands (up, down, left, right)
2. The snake must eat all apples on the grid to win
3. When the snake eats an apple, it grows longer by one segment
4. The snake cannot collide with walls or itself
5. The snake moves one cell at a time in the chosen direction

### Input:
- A grid showing the initial state with the snake and apples
- Snake head is marked with 'H', snake body with 'S', apples with 'A', empty cells with '.'

### The Snake State:
{}

### Goal:
Find a sequence of directional moves to eat all apples without the snake colliding with walls or itself.

### Output Format Requirements:
Your answer should be a sequence of directional moves separated by spaces.
Valid moves are: up, down, left, right
Example: up right down left up
"""

