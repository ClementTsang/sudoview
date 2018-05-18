#define NONE 0
#define N 9

bool solve_sudoku(int grid[N][N]);

bool find_unassign(int grid[N][N], int &row, int &col);

bool used_in_row(int grid[N][N], int row, int num);

bool used_in_col(int grid[N][N], int col, int num);

bool used_in_box(int grid[N][N], int boxStartRow, int boxStartCol, int num);

bool is_safe(int grid[N][N], int row, int col, int num);

