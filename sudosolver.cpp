#include <cstdlib>
#include <stdio.h>
#include <iostream>
#include "sudosolver.h"

bool solve_sudoku(int grid[N][N]) {
    int row = 0, col = 0;    
    if (!find_unassign(grid, row, col))
        return true; 

    for (int num = 1; num <= 9; ++num) {
        // if looks promising
        if (is_safe(grid, row, col, num)) {
            grid[row][col] = num;

            if (solve_sudoku(grid)) return true;
            
            grid[row][col] = NONE;
        }
    }
    return false;
}

bool find_unassign(int grid[N][N], int &row, int &col) {
    for (row = 0; row < N; ++row)
        for (col = 0; col < N; ++col) {
            if (grid[row][col] == NONE) return true;
        }
        return false;
}

bool used_in_row(int grid[N][N], int row, int num) {
    for (int col = 0; col < N; ++col) {
        if (grid[row][col] == num) return true;
    }
    return false;
}

bool used_in_col(int grid[N][N], int col, int num) {
    for (int row = 0; row < N; ++row) {
        if (grid[row][col] == num) return true;
    }
    return false;
}

bool used_in_box(int grid[N][N], int boxStartRow, int boxStartCol, int num) {
    for (int row = 0; row < 3; ++row) {
        for (int col = 0; col < 3; ++col) {
            if (grid[row+boxStartRow][col+boxStartCol] == num) return true;
        }
    }
        return false;
}

bool is_safe(int grid[N][N], int row, int col, int num) {
    return !used_in_row(grid, row, num) && !used_in_col(grid, col, num) && !used_in_box(grid, row - row%3 , col - col%3, num);
}

