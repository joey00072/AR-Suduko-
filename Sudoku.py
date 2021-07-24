from ImageProssesing import is_image_empty
import numpy as np
import cv2
import matplotlib.pyplot as plt
import torch


def predict_digit(svm_model, hog, image):
    SIZE = 32
    img = cv2.resize(image, (SIZE, SIZE)).reshape(1, 1, SIZE, SIZE) / 255
    img = torch.tensor(img).float()
    pred = svm_model(img)
    res = np.argmax(pred.detach().numpy())
    # print(res)
    return [res]


def seperate_grid_digits(projected_grid):
    pading = 9
    num_width = 50
    digits_imgs = []
    for i in range(0, 9):
        row = []
        for j in range(0, 9):
            dig = projected_grid[
                i * num_width + pading : i * num_width + num_width - pading,
                j * num_width + pading : j * num_width + num_width - pading,
            ]
            if not is_image_empty(dig, thresh=20):
                row.append(dig)
            else:
                row.append(None)

        digits_imgs.append(row)
    return digits_imgs


cnt = 0


def pridict_digits(digits_imgs, svm_model, hog):
    global cnt
    sudoku_grid = np.zeros((9, 9), np.uint8)
    for i in range(0, 9):
        for j in range(0, 9):
            dig = digits_imgs[i][j]

            if dig is not None:
                sudoku_grid[i, j] = int(predict_digit(svm_model, hog, dig)[0])
                cnt += 1
    return sudoku_grid


def draw_solved_grid(to_draw_on, old_grid, solved_grid):
    num_width = 50
    pading = 10
    for i in range(0, 9):
        for j in range(0, 9):
            if old_grid[i, j] == 0:
                cv2.putText(
                    to_draw_on,
                    str(solved_grid[i, j]),
                    (
                        j * num_width + pading,
                        i * num_width + pading + int(num_width * 0.5),
                    ),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (255, 5, 5),
                    thickness=2,
                    lineType=cv2.LINE_AA,
                )
    return to_draw_on


def draw_original_grid(to_draw_on, old_grid):
    num_width = 50
    pading = 10
    for i in range(0, 9):
        for j in range(0, 9):
            if old_grid[i, j] != 0:
                cv2.putText(
                    to_draw_on,
                    str(old_grid[i, j]),
                    (
                        j * num_width + pading + 25,
                        i * num_width + pading + 10 + int(num_width * 0.5),
                    ),
                    cv2.FONT_HERSHEY_DUPLEX,
                    0.7,
                    (0, 0, 255),
                    thickness=1,
                    lineType=cv2.LINE_4,
                )

    return to_draw_on


def draw_digits(digits):
    image = np.zeros((450, 450), np.uint8)
    image = image * 50
    pading = 9
    num_width = 50
    for i in range(9):
        for j in range(9):
            if digits[i][j] is not None:
                image[
                    i * num_width + pading : i * num_width + num_width - pading,
                    j * num_width + pading : j * num_width + num_width - pading,
                ] = digits[i][j]
    return image


def measure_different(grid1, grid2):
    m = grid1 - grid2
    d = np.count_nonzero(m)
    return d


class EntryData:
    def __init__(self, r, c, n):
        self.row = r
        self.col = c
        self.choices = n

    def set_data(self, r, c, n):
        self.row = r
        self.col = c
        self.choices = n


def solve_sudoku(matrix):
    cont = [True]
    for i in range(9):
        for j in range(9):
            if not can_be_correct(matrix, i, j):
                return
    sudoku_helper(matrix, cont)


def sudoku_helper(matrix, cont):
    if not cont[0]:
        return
    best_candidate = EntryData(-1, -1, 100)
    for i in range(9):
        for j in range(9):
            if matrix[i][j] == 0:
                num_choices = count_choices(matrix, i, j)
                if best_candidate.choices > num_choices:
                    best_candidate.set_data(i, j, num_choices)

    if best_candidate.choices == 100:
        cont[0] = False
        return

    row = best_candidate.row
    col = best_candidate.col

    for j in range(1, 10):
        if not cont[0]:
            return

        matrix[row][col] = j

        if can_be_correct(matrix, row, col):
            sudoku_helper(matrix, cont)

    if not cont[0]:
        return
    matrix[row][col] = 0


def count_choices(matrix, i, j):
    can_pick = [True, True, True, True, True, True, True, True, True, True]
    # From 0 to 9 - drop 0

    for k in range(9):
        can_pick[matrix[i][k]] = False

    for k in range(9):
        can_pick[matrix[k][j]] = False

    r = i // 3
    c = j // 3
    for row in range(r * 3, r * 3 + 3):
        for col in range(c * 3, c * 3 + 3):
            can_pick[matrix[row][col]] = False

    count = 0
    for k in range(1, 10):
        if can_pick[k]:
            count += 1

    return count


def can_be_correct(matrix, row, col):
    for c in range(9):
        if matrix[row][col] != 0 and col != c and matrix[row][col] == matrix[row][c]:
            return False

    for r in range(9):
        if matrix[row][col] != 0 and row != r and matrix[row][col] == matrix[r][col]:
            return False

    r = row // 3
    c = col // 3
    for i in range(r * 3, r * 3 + 3):
        for j in range(c * 3, c * 3 + 3):
            if (
                row != i
                and col != j
                and matrix[i][j] != 0
                and matrix[i][j] == matrix[row][col]
            ):
                return False

    return True


def all_board_non_zero(matrix):
    for i in range(9):
        for j in range(9):
            if matrix[i][j] == 0:
                return False
    return True
