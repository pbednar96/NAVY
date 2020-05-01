import pygame
import numpy as np

SIZE = 100
PROB_TREE = .05
PROB_FIRE = .001

# cell size in game board
CELL_SIZE = 10

light_green = (125, 254, 0)
green = (0, 128, 0)
red = (255, 0, 0)


# EMPTY = 0
# TREE = 1
# FIRE = 2

def get_size(matrix):
    # tmp func for PyGame
    heigth = len(matrix)
    width = len(matrix[0])
    return heigth * CELL_SIZE, width * CELL_SIZE


def init_forest():
    # init forest
    forest = np.zeros((SIZE, SIZE))
    for i in range(SIZE):
        for j in range(SIZE):
            forest[i][j] = np.random.choice(3, p=[.15, .75, .10])
    return forest


def next_generation(forest):
    # next generation for fire
    next_gen_forest = np.zeros((SIZE, SIZE))
    for i in range(1, len(forest) - 1):
        for j in range(1, len(forest) - 1):
            if forest[i][j] == 0 and np.random.random() < PROB_TREE:
                next_gen_forest[i][j] = 1
            if forest[i][j] == 1:
                next_gen_forest[i][j] = 1
                if fire_in_neighbourhood(forest, i, j):
                    next_gen_forest[i][j] = 2
            else:
                if np.random.random() < PROB_FIRE:
                    next_gen_forest[i][j] = 2
    return next_gen_forest


def fire_in_neighbourhood(matrix, y, x):
    # return True if one of neighbourhoods fire
    neighbourhood_fire = False

    if matrix[y - 1][x - 1] == 2:
        neighbourhood_fire = True
    if matrix[y - 1][x] == 2:
        neighbourhood_fire = True
    if matrix[y - 1][x + 1] == 2:
        neighbourhood_fire = True
    if matrix[y][x - 1] == 2:
        neighbourhood_fire = True
    if matrix[y][x + 1] == 2:
        neighbourhood_fire = True
    if matrix[y + 1][x - 1] == 2:
        neighbourhood_fire = True
    if matrix[y + 1][x] == 2:
        neighbourhood_fire = True
    if matrix[y + 1][x + 1] == 2:
        neighbourhood_fire = True

    return neighbourhood_fire


def main():
    forest = init_forest()

    pygame.init()
    height, width = get_size(forest)
    gameDisplay = pygame.display.set_mode((width, height))
    pygame.display.set_caption('ForestFire')
    clock = pygame.time.Clock()
    while True:
        clock.tick(3)
        forest = next_generation(forest)
        gameDisplay.fill(light_green)
        for index_x, row in enumerate(forest):
            for index_y, cell in enumerate(row):
                if cell == 1:
                    pygame.draw.rect(gameDisplay, green,
                                     (index_y * CELL_SIZE, index_x * CELL_SIZE, CELL_SIZE, CELL_SIZE))
                if cell == 2:
                    pygame.draw.rect(gameDisplay, red,
                                     (index_y * CELL_SIZE, index_x * CELL_SIZE, CELL_SIZE, CELL_SIZE))
        pygame.display.update()
    pygame.quit()
    quit()


if __name__ == "__main__":
    main()
