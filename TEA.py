from PIL import Image
import numpy as np

WIDTH = 720
HEIGHT = 720

MAX_ITER = 80


def to_gray_scale(i):
    shade_of_gray = int(255 * (i / MAX_ITER))
    return shade_of_gray, shade_of_gray, shade_of_gray


def main():
    img = Image.new('RGB', (WIDTH, HEIGHT), (0, 0, 0))
    pixels = img.load()

    x_values = np.linspace(-2, 2, WIDTH)
    y_values = np.linspace(-2, 2, HEIGHT)

    for index_x, x in enumerate(x_values):
        for index_y, y in enumerate(y_values):
            z = 0
            c = np.complex(x, y)
            for index in range(MAX_ITER):
                z = z * z + c
                if abs(z) > 2.0:
                    pixels[index_x, index_y] = to_gray_scale(index)
                    break

    img.show()


if __name__ == "__main__":
    main()
