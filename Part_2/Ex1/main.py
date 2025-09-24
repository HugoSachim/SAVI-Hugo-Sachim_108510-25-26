#!/usr/bin/env python3
# shebang line for linux / mac

from copy import deepcopy
import cv2
import numpy as np

def main():
    print("python main function")

    # Reading the image from disk
    original_image = cv2.imread('lake.jpg', cv2.IMREAD_COLOR)
    if original_image is None:
        print("Erro: não encontrei lake.jpg")
        return

    h, w, channels = original_image.shape

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # ou *'X264' se suportado
    out = cv2.VideoWriter('output.mp4', fourcc, 5.0, (w, h))

    # Factors para escurecer
    factors = np.linspace(1, 0, 20)
    print('factors = ' + str(factors))

    middle_width = round(w / 2)

    for factor in factors:  # progressive nightfall
        image = deepcopy(original_image)

        for y in range(0, h):  # iterate all rows
            for x in range(middle_width, w):  # iterating cols from middle to last
                bgr = original_image[y, x, :]
                bgr_darkened = (bgr * factor).astype(np.uint8)
                image[y, x, :] = bgr_darkened

        cv2.imshow('Darkened', image)
        cv2.waitKey(50)
        out.write(image)  # grava frame

    cv2.waitKey(0)  # espera tecla
    out.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
