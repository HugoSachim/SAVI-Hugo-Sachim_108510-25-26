import cv2

def main():
    image_path = r"C:\Users\hogu\Desktop\SAVI-Hugo-Sachim_108510-25-26\Part_2\lake.jpg"
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)



    cv2.imshow('Image', image)
    cv2.waitKey(0)

if __name__ == '__main__':
    main()
