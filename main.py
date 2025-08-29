from ezonnx import DinoV2,DinoV3
import cv2
def main():
    # dino2 = DinoV2("small",quantize=None, size=384*3)
    # result = dino2("examples/cat.jpg")
    # print(result.class_token)
    # cv2.imwrite("examples/patch_token2.jpg", result.pca_image_rgb)

    dino3 = DinoV3("vits16plus", quantize=None, size=384*2)
    result = dino3("examples/cat.jpg")
    cv2.imwrite("examples/patch_token3.jpg", result.pca_image_rgb)

if __name__ == "__main__":
    main()
