import cv2
import os


def imgs2File(path, outName):
    images = [img for img in os.listdir(path) if img.endswith(".jpg")]
    images.sort(key=lambda x:int(x.split('.')[0]))

    # 选择编码器和帧率
    frame = cv2.imread(os.path.join(path, images[0]))
    height, width, layers = frame.shape
    video = cv2.VideoWriter(outName + '.avi', cv2.VideoWriter_fourcc(*'DIVX'), 10, (width, height))

    # 将图片写入视频
    for image in images:
        video.write(cv2.imread(os.path.join(path, image)))

    cv2.destroyAllWindows()
    video.release()
