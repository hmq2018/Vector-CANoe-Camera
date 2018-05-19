# -*- coding:utf-8 -*-
import threading
from time import sleep
import cv2
import numpy as np
import traceback
import socketserver
from datetime import datetime
import os
from pylab import array, plot, show, axis, arange, figure, uint8

def image_process(image, level=12):
    # image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # # image = cv2.erode(image, kernel, iterations=1)
    # image = cv2.medianBlur(image, 3)

    maxIntensity = 255.0  # depends on dtype of image data
    x = arange(maxIntensity)
    # Parameters for manipulating image data
    phi = 1
    theta = 1
    image = (maxIntensity / phi) * (image / (maxIntensity / theta))**level
    image = array(image, dtype=uint8)
    # ret1, image = cv2.threshold(image, 10, 255, cv2.THRESH_BINARY)
    # image = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    image = cv2.medianBlur(image, 3)
    # image = cv2.Canny(image, 50, 150)

    # convert the bottom image to grayscale, then apply a blackhat
    # morphological operator to find dark regions against a light
    # background (i.e., the routing and account numbers)
    # rectKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (17, 7))
    # gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, rectKernel)

    # kernel = np.ones((3, 3), np.uint8)
    # image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # ret1, image = cv2.threshold(image, 196, 255, cv2.THRESH_BINARY)
    # # image = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    # # image = cv2.erode(image, kernel, iterations=1)
    # image = cv2.medianBlur(image, 3)
    # # image = cv2.Canny(image, 50, 150)
    return image

def inverse_color(image):
    height,width = image.shape
    img2 = image.copy()

    for i in range(height):
        for j in range(width):
            img2[i,j] = (255-image[i,j]) # For GRAY_SCALE image ;
                                         # for R.G.B image: img2[i,j] = (255-image[i,j][0],255-image[i,j][1],255-image[i,j][2])
    return img2

def detect_black(Img):
    # Img = cv2.imread(imagename)#读入一幅图像
    I_h, I_w = Img.shape[:2]
    I_h2,I_w2 = int(I_h * 0.8),int(I_w*0.8)
    I_h1,I_w1 = int((I_h-I_h2)/2),int((I_w-I_w2)/2)
    Img = Img[I_h1:I_h2, I_w1:I_w2]
    kernel_4 = np.ones((2,2),np.uint8)#4x4的卷积核

    if Img is not None:#判断图片是否读入
        HSV = cv2.cvtColor(Img, cv2.COLOR_BGR2HSV)#把BGR图像转换为HSV格式
        '''
        HSV模型中颜色的参数分别是：色调（H），饱和度（S），明度（V）
        下面两个值是要识别的颜色范围
        '''
        Lower = np.array([0, 0, 0])#要识别颜色的下限
        Upper = np.array([255, 255, 90])#要识别的颜色的上限
        #mask是把HSV图片中在颜色范围内的区域变成白色，其他区域变成黑色
        mask = cv2.inRange(HSV, Lower, Upper)
        # mask = inverse_color(mask)
        #下面四行是用卷积进行滤波
        erosion = cv2.erode(mask,kernel_4,iterations = 1)
        # erosion = cv2.erode(erosion,kernel_4,iterations = 1)
        dilation = cv2.dilate(erosion,kernel_4,iterations = 1)
        # dilation = cv2.dilate(dilation,kernel_4,iterations = 1)
        # #target是把原图中的非目标颜色区域去掉剩下的图像
        # target = cv2.bitwise_and(Img, Img, mask=dilation)
        #将滤波后的图像变成二值图像放在binary中
        ret, binary = cv2.threshold(dilation,127,255,cv2.THRESH_BINARY)
        #在binary中发现轮廓，轮廓按照面积从小到大排列
        binary, contours, hierarchy = cv2.findContours(binary,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)

        if len(contours) > 3:
            # print("not found")
            return False
        else:
            # print("black")
            return True

    # inital camera



def camera_start():
    global capture_val,quit_value,black_value,black_result,camerano,filename
    # create folder
    try:
        isExists1 = os.path.exists('../Capture')
        if not isExists1:
            os.mkdir('../Capture')
    except:
        print("folder create error")

    while(camerano == 9999):
        if camerano < 9999:
            break
        sleep(0.1)

    # inital camera
    try:
        cap = cv2.VideoCapture(camerano)
        # cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)
        # cap.set(cv2.CAP_PROP_FOCUS, 118)
        # cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        # cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        # sleep(1)
        # cap.release()
        # cv2.destroyAllWindows()
    except:
        cap = cv2.VideoCapture(0)
        traceback.print_exc()
        print("[ERROR]: Camera Connect Error!")

    while (1):
        # get a frame
        ret, frame = cap.read()

        # show a frame
        if ret == 1:
            cv2.imshow("CANoe Camera", frame)
            # counter1 = counter.get()
            # counter1 == 1 means capture, ==6 means compare, ==9 means exit
            if capture_val == 1:
                try:
                    # filename = datetime.now().strftime("%Y%m%d_%H%M%S_%f") + ".png"
                    cv2.imwrite("../Capture/"+filename, frame)
                    capture_val = 0
                    print(filename + " capture ok")
                except:
                    print("capture error")
            if quit_value == 1:
                try:
                    break
                except:
                    print("capture error")
            if black_value == 1:
                frame = image_process(frame)
                if detect_black(frame):
                    black_result = 1
                    black_value = 0
                    print("Black SCREEN")
                else:
                    black_result = 0
                    black_value = 0
                    print("WHITE SCREEN")

            ch = cv2.waitKey(1)
            # print(ch)
            # sleep(1)
            if ch == ord('q'):
                quit_value = 1
                sleep(1)
                print("service quit")
                break
            elif ch == ord(' '):
                cv2.imwrite('./Standard.png', frame)
            else:
                pass
        else:
            print("Camera Error!!!")
    try:
        cap.release()
        cv2.destroyAllWindows()
    except:
        print("Camera Close Error!!!")

class MyTCPhandler(socketserver.BaseRequestHandler):  # 必须继承这个类
    # timeout = 3
    # def handle_timeout(self):
    #     print ("No message received in {0} seconds".format(self.timeout))

    def handle(self):
        global capture_val,camerano,quit_value,black_value,black_result,server_timeout,server,filename
        print(self.request)  #打印出来的就是conn
        print(self.client_address)   #打印出来的就是addr
        # self.request.timeout = 2
        # print(self.server.timeout)
        while True:
            try:
                # if quit_value == 1:
                #     break
                #     print("service quit")
                data = self.request.recv(1024)
                if not data: break
                if "CAP" in data:
                    capture_val = 1
                    filename = datetime.now().strftime("%Y%m%d_%H%M%S_%f") + ".png"
                    self.request.send(filename)
                    # print("capture")
                elif "CAMS" in data:
                    camerano = int(data.split("=")[1])
                    print("Select Camera no:",camerano)
                elif "BLACK" in data:
                    black_value = 1
                    sleep(0.3)
                    if black_result == 1:
                        self.request.send("BLACK_ONN")
                        # print("SCREEN IS: black")
                    else:
                        self.request.send("BLACK_OFF")
                        # print("SCREEN IS: light")
                elif "QUIT" in data:
                    quit_value = 1
                    sleep(1)
                    print("quit")
                    break
                else:
                    print(data.strip())
                # self.request.send(data.upper())
            except Exception:
                break
        quit_value = 0
        self.request.close()
        server.shutdown()


if __name__ == "__main__":
    global capture_val,host,port,quit_value,black_value,black_result,server_timeout,server,filename
    capture_val = 0
    quit_value = 0
    black_result = 0
    black_value = 0
    camerano = 9999
    server_timeout = 3
    filename = ""
    threads = []
    host,port ='127.0.0.1',17778


    #camera start thread
    th1 = threading.Thread(target=camera_start, args=())
    # th1.setDaemon(True)
    th1.start()
    threads.append(th1)
    # th2 = threading.Thread(target=service_start, args=(host,port,))
    # # th2.setDaemon(True)
    # th2.start()
    # threads.append(th2)

    # tcp ip server start
    print("server start")
    server = socketserver.ThreadingTCPServer((host,port),MyTCPhandler) #实现了多线程的socket通话
    server.allow_reuse_address=True
    # server.handle_timeout()
    # server.timeout = 10
    server.serve_forever()
    print("server quit")

    th1.join()

    print "All threads finish"