import paddlehub as hub
import cv2
import numpy as np
# import pygame as pg
import time
import random
import os
import math

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

currentSeg = None
currentSeg3 = None
currentTime = 0
lastTime = -1
genTime = [5, 10, 20, 30]
currentIndex = 0
gm = [0, 1, 2, 1]
W = 0
H = 0
showimg = None
minPIXEL = 2500
dangerousPIXEL = 4500
balls = []


class segUtils:
    def __init__(self):
        super(segUtils, self).__init__()
        self.ace2p = hub.Module(name='ace2p')

    def getMask(self, frame):
        res = self.ace2p.segmentation([frame], use_gpu=True)
        if isinstance(res, list):
            resint = res[0]['data']
            resint[resint != 13] = 0
            resint[resint == 13] = 1
            return resint
        else:
            return None

class segHuman:
    def __init__(self):
        super(segHuman, self).__init__()
        self.module = hub.Module(name='humanseg_mobile')
        self.prev_gray = None
        self.prev_cfd = None

    def getMask(self, frame, cap=None):
        if cap is None:
            res = self.module.segment(images=[frame], use_gpu=True)
            if isinstance(res, list):
                return np.around(res[0]['data'] / 255)
        else:
            res, gray, cfd = self.module.video_stream_segment(images=frame, frame_id=cap.get(1), use_gpu=True, prev_gray=self.prev_gray, prev_cfd=self.prev_cfd)
            self.prev_gray = gray
            self.prev_cfd = cfd
            res[res < 100] = 0
            res[res >= 100] = 1
            return res

def getPIXEL(x, y, radius):
    t = y - radius if y - radius > 0 else 0
    l = x - radius if x - radius > 0 else 0
    b = y + radius if y + radius < H else H - 1
    r = x + radius if x + radius < W else W - 1
    return t,l,b,r

class Ball:
    x = None
    y = None
    speed_x = None
    speed_y = None
    radius = None
    color = None

    def __init__(self, x, y, speed_x, speed_y, radius, color):
        self.x = x
        self.y = y
        self.speed_x = speed_x
        self.speed_y = speed_y
        self.radius = radius
        self.color = color

    def draw(self, screen):
        t,l,b,r = getPIXEL(self.x, self.y, self.radius)
        if isinstance(self.color, int):
            screen[t:b,l:r,:] = self.color
        else:
            screen[t:b,l:r,0] = self.color[0]
            screen[t:b,l:r,1] = self.color[1]
            screen[t:b,l:r,2] = self.color[2]
        

    def move(self, screen):
        self.x += self.speed_x
        self.y += self.speed_y
        
        if self.x > W - self.radius or self.x < self.radius:
            self.speed_x = -self.speed_x

        if self.y > H - self.radius or self.y < self.radius:
            self.speed_y = -self.speed_y

        time.sleep(0.001)
        if inseg(self.x, self.y, self.radius):
            return True
        else:
            self.draw(screen)
            return False

def randomXY():
    x = random.randint(0, W)
    y = random.randint(0, H)
    return x, y

def inseg(x, y, radius):
    global currentSeg
    if currentSeg is None:
        return False
    else:
        t,l,b,r = getPIXEL(x, y, radius)
        if np.sum(currentSeg[t:b,l:r]) > 0:
            return True
        else:
            return False


def create_ball():

    r = 3
    color = 0

    x, y = randomXY()
    if inseg(x,y,r+30):
        x, y = randomXY()

    speed_x = random.randint(-10, 10)
    speed_y = random.randint(-10, 10)
    
    b = Ball(x, y, speed_x, speed_y, r, color)
    balls.append(b) 


def ball_manager():
    global showimg
    global currentIndex
    global currentTime
    global lastTime
    global gm
    global genTime
    if currentTime != lastTime:
        if currentIndex < len(gm):
            if currentTime < genTime[currentIndex]:
                for i in range(gm[currentIndex]):
                    create_ball()
            else:
                currentIndex += 1
                if currentIndex >= len(gm):
                    currentIndex = len(gm) - 1
        
        lastTime = currentTime

    for b in balls:
        if b.move(showimg):
            return True
    
    return False
        

def main():
    global showimg
    global H
    global W
    global currentTime
    global currentSeg
    global startTime
    global balls
    global currentIndex
    videostream = 0

    cap = cv2.VideoCapture(videostream)
    su = segUtils()
    sh = segHuman()
    
    restart = True
    while restart:
        restart = False
        currentSeg = None
        currentSeg3 = None
        lastTime = -1
        showimg = None
        currentIndex = 0
        balls = []
        startTime = time.time()
        while True:

            ret, frame = cap.read()

            if videostream == 0:
                frame = cv2.flip(frame, 1)

            if ret == True:

                H, W = frame.shape[:2]
                currentTime = math.floor(time.time() - startTime)
                currentSeg = su.getMask(frame)
                if currentSeg is not None:
                    sumSeg = np.sum(currentSeg)
                    if sumSeg < minPIXEL:
                        if showimg is None:
                            showimg = np.ones_like(frame) * 255
                        showimg = cv2.putText(showimg, "Pixel: " + str(sumSeg), (0, 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,0), 3)
                        showimg = cv2.putText(showimg, "Your face is too small!", (0, int(H/2)), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,0), 3)
                        cv2.imshow('Game', showimg)
                        if cv2.waitKey(0) == ord('r'):
                            restart = True
                        break
                    else:
                        showimg = np.ones_like(frame) * 255
                        currentSeg3 = np.repeat(currentSeg[:,:,np.newaxis], 3, axis=2)
                        if np.sum(currentSeg) < dangerousPIXEL:
                            frame[:,:,2] = 255
                        showimg = frame * currentSeg3 + showimg * (1 - currentSeg3)
                        gameover = ball_manager()
                        showimg = showimg.astype(np.uint8)
                        showimg = cv2.putText(showimg, "Pixel: " + str(sumSeg)+" Time: %2f"% (time.time() - startTime), (0, 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,0), 3)
                        if gameover:
                            showimg = cv2.putText(showimg, "You Lose! Time: %2f" % (time.time() - startTime), (0, int(H/2)), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 3)
                            cv2.imshow('Game', showimg)
                            if cv2.waitKey(0) == ord('r'):
                                restart = True
                            break
                        else:
                            cv2.imshow('Game', showimg)
                            cv2.waitKey(1)
                else:
                    if showimg is None:
                        showimg = np.ones_like(frame) * 255
                    showimg = cv2.putText(showimg, "Keep your face in camera", (0, int(H/2)), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,0), 3)
                    cv2.imshow('Game', showimg)
                    if cv2.waitKey(0) == ord('r'):
                        restart = True
                    break
            else:
                if showimg is None:
                    showimg = np.ones_like(frame) * 255
                showimg = cv2.putText(showimg, "Check your camera!", (0, int(H/2)), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,0), 3)
                cv2.imshow('Game', showimg)
                if cv2.waitKey(0) == ord('r'):
                    restart = True
                break
    cv2.destroyAllWindows()
    cap.release()

if __name__ == '__main__':
    main()