# 参赛作品名
## 是男人就坚持100秒（DodgeFace）   
   
## 作品简介
### 和传统的《是男人就坚持X秒》一样，只不过这次你需要让你的脸来躲避那些飞来的小球
    
## 使用方式
### 保证你的脸在摄像头可以拍摄的范围内，运行程序```
    python DodgeFace.py
```   
### 移动你的身体、转动你的头颅躲避那些飞来的小球   
### 如果没检测到你的脸会直接Game Over哦   
### 程序中设定了危险像素大小，当你的脸小于某个像素量（dangerPixel）的时候，你的脸会变红，提示你靠近摄像头一点，否则当脸的像素小于最小像素量（minPixel）的时候则会直接Game Over。   
### 如果不小心死掉了，按**r**重新开始，按其他的则退出游戏      

## 一些提示
### 程序默认开启了GPU以保证程序的正常运行，如果没有GPU可能会很卡，咳咳   
### 当然可以换成整个人来躲避球，这个只需要改动一行代码就行，不过建议同时改大球的半径，不然我怕你看不清球，哈哈
### 有点忙，还有些优化的地方没有做，之后再补吧，嘻嘻  
### 希望大家玩得开心~
