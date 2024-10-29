import cv2 #görüntü daha sonrasında mediapipe göndererek işlenecek
import mediapipe
import pyttsx3

camera = cv2.VideoCapture(0) #kamera çağırmak için 
engine = pyttsx3.init()
mpHands = mediapipe.solutions.hands
hands =mpHands.Hands() #el oluşacak
mpDraw =mediapipe.solutions.drawing_utils#nokta çizimi
checkTamam=False

while True:

    success,img =camera.read()
    
    imgRGB =cv2.cvtColor(img,cv2.COLOR_BGR2RGB)

    hlms =hands.process(imgRGB) #noktaları bulacak
    
    height,width,channel=img.shape
    
    if hlms.multi_hand_landmarks:
        for handlandmarks in hlms.multi_hand_landmarks: #ekrana bastırmak için
            
            for fingerNum,landmark in enumerate(handlandmarks.landmark): #xyz değişkenlerini sıralıyor
                positionX,positionY =int(landmark.x * width),int(landmark.y * height)
                
                
                cv2.putText(img,str(fingerNum),(positionX,positionY), #parmak numaralandıması
                            cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,0),2)
                
                if fingerNum >4 and landmark.y < handlandmarks.landmark[2].y: #y ekseni en büyük olanı al
                    break
                
                if fingerNum == 20 and landmark.y > handlandmarks.landmark[2].y:
                    #print("Tamam")
                    checkTamam=True
                
                if fingerNum ==4: #4 numaralıyı beyaz daire içine al
                    cv2.circle(img,(positionX,positionY),30,(255,255,255),cv2.FILLED)
                
                
            
            mpDraw.draw_landmarks(img,handlandmarks,mpHands.HAND_CONNECTIONS)
    
    
    
    
    cv2.imshow("Kamera",img) #q ya bas kapan  
    
    if checkTamam:
        engine.say("Ups")
        engine.runAndWait()
        break
    
    if cv2.waitKey(1) & 0xFF ==ord("q"):
        break
    
    