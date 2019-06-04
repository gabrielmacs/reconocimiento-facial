import cv2, os
import numpy as np#trabajar conarreglos, son funciones poptimizadas para el manejor de matrices
from PIL import Image

def create_dataset():
    cam = cv2.VideoCapture(0)
    detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml' )#este es el clasificador y sirve para detectar la cara frontal


    Id = input('Dame un Id> ')
    sampleNum = 0
    print(cam) 
    while(True):
        ret, img = cam.read()
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)#para leer la pasamos a un tono gris, con color no vale
        faces = detector.detectMultiScale(gray, 1.3, 5)#es el tamaño de la imagen
        for (x,y,w,h) in faces:
            print(x,y,w,h)
            #dibuja el tamaño de la imagen donde el punto de origen es el superior isq, el color y ancho del cuadrado
            cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)

            sampleNum=sampleNum + 1#el nombre con el que se guarda, es el numero de ejemplos
            cv2.imwrite("dataSet/User."+Id +'.'+ str(sampleNum) + ".jpg", gray[y:y+h,x:x+w])

            cv2.imshow('Creando dataset', img)

        if cv2.waitKey(100) & 0xFF == ord('q'):
            break

        elif sampleNum > 50:
            break

    cam.release()
    cv2.destroyAllWindows()


def recognize():
    recognizer = cv2.face.LBPHFaceRecognizer_create()  #instancia de objeto para usar en entrenamiento 
    recognizer.read('trainner/trainner.yml') # lee la data de entrenamiento 
    cascadePath = "haarcascade_frontalface_default.xml" 
    faceCascade = cv2.CascadeClassifier(cascadePath) #reconoce las caras de frente 

    cam = cv2.VideoCapture(0)
    isIdentifyed = False

    font = cv2.FONT_HERSHEY_SIMPLEX #normal size sans-serif font
    while True:
        ret, im = cam.read()
        gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY) #pasamos a gris
        faces = faceCascade.detectMultiScale(gray, 1.2,5)
        for(x, y, w, h) in faces:
            cv2.rectangle(im,(x, y),(x+w, y+h),(225, 0, 0), 2)
            Id, conf = recognizer.predict(gray[y:y+h,x:x+w]) #aqui coge las nuevas imagenes de la caata y predice segun la data que tiene
                if(Id==1): # esto va a la base de datos 
                    Id="Persona 1"
                    isIdentifyed = True
                elif(Id==2):
                    Id="Persona 2"
                    isIdentifyed = True
            else:
                Id="Buscando..."
            cv2.putText(im,str(Id), org=(x,y+h),fontFace=font, color=(255,255,255), fontScale=1) #pone el texto con esta fuente

            if isIdentifyed:
                break
        cv2.imshow('im',im)
        if isIdentifyed:
            break
        if cv2.waitKey(10) & 0xFF==ord('q'):
            break


    cam.release()
    cv2.destroyAllWindows()

    return isIdentifyed


