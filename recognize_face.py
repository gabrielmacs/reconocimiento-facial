import cv2,os
import numpy as np
from PIL import Image

recognizer = cv2.face.LBPHFaceRecognizer_create()  #instancia de objeto para usar en entrenamiento 
detector= cv2.CascadeClassifier("haarcascade_frontalface_default.xml")  #reconocer rostro de frente 

# crea la data de entrenamiento con modelo KNN y 8 vecinos

def getImagesAndLabels(path):# es el directorio de donde estas las imagenees 
    imagePaths=[os.path.join(path,f) for f in os.listdir(path)]
    faceSamples=[] #ejemplos de la imagen 
    Ids=[]#ids de las imagenes
    for imagePath in imagePaths:#recorrer todas las imagenes 
        pilImage = Image.open(imagePath).convert('L')# abrimos la imagen usando pillot
        imageNp = np.array(pilImage,'uint8')#aqui la convertimos en un array de numpy, ya que la imagen es una matriz
        Id = int(os.path.split(imagePath)[-1].split(".")[1])#guardamos el id
        faces = detector.detectMultiScale(imageNp)#guarda las imagenes
        for (x,y,w,h) in faces:
            faceSamples.append(imageNp[y:y+h,x:x+w])
            Ids.append(Id)

    return faceSamples,Ids#retorna todos las caras y ids

def run():
    faces,Ids = getImagesAndLabels('dataSet')# de todo el data se coge las caras y ids
    recognizer.train(faces, np.array(Ids))# aqui se le pasa la data con el id 
    recognizer.write('trainner/trainner.yml')