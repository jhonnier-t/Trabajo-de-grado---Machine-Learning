import numpy as np
import cv2
import dropbox
import time
import threading
from os import system, remove

def show():
    serv = dropbox.Dropbox('token')
    for i in range(0, 1200):
        system("fswebcam -r 1920x1080 --jpeg 99 -D 3 -S 15/home/pi/park"+str(i)+".jpg")
        ruta = '/Dropbox/Pruebas/parking'+str(i)+'.jpg'
        try:
            serv.files_upload(open("/home/pi/park"+str(i)+".jpg",'rb').read(), ruta)
        except:
            pass
        remove("/home/pi/park"+str(i)+".jpg")
        time.sleep(400)
        print('Thead Finalizado')

t1 = threading.Thread(name='Show', target=show, args=(1,))
t1.start()
t1.join()