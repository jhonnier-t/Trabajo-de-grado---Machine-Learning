from flask import Flask, render_template
import datetime
from datetime import datetime
from threading import Thread
from pickle import load

app = Flask(__name__)

@app.route('/')
def home():
    hora=datetime.now()
    hora=hora.strftime("%H:%M:%S")
    rutaF = '/home/pi/Plazas'
    counter = 0
    while(True):
        celdas = list()
        try:
            fichero = open(rutaF, 'rb')
            lista = load(fichero)
            for celda in lista:
                add = 1 if celda == "L" else 0
                celdas.append("libre" if celda == "L" else "ocupada")
                counter += add
                dis = str(counter)
        except:
            pass
        finally:
            print(celdas)
            if(len(celdas) == 21):
                break
    return render_template('index.html',celda1=celdas[0],celda2=celdas[1],celda3=celdas[2],celda4=celdas[3],
    celda5=celdas[4],celda6=celdas[5],celda7=celdas[6],celda8=celdas[7],celda9=celdas[8],celda10=celdas[9],
    celda11=celdas[10],celda12=celdas[11],celda13=celdas[12],celda14=celdas[13],celda15=celdas[14],
    celda16=celdas[15],celda17=celdas[16],celda18=celdas[17],celda19=celdas[18],celda20=celdas[19],
    celda21=celdas[20],Hora=hora, Dis=dis)

if __name__=='__main__':
    app.run(debug=True)
