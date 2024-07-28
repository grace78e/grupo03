"""
from flask import Flask, request, render_template
import cv2
import numpy as np
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import os

app = Flask(__name__)


# Definición de la clase del modelo
class BurnDetectionModel(nn.Module):
    def __init__(self):
        super(BurnDetectionModel, self).__init__()
        self.model = models.resnet18(pretrained=True)
        num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(num_ftrs, 3)  # Cambia 3 por el número de clases que tienes
        )

    def forward(self, x):
        return self.model(x)

# Cargar modelo YOLO
net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")

# Cargar las clases
classes = []
with open("burn.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]

layer_names = net.getLayerNames()

# Obtener capas de salida no conectadas
output_layers_indices = net.getUnconnectedOutLayers()
if isinstance(output_layers_indices, np.ndarray):
    output_layers = [layer_names[i - 1] for i in output_layers_indices]
else:
    output_layers = [layer_names[output_layers_indices - 1]]

def predict_image(file_path, model):
    img = cv2.imread(file_path)
    if img is None:
        print(f'Error al cargar la imagen: {file_path}')
        return
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    height, width, channels = img.shape

    # Detectar objetos con YOLO
    blob = cv2.dnn.blobFromImage(img, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)

    class_ids = []
    confidences = []
    boxes = []

    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.85:
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)

                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
    print(f'Detecciones: {len(boxes)}')
    plt.imshow(img)
    plt.axis('off')
    plt.show()


    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]
            if class_ids[i] < len(classes):
                label = str(classes[class_ids[i]])
                print(f'Detección: {label} con confianza {confidences[i]:.2f}')

                if label == 'burn':
                    print('Se detectó una quemadura.')
                    # Si se detecta una quemadura, utilizar el modelo de PyTorch para clasificar el grado
                    try:
                        # Cargar y preprocesar la imagen
                        transform = transforms.Compose([
                            transforms.Resize((224, 224)),
                            transforms.ToTensor(),
                            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                        ])
                        image = Image.open(file_path)
                        image = transform(image).unsqueeze(0)  # Añadir dimensión de batch

                        # Cargar el modelo
                        model.eval()  # Cambiar a modo evaluación
                        with torch.no_grad():
                          prediction = model(image) # Predict using the model, not model.predict
                          predicted_class = torch.argmax(prediction, dim=1) # Get index of highest probability
                          confidence = torch.nn.functional.softmax(prediction, dim=1)[0][predicted_class].item() * 100 # Calculate confidence

                        # Asignar etiqueta según la clase predicha
                        if predicted_class == 0:
                            class_label = 'Primer grado'
                        elif predicted_class == 1:
                            class_label = 'Segundo grado'
                        elif predicted_class == 2:
                            class_label = 'Tercer grado'
                        else:
                            class_label = 'No es una imagen de quemadura'

                        plt.imshow(img_rgb)
                        plt.axis('off')
                        plt.title(f'Predicción: {class_label} - Confianza: {confidence:.2f}%')
                        plt.show()

                    except Exception as e:
                        print(f'Error al cargar el modelo o realizar la predicción: {e}')

# Cargar el modelo completo
model = BurnDetectionModel()
state_dict = torch.load('C:/Users/Jhon/Desktop/proyecto_burnIA/static/modelo_RESNET18_quemaduras_final3.pth')
#model.load_state_dict(torch.load('/content/drive/MyDrive/modelo_RESNET18_quemaduras.h5'))
model.eval()  # Cambiar a modo evaluación

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        if 'file' not in request.files:
            return 'No file part'
        file = request.files['file']
        if file.filename == '':
            return 'No selected file'
        file_path = os.path.join('C:/Users/Jhon/Desktop/proyecto_burnIA/static', file.filename)
        file.save(file_path)
        result = predict_image(file_path)
        return render_template('index.html', result=result)  # Asegúrate de pasar el resultado
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
"""
"""
from flask import Flask, request, render_template
import cv2
import numpy as np
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import os
import matplotlib.pyplot as plt

app = Flask(__name__)

# Definición de la clase del modelo
class BurnDetectionModel(nn.Module):
    def __init__(self):
        super(BurnDetectionModel, self).__init__()
        self.model = models.resnet18(pretrained=True)
        num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(num_ftrs, 3)  # Cambia 3 por el número de clases que tienes
        )

    def forward(self, x):
        return self.model(x)

# Cargar modelo YOLO
net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")

# Cargar las clases
classes = []
with open("burn.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]

layer_names = net.getLayerNames()
output_layers_indices = net.getUnconnectedOutLayers()
if isinstance(output_layers_indices, np.ndarray):
    output_layers = [layer_names[i - 1] for i in output_layers_indices]
else:
    output_layers = [layer_names[output_layers_indices - 1]]

# Cargar el modelo completo
model = BurnDetectionModel()
state_dict = torch.load('C:/Users/Jhon/Desktop/proyecto_burnIA/static/modelo_RESNET18_quemaduras_final4.pth')
model.load_state_dict(state_dict)
model.eval()  # Cambiar a modo evaluación

# Ajustar el umbral de confianza para la detección de objetos
confidence_threshold = 0.3

def predict_image(file_path, model):
    img = cv2.imread(file_path)
    if img is None:
        print(f'Error al cargar la imagen: {file_path}')
        return "Error al cargar la imagen."

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    height, width, channels = img.shape

    # Detectar objetos con YOLO
    blob = cv2.dnn.blobFromImage(img, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)

    class_ids = []
    confidences = []
    boxes = []

    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > confidence_threshold:
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)

                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
    print(f'Detecciones: {len(boxes)}')

    # Ajustar el umbral de confianza y superposición no máxima
    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]
            if class_ids[i] < len(classes) and confidences[i] > confidence_threshold:
                label = str(classes[class_ids[i]])
                print(f'Detección: {label} con confianza {confidences[i]:.2f}')

                if label == 'burn':
                    print('Se detectó una quemadura.')
                    # Clasificar el grado de la quemadura
                    try:
                        transform = transforms.Compose([
                            transforms.Resize((224, 224)),
                            transforms.ToTensor(),
                            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                        ])
                        image = Image.open(file_path)
                        image = transform(image).unsqueeze(0)  # Añadir dimensión de batch

                        # Realizar la predicción
                        model.eval()  # Cambiar a modo evaluación
                        with torch.no_grad():
                            prediction = model(image)
                            predicted_class = torch.argmax(prediction, dim=1).item()
                            confidence = torch.nn.functional.softmax(prediction, dim=1)[0][predicted_class].item() * 100

                        # Asignar etiqueta según la clase predicha
                        if predicted_class == 0:
                            class_label = 'Primer grado'
                        elif predicted_class == 1:
                            class_label = 'Segundo grado'
                        elif predicted_class == 2:
                            class_label = 'Tercer grado'
                        else:
                            class_label = 'No es una imagen de quemadura'

                        plt.imshow(img_rgb)
                        plt.axis('off')
                        plt.title(f'Predicción: {class_label} - Confianza: {confidence:.2f}%')
                        plt.show()

                        return f'Predicción: {class_label} - Confianza: {confidence:.2f}%'

                    except Exception as e:
                        print(f'Error al cargar el modelo o realizar la predicción: {e}')
                        return "Error al realizar la predicción."

    return "No se detectaron quemaduras."

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        if 'file' not in request.files:
            return 'No file part'
        file = request.files['file']
        if file.filename == '':
            return 'No selected file'
        
        # Verificar la extensión del archivo
        allowed_extensions = {'jpg', 'jpeg', 'png'}
        file_extension = file.filename.rsplit('.', 1)[1].lower() if '.' in file.filename else ''
        
        if file_extension not in allowed_extensions:
            return 'Formato de archivo no permitido. Solo se permiten JPG y PNG.'

        # Guardar la imagen en el directorio estático
        file_path = os.path.join('C:/Users/Jhon/Desktop/proyecto_burnIA/static', file.filename)
        file.save(file_path)
        
        # Realizar la predicción
        result = predict_image(file_path, model)
        return render_template('index.html', result=result)  # Asegúrate de pasar el resultado

    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
"""

from flask import Flask, request, render_template, send_from_directory
import cv2
import numpy as np
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import os
import matplotlib.pyplot as plt

app = Flask(__name__)

# Definir la ruta para la carpeta de medios
MEDIA_FOLDER = 'C:/Users/Jhon/Desktop/proyecto_burnIA/media'
app.config['MEDIA_FOLDER'] = MEDIA_FOLDER

# Crear un endpoint para servir archivos de la carpeta de medios
@app.route('/media/<path:filename>')
def media(filename):
    return send_from_directory(app.config['MEDIA_FOLDER'], filename)


# Definición del modelo de clasificación de quemaduras
class BurnDetectionModel(nn.Module):
    def __init__(self):
        super(BurnDetectionModel, self).__init__()
        self.model = models.resnet18(pretrained=True)
        num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(num_ftrs, 3)  # Número de clases (grados de quemaduras)
        )

    def forward(self, x):
        return self.model(x)

# Función para predecir el grado de quemadura en una imagen
def predict_image(file_path, model):
    # Cargar y preprocesar la imagen
    img = cv2.imread(file_path)
    if img is None:
        print(f'Error al cargar la imagen: {file_path}')
        return
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    try:
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        
        # Abrir la imagen con PIL para manejar transparencia
        image = Image.open(file_path)
        
        # Convertir la imagen a RGB si tiene canal alfa
        if image.mode == 'RGBA':
            image = image.convert('RGB')
        
        image = transform(image).unsqueeze(0)  # Añadir dimensión de batch

        # Realizar la predicción utilizando el modelo
        model.eval()  # Cambiar a modo evaluación
        with torch.no_grad():
            prediction = model(image)
            predicted_class = torch.argmax(prediction, dim=1)  # Obtener índice de la mayor probabilidad
            confidence = torch.nn.functional.softmax(prediction, dim=1)[0][predicted_class].item() * 100  # Calcular confianza

        # Asignar etiqueta según la clase predicha
        if confidence >= 0.8:
            if predicted_class == 0:
                class_label = 'Primer grado'
            elif predicted_class == 1:
                class_label = 'Segundo grado'
            elif predicted_class == 2:
                class_label = 'Tercer grado'
            else:
                class_label = 'No es una imagen de quemadura'


            # Guardar la imagen de predicción
            output_image_path = os.path.join(app.config['MEDIA_FOLDER'], 'prediction.png')
            plt.imshow(img_rgb)
            plt.axis('off')
            plt.title(f'Predicción: {class_label} - Confianza: {confidence:.2f}%')
            plt.savefig(output_image_path, bbox_inches='tight', pad_inches=0)
            plt.close()  # Cerrar la figura para evitar abrir una nueva ventana

            return class_label  # Retornar solo la etiqueta de clase

        else:
            return "La confianza de la predicción no alcanza el umbral del 70%."

    except Exception as e:
        print(f'Error al realizar la predicción: {e}')
        return "Error al realizar la predicción."
# Cargar el modelo completo
model = BurnDetectionModel()
state_dict = torch.load('C:/Users/Jhon/Desktop/proyecto_burnIA/static/modelo_RESNET18_quemaduras_final4.pth')
model.load_state_dict(state_dict)
model.eval()  # Cambiar a modo evaluación




@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        if 'file' not in request.files:
            return render_template('index.html', error='No file part')
        file = request.files['file']
        if file.filename == '':
            return render_template('index.html', error='No selected file')
        
        # Verificar la extensión del archivo
        allowed_extensions = {'jpg', 'jpeg', 'png'}
        file_extension = file.filename.rsplit('.', 1)[1].lower() if '.' in file.filename else ''
        
        if file_extension not in allowed_extensions:
            return render_template('index.html', error='Formato de archivo no permitido. Solo se permiten JPG y PNG.')

        # Guardar la imagen en el directorio estático
        
        file_path = os.path.join('C:/Users/Jhon/Desktop/proyecto_burnIA/media', file.filename)
        file.save(file_path)
        
        # Realizar la predicción
        result = predict_image(file_path, model)
        return render_template('index.html', result=result)  # Asegúrate de pasar el resultado

    return render_template('index.html')


if __name__ == '__main__':
    app.run(debug=True)
