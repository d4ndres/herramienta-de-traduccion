from pdf2image import convert_from_path
poppler_path = r"C:\py_pdf_2_img\poppler-24.02.0\Library\bin"
import matplotlib.pyplot as plt
import os
import cv2
import numpy as np
import pytesseract as tst
tst.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract'
import keras_ocr as ocr


def find_intersection(line1, line2):
    # Convierte las líneas a formato de ecuación general: Ax + By = C
    A1 = line1[3] - line1[1]
    B1 = line1[0] - line1[2]
    C1 = A1 * line1[0] + B1 * line1[1]

    A2 = line2[3] - line2[1]
    B2 = line2[0] - line2[2]
    C2 = A2 * line2[0] + B2 * line2[1]

    # Calcula el determinante
    det = A1 * B2 - A2 * B1

    if det == 0:
        return None  # Líneas paralelas
    else:
        x = (B2 * C1 - B1 * C2) / det
        y = (A1 * C2 - A2 * C1) / det
        return (x, y)

def pdf_to_image(pdf_path, poppler_path, dpi=200):
    pages = convert_from_path(pdf_path, dpi, poppler_path=poppler_path)
    first_page_path = pages[0]
    first_page_path.save('first_page.jpg', 'JPEG')
    return 'first_page.jpg'

def find_table( img ):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Umbralizando la imagen para aislar el área verde
    _, thresh = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY_INV)
    # Encontrando contornos en la imagen umbralizada
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # Ordenando contornos por área y encontrando el más grande
    contours = sorted(contours, key=cv2.contourArea, reverse=True)

    if contours:
        # El primer contorno debe ser el más grande, cubriendo el área deseada
        x, y, w, h = cv2.boundingRect(contours[0])
        # Recortando el área de la imagen original
        cropped_image = image[y:y+h, x:x+w]
        return cropped_image
    else: 
        return None

# regresa la dos elementos la mediaX y la mediaY
def meanLongLines( lines ):
    meanX = 0
    meanY = 0
    for line in lines:
        x1, y1, x2, y2 = line[0]
        meanX += abs(x2 - x1)
        meanY += abs(y2 - y1)
    return meanX/len(lines), meanY/len(lines)

# regresa las distancias mas grandes x y y
def greatestDistance( lines ):
    maxDistanceX = 0
    maxDistanceY = 0
    for line in lines:
        x1, y1, x2, y2 = line[0]
        if abs(x2 - x1) > maxDistanceX:
            maxDistanceX = abs(x2 - x1)
        if abs(y2 - y1) > maxDistanceY:
            maxDistanceY = abs(y2 - y1)
    return maxDistanceX, maxDistanceY

def houghtTransform( img ):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray_blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    # Detectando bordes con Canny
    edges = cv2.Canny(gray_blurred, 50, 150, apertureSize=3)
    # Utilizando la transformada de Hough para detectar líneas
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=50, minLineLength=50, maxLineGap=10)

    image_copy = np.copy(img)

    # meanX, meanY = meanLongLines(lines)
    greatestX, greatestY = greatestDistance(lines)
    count = 0

    acceptedLines = []
    for (index, line) in enumerate(lines):
        x1, y1, x2, y2 = line[0]
        
        if abs(x1-x2) > greatestX * 0.9 or abs(y1-y2) > greatestY * 0.7:
            acceptedLines.append(line)
            # print(line)
            # count += 1
            # print(count)
            # cv2.line(image_copy, (x1, y1), (x2, y2), (0, 255, 0), 3)
            # escribe el indice en x1, y1
            # cv2.putText(image_copy, str(count), (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
    
    # Agregamos las lineas que representan los bordes de la imagen
    # recordar que acceptedLines es una lista de listas

    yf = image_copy.shape[0]
    xf = image_copy.shape[1]
    # print(np.shape(image_copy))
    acceptedLines.append([[0, 0, 0, yf]])
    acceptedLines.append([[xf, 0, xf, yf]])
    acceptedLines.append([[0, 0, xf, 0]])
    acceptedLines.append([[0, yf, xf, yf]])


    # Eliminamos superposiciones de lineas
    result = []
    for i in range(len(acceptedLines)):
        if ( len(result) == 0):
            result.append(acceptedLines[i])
            continue
        x1, y1, x2, y2 = acceptedLines[i][0]
        for j in range(len(result)):
            x3, y3, x4, y4 = result[j][0]
            if abs(x1-x3) < 10 and abs(y1-y3) < 10:
                break
            if j == len(result) - 1:
                result.append(acceptedLines[i])

    result = sorted(result, key=lambda x: x[0][0])

    for (index, line) in enumerate(result):
        x1, y1, x2, y2 = line[0]
        cv2.line(image_copy, (x1, y1), (x2, y2), (0, 0, 255), 3)
        cv2.putText(image_copy, str(index), (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
    


    #agagregamos un margo rojo
    # cv2.rectangle(image_copy, (0, 0), (image_copy.shape[1], image_copy.shape[0]), (0, 0, 255), 8)
    return image_copy, result

def insertIntersections(img, xlines):
    # lines son elementos [[1,2,3,4]]. es necesario convertirlo a un array de numpy y que sea [1,2,3,4]
    copyImg = np.copy(img)
    lines = [line[0] for line in xlines]

    intersections = []
    for i, line1 in enumerate(lines):
        for line2 in lines[i+1:]:
            intersect = find_intersection(line1, line2)
            if intersect:
                intersections.append(intersect)

    # agrega los puntos esquineros de la imagen
    intersections.append((0, 0))
    intersections.append((0, img.shape[0]))
    intersections.append((img.shape[1], 0))
    intersections.append((img.shape[1], img.shape[0]))
    
    for point in intersections:
        cv2.circle(copyImg, (int(point[0]), int(point[1])), 5, (0, 255, 0), -1)

    return copyImg, intersections

def getMatrixByCoordinates( coordinates ):
    coords_sorted = sorted(coordinates, key=lambda x: (x[0], x[1]))

    unique_xs = sorted(set([x for x, y in coords_sorted]))
    unique_ys = sorted(set([y for x, y in coords_sorted]), reverse=True)  # Invertimos para que Y vaya de mayor a menor.
    
    matrix = [[None for _ in unique_xs] for _ in unique_ys]

    for x, y in coords_sorted:
        x_index = unique_xs.index(x)
        y_index = unique_ys.index(y)
        matrix[y_index][x_index] = (x, y)

    # revertir el orden de la lista matrix
    return matrix[::-1]    

def createSubImageByCoordinates( coordinates, img ):

    matrix = getMatrixByCoordinates(coordinates)
    imageMatrix = np.copy(img)

    # necesitamos crear un rectangulo segun la matrix donde encontraremos el punto mas cercano que tenga coordenadas de none, none
    # para encontrar los limites de la celda y sacarlo comoimagen independiente
    for i in range(len(matrix) - 1):
        for j in range(len(matrix[i]) - 1):
            if matrix[i][j] is not None:

                for k in range( j+1, len(matrix[i])):
                    if matrix[i][k] is not None:
                        x1, y1 = matrix[i][j]
                        x2, _ = matrix[i][k]
                        
                        if matrix[i+1][j] is not None and matrix[i+1][k] is not None:
                            _, y2 = matrix[i+1][j]
                            h = int(abs(y2 - y1))
                            w = int(abs(x2 - x1))
                            # cropped_image = image[y:y+h, x:x+w]
                            subImage = img[ int(y1):int(y1+h), int(x1):int(x1+w)]
                            text = tst.image_to_string(subImage)

                            # cv2.write(f'./subImages/{x1}_{y1}.jpg', subImage)
                            cv2.imwrite(f'./subImages/{i}_{j}.jpg', subImage)

                            print(y1, y1+h, x1, x1+w, text)
                            cv2.imshow('imagen', subImage)
                            cv2.waitKey(0)
                            cv2.destroyAllWindows()

                            # cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 255), 2)
                            # cv2.imshow('imagen', img)
                            # cv2.waitKey(0)
                            # cv2.destroyAllWindows()
                        break

    print(matrix)

    # print(coordinates)
    # pass
        

def useKeras():
    pipeline = ocr.pipeline.Pipeline()
    miImage = 'transform.jpg'
    images = [ocr.tools.read(miImage)]
    prediction_groups = pipeline.recognize(images)
    print(prediction_groups)
    fig, axs = plt.subplots(nrows=len(images), figsize=(20, 20))
    for ax, image, predictions in zip(axs, images, prediction_groups):
        ocr.tools.drawAnnotations(image=image, predictions=predictions, ax=ax)
        plt.show()



if __name__ == '__main__':
    pdfFiles = [f for f in os.listdir('.') if f.endswith(".pdf")]
    pathFile = pdfFiles[0]
    path = pdf_to_image(pathFile, poppler_path, dpi=200)
    image = cv2.imread(path)
    
    newImage = find_table(image)

    transform, lines = houghtTransform(newImage)
    _, coordinates = insertIntersections(transform, lines)
    createSubImageByCoordinates(coordinates, newImage)

    # salva la imagen como jpg
    cv2.imwrite('transform.jpg', newImage)
    plt.imshow(newImage)
    plt.show()

