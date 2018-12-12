# Imports necessarios para execucao
import cv2
import csv
import glob
import numpy as np
from skimage import feature
import mahotas.features
import os
from scipy.stats import kurtosis, skew


def statisticalMoments(files, images):
    # Codigo para momentos estatisticos

    csv.register_dialect('dial', delimiter=';')
    myFile = open('CSVs\\StatiscalMoments_High.csv', 'w', newline="") # High file
    writer = csv.writer(myFile, dialect='dial')
    myFile1 = open('CSVs\\StatiscalMoments_Low.csv', 'w', newline="") # Low file
    writer1 = csv.writer(myFile1, dialect='dial')
    myFile2 = open('CSVs\\StatiscalMoments_Severe.csv', 'w', newline="") # Lesao Grave file
    writer2 = csv.writer(myFile2, dialect='dial')
    myFile3 = open('CSVs\\StatiscalMoments_Normal.csv', 'w', newline="") # Lesao Normal file
    writer3 = csv.writer(myFile3, dialect='dial')

    row = []
    i = 0
    for img in images:
        row.append(files[i].rsplit('\\', 1)[1])  # Adiciona o nome da imagem no arquivo
        row.append(np.mean(img))
        row.append(np.var(img))
        row.append(kurtosis(img, axis=None, fisher=False))  # Nao usa a definicao de Fisher e calcula sobre a imagem inteira
        row.append(skew(img, axis=None))
        if 'high' in files[i]:
            writer.writerow(row)
        if 'low' in files[i]:
            writer1.writerow(row)
        if 'severe' in files[i]:
            writer2.writerow(row)
        if 'normal' in files[i]:
            writer3.writerow(row)
        row[:] = []
        i = i + 1

    myFile.close()
    myFile1.close()
    myFile2.close()
    myFile3.close()


def huMoments(files, images):

    csv.register_dialect('dial', delimiter=';')
    myFile = open('CSVs\\HuMoments_High.csv', 'w', newline="") # High file
    writer = csv.writer(myFile, dialect='dial')
    myFile1 = open('CSVs\\HuMoments_Low.csv', 'w', newline="") # Low file
    writer1 = csv.writer(myFile1, dialect='dial')
    myFile2 = open('CSVs\\HuMoments_Severe.csv', 'w', newline="") # Lesao Grave file
    writer2 = csv.writer(myFile2, dialect='dial')
    myFile3 = open('CSVs\\HuMoments_Normal.csv', 'w', newline="") # Lesao Normal file
    writer3 = csv.writer(myFile3, dialect='dial')

    row = []
    i = 0
    for img in images:
        row.append(files[i].rsplit('\\', 1)[1])  # Adiciona o nome da imagem no arquivo
        ret, th = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
        M = cv2.moments(th, True)
        row.extend(list(cv2.HuMoments(M).flatten()))
        if 'high' in files[i]:
            writer.writerow(row)
        if 'low' in files[i]:
            writer1.writerow(row)
        if 'severe' in files[i]:
            writer2.writerow(row)
        if 'normal' in files[i]:
            writer3.writerow(row)
        row[:] = []
        i = i + 1

    myFile.close()
    myFile1.close()
    myFile2.close()
    myFile3.close()


def haralickMoments(files, images):
    # CODIGO PARA HARALICK

    csv.register_dialect('dial', delimiter=';')
    myFile = open('CSVs\\Haralick_High.csv', 'w', newline="")  # High file
    writer = csv.writer(myFile, dialect='dial')
    myFile1 = open('CSVs\\Haralick_Low.csv', 'w', newline="")  # Low file
    writer1 = csv.writer(myFile1, dialect='dial')
    myFile2 = open('CSVs\\Haralick_Severe.csv', 'w', newline="")  # Lesao Grave file
    writer2 = csv.writer(myFile2, dialect='dial')
    myFile3 = open('CSVs\\Haralick_Normal.csv', 'w', newline="")  # Lesao Normal file
    writer3 = csv.writer(myFile3, dialect='dial')

    row = []
    i = 0
    for img in images:
        row.append(files[i].rsplit('\\', 1)[1])  # Adiciona o nome da imagem no arquivo
        row.extend(list(mahotas.features.haralick(img).mean(0)))
        if 'high' in files[i]:
            writer.writerow(row)
        if 'low' in files[i]:
            writer1.writerow(row)
        if 'severe' in files[i]:
            writer2.writerow(row)
        if 'normal' in files[i]:
            writer3.writerow(row)
        row[:] = []
        i = i + 1

    myFile.close()
    myFile1.close()
    myFile2.close()
    myFile3.close()

def lbp(files, images):
    # CODIGO PARA LBP

    lbp_sampling_points = 8  # Pontos tomados na vizinhança
    lbp_sampling_radius = 2  # Raio ao redor do pixel central
    method = "uniform"       # Metodo adotado para o LBP
    eps = 1e-7

    csv.register_dialect('dial', delimiter=';')
    myFile = open('CSVs\\LBP_High.csv', 'w', newline="")  # High file
    writer = csv.writer(myFile, dialect='dial')
    myFile1 = open('CSVs\\LBP_Low.csv', 'w', newline="")  # Low file
    writer1 = csv.writer(myFile1, dialect='dial')
    myFile2 = open('CSVs\\LBP_Severe.csv', 'w', newline="")  # Lesao Grave file
    writer2 = csv.writer(myFile2, dialect='dial')
    myFile3 = open('CSVs\\LBP_Normal.csv', 'w', newline="")  # Lesao Normal file
    writer3 = csv.writer(myFile3, dialect='dial')

    row = []
    i = 0
    for img in images:
        row.append(files[i].rsplit('\\', 1)[1])  # Adiciona o nome da imagem no arquivo
        lbp = feature.local_binary_pattern(img, lbp_sampling_points, lbp_sampling_radius, method=method)
        (hist, _) = np.histogram(lbp.ravel(), bins=np.arange(0, lbp_sampling_points + 3), range=(0, lbp_sampling_points + 2))
        hist = hist.astype("float")
        hist /= (hist.sum() + eps)
        row = list(hist)
        if 'high' in files[i]:
            writer.writerow(row)
        if 'low' in files[i]:
            writer1.writerow(row)
        if 'severe' in files[i]:
            writer2.writerow(row)
        if 'normal' in files[i]:
            writer3.writerow(row)
        row[:] = []
        i = i + 1

    myFile.close()
    myFile1.close()
    myFile2.close()
    myFile3.close()


def main():
    files = []
    my_dirs = [d for d in os.listdir('output') if os.path.isdir(os.path.join('output', d))]
    for dir in my_dirs:
        # Descoberta de todas as imagens no diretorio passado
        imdir = 'output\\' + dir + '\\'
        # print(imdir)
        ext = ['png', 'jpg']
        [files.extend(glob.glob(imdir + '*.' + e)) for e in ext]

    images = [cv2.imread(file, cv2.IMREAD_GRAYSCALE) for file in files]

    statisticalMoments(files, images)
    huMoments(files, images)
    haralickMoments(files, images)
    lbp(files, images)

    print("Finalizado com sucesso!")

if __name__ == '__main__':
    main()