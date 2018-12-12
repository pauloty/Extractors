# Imports necessarios para execucao
import cv2
import csv
import glob
import numpy as np
from skimage import feature
import mahotas.features
import os
from scipy.stats import kurtosis, skew


def extractors(path):
    # Descoberta de todas as imagens no diretorio passado
    imdir = 'output\\' + path + '\\'
    print(imdir)
    specifier = '_from_' + path  # Especificador para cada .csv
    ext = ['png', 'jpg']
    files = []
    [files.extend(glob.glob(imdir + '*.' + e)) for e in ext]
    images = [cv2.imread(file, cv2.IMREAD_GRAYSCALE) for file in files]

    # Codigo para momentos estatisticos
    myFile = open('CSVs\\StatiscalMoments' + specifier + '.csv', 'w', newline="")
    csv.register_dialect('dial', delimiter=';')
    writer = csv.writer(myFile, dialect='dial')
    row = []
    i = 0
    for img in images:
        row.append(files[i].rsplit('\\', 1)[1])   # Adiciona o nome da imagem no arquivo
        # row.append(files[i])                    # Adiciona o caminho da imagem no arquivo
        row.append(np.mean(img))
        row.append(np.var(img))
        # row.extend(list(skew(img)))  // axis = 0
        # row.extend(list(kurtosis(img))) // axis = 0
        row.append(kurtosis(img, axis=None, fisher=False))  # Nao usa a definicao de Fisher e calcula sobre a imagem inteira
        row.append(skew(img, axis=None))
        if 'suspect' in files[i]:
            row.append(1)
        else:
            row.append(2)
        writer.writerow(row)
        row[:] = []
        i = i + 1
    myFile.close()

    # CODIGO MOMENTOS DE HU
    myFile = open('CSVs\\HuMoments' + specifier + '.csv', 'w', newline="")
    csv.register_dialect('dial', delimiter=';')
    writer = csv.writer(myFile, dialect='dial')
    row = []
    i = 0
    for img in images:
        row.append(files[i])  # Adiciona o nome da imagem no arquivo
        ret, th = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
        M = cv2.moments(th, True)
        row.extend(list(cv2.HuMoments(M).flatten()))
        if 'suspect' in files[i]:
            row.append(1)
        else:
            row.append(2)
        writer.writerow(row)
        row[:] = []
        i = i + 1
    myFile.close()

    # CODIGO PARA HARALICK

    # Foi utilizada o pacote Mahotas para a extracao, esse pacote extrai 13 caracteristicas
    # Como na definicao de Haralick, o calculo é feito em 4 direcoes distintas, na literatura é feito
    # a media entre elas

    myFile = open('CSVs\\Haralick' + specifier + '.csv', 'w', newline="")
    csv.register_dialect('dial', delimiter=';')
    writer = csv.writer(myFile, dialect='dial')
    row = []
    i = 0
    for img in images:
        row.append(files[i])  # Adiciona o nome da imagem no arquivo
        # print(mahotas.features.haralick(img).mean(0))
        row.extend(list(mahotas.features.haralick(img).mean(0)))
        if 'suspect' in files[i]:
            row.append(1)
        else:
            row.append(2)
        writer.writerow(row)
        row[:] = []
        i = i + 1
    myFile.close()


    # CODIGO PARA LBP

    # A abordagem utilizada foi aplicar o LBP com alguns parametros pre-determinados
    # o calculo foi feito gerando o histograma, essa é a abordagem feita na literatura

    lbp_sampling_points = 8  # Pontos tomados na vizinhança
    lbp_sampling_radius = 2  # Raio ao redor do pixel central
    method = "uniform"       # Metodo adotado para o LBP
    eps = 1e-7

    myFile = open('CSVs\\LBP' + specifier + '.csv', 'w', newline="")
    csv.register_dialect('dial', delimiter=';')
    writer = csv.writer(myFile, dialect='dial')
    row = []
    i = 0
    for img in images:
        row.append(files[i])  # Adiciona o nome da imagem no arquivo
        lbp = feature.local_binary_pattern(img, lbp_sampling_points, lbp_sampling_radius, method=method)
        (hist, _) = np.histogram(lbp.ravel(), bins=np.arange(0, lbp_sampling_points + 3), range=(0, lbp_sampling_points + 2))
        hist = hist.astype("float")
        hist /= (hist.sum() + eps)
        row = list(hist)
        if 'suspect' in files[i]:
            row.append(1)
        else:
            row.append(2)
        writer.writerow(row)
        row[:] = []
        i = i + 1
    myFile.close()


def main():
    my_dirs = [d for d in os.listdir('output') if os.path.isdir(os.path.join('output', d))]
    extractors(my_dirs[0])
    for dir in my_dirs:
        extractors(dir)
    print("Finalizado com sucesso!")

if __name__ == '__main__':
    main()