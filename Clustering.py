import numpy as np
from Meddleman import Matrix
from utils import min_distance, max_distance, average_distance

"""
Hacer las 3 estrategias maximo,minimo, promedio
Input sera un matrix de distancias
Tambien debemos hacer matrix cogenitica para saber cual es mejor 
Quienes se unen y en que valor se unen en cada iteracón
porque esta matrix de congenetica se usan estos valores
Tambien debemos generar la matrix cogenetica
y el ccc de cada matrix cogenitica de cada estrategica
24 x 24
"""


def generateNewDistanceMatrix(oldDMatrix, index_minor_value, n):
    new_matrix_D = np.copy(oldDMatrix)
    # ?mv = Minor value
    i_mv = index_minor_value[0]
    j_mv = index_minor_value[1]
    new_matrix_D = np.delete(new_matrix_D, obj=j_mv, axis=0)
    new_matrix_D = np.delete(new_matrix_D, obj=j_mv, axis=1)
    for i in range(n - 1):
        if i != i_mv:

            if i < j_mv:
                new_matrix_D[i_mv][i] = (1 / 2) * (oldDMatrix[i_mv, i] + oldDMatrix[j_mv, i] - oldDMatrix[i_mv, j_mv])
                new_matrix_D[i, i_mv] = new_matrix_D[i_mv, i]
            else:
                new_matrix_D[i_mv, i] = (1 / 2) * (
                        oldDMatrix[i_mv, i + 1] + oldDMatrix[j_mv, i + 1] - oldDMatrix[i_mv, j_mv])
                new_matrix_D[i, i_mv] = new_matrix_D[i_mv, i]
            # !Problema cuando la distancia es negativa
            if new_matrix_D[i, i_mv] < 0:
                new_matrix_D[i_mv, i] = VALUE_PROBLEM
                new_matrix_D[i, i_mv] = VALUE_PROBLEM

    new_matrix_D = np.round(new_matrix_D, decimals=2)
    return new_matrix_D


def generateDistance(string1, string2):
    different_positions = 0
    alignments_positions = 0
    for c_1, c_2 in zip(string1, string2):
        if c_1 != "-" and c_2 != "-":
            alignments_positions += 1
        elif (c_1 == "-" and c_2 != "-") or (c_1 != "-" and c_2 == "-"):
            different_positions += 1
    return different_positions / alignments_positions


class Cluster:
    path_to_save_debug = "output/"

    def __init__(self, criterion="Min", list_inputs=[], debug=False):
        self.list_inputs = list_inputs
        if criterion == "Min":
            self.criterion = min_distance
        elif criterion == "Max":
            self.criterion = max_distance
        elif criterion == "Avg":
            self.criterion = average_distance
        else:
            raise Exception("Estrategia puesta no encontrada solo hay Min, Max y Avg")
        self.criterion_str = criterion
        self.debug = debug
        if len(list_inputs) == 0:
            self.matrix_distance = np.zeros((2, 2))
        else:
            print("Debe haber ingresado una matrix de distancias")

    def getIndexMinValue(self, n, matrix):
        # *Indices del triangulo superior para obtener el menor valor
        # https://stackoverflow.com/questions/17368947/generating-indices-using-np-triu-indices

        r, c = np.triu_indices(n, 1)
        list_indexes = list(zip(r, c))

        list_values = [matrix[i, j] for [i, j] in list_indexes]

        # min_value = list_values.min()
        index_min_value = np.argmin(list_values)
        minorValue = matrix[list_indexes[index_min_value]]
        if self.debug:
            with open(self.path_to_save_debug + "iteration" + self.criterion_str + ".txt", "a") as f:
                f.write("\nMenor Valor:\n")
                f.write(str(minorValue))
                f.write("\nIndices del menor Valor en la Matriz: \n")
                f.write(str(list_indexes[index_min_value]))

        return list(list_indexes[index_min_value]), minorValue

    def getNewRow(self, old_matrix, index_min):
        # *Obtener las filas que se uniran
        row_1_delete = old_matrix[index_min[0]]
        row_2_delete = old_matrix[index_min[1]]
        # *Eliminar columnas los elementos a unirse
        row_1_delete = np.delete(row_1_delete, index_min)
        row_2_delete = np.delete(row_2_delete, index_min)
        new_row = []
        for dist_1, dist_2 in zip(row_1_delete, row_2_delete):
            new_row.append(self.criterion(dist_1, dist_2))
        return new_row

    def getNewMatrix(self, old_distance_matrix, index_min):
        # *Eliminar las fila y columnas que se uniran
        new_distance_matrix = np.delete(old_distance_matrix, index_min, axis=0)
        new_distance_matrix = np.delete(new_distance_matrix, index_min, axis=1)

        new_row = self.getNewRow(old_distance_matrix, index_min)
        # *Agregar Columna
        new_distance_matrix = np.append(new_distance_matrix, [new_row], axis=0)
        # *Agregar Fila
        new_row.append(0)
        # https://stackoverflow.com/questions/5954603/transposing-a-1d-numpy-array
        new_row = np.array([new_row])
        new_row = new_row.reshape((-1, 1))
        new_distance_matrix = np.append(new_distance_matrix, new_row, axis=1)

        return new_distance_matrix

    def execute(self, old_distance_matrix):

        if self.debug:
            with open(self.path_to_save_debug + "iteration" + self.criterion_str + ".txt", "w") as f:
                np.savetxt(f, old_distance_matrix, fmt='%1.2f')

        indexes_per_iterations = []
        minorValue_per_iterations = []
        while old_distance_matrix.shape[0] != 2:
            # https://stackoverflow.com/questions/47314754/how-to-get-triangle-upper-matrix-without-the-diagonal-using-numpy

            n = old_distance_matrix.shape[0]

            # -----------------Obtener el indice del menor valor-----------
            index_min, minorValue = self.getIndexMinValue(n, old_distance_matrix)
            indexes_per_iterations.append(index_min)
            minorValue_per_iterations.append(minorValue)

            # -----------------Generar nueva fila y columna-----------
            new_distance_matrix = self.getNewMatrix(old_distance_matrix, index_min)
            # *Actualizar el valor
            old_distance_matrix = new_distance_matrix
            if self.debug:
                with open(self.path_to_save_debug + "iteration" + self.criterion_str + ".txt", "a") as f:
                    f.write("\nMatrix Reducida:\n")
                    np.savetxt(f, old_distance_matrix, fmt='%1.2f')
            # ?Agregar los indices y valores cuando la matrix tieene forma 2x2
        minorValue_per_iterations.append(new_distance_matrix[0, 1])
        indexes_per_iterations.append([0, 1])
        return indexes_per_iterations, minorValue_per_iterations

    def getMatrixCofenetica(self, original_matrix, indexes_per_iterations, minorValue_per_iterations):

        matrixCofenetica = np.zeros(original_matrix.shape)

        list_index = [x for x in range(len(original_matrix[0]))]
        # print(list_index)
        # print(indexes_per_iterations)
        # print(minorValue_per_iterations)

        for index_ite, minorValue_ite in zip(indexes_per_iterations, minorValue_per_iterations):
            new_group_index = []
            temp_new_group_index = []
            new_list_index = []
            for i, index in enumerate(list_index):
                if i in index_ite:
                    if type(index) == list:
                        for item in index:
                            new_group_index.append(item)
                    else:
                        new_group_index.append(index)
                    # temp_new_group_index.append(index) #Funca
                    if type(index) == list:
                        temp_new_group_index.append(index)
                    else:
                        temp_new_group_index.append([index])
                else:
                    new_list_index.append(index)
            new_list_index.append(new_group_index)
            list_index = new_list_index

            # *Rellenar la Matrix Cofenetica
            # print("Temp Group:", temp_new_group_index)
            for group_1 in temp_new_group_index[0]:
                for group_2 in temp_new_group_index[1]:
                    matrixCofenetica[group_1][group_2] = minorValue_ite
                    matrixCofenetica[group_2][group_1] = minorValue_ite
        return matrixCofenetica


# ?Si ingresamos entradas como string
def generate_matrix_distance(self):
    listNone = []

    n_string = len(self.list_inputs)
    matrix_distance = np.full(shape=(n_string, n_string), fill_value=0.0, dtype=np.float_)
    matrix_MatrixGlobal = []
    matrix_alignments = np.full(shape=(n_string, n_string), fill_value="").tolist()
    n_string_inputs = len(self.list_inputs)
    for n in range(n_string_inputs):
        matrix_MatrixGlobal.append(listNone)

    for i in range(n_string):
        for j in range(n_string):
            if i != j:
                s1, s2 = self.list_inputs[i], self.list_inputs[j]
                MatrixGlobal = Matrix(s1, s2)
                MatrixGlobal.fun(s1, s2)
                matrix_MatrixGlobal[i][j] = MatrixGlobal
                matrix_alignments[i][j] = MatrixGlobal.getOneAligment()
                distance = generateDistance(matrix_alignments[i][j][0], matrix_alignments[i][j][1])
                matrix_distance[i][j] = distance
    return matrix_distance