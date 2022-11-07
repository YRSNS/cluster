import numpy as np
from treelib import Node, Tree
import graphviz
import networkx as nx
import matplotlib.pyplot as plt
from utils import *


class ValueCondition:
    def __init__(self, value, index):
        self.value = value
        self.index = index

    def __str__(self):
        return self.value


# ?Link del simuladore del algoritmo de Meddleman
# https://bioboot.github.io/bimm143_W20/class-material/nw/


class Matrix:
    value_interval = -2  # gap
    values_matrix = None  # scores
    # matrix_node = None
    matrix_coordinates = None
    debug = False
    ways = []
    string1 = None
    string2 = None

    def __init__(self, string1, string2, debug=False):
        self.G = nx.DiGraph()
        self.path_simple = []

        self.string1 = string1
        self.string2 = string2
        # ? el +1  es por simular el añadido de al inicio "-"
        n, m = len(string1) + 1, len(string2) + 1
        self.debug = debug
        # --------------Matrix de Valores--------------
        self.values_matrix = np.zeros((n, m), int)
        # ? Rellenar las primera fila y columna con la serie
        # * 0 -2 -4 -6 ...
        self.values_matrix[0] = np.arange(m) * self.value_interval
        self.values_matrix[:, 0] = np.arange(n) * self.value_interval

        # --------------Matrix de Coordenadas--------------
        # ? Matrix donde se guarda lista de tuplas (indices)
        self.matrix_coordinates = []
        for i in range(n):
            self.matrix_coordinates.append([])

        # *Rellenar la primera fila
        for i in range(m):
            tuple_index = (0, i - 1)
            self.matrix_coordinates[0].append([tuple_index])
        # *Rellenar la primera columna sin el 0,0
        for i in range(1, n):
            tuple_index = (i - 1, 0)
            self.matrix_coordinates[i].append([tuple_index])
        self.matrix_coordinates[0][0] = [()]
        # -------------------------------------------------

        # -------Debug--------
        if self.debug:
            print("Cadena 1:", string1)
            print("Cadena 2:", string2)
            print("Matrix de Valores Inicial:", self.values_matrix, end="\n\n")
            print("Matrix de Coordenadas Inicial:", self.matrix_coordinates, end="\n\n")
            print("N = ", n, "M = ", m)

    # *Recursivo a bucle
    def travel_matrix_to_one_simple_path(self, x_index, y_index):
        while not (x_index == 0 and y_index == 0):
            self.path_simple.append(self.matrix_coordinates[x_index][y_index][0])
            x_index = self.path_simple[-1][0]
            y_index = self.path_simple[-1][1]

    def get_one_path(self):
        x_start = len(self.matrix_coordinates) - 1
        y_start = len(self.matrix_coordinates[0]) - 1
        # self.path_simple.append((len(self.matrix_coordinates) - 1, len(self.matrix_coordinates[0]) - 1))
        self.path_simple.append((x_start, y_start))
        self.travel_matrix_to_one_simple_path(x_start, y_start)

    def fix_bool_list(self):
        index = self.path_simple[0]
        list_bool = []
        way = self.path_simple[1::]

        for i, next_node in enumerate(way):

            index_diagonal = (index[0] - 1, index[1] - 1)
            index_izquierda = (index[0], index[1] - 1)

            if index_diagonal == next_node:
                list_bool.append(1)
            elif index_izquierda == next_node:
                list_bool.append(2)
            else:  # Index Arriba
                list_bool.append(3)
            index = next_node
        return list_bool

    def fun(self, string1, string2):
        # +1  es por simular el añadido de al inicio "-"
        n, m = len(string1) + 1, len(string2) + 1

        for i in range(1, n):
            for j in range(1, m):
                # ---------Obtener valores de condiciones---------
                value_first_condition = 1
                if string1[i - 1] != string2[j - 1]:
                    value_first_condition = -1

                # *Valores en orden de Esquina izquierda, solo Derecha y solo izquierda
                index_1, index_2, index_3 = (i - 1, j - 1), (i - 1, j), (i, j - 1)
                value_1 = self.values_matrix[i - 1][j - 1]
                value_2 = self.values_matrix[i - 1][j]
                value_3 = self.values_matrix[i][j - 1]

                # *Guardar el valor junto al indice de donde proviene
                values_matrix = [ValueCondition(value_1 + value_first_condition, index_1),
                                 ValueCondition(value_2 - 2, index_2),
                                 ValueCondition(value_3 - 2, index_3)]
                # ------Mantener solo el mayor valor-----
                # *Ordenar
                sorted_values_conditions = sorted(values_matrix, key=lambda x: x.value)
                # ?Reverse no retorna nada solo actualiza los indices
                sorted_values_conditions.reverse()
                # *Filtrar
                sorted_values_conditions = KeepWay(sorted_values_conditions)
                list_value_indexs = [classValue.index for classValue in sorted_values_conditions]

                # ------Agregar a la matrix de valores y coordenadas-----
                self.matrix_coordinates[i].append(list_value_indexs)
                self.values_matrix[i][j] = sorted_values_conditions[0].value

                # -------Debug--------
                if self.debug:
                    print("-" * 7, i, "-", j, "-" * 7)
                    print("Valores ordenados de mayor a menor:")
                    [print(classValue.__str__(), end=" ") for classValue in sorted_values_conditions]
                    print("Mayores valor con sus indices:", list_value_indexs)
        self.get_one_path()
        if self.debug:
            print("Matrix de Valores:", self.values_matrix)
            print("Matrix de Coordenadas:", self.matrix_coordinates)

    def getAlignmentFix(self, list_bool):
        stringAlignment1 = ""
        stringAlignment2 = ""
        string1_inverse = self.string1[::-1]
        string2_inverse = self.string2[::-1]

        j = 0
        k = 0
        for i, bool_way in enumerate(list_bool):
            if bool_way == 1:
                stringAlignment1 += string1_inverse[j]
                stringAlignment2 += string2_inverse[k]
                j += 1
                k += 1
                # print("1:", "j=" + str(j) + "   k=" + str(k))
            elif bool_way == 2:
                stringAlignment1 += "-"
                stringAlignment2 += string2_inverse[k]
                k += 1
                # print("2:", "j=" + str(j) + "   k=" + str(k))
            elif bool_way == 3:
                stringAlignment1 += string1_inverse[j]
                stringAlignment2 += "-"
                j += 1
                # print("3:", "j=" + str(j) + "   k=" + str(k))
        stringAlignment1, stringAlignment2 = stringAlignment1[::-1], stringAlignment2[::-1]
        return stringAlignment1, stringAlignment2

    def getOneAligment(self):
        list_bool_to_alignment = self.fix_bool_list()
        if self.debug:
            print("Len List Booleano:", len(list_bool_to_alignment))
            print("Lista Booleana:", list_bool_to_alignment)
            print("Len S1:", len(self.string1))
            print("Len S2:", len(self.string2))
        # print("Alineación:")
        alignments = self.getAlignmentFix(list_bool_to_alignment)
        # [print(a) for a in alignments]
        return alignments


# ----------------------Generar Matrix de Distancias a partir de strings--------------------
import pandas as pd
import numpy as np

colums_header = []
listNone = []
paresCaminos = []
index_center_star = None


def MatrixScoreAllString(list_inputs):
    n_string = len(list_inputs)
    # *dtype: https://numpy.org/doc/stable/reference/arrays.dtypes.html
    matrix_distance = np.full(shape=(n_string, n_string), fill_value=0.0, dtype=np.float_)
    matrix_MatrixGlobal = []
    matrix_alignments = np.full(shape=(n_string, n_string), fill_value="").tolist()
    for n in range(len(list_inputs)):
        matrix_MatrixGlobal.append(listNone)

    for i in range(n_string):
        for j in range(n_string):
            if i != j:
                s1, s2 = list_inputs[i], list_inputs[j]
                MatrixGlobal = Matrix(s1, s2)
                MatrixGlobal.fun(s1, s2)
                matrix_MatrixGlobal[i][j] = MatrixGlobal
                matrix_alignments[i][j] = MatrixGlobal.getOneAligment()
                distance = generateDistance(matrix_alignments[i][j][0], matrix_alignments[i][j][1])
                matrix_distance[i][j] = distance
    return matrix_distance



