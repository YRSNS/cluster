from itertools import groupby
import numpy as np


def all_equal(iterable):
    g = groupby(iterable)
    return next(g, True) and not next(g, False)


def KeepWay(values_conditions):
    list_values = [values_conditions[0].value, values_conditions[1].value, values_conditions[2].value]
    if all_equal(list_values):
        return values_conditions
    elif values_conditions[0].value == values_conditions[1].value:
        del values_conditions[2]
        return values_conditions
    elif values_conditions[0].value == values_conditions[2].value:
        del values_conditions[1]
        return values_conditions
    return [values_conditions[0]]


# ----------------------Generar Matrix de Distancias a partir de strings--------------------


# https://stackoverflow.com/questions/31247198/python-pandas-write-content-of-dataframe-into-text-file
def saveTxt(df):
    with open('output.txt', mode='w') as file_object:
        print("Matrix de Score:\n", file=file_object)
        print(df, file=file_object)


def saveStartTXT(center_string, index_center_star, pares_alignment, multiple_alignment):
    f = open("output.txt", "a")
    f.write("Cadena central: " + center_string + '\n')
    f.write("Indice de la cadena central: " + str(index_center_star) + '\n')
    f.write("\nAlineaciones con la cadena central:\n")

    for i in pares_alignment:
        f.write(i + "\n")

    f.write("\nAlinieación Multiple\n")
    for i in multiple_alignment:
        f.write(i + "\n")


def generateDistance(string1, string2):
    different_positions = 0
    alignments_positions = 0
    for c_1, c_2 in zip(string1, string2):
        if c_1 != "-" and c_2 != "-":
            alignments_positions += 1
        elif (c_1 == "-" and c_2 != "-") or (c_1 != "-" and c_2 == "-"):
            different_positions += 1
    return different_positions / alignments_positions


# https://www.ugr.es/~gallardo/pdf/cluster-3.pdf
# https://slideplayer.com/slide/14121882/
# *Estrategias
def min_distance(distance_1, distance_2):
    if distance_1 < distance_2:
        return distance_1
    return distance_2


def max_distance(distance_1, distance_2):
    if distance_1 > distance_2:
        return distance_1
    return distance_2


def average_distance(distance_1, distance_2):
    return (distance_2 + distance_1) / 2


dict_strategics = {0: "Distancia Minima", 1: "Distancia Maxima", 2: "Distancia Promedio"}


def print_Winner(list_cccs):
    print("Min distance:", list_cccs[0])
    print("Max distance:", list_cccs[1])
    print("Promedio ponderado:", list_cccs[2])

    max_ccc = max(list_cccs)
    indexes = [i for i, x in enumerate(list_cccs) if x == max_ccc]
    print("\n\n")
    if len(indexes) == 1:
        print("Estrategia Ganador es ", dict_strategics[indexes[0]], " con ", max_ccc)
    else:
        print("Empate... Las estrategias")
        for index in indexes:
            print(dict_strategics[index])
        print("Con el valor de ", max_ccc)

