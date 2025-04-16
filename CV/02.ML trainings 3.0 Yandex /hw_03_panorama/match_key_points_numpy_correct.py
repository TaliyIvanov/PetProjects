import numpy as np


# o not change the code in the block below
# __________start of block__________
class DummyMatch:
    def __init__(self, queryIdx, trainIdx, distance):
        self.queryIdx = queryIdx  # index in des1
        self.trainIdx = trainIdx  # index in des2
        self.distance = distance
# __________end of block__________


def match_key_points_numpy(des1: np.ndarray, des2: np.ndarray) -> list:
    """
    Match descriptors using brute-force matching with cross-check.

    Args:
        des1 (np.ndarray): Descriptors from image 1, shape (N1, D)
        des2 (np.ndarray): Descriptors from image 2, shape (N2, D)

    Returns:
        List[DummyMatch]: Sorted list of mutual best matches.
    """
    # YOUR CODE HERE
    # Как я понял задание:

    # Шаг 1: Ищем пары для des1
        # Создаем словарь для хранения пары и расстояния для des1[i] (привет LeetCode)
        # best_des1 = {}
        # Итерируюсь по des1
            # Создаю переменные, которые хранят: 
            #                                   минимальную дистанцию 
            #                                   лучшую пару для des1[i]              
            # Итерируюсь по des2:
                # вычисляю Евклидово расстояние между дискрипторами
                # если оно короче минимального, то обновляю переменные
            # добавляю полученные переменные в словарь для des1
            # best_des1[i] = {best_j, min_dist} 

    # Шаг 2: Ищем пары для des2
        # повторяет Шаг1, только для des2

    # Шаг 3: Оставляем только взаимные пары значений
        # Проверка на взаимные пары в словарях привет LeetCode 205. Isomorphic Strings
        # создаем результирующий список
        # за O(N^2) перебираем значения в словарях
        # если пары совпадают, то добавляем их в результирующий список

    # Шаг 4:
        # сортируем список по ключу = distance
        # возвращаем результат


    # Шаг 1: Поиск пар для des1
    best_des1 = {}
    for i in range(len(des1)):
        min_dist = float('inf')
        best_j = -1
        # второй цикл полного перебора des2
        for j in range(len(des2)):
            curr_dist = np.linalg.norm(des1[i] - des2[j]) # считаем Евклидово расстояние
            if curr_dist < min_dist:
                min_dist = curr_dist
                best_j = j
        # добавляем пару индексов и расстояние в словарь
        best_des1[i] = (best_j, min_dist)

    # Шаг 2: Поиск пар для des2
    best_des2 = {}
    for i in range(len(des2)):
        min_dist = float('inf')
        best_j = -1
        for j in range(len(des1)):
            curr_dist = np.linalg.norm(des2[i] - des1[j])
            if curr_dist < min_dist:
                min_dist = curr_dist
                best_j = j
        best_des2[i] = (best_j, min_dist)
    
    # Шаг 3: Оставляем только взаимные пары значений
    # инициализирую результирующий массив
    matches = []
    # проверяем оба словаря на совпадение пар
    for i, (j, dist) in best_des1.items():
        if best_des2.get(j, (-1,))[0] == i:
            matches.append(DummyMatch(i,j,dist))

    # Шаг 4:
    # сортируем по месту
    matches.sort(key=lambda m: m.distance)
    return matches