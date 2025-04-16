import numpy as np

### Реализация в формате с numpy матрицами, спасибо нейросети_))))
### Написал не сам, сам написал предыдущий вариант, но все равно
### Главное, что разобрался в алгоритме)

def match_key_points_numpy(des1, des2):
    # Расстояние: матрица (len(des1), len(des2))
    dists = np.linalg.norm(des1[:, np.newaxis] - des2[np.newaxis, :], axis=2)

    # Лучшие соответствия из des1 → des2
    best_des1 = np.argmin(dists, axis=1)
    best_des1_dists = np.min(dists, axis=1)

    # Лучшие соответствия из des2 → des1
    best_des2 = np.argmin(dists.T, axis=1)

    # Взаимные совпадения
    matches = []
    for i, j in enumerate(best_des1):
        if best_des2[j] == i:
            matches.append(DummyMatch(i, j, best_des1_dists[i]))

    # Сортировка
    matches.sort(key=lambda m: m.distance)
    return matches