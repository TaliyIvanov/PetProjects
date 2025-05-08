"""
В данном файлике разобран код метрики Intersection over Union для двух рамок.
В основном ноутбуке есть данная метрика, но она используется именно для 
задачи сегментации.
Здесь же я базово разбираю данную метрику для задачи Детектирования.

Кратко:
- На вход подаются 4 координаты двух рамок - bboxes
- Находим координаты фигуры-пересечения
	- $x_{left}$ - максимальный из x левых границ рамок
	- $x_{right}$ - минимальный из х правых границ рамок
	- $y_{up}$ - максимальный их y верхних границ рамок
	- $y_{down}$ - минимальный из y верхних границ рамок
- Считаем площадь пересечения (y_down-y_up) x (x_right - x_left)
- Считаем площадь объединения площадь bbox_1 + площадь bbox_2 - пересечение
- Вычисляем метрику: Пересечение / Объединение
"""



def IoU_metric(p_1, p_2, p_3, p_4):
    """
    p_1 = (x_1, y_1)
    p_2 = (x_2, y_2)
    p_3 = (x_3, y_3)
    p_4 = (x_4, y_4)
    """
    # find intersection coordinates
    x_left = max(p_1[0], p_3[0])
    x_right = min(p_2[0], p_4[0])
    y_up = max(p_3[1], p_1[1])
    y_down = min(p_2[1], p_4[1])

    # проверка на отсутствие пересечения
    if x_right < x_left or y_down < y_up:
        return -1
    
    # find intersection area
    inter_area = (x_right - x_left) * (y_down - y_up)
    
    
    # find bboxes areas
    bbox_1_area = (p_2[0]-p_1[0]) * (p_2[1] - p_1[1])
    bbox_2_area = (p_4[0]-p_3[0]) * (p_4[1] - p_3[1])
    # find union area
    union_area = bbox_1_area + bbox_2_area - inter_area
    # find IoU metric
    IoU = inter_area / union_area

    return IoU

p_1 = (4,3)
p_2 = (9,8)
p_3 = (7,2)
p_4 = (12,6)

print(IoU_metric(p_1, p_2, p_3, p_4))