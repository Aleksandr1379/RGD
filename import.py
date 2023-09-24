import cv2
import numpy as np
import time
import openpyxl
import tkinter as tk
from tkinter import filedialog
from tkinter import messagebox

CONFIDENCE = 0.5
SCORE_THRESHOLD = 0.5
IOU_THRESHOLD = 0.5
config_path = "C:\\yolov3.cfg"
weights_path = "C:\\yolov3.weights"
font_scale = 1
thickness = 1
labels = open("C:\\coco.names").read().strip().split("\n")

# Проверка наличия класса "телефон" в файле coco.names
if "cell phone" not in labels:
    print("Класс 'cell phone' не найден в файле coco.names.")
    exit(1)

# Генерация случайных цветов для каждого класса
colors = np.random.randint(0, 255, size=(len(labels), 3), dtype="uint8")

net = cv2.dnn.readNetFromDarknet(config_path, weights_path)

ln = net.getLayerNames()
ln = [ln[idx - 1] for idx in net.getUnconnectedOutLayers()]

# Функция для выбора видеофайла
def browse_file():
    file_path = filedialog.askopenfilename()
    video_file_entry.delete(0, tk.END)
    video_file_entry.insert(0, file_path)

# Функция для запуска обнаружения
def start_detection():
    video_file = video_file_entry.get()
    if not video_file:
        messagebox.showerror("Ошибка", "Выберите видеофайл.")
        return

    # Используйте опцию cv2.CAP_DSHOW для увеличения нагрузки на CPU
    cap = cv2.VideoCapture(video_file, cv2.CAP_FFMPEG)

    fps = cap.get(cv2.CAP_PROP_FPS)  # Считываем скорость кадров видео

    # Установите размер кадра для обработки
    frame_width = 200  # Установите желаемую ширину кадра
    frame_height = 200  # Установите желаемую высоту кадра
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, frame_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_height)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter("output.mp4", fourcc, fps, (frame_width, frame_height))

    phone_usage_start_time = None  # Время начала использования телефона
    phone_usage_end_time = None    # Время окончания использования телефона
    phone_in_use = False 

    # Создание Excel-файла и рабочей книги
    excel_file = "phone_usage_report.xlsx"
    workbook = openpyxl.Workbook()
    sheet = workbook.active

    # Заголовки для отчета
    sheet["A1"] = "Время начала использования"
    sheet["B1"] = "Время окончания использования"
    sheet["C1"] = "Продолжительность (секунды)"

    # Установка начальной строки для записи отчета
    current_row = 2

    while True:
        ret, image = cap.read()
        if not ret:
            break

        blob = cv2.dnn.blobFromImage(image, 1/255.0, (416, 416), swapRB=True, crop=False)
        net.setInput(blob)
        layer_outputs = net.forward(ln)

        phone_detected = False  # Флаг, указывающий, что обнаружен телефон

        for output in layer_outputs:
         for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]

        if confidence > CONFIDENCE and labels[class_id] in ['cell phone', 'person', 'car']:
            phone_detected = True
            break

    

        if phone_detected:
            if not phone_in_use:
                phone_in_use = True
                phone_usage_start_time = time.time()
        else:
            if phone_in_use:
                phone_in_use = False
                if phone_usage_start_time is not None:
                    phone_usage_end_time = time.time()
                    phone_usage_duration = phone_usage_end_time - phone_usage_start_time
                    print(f"Машинист использовал телефон {phone_usage_duration:.2f} секунд.")
                    
                    # Запись данных в Excel-файл
                    sheet[f"A{current_row}"] = time.strftime("%H:%M:%S", time.localtime(phone_usage_start_time))
                    sheet[f"B{current_row}"] = time.strftime("%H:%M:%S", time.localtime(phone_usage_end_time))
                    sheet[f"C{current_row}"] = phone_usage_duration
                    
                    current_row += 1
                    phone_usage_start_time = None

        out.write(image)
        cv2.imshow("image", image)

        if ord("q") == cv2.waitKey(1):
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()
    
    # Сохранение Excel-файла
    workbook.save(excel_file)

# Создание графического интерфейса
root = tk.Tk()
root.title("Обнаружение объектов на видео")

video_file_label = tk.Label(root, text="Выберите видеофайл:")
video_file_label.pack()

video_file_entry = tk.Entry(root)
video_file_entry.pack()

browse_button = tk.Button(root, text="Обзор", command=browse_file)
browse_button.pack()

start_button = tk.Button(root, text="Начать обнаружение", command=start_detection)
start_button.pack()

root.mainloop()
