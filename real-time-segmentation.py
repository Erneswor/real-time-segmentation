import cv2
import mediapipe as mp
import numpy as np
from tkinter import Tk, Button, Toplevel
from tkinter import filedialog

#Este metodo permite seleccionar una imagen de fondo para la pantalla verde
def pantalla_verde_custom():
    """Función para efecto de pantalla verde con imagen personalizada.
    Abre el explorador de archivos para seleccionar la imagen de fondo.
    """
    # Abrir diálogo para seleccionar la imagen de fondo
    file_path = filedialog.askopenfilename(
        title="Seleccione la imagen de fondo",
        filetypes=[("Image Files", "*.jpg *.jpeg *.png *.bmp")]
    )
    if not file_path:
        print("Operación cancelada")
        return

    fondo = cv2.imread(file_path)
    if fondo is None:
        print("Error cargando imagen de fondo. Asegúrese de que la ruta sea correcta.")
        return

    mp_selfie_segmentation = mp.solutions.selfie_segmentation

    try:
        cap = cv2.VideoCapture(0)  # Ajusta el índice de la cámara si es necesario
        if not cap.isOpened():
            print("Error al abrir la cámara")
            return

        with mp_selfie_segmentation.SelfieSegmentation(model_selection=1) as segmenter:
            print("Presione ESC para salir...")
            while True:
                ret, frame = cap.read()
                if not ret:
                    print("No se pudo leer el frame. Terminando...")
                    break

                # Procesar la imagen: espejar, convertir y segmentar
                frame = cv2.flip(frame, 1)
                frame.flags.writeable = False
                results = segmenter.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                frame.flags.writeable = True

                # Ajustar la máscara al tamaño del frame y aplicar umbral
                mask = cv2.resize(results.segmentation_mask, (frame.shape[1], frame.shape[0]))
                condition = mask > 0.5

                # Redimensionar el fondo a las dimensiones del frame
                fondo_resized = cv2.resize(fondo, (frame.shape[1], frame.shape[0]))
                # Componer la imagen: donde se detecta a la persona se muestra el frame original,
                # en el resto se coloca el fondo seleccionado
                output = np.where(condition[..., None], frame, fondo_resized)

                cv2.imshow("Pantalla Verde Personalizada", output)
                if cv2.waitKey(1) & 0xFF == 27:  # ESC para salir
                    break

    finally:
        cap.release()
        cv2.destroyAllWindows()

#Este metodo permite aplicar un filtro de desenfoque en la región del rostro detectado
def filtro_rostro():
    """Función para aplicar desenfoque sobre la región del rostro detectado."""
    mp_face_detection = mp.solutions.face_detection

    try:
        cap = cv2.VideoCapture(0)  # Ajusta el índice de la cámara si es necesario
        if not cap.isOpened():
            print("Error al abrir la cámara")
            return

        with mp_face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.5) as face_detection:
            print("Presione ESC para salir...")
            while True:
                ret, frame = cap.read()
                if not ret:
                    print("No se pudo leer el frame. Terminando...")
                    break

                frame.flags.writeable = False
                results = face_detection.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                frame.flags.writeable = True

                if results.detections:
                    for detection in results.detections:
                        bboxC = detection.location_data.relative_bounding_box
                        ih, iw, _ = frame.shape
                        x = max(0, int(bboxC.xmin * iw))
                        y = max(0, int(bboxC.ymin * ih))
                        w = min(int(bboxC.width * iw), iw - x)
                        h = min(int(bboxC.height * ih), ih - y)

                        if w > 0 and h > 0:
                            try:
                                rostro = frame[y:y+h, x:x+w]
                                rostro_blur = cv2.GaussianBlur(rostro, (99, 99), 30)
                                frame[y:y+h, x:x+w] = rostro_blur
                                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                            except Exception as e:
                                print(f"Error procesando rostro: {e}")

                cv2.imshow("Filtro en Rostro", frame)
                if cv2.waitKey(1) & 0xFF == 27:
                    break

    finally:
        cap.release()
        cv2.destroyAllWindows()

#Este metodo permite aplicar ver a las personas de color gris y el entorno color negro 
def selfie_segmentation():
    """
    Realiza la segmentación de selfie, mostrando a la persona en gris y el fondo en negro.
    """
    mp_selfie_segmentation = mp.solutions.selfie_segmentation
    segmenter = mp_selfie_segmentation.SelfieSegmentation(model_selection=1)
    cap = cv2.VideoCapture(0)
    print("Presione ESC para salir...")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = segmenter.process(img_rgb)
        mask = results.segmentation_mask

        condition = mask > 0.5
        gris = np.full(frame.shape, (128, 128, 128), dtype=np.uint8)
        negro = np.zeros(frame.shape, dtype=np.uint8)

        output = np.where(condition[..., None], gris, negro)

        cv2.imshow("Selfie Segmentation", output)
        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()

#Este metodo son los estilos del menu
def main_gui():
    """Interfaz gráfica de menú mejorada usando Tkinter."""
    root = Tk()
    root.title("Menú de Filtros")
    root.geometry("400x400")
    root.configure(bg="#2c3e50")  # Fondo oscuro

    # Estilo de botones
    button_style = {
        "font": ("Arial", 14, "bold"),
        "bg": "#3498db",  # Azul
        "fg": "white",
        "activebackground": "#2980b9",  # Azul más oscuro
        "activeforeground": "white",
        "relief": "raised",
        "bd": 3,
        "width": 25
    }

    # Crear botones con estilo
    Button(root, text="Pantalla Verde Personalizada", command=pantalla_verde_custom, **button_style).pack(pady=15)
    Button(root, text="Filtro en Rostro", command=filtro_rostro, **button_style).pack(pady=15)
    Button(root, text="Segmentación (gris/negro)", command=selfie_segmentation, **button_style).pack(pady=15)
    Button(root, text="Salir", command=root.quit, **button_style).pack(pady=15)

    # Agregar un título decorativo
    title_label = Button(
        root, text="Menú de Filtros", font=("Arial", 18, "bold"), bg="#e74c3c", fg="white",
        relief="flat", bd=0, width=30
    )
    title_label.pack(pady=20)

    root.mainloop()

if __name__ == "__main__":
    main_gui()
