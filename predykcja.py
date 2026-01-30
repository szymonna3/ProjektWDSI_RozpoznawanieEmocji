import cv2
import numpy as np
from tensorflow.keras.models import load_model

SCIEZKA_DO_ZDJECIA = "przykladowe/2.jpg"

try:
    model = load_model('moj_model_emocji.h5')
except:
    print("Brak modelu")
    exit()

emocje_dict = {0: "Zlosc", 1: "Odraza", 2: "Strach", 3: "Radosc", 4: "Smutek", 5: "Zaskoczenie", 6: "Neutralny"}

image = cv2.imread(SCIEZKA_DO_ZDJECIA)

if image is None:
    print("Brak pliku")
    exit()

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
faces = face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=6, minSize=(30, 30))

if len(faces) > 0:
    for (x, y, w, h) in faces:
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

        roi_gray = gray[y:y + h, x:x + w]
        roi_gray = cv2.resize(roi_gray, (48, 48))
        roi_gray = roi_gray.astype('float32') / 255.0
        roi_gray = np.expand_dims(roi_gray, axis=0)
        roi_gray = np.expand_dims(roi_gray, axis=-1)

        prediction = model.predict(roi_gray)
        max_index = int(np.argmax(prediction))
        predicted_emotion = emocje_dict[max_index]
        pewnosc = prediction[0][max_index] * 100

        tekst = f"{predicted_emotion} ({pewnosc:.0f}%)"
        tekst_y = y + 25 if y < 30 else y - 10

        cv2.putText(image, tekst, (x, tekst_y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        print(tekst)

    cv2.imshow('Wynik', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
else:
    print("Nie wykryto twarzy")