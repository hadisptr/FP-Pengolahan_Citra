import cv2
import numpy as np

# Parameter-parameter (Membuat kotak2 pada setiap kendaraan yang terdeteksi)
min_contour_width = 40
min_contour_height = 40
offset = 10
line_height = 550
matches = []
cars = 0

# Fungsi untuk mendapatkan titik tengah (centroid)
# Membuat titik tengah pada setiap kotak yang muncul
def get_centroid(x, y, w, h):
    x1 = int(w / 2)
    y1 = int(h / 2)
    cx = x + x1
    cy = y + y1
    return cx, cy

# Mengambil contoh video jalanan yang sudah ada
cap = cv2.VideoCapture('traffic.mp4')
cap.set(3, 1920)
cap.set(4, 1080)

# Membaca frame-frame dari video
ret, frame1 = cap.read()
ret, frame2 = cap.read()

# Membaca frame dari dari video sebelumnya
while ret:
    # Segmentasi (Menghitung perbedaan absolut antara dua frame berturut-turut untuk mendeteksi perubahan atau gerakan dalam video)
    d = cv2.absdiff(frame1, frame2)

    # Segmentasi (Mengubah citra perbedaan menjadi citra skala keabuan)
    grey = cv2.cvtColor(d, cv2.COLOR_BGR2GRAY)

    # Morphologi (Menggunakan filter Gaussian untuk meratakan citra dan mengurangi noise)
    blur = cv2.GaussianBlur(grey, (5, 5), 0)

    # Morphology (Melakukan thresholding untuk memisahkan objek dari latar belakang)
    ret, th = cv2.threshold(blur, 20, 255, cv2.THRESH_BINARY)

    # Morpholoi (Melakukan dilasi untuk memperbesar area objek dan menghubungkan bagian yang terputus)
    dilated = cv2.dilate(th, np.ones((3, 3)))

    # Morphologi (Membuat kernel berbentuk elips sebagai elemen struktural untuk digunakan dalam operasi morfologi)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))

    # Morphologi (Mengaplikasikan operasi morfologi penutupan untuk mengisi lubang kecil dan menghaluskan tepi objek)
    closing = cv2.morphologyEx(dilated, cv2.MORPH_CLOSE, kernel)

    # Menemukan kontur
    contours, _ = cv2.findContours(closing, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    for (i, c) in enumerate(contours):
        (x, y, w, h) = cv2.boundingRect(c)
        contour_valid = (w >= min_contour_width) and (h >= min_contour_height)

        if not contour_valid:
            continue

        # membuat line
        cv2.rectangle(frame1, (x - 10, y - 10), (x + w + 10, y + h + 10), (255, 0, 0), 2)
        cv2.line(frame1, (0, line_height), (1200, line_height), (0, 255, 0), 2)

        centroid = get_centroid(x, y, w, h)
        matches.append(centroid)
        cv2.circle(frame1, centroid, 5, (0, 255, 0), -1)

        # Mendeteksi apakah object melewati garis referensi 
        cx, cy = centroid
        for (x, y) in matches:
            if y < (line_height + offset) and y > (line_height - offset):
                cars += 1
                matches.remove((x, y))
                print(cars)

    # Menampilkan jumlah kendaraan
    cv2.putText(frame1, "Total Kendaraan Terdeteksi: " + str(cars), (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 1,
                (0, 170, 0), 2)
    cv2.putText(frame1, "Kendaraan", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 170, 0), 2)

    # Menampilkan frame
    cv2.imshow("Original", frame1)
    cv2.imshow("Difference", th)

    # Keluar dari program jika tombol 'Esc' ditekan
    if cv2.waitKey(1) == 27:
        break

    frame1 = frame2
    ret, frame2 = cap.read()

# Membebaskan sumber daya
cv2.destroyAllWindows()
cap.release()
