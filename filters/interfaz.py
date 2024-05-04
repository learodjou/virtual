"""Interfaz para procesamiento de señales."""

import os

import matplotlib.pyplot as plt
import numpy as np
from pydub import AudioSegment
from PyQt5.QtCore import QSize, Qt, QUrl
from PyQt5.QtGui import QFont, QIcon, QPixmap
from PyQt5.QtMultimedia import QMediaContent, QMediaPlayer
from PyQt5.QtWidgets import (
    QApplication,
    QComboBox,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QSlider,
    QSpinBox,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)
from scipy.io import wavfile
from scipy.signal import filtfilt, firwin, iirfilter, kaiserord, lfilter


class MyWindow(QWidget):
    """Creacion de Interfaz."""

    def __init__(self):
        """Inicialización de la Interfaz."""
        super().__init__()
        self.setGeometry(
            0, 0, 2300, 1400
        )  # Esta geometria debe ser cambiada dependiendo la pantalla
        self.setWindowTitle("HMI para procesamiento de señales")
        self.media_player = QMediaPlayer()
        self.media_player.setVolume(50)
        self.sampFreq = 4410  # Establecer frecuencia de muestreo inicial
        self.sound = None
        self.mainLayout()

    def mainLayout(self):
        """Establecer el diseño de la interfaz."""
        layout = QVBoxLayout()

        # Titulo de la interfaz
        self.mainlabel = QLabel("HMI para procesamiento de señales")
        font = QFont("Times New Roman", 18, QFont.Bold)
        self.mainlabel.setFont(font)
        self.mainlabel.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.mainlabel)

        self.name = QLabel("Actividad 1 \nLea Rodriguez Jouault A01659896")
        font2 = QFont("Times New Roman", 12, QFont.Medium)
        self.name.setFont(font2)
        self.name.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.name)

        # Crear pestañas
        self.tabs = QTabWidget()
        layout.addWidget(self.tabs)

        # Pestaña de carga de audio
        self.audio_tab = QWidget()
        self.audio_tabLayout()
        self.tabs.addTab(self.audio_tab, "Audio")

        # Pestaña de Transformada de Fourier
        self.ft_tab = QWidget()
        self.ft_tabLayout()
        self.tabs.addTab(self.ft_tab, "Fourier")

        # Pestaña de Filtrado de audio
        self.filtro_tab = QWidget()
        self.filtros_tabLayout()
        self.tabs.addTab(self.filtro_tab, "Filtros")

        self.setLayout(layout)

    # ----------- AUDIO TAB -----------

    def audio_tabLayout(self):
        """Establecer el diseño de la pestaña de Audio."""
        layout = QVBoxLayout()

        # Diseño del boton
        self.load_button = QPushButton()
        self.load_button.clicked.connect(
            self.load_audio_file
        )  # se redirige a una funcion para poder cargar el audio
        icon_pos = QIcon(
            "/home/lea/Virtual/virtual/img/download.png"
        )  # cambia dependiendo de computadora
        self.load_button.setIcon(icon_pos)
        self.load_button.setIconSize(QSize(80, 80))
        self.load_button.setFixedSize(500, 150)
        self.load_button.setText("Cargar archivo")
        self.load_button.setStyleSheet(
            "background-color : #D6396D; color: white; border-radius: 15px;"
        )
        layout.addWidget(self.load_button, alignment=Qt.AlignCenter)

        self.audio_tab.setLayout(layout)

    def load_audio_file(self):
        """Cargar audio, procesamiento de la señal y guardar grafica."""
        # Se seleccionan solo audios en esos formatos
        file_dialog = QFileDialog()
        file_dialog.setNameFilter("Archivos de audio (*.wav *.mp3 *.aac)")
        file_dialog.selectNameFilter("Archivos de audio (*.wav *.mp3 *.aac)")
        file_dialog.setViewMode(QFileDialog.List)

        if file_dialog.exec_():
            file_paths = file_dialog.selectedFiles()
            if file_paths:
                file_path = file_paths[0]
                self.play_audio(file_path)  # reproducir audio escogido

                # Aplicar procesamiento dependiendo del tipo de audio
                if file_path.endswith(".wav"):
                    self.sampFreq, self.sound = wavfile.read(file_path)
                    if self.sound.ndim == 2:  # convertir a mono si es stereo
                        self.sound = np.mean(self.sound, axis=1)

                elif file_path.endswith(".mp3"):
                    temp = AudioSegment.from_mp3(file_path)
                    if temp.channels > 1:  # convertir a mono si es stereo
                        temp = temp.set_channels(1)
                    self.sound = np.array(temp.get_array_of_samples())
                    self.sampFreq = (
                        temp.frame_rate
                    )  # recuperar frecuencia de muestreo del audio

                elif file_path.endswith(".aac"):
                    temp = AudioSegment.from_file(file_path, format="aac")
                    if temp.channels > 1:  # convertir a mono si es stereo
                        temp = temp.set_channels(1)
                    self.sound = np.array(temp.get_array_of_samples())
                    self.sampFreq = (
                        temp.frame_rate
                    )  # recuperar frecuencia de muestreo del audio

                # Normalizar la señal
                self.sound = self.sound / 2.0**15

                # Graficar la señal original
                plt.plot(self.sound[:], "#D6396D")
                plt.xlabel("Señal de audio")
                plt.title("Gráfico de la señal original")
                plt.tight_layout()
                plt.savefig("original_signal.png")  # se guarda la grafica del audio
                plt.close()

                self.update_interface(file_path)

    def update_interface(self, file_path):
        """Pestaña de Audio se actualiza."""
        # Nombre del audio seleccionado
        self.label = QLabel()
        self.label.setText(f"Archivo de audio cargado: {os.path.basename(file_path)}")
        self.label.setAlignment(Qt.AlignCenter)
        font_file = QFont("Times New Roman", 12, QFont.Bold)
        self.label.setFont(font_file)
        layout = self.audio_tab.layout()
        layout.insertWidget(1, self.label)

        # Cargar la imagen de la grafica del audio
        images_layout = QHBoxLayout()
        images_layout.addWidget(self.show_image("original_signal.png", 1500, 800))
        layout.insertLayout(2, images_layout)

    # ----------- FOURIER TAB -----------

    def ft_tabLayout(self):
        """Establecer el diseño de la pestaña de la Transformada de Fourier."""
        layout = QVBoxLayout()

        # Diseño del boton
        self.ft_button = QPushButton()
        self.ft_button.clicked.connect(
            self.fourier_transform
        )  # se redirige a una funcion para aplicar la ft
        icon_transform = QIcon(
            "/home/lea/Virtual/virtual/img/fourier.png"
        )  # cambia dependiendo de computadora
        self.ft_button.setIcon(icon_transform)
        self.ft_button.setIconSize(QSize(90, 90))
        self.ft_button.setFixedSize(500, 150)
        self.ft_button.setText("Transformada de Fourier")
        self.ft_button.setStyleSheet(
            "background-color : #0CC0DF; color: white; border-radius: 15px;"
        )

        layout.addWidget(self.ft_button, alignment=Qt.AlignCenter)
        self.ft_tab.setLayout(layout)

    def fourier_transform(self):
        """Aplicar la ft y graficar la nueva señal."""
        fft_spectrum = np.fft.rfft(self.sound)
        freq = np.fft.rfftfreq(self.sound.size, d=1.0 / self.sampFreq)
        fft_spectrum_abs = np.abs(fft_spectrum)

        plt.plot(freq, fft_spectrum_abs, "#0CC0DF")
        plt.xlabel("frequency, Hz")
        plt.ylabel("Amplitude, units")
        plt.title("Transformada de Fourier")
        plt.tight_layout()
        plt.savefig("ft.png")  # se guarda la grafica de la ft de la señal
        plt.close()

        self.update_fourier_interface()

    def update_fourier_interface(self):
        """Pestaña de Fourier se actualiza."""
        # EStablecer titulo de la pestaña
        self.label = QLabel()
        self.label.setText("Transformada de Fourier")
        self.label.setAlignment(Qt.AlignCenter)
        font_file = QFont("Times New Roman", 14, QFont.Medium)
        self.label.setFont(font_file)
        layout = self.ft_tab.layout()
        layout.insertWidget(1, self.label)

        # Carga la imagen de la grafica de la ft
        img_layout = QVBoxLayout()
        img_layout.addWidget(
            self.show_image("ft.png", 1500, 1000), alignment=Qt.AlignCenter
        )
        layout.insertLayout(2, img_layout)

    # ----------- FILTRO TAB -----------

    def filtros_tabLayout(self):
        """Establecer el diseño de la pestaña de Filtros."""
        layout = QVBoxLayout()
        layout.setAlignment(Qt.AlignTop)

        filter_type_layout = QHBoxLayout()

        # Diseño para escoger tipo de filtro
        filter_type_label = QLabel("Tipo de filtro:")
        font = QFont("Times New Roman", 12, QFont.Bold)
        filter_type_label.setFont(font)
        self.filter_type = QComboBox()
        self.filter_type.addItems(["IIR", "FIR"])
        self.filter_type.setStyleSheet(
            "background-color: #f0baff; border: 2px; color: #000000; border-radius: 5px;"
        )
        filter_type_layout.addWidget(filter_type_label)
        filter_type_layout.addWidget(self.filter_type)
        layout.addLayout(filter_type_layout)

        # Diseño para escoger el orden del filtro
        order_layout = QHBoxLayout()
        order_label = QLabel("Orden:")
        order_label.setFont(font)
        self.order_spinbox = QSpinBox()
        self.order_spinbox.setRange(1, 10)
        self.order_spinbox.setValue(4)
        self.order_spinbox.setStyleSheet(
            "background-color: #f0baff; border: 2px; color: #000000; border-radius: 5px;"
        )
        order_layout.addWidget(order_label)
        order_layout.addWidget(self.order_spinbox)
        layout.addLayout(order_layout)

        # Diseño para escoger la frecuencia del filtro
        bandpass_layout = QHBoxLayout()
        bandpass_label = QLabel("Frecuencia:")
        bandpass_label.setFont(font)
        self.bandpass = QComboBox()
        self.bandpass.addItems(["lowpass", "highpass", "bandpass"])
        self.bandpass.setStyleSheet(
            "background-color: #f0baff; border: 2px; color: #000000; border-radius: 5px;"
        )
        bandpass_layout.addWidget(bandpass_label)
        bandpass_layout.addWidget(self.bandpass)
        layout.addLayout(bandpass_layout)
        self.bandpass.currentIndexChanged.connect(
            self.updateBandpass
        )  # funcion para agregar o no otra frec e corte

        # Diseño para escoger la frecuencia de corte 1
        slider_layout = QHBoxLayout()
        self.parameter_edit = QLineEdit()
        self.parameter_edit.setFixedWidth(100)
        self.parameter_edit.setText("50")
        self.parameter_edit.setStyleSheet(
            "background-color: #f0baff; border: 2px; color: #000000; border-radius: 5px;"
        )
        slider_label = QLabel("Frec de corte 1:")
        slider_label.setFont(font)
        self.parameter_slider = QSlider(Qt.Horizontal)
        self.parameter_slider.setMinimum(1)
        self.parameter_slider.setMaximum(self.sampFreq / 2)  # maximo la freq de Nyquist
        self.parameter_slider.setValue(50)
        self.parameter_slider.setTickInterval(1000)
        self.parameter_slider.setTickPosition(QSlider.TicksBelow)
        slider_layout.addWidget(slider_label)
        slider_layout.addWidget(self.parameter_edit)
        slider_layout.addWidget(self.parameter_slider)
        layout.addLayout(slider_layout)
        self.parameter_slider.valueChanged.connect(
            self.updateLineEdit
        )  # conectar caja con slider
        self.parameter_edit.textChanged.connect(
            self.updateSlider
        )  # conectar caja con slider

        # Diseño para escoger la frecuencia de corte 2
        second_slider_layout = QHBoxLayout()
        self.parameter_edit2 = QLineEdit()
        self.parameter_edit2.setFixedWidth(100)
        self.parameter_edit2.setText("50")
        self.parameter_edit2.setStyleSheet(
            "background-color: #f0baff; border: 2px; color: #000000; border-radius: 5px;"
        )
        self.second_slider_label = QLabel("Frec de corte 2:")
        self.second_slider_label.setFont(font)
        self.second_parameter_slider = QSlider(Qt.Horizontal)
        self.second_parameter_slider.setMinimum(1)
        self.second_parameter_slider.setMaximum(self.sampFreq / 2)
        self.second_parameter_slider.setValue(50)
        self.second_parameter_slider.setTickInterval(1000)
        self.second_parameter_slider.setTickPosition(QSlider.TicksBelow)
        second_slider_layout.addWidget(self.second_slider_label)
        second_slider_layout.addWidget(self.parameter_edit2)
        second_slider_layout.addWidget(self.second_parameter_slider)
        layout.addLayout(second_slider_layout)
        self.second_parameter_slider.valueChanged.connect(self.updateLineEdit2)
        self.parameter_edit2.textChanged.connect(self.updateSlider2)
        self.parameter_edit2.hide()  # se esconde porque lowpass filter esta de default
        self.second_slider_label.hide()  # solo bandpass exige 2 frecuencias de corte
        self.second_parameter_slider.hide()

        # Diseño del boton para aplicar filtro con parametros previamente escogidos
        self.apply_filter_button = QPushButton("Aplicar Filtro")
        icon_pos = QIcon(
            "/home/lea/Virtual/virtual/img/click.png"
        )  # cambia dependiendo de computadora
        self.apply_filter_button.setIcon(icon_pos)
        self.apply_filter_button.setIconSize(QSize(40, 40))
        self.apply_filter_button.setFixedSize(300, 80)
        self.apply_filter_button.setStyleSheet(
            "background-color: #cb6ce6; border: 2px; color: #ffffff; border-radius: 5px;"
        )

        self.apply_filter_button.clicked.connect(
            lambda: self.apply_filter(
                self.filter_type.currentText(),
                self.bandpass.currentText(),
                float(self.parameter_edit.text()),
                (
                    float(self.parameter_edit2.text())
                    if self.bandpass.currentText() == "bandpass"
                    else None
                ),
                self.order_spinbox.value(),
            )
        )

        layout.addWidget(self.apply_filter_button, alignment=Qt.AlignCenter)
        self.filtro_tab.setLayout(layout)

    def apply_filter(
        self, filterType, bandType, f_cutoff1, f_cutoff2=None, order=4, ftype="butter"
    ):
        """Aplicar filtro a la señal dependiendo de lo escogido por el usuario."""
        # Filtro IIR
        if filterType == "IIR":
            # Checar si se selecciono bandpass para garantizar 2da frec de corte
            if bandType == "bandpass" and f_cutoff2 is not None:
                f_cutoff = [f_cutoff1, f_cutoff2]
            else:
                f_cutoff = f_cutoff1

            # Crear filtro IIR
            b, a = iirfilter(
                N=order, Wn=f_cutoff, fs=self.sampFreq, btype=bandType, ftype=ftype
            )
            # Aplicar filtro IIR al audio
            self.newSignal = filtfilt(b, a, self.sound)

        # Filtro FIR
        elif filterType == "FIR":
            width = 5.0 / (self.sampFreq / 2)  # ancho de transición del filtro
            ripple_db = 20.0  # rizado máximo a 20 dB
            N, beta = kaiserord(
                ripple_db, width
            )  # calcula el orden y beta para la ventana de Kaiser

            # Asegura que el orden sea impar para simetria
            if N % 2 == 0:
                N += 1

            # Checar si se selecciono bandpass para garantizar 2da frec de corte
            if bandType == "bandpass" and f_cutoff2 is not None:
                f_cutoff = [f_cutoff1, f_cutoff2]
                f_cutoff = [fc / (self.sampFreq / 2) for fc in f_cutoff]
            else:
                f_cutoff = f_cutoff1 / (self.sampFreq / 2)

            # Crear filtro FIR con ventna de Kaiser
            taps = firwin(
                N,
                f_cutoff,
                window=("kaiser", beta),
                pass_zero=bandType,
                fs=self.sampFreq,
            )
            # Aplicar filtro FIR al audio
            self.newSignal = lfilter(taps, 1.0, self.sound)

        self.plot_filter()  # Graficar audio filtrado
        self.fourier_filter()  # Graficar ft del audio filtrado
        self.update_filter_int()

    def fourier_filter(self):
        """Aplicar la ft y graficar la señal filtrada."""
        fft_spectrum = np.fft.rfft(self.newSignal)
        freq = np.fft.rfftfreq(self.newSignal.size, d=1.0 / self.sampFreq)
        fft_spectrum_abs = np.abs(fft_spectrum)

        plt.plot(freq, fft_spectrum_abs, "#ff81c1")
        plt.xlabel("frequency, Hz")
        plt.ylabel("Amplitude, units")
        plt.title("Transformada de Fourier filtrada")
        plt.tight_layout()
        plt.savefig("Filtered_Fourier.png")
        plt.close()

    def plot_filter(self):
        """Graficar y guardar señal filtrada."""
        plt.plot(self.sound, label="Original Audio", color="#b432d9")
        plt.plot(self.newSignal, label="Filtered Audio", color="#ff81c1")
        plt.title("Original vs. Filtered Audio Signal")
        plt.xlabel("Samples")
        plt.ylabel("Amplitude")
        plt.legend()
        plt.tight_layout()
        plt.savefig("Filtered_Audio.png")
        plt.close()

    def update_filter_int(self):
        """Pestaña de Filtros se actualiza."""
        layout = self.filtro_tab.layout()

        # Mostrar graficas del audio filtrado
        self.img_layout = QHBoxLayout()
        self.img_layout.addWidget(self.show_image("Filtered_Audio.png", 1300, 700))
        self.img_layout.addWidget(self.show_image("Filtered_Fourier.png", 1300, 700))

        # Diseño de botones
        self.button_layout = QHBoxLayout()
        self.clear_button = QPushButton("Borrar")
        self.clear_button.setFixedSize(250, 80)
        self.clear_button.setStyleSheet(
            "background-color: #cb6ce6; border: 2px; color: #ffffff; border-radius: 5px;"
        )
        self.clear_button.clicked.connect(self.clear)  # Aplicar nuevo filtro

        self.save_button = QPushButton("Guardar")
        self.save_button.setFixedSize(250, 80)
        self.save_button.setStyleSheet(
            "background-color: #cb6ce6; border: 2px ;color: #ffffff; border-radius: 5px;"
        )
        self.save_button.clicked.connect(self.save_filter)  # Guardar audio filtrado

        self.button_layout.addWidget(self.clear_button, alignment=Qt.AlignCenter)
        self.button_layout.addWidget(self.save_button, alignment=Qt.AlignCenter)

        layout.addLayout(self.button_layout)
        layout.insertLayout(6, self.img_layout)
        layout.insertLayout(7, self.button_layout)

    # ----------- CONFIGURACIONES -----------

    def clear(self):
        """Resetear valores."""
        self.newSignal = None  # borrar senal filtrada
        self.parameter_edit.setText("50")  # reset de slider de frec de corte 1
        self.parameter_edit2.setText("50")  # reset de slider de frec de corte 2
        self.order_spinbox.setValue(4)  # reset de orden

        if self.img_layout is not None:  # elimina graficas
            self.clearLayout(self.img_layout)
            self.img_layout = None

        if self.button_layout is not None:  # elimina botones
            self.clearLayout(self.button_layout)
            self.button_layout = None

    def clearLayout(self, layout):
        """Borrar elementos graficos."""
        if layout is not None:
            while layout.count():
                item = layout.takeAt(0)
                widget = item.widget()
                if widget is not None:
                    widget.deleteLater()
                else:
                    self.clearLayout(item.layout())

    def save_filter(self):
        """Guardar nueva señal filtrada en formato escogido."""
        # Normalizar señal de audio para estar dentro del rango de 16 bits
        self.newSignal = np.int16(
            self.newSignal / np.max(np.abs(self.newSignal)) * 32767
        )
        # Convierte la matriz de señal normalizada y escalada en un string de bytes
        audio_segment = AudioSegment(
            self.newSignal.tobytes(),
            sample_width=2,
            frame_rate=self.sampFreq,
            channels=1,
        )

        # Seleccionar tipo de archivo a guardar
        options = QFileDialog.Options()
        fileName, selectedFilter = QFileDialog.getSaveFileName(
            self,
            "Save File",
            "",
            "WAV Audio Files (*.wav);;MP3 Audio Files (*.mp3);;AAC Audio Files (*.aac)",
            options=options,
        )

        if fileName:
            # Dependiendo de la seleccion y de la terminacion del archivo, se le agrega
            if ".wav" in selectedFilter and not fileName.endswith(".wav"):
                fileName += ".wav"
            elif ".mp3" in selectedFilter and not fileName.endswith(".mp3"):
                fileName += ".mp3"
            elif ".aac" in selectedFilter and not fileName.endswith(".aac"):
                fileName += ".aac"

            # DEpendiendo del tipo se archivo seleccionado, se guarda en ese formato
            if fileName.endswith(".wav"):
                wavfile.write(fileName, self.sampFreq, self.newSignal)
            elif fileName.endswith(".mp3"):
                audio_segment.export(fileName, format="mp3")
            elif fileName.endswith(".aac"):
                audio_segment.export(fileName, format="ipod", codec="aac")

    def updateSlider(self):
        """Enlazar caja 1 con slider 1."""
        value = int(self.parameter_edit.text())
        self.parameter_slider.setValue(value)

    def updateLineEdit(self):
        """Enlazar caja 1 con slider 1, texto."""
        value = self.parameter_slider.value()
        self.parameter_edit.setText(str(value))

    def updateSlider2(self):
        """Enlazar caja 2 con slider 2."""
        value = int(self.parameter_edit2.text())
        self.second_parameter_slider.setValue(value)

    def updateLineEdit2(self):
        """Enlazar caja 2 con slider 2, texto."""
        value = self.second_parameter_slider.value()
        self.parameter_edit2.setText(str(value))

    def updateBandpass(self, index):
        """Mostrar segunda frecuencia de corte."""
        if index == 2:
            self.parameter_edit2.show()
            self.second_slider_label.show()
            self.second_parameter_slider.show()
        else:
            self.parameter_edit2.hide()
            self.second_slider_label.hide()
            self.second_parameter_slider.hide()

    def show_image(self, image_path, width, height):
        """Mostrar imagen en interfaz."""
        pixmap = QPixmap(image_path)
        pixmap = pixmap.scaled(width, height, Qt.KeepAspectRatio)
        label = QLabel(self)
        label.setPixmap(pixmap)
        label.setAlignment(Qt.AlignCenter)
        return label

    def play_audio(self, file_path):
        """Reproducir audio."""
        media_content = QMediaContent(QUrl.fromLocalFile(file_path))
        self.media_player.setMedia(media_content)
        self.media_player.play()


def main():
    """Establecer Main."""
    app = QApplication([])
    window = MyWindow()
    window.show()
    app.exec_()


if __name__ == "__main__":
    main()
