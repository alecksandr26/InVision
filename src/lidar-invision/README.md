# LIDAR Project - Raspberry Pi

Este proyecto usa un LiDAR para obtener:
- Ángulo (grados)
- Distancia (cm)

## Requisitos

- Raspberry Pi
- SDK de YDLIDAR

## Instalación del SDK

git clone https://github.com/YDLIDAR/YDLidar-SDK.git
cd YDLidar-SDK
mkdir build
cd build
cmake ..
make
sudo make install

## Compilar este proyecto

mkdir build
cd build
cmake ..
make

## Ejecutar

./tmini_test
