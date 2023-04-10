import numpy as np
import streamlit as st
import os
import matplotlib.pyplot as plt
import math
from bresenham import bresenham


def radon_transform(image, num_angles, num_detect, theta, rotation):
    global min
    shape_min = min(image.shape[:2])
    r = shape_min // 2 - 1

    angles = np.linspace(0, np.pi*2, num_angles, endpoint=False)[:rotation]
    sinogram = np.zeros((num_angles, num_detect))

    for i, angle in enumerate(angles):
        x1 = int(r + r * math.cos(angle))
        y1 = int(r + r * math.sin(angle))
        for j in range(num_detect):
            x2 = int(r + r * math.cos(angle + np.pi - theta/2 + j * theta/(num_detect-1)))
            y2 = int(r + r * math.sin(angle + np.pi - theta/2 + j * theta/(num_detect-1)))
            line = list(bresenham(x1, y1, x2, y2))
            value = 0

            #print(angle, r, x1,y1,x2,y2)
            for element in line:
                if len(image.shape) == 3:
                    value += image[element[0], element[1], 0]
                if len(image.shape) == 2:
                    value += image[element[0], element[1]]
            sinogram[i, j] += value
    max = sinogram.max()
    min = sinogram.min()
    #print(min, max)
    for x in range(len(sinogram)):
        for y in range(len(sinogram[0])):
            sinogram[x, y] = (sinogram[x, y]-min)/(max-min)
    return sinogram


def inverse_radon_transform(sinogram, theta):
    num_angles, num_detect = sinogram.shape[:2]
    shape_min = 500
    r = shape_min // 2 - 1

    angles = np.linspace(0, np.pi * 2, num_angles, endpoint=False)
    reconstruction = np.zeros((shape_min, shape_min))

    for i, angle in enumerate(angles):
        x1 = int(r + r * math.cos(angle))
        y1 = int(r + r * math.sin(angle))
        for j in range(num_detect):
            x2 = int(r + r * math.cos(angle + np.pi - theta/2 + j * theta/(num_detect-1)))
            y2 = int(r + r * math.sin(angle + np.pi - theta/2 + j * theta/(num_detect-1)))
            line = list(bresenham(x1, y1, x2, y2))
            for element in line:
                reconstruction[element[0], element[1]] += sinogram[i, j]

    max = np.max(reconstruction)
    min = np.min(np.nonzero(sinogram))
    print(min, max)
    for x in range(len(reconstruction)):
        for y in range(len(reconstruction[0])):
            if reconstruction[x, y] < min:
                value = 0
            else:
                value = (reconstruction[x, y] - min) / (max - min)
            reconstruction[x, y] = value
    return reconstruction


def convolution_filter(sinogram, size):
    filter = np.zeros(2*size+1)
    filter[size] = 1
    for i in range(1, size):
        if i % 2 == 0:
            filter[i] = 0
        else:
            filter[size + i] = (-4 / (np.pi ** 2)) / (i ** 2)
            filter[size - i] = (-4 / (np.pi ** 2)) / (i ** 2)
    filtered_sinogram = np.zeros(sinogram.shape[:2])
    for i in range(len(sinogram)):
        filtered_sinogram[i] = (np.convolve(sinogram[i], filter, 'same'))
    max = filtered_sinogram.max()
    min = filtered_sinogram.min()
    for x in range(len(filtered_sinogram)):
        for y in range(len(filtered_sinogram[0])):
            filtered_sinogram[x, y] = (filtered_sinogram[x, y] - min) / (max - min)
    return filtered_sinogram



def main():
    st.title('CT scan simulator')
    st.subheader('by Agnieszka Grzymska and Michał Pawlicki')
    st.markdown('---')
    file, num_angles, num_detect, theta, rotation = side_bar()
    image = plt.imread('./images/'+file, format='gray')
    st.image(image, width=300)
    sinogram = radon_transform(image, num_angles, num_detect, theta, rotation)
    st.image(sinogram, width=300)
    filtered_sinogram = convolution_filter(sinogram, 15)
    print(filtered_sinogram.shape)
    st.image(filtered_sinogram, width=300)
    reconstruction = inverse_radon_transform(filtered_sinogram, theta)
    st.image(reconstruction, width=300)


def side_bar():
    st.sidebar.markdown('# Set scanner options')
    st.sidebar.markdown(""" $\Delta$ $\\alpha$ $value$""")
    angles = [1,2,3,5,10,15,18,20,30,36,40,45,60,72,90,120,180,360]
    alpha = st.sidebar.select_slider("Select value of alpha", options=angles)
    num_angles = int(360/alpha)
    st.sidebar.markdown("""$Rotation$ $progress$""")
    rotation = st.sidebar.select_slider("Select number of steps", options=range(1, num_angles + 1, 1))
    st.sidebar.markdown("""$Number$ $of$ $detectors$""")
    num_detect = st.sidebar.select_slider("Select value of n", options=range(10, 101, 10))
    st.sidebar.markdown("""$Span$ $of$ $the$ $emitter$ $system$""")
    span = st.sidebar.select_slider("Select value of l", options=range(1, 361))
    theta = math.radians(span)
    files = os.listdir('./images')
    st.sidebar.markdown("""$File$ $to$ $scan$""")
    file = st.sidebar.selectbox(
        'Choose a file to read',
        files)
    return file, num_angles, num_detect, theta, rotation


if __name__ == '__main__':
    main()
