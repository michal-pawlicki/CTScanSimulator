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
    sinogram = np.zeros((num_angles, num_detect, 3))

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
            sinogram[i, j, 0] += value
    max = sinogram.max()
    min = sinogram.min()
    #print(min, max)
    for x in range(len(sinogram)):
        for y in range(len(sinogram[0])):
            value = (sinogram[x, y, 0]-min)/(max-min)
            sinogram[x, y, 0] = value
            sinogram[x, y, 1] = value
            sinogram[x, y, 2] = value
    return sinogram


def main():
    st.title('CT scan simulator')
    st.subheader('by Agnieszka Grzymska and Michał Pawlicki')
    st.markdown('---')
    file, num_angles, num_detect, theta, rotation = side_bar()
    image = plt.imread('./images/'+file, format='gray')
    sinogram = radon_transform(image, num_angles, num_detect, theta, rotation)
    st.image(image, width=300)
    st.image(sinogram, width=300)


def side_bar():
    st.sidebar.markdown('# Set scanner options')
    st.sidebar.markdown(""" $\Delta$ $\\alpha$ $value$""")
    angles = [1,2,3,4,5,6,8,9,10]
    alpha = st.sidebar.select_slider("Select value of alpha", options=angles)
    num_angles = int(360/alpha)
    st.sidebar.markdown("""$Rotation$ $progress$""")
    rotation = st.sidebar.select_slider("Select number of steps", options=range(1, num_angles + 1, 1))
    st.sidebar.markdown("""$Number$ $of$ $detectors$""")
    num_detect = st.sidebar.select_slider("Select value of n", options=range(50, 151, 10))
    st.sidebar.markdown("""$Span$ $of$ $the$ $emitter$ $system$""")
    span = st.sidebar.select_slider("Select value of l", options=range(20, 181))
    theta = math.radians(span)*2
    files = os.listdir('./images')
    st.sidebar.markdown("""$File$ $to$ $scan$""")
    file = st.sidebar.selectbox(
        'Choose a file to read',
        files)
    return file, num_angles, num_detect, theta, rotation


if __name__ == '__main__':
    main()
