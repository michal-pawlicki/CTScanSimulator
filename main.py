import os

import streamlit as st
import os
from PIL import Image


def main():
    st.title('CT scan simulator')
    st.subheader('by Agnieszka Grzymska and Michał Pawlicki')
    st.markdown('---')
    file = side_bar()
    image = Image.open('./images/'+file)
    st.image(image, width=300)
    st.text(image.getpixel((1,2)))


def side_bar():
    st.sidebar.markdown('# Set scanner options')
    st.sidebar.markdown(""" $\Delta$ $\\alpha$ $value$""")
    st.sidebar.select_slider("Select value of alpha", options=range(1, 20, 2))
    st.sidebar.markdown("""$Number$ $of$ $detectors$""")
    st.sidebar.select_slider("Select value of n", options=range(1, 20, 2))
    st.sidebar.markdown("""$Span$ $of$ $the$ $emitter$ $system$""")
    st.sidebar.select_slider("Select value of l", options=range(1, 20, 2))
    files = os.listdir('./images')
    st.sidebar.markdown("""$File$ $to$ $scan$""")
    file = st.sidebar.selectbox(
        'Choose a file to read',
        files)
    return file


if __name__ == '__main__':
    main()
