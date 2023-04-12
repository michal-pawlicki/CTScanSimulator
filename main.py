import numpy as np
import streamlit as st
import os
import matplotlib.pyplot as plt
import math
from bresenham import bresenham
from pydicom.dataset import Dataset, FileDataset
from pydicom.uid import ExplicitVRLittleEndian
import pydicom._storage_sopclass_uids
from skimage.util import img_as_ubyte
from skimage.exposure import rescale_intensity

@st.cache_data
def radon_transform(image, num_angles, num_detect, theta):
    global min
    shape_min = min(image.shape[:2])
    r = shape_min // 2 - 1

    angles = np.linspace(0, np.pi*2, num_angles, endpoint=False)
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


@st.cache_data
def inverse_radon_transform(sinogram, theta):
    num_angles, num_detect = sinogram.shape[:2]
    shape_min = 500
    r = shape_min // 2 - 1

    angles = np.linspace(0, np.pi * 2, num_angles, endpoint=False)
    reconstruction = np.zeros((shape_min, shape_min))
    matrix = np.zeros((shape_min, shape_min))

    for i, angle in enumerate(angles):
        x1 = int(r + r * math.cos(angle))
        y1 = int(r + r * math.sin(angle))
        for j in range(num_detect):
            x2 = int(r + r * math.cos(angle + np.pi - theta/2 + j * theta/(num_detect-1)))
            y2 = int(r + r * math.sin(angle + np.pi - theta/2 + j * theta/(num_detect-1)))
            line = list(bresenham(x1, y1, x2, y2))
            for element in line:
                reconstruction[element[0], element[1]] += sinogram[i, j]
                matrix[element[0], element[1]] += 1

    for i in matrix:
        for j in i:
            if j == 0:
                j = 1
    reconstruction = reconstruction/matrix

    max = np.max(reconstruction)
    min = np.min(np.nonzero(sinogram))
    #print(min, max)
    """for x in range(len(reconstruction)):
        for y in range(len(reconstruction[0])):
            if reconstruction[x, y] < min:
                value = 0
            else:
                value = (reconstruction[x, y] - min) / (max - min)
            reconstruction[x, y] = value"""
    return reconstruction


@st.cache_data
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


def convert_image_to_ubyte(img):
    return img_as_ubyte(rescale_intensity(img, out_range=(0.0, 1.0)))

def save_as_dicom(file_name, img, patient_data):
    print("Saving to file...")
    img_converted = convert_image_to_ubyte(img)
    
    meta = Dataset()
    meta.MediaStorageSOPClassUID = pydicom._storage_sopclass_uids.CTImageStorage
    meta.MediaStorageSOPInstanceUID = pydicom.uid.generate_uid()
    meta.TransferSyntaxUID = pydicom.uid.ExplicitVRLittleEndian  

    ds = FileDataset(None, {}, preamble=b"\0" * 128)
    ds.file_meta = meta
    ds.is_little_endian = True
    ds.is_implicit_VR = False

    ds.SOPClassUID = pydicom._storage_sopclass_uids.CTImageStorage
    ds.SOPInstanceUID = meta.MediaStorageSOPInstanceUID
    ds.PatientName = patient_data["PatientName"]
    ds.PatientID = patient_data["PatientID"]
    ds.ImageComments = patient_data["ImageComments"]
    ds.Modality = "CT"
    ds.SeriesInstanceUID = pydicom.uid.generate_uid()
    ds.StudyInstanceUID = pydicom.uid.generate_uid()
    ds.FrameOfReferenceUID = pydicom.uid.generate_uid()
    ds.BitsStored = 8
    ds.BitsAllocated = 8
    ds.SamplesPerPixel = 1
    ds.HighBit = 7
    ds.ImagesInAcquisition = 1
    ds.InstanceNumber = 1
    ds.Rows, ds.Columns = img_converted.shape
    ds.ImageType = r"ORIGINAL\PRIMARY\AXIAL"
    ds.PhotometricInterpretation = "MONOCHROME2"
    ds.PixelRepresentation = 0

    pydicom.dataset.validate_file_meta(ds.file_meta, enforce_standard=True)

    ds.PixelData = img_converted.tobytes()
    ds.save_as(file_name, write_like_original=False)

def read_dicom(file):
    ds = pydicom.dcmread(file)
    patient_data = {}
    patient_data["PatientName"] = ds.PatientName
    patient_data["PatientID"] = ds.PatientID
    patient_data["ImageComments"] = ds.ImageComments
    ct = ds.PixelData
    rows = ds.Rows
    cols = ds.Columns
    if "SamplesPerPixel" in ds:
        channels = ds.SamplesPerPixel
    else:
        channels = 1
    image = np.frombuffer(ct, dtype=np.uint8).reshape((rows, cols, channels))
    return image, patient_data


def main():
    st.title('CT scan simulator')
    st.subheader('by Agnieszka Grzymska and Michał Pawlicki')
    st.markdown('---')

    file_side_bar, num_angles, num_detect, theta = side_bar()
    image = None
    st.markdown("### Import from DICOM")
    form = st.form("import", clear_on_submit=True)
    files = os.listdir('./examples')
    file = form.selectbox('Choose a file to read', files)
    if form.form_submit_button("Read"):
        image, patient_data = read_dicom('./examples/'+file)
        st.markdown("Patient: " + str(patient_data["PatientName"]))
        st.markdown("Patient ID number: " + str(patient_data["PatientID"]))
        st.markdown("Comments: " + str(patient_data["ImageComments"]))
    else:
        image = plt.imread('./images/'+file_side_bar, format='gray')
    st.markdown("### Original image")
    st.image(image, width=300)

    st.markdown("### Generated sinogram")
    sinogram = radon_transform(image, num_angles, num_detect, theta)
    rotation = num_angles
    steps = st.checkbox(label="Show steps")
    if steps:
        st.markdown("""$Rotation$ $progress$""")
        rotation = st.select_slider("Select number of steps", options=range(1, num_angles + 1, 1), value=num_angles)
    sinogram_steps = np.zeros((num_angles, num_detect))
    sinogram_steps[:rotation] = sinogram[:rotation]
    st.image(sinogram_steps, width=300)

    st.markdown("### Filtered sinogram")
    filtered_sinogram = convolution_filter(sinogram_steps, 15)
    st.image(filtered_sinogram, width=300)

    st.markdown("### Reconstructed image")
    filtering = st.checkbox(label="Filtering")
    if filtering:
        reconstruction = inverse_radon_transform(filtered_sinogram, theta)
    else:
        reconstruction = inverse_radon_transform(sinogram_steps, theta)
    st.image(reconstruction, width=300)
    st.markdown('---')

    patient_data = {}
    st.markdown("### Save as DICOM")
    form = st.form("save", clear_on_submit=True)
    patient_data["PatientName"] = form.text_input(label="Patient name and surname", value="")
    patient_data["PatientID"] = form.text_input(label="Patient ID number", value="")
    patient_data["ImageComments"] = form.text_input(label="Comments about the image", value="")
    if form.form_submit_button("Save"):
        save_as_dicom("./dicom/"+patient_data["PatientID"], reconstruction, patient_data)
    

def side_bar():
    form = st.sidebar.form("user_input")
    form.markdown('# Set scanner options')
    form.markdown(""" $\Delta$ $\\alpha$ $value$""")
    angles = [1,2,3,4,5,6,8,9,10]
    alpha = form.select_slider("Select value of alpha", options=angles, value=1)
    num_angles = int(360/alpha)
    form.markdown("""$Number$ $of$ $detectors$""")
    num_detect = form.select_slider("Select value of n", options=range(100, 501, 10), value=300)
    form.markdown("""$Span$ $of$ $the$ $emitter$ $system$""")
    span = form.select_slider("Select value of l", options=range(20, 181), value=90)
    theta = math.radians(span)*2
    files = os.listdir('./images')
    form.markdown("""$File$ $to$ $scan$""")
    file = form.selectbox(
        'Choose a file to read',
        files)
    form.form_submit_button("Submit")
    global from_file
    from_file = False
    return file, num_angles, num_detect, theta


if __name__ == '__main__':
    main()
