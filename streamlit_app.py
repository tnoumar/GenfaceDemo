import numpy as np
import os
import pickle
import streamlit as st
import sys
import tensorflow as tf
import urllib
import subprocess
from PIL import Image
import os
import glob
import time

# global variables
favicons_dir = "/content/GenfaceDemo/Favicon/"
age_dict = {"Child": -2.5, "Young": 0, "Old": 2.5}
gender_dict = {"Female": -2.5, "Neutral": 0, "Male": 2.5}
pose_dict = {"Left-sided": -2.5, "Symmetrical": 0, "Right-sided": 2.5}
smile_dict = {"Sad": -2.5, "Neutral": 0, "Happy": 2.5}


def clear_img_dir():
    files = glob.glob("/content/downloaded_imgs/*.jpg")
    for f in files:
        os.remove(f)


def generate_image(age, eyeglasses, gender, pose, smile):
    var = subprocess.check_output(
        [
            "python",
            "ganface_gen.py",
            str(age),
            str(eyeglasses),
            str(gender),
            str(pose),
            str(smile),
        ]
    )
    var_name = var.splitlines()[-1]
    file_name = var_name.decode("utf-8")
    with open(file_name, "rb") as file:
        img = Image.open(file_name)
    return img


def progress_bar():
    my_bar = st.progress(0)
    for percent_complete in range(100):
        time.sleep(0.1)
        my_bar.progress(percent_complete + 1)


def main():
    st.set_page_config("GenFace", favicons_dir + "favicon.ico")
    st.title("GenFace's StyleGAN generator")

    st.sidebar.title("Features")
    st.sidebar.title("Facial attributes")

    # sliders
    age = st.sidebar.select_slider("Age", options=["Child", "Young", "Old"])
    age = age_dict[age]
    st.sidebar.write("Age:", age)

    eyeglasses = st.sidebar.select_slider("Eyeglasses", options=["Without", "With"])
    eyeglasses = 3.0 if eyeglasses == "With" else -3.0
    st.sidebar.write("Eyeglasses:", eyeglasses)

    gender = st.sidebar.select_slider("Gender", options=["Female", "Neutral", "Male"])
    gender = gender_dict[gender]
    st.sidebar.write("Gender:", gender)
    pose = st.sidebar.select_slider(
        "Pose", options=["Left-sided", "Symmetrical", "Right-sided"]
    )
    pose = pose_dict[pose]
    st.sidebar.write("Pose:", pose)

    smile = st.sidebar.select_slider("Smile", options=["Sad", "Neutral", "Happy"])
    smile = smile_dict[smile]
    st.sidebar.write("Smile:", smile)

    if st.sidebar.button("Generate"):
        clear_img_dir()
        image_out = generate_image(age, eyeglasses, gender, pose, smile)
        st.image(image_out, use_column_width=True)

    st.sidebar.write(
        """Playing with the sliders, you _will_ generate new **faces** that never existed before.
        """
    )
    st.sidebar.write(
        """For example, moving the `Smiling` slider can turn a face from grinching to carrying a smile. 
        """
    )
    st.sidebar.caption(f"Streamlit version `{st.__version__}`")

    # Generate a new image from this feature vector (or retrieve it from the cache).


if __name__ == "__main__":
    main()
