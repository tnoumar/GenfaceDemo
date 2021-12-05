import numpy as np
import os
import pickle
import streamlit as st
import sys
import tensorflow as tf
import urllib





def main():
    st.title("GenFace demo")
    
    # Download all data files if they aren't already in the working directory.


    st.sidebar.title("Features")
    # If the user doesn't want to select which features to control, these will be used.
    default_control_features = ["Age", "Eyeglasses", "Gender", "Pose", "Smile"]

    if st.sidebar.checkbox("Show advanced options"):
        # Randomly initialize feature values.
        features = [1,1,1]

        # Some features are badly calibrated and biased. Removing them
        block_list = ["Attractive", "Big_Lips", "Big_Nose", "Pale_Skin"]
        sanitized_features = [
            feature for feature in features if feature not in block_list
        ]

        # Let the user pick which features to control with sliders.
        control_features = st.sidebar.multiselect(
            "Control which features?",
            sorted(sanitized_features)
        )
    else:
        features =[1,1,1]
        # Don't let the user pick feature values to control.
        control_features = default_control_features

    # # Insert user-controlled values from sliders into the feature vector.
    # for feature in features:
    #     features[feature] = st.sidebar.slider(feature, 0, 100, 50, 5)

    st.sidebar.title("Note")
    st.sidebar.write(
        """Playing with the sliders, you _will_ find **biases** that exist in this
        model.
        """
    )
    st.sidebar.write(
        """For example, moving the `Smiling` slider can turn a face from masculine to
        feminine or from lighter skin to darker. 
        """
    )
    st.sidebar.write(
        """Apps like these that allow you to visually inspect model inputs help you
        find these biases so you can address them in your model _before_ it's put into
        production.
        """
    )
    st.sidebar.caption(f"Streamlit version `{st.__version__}`")

    # Generate a new image from this feature vector (or retrieve it from the cache).
    # with session.as_default():
    #     image_out = generate_image(
    #         session, pg_gan_model, tl_gan_model, features, feature_names
    #     )

    # st.image(image_out, use_column_width=True)



def generate_image(styleGAN, features, feature_names):
    """
    Converts a feature vector into an image.
    """
    # return image

USE_GPU = False
if __name__ == "__main__":
    main()
