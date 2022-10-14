import torch
import utils
import streamlit as st
import cv2
import os
from PIL import Image
from matplotlib import image
import matplotlib.pyplot as plt
import subprocess


def save_uploaded_file(uploadedfile):
    if not os.path.exists("tempDir"):
        os.makedirs("tempDir")
    with open(os.path.join("tempDir", uploadedfile.name), "wb") as f:
        f.write(uploadedfile.getbuffer())
    return st.success("Saved file :{} in tempDir".format(uploadedfile.name))


RESULTS_PATH = "temp"


video_file = st.file_uploader(
    "Choose Files", accept_multiple_files=False, type=["mp4", "mov"]
)

if video_file is not None:
    file_details = {"FileName": video_file.name, "FileType": video_file.type}
    st.write(file_details)
    save_uploaded_file(video_file)

    command = " python detect2.py --weights yolov5s.pt --img 640 --conf 0.25 --source tempDir/{} --project {}".format(
        file_details["FileName"], RESULTS_PATH
    )
    subprocess.call(
        command,
        shell=True,
    )

    # get latest inferred video
    experiments = sorted(os.listdir(RESULTS_PATH), reverse=True)
    st.write(experiments)
    latest_folder = experiments[0]

    st.write(os.path.join(RESULTS_PATH, latest_folder, file_details["FileName"]))
    st.video(os.path.join(RESULTS_PATH, "exp", file_details["FileName"]))
