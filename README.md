Video Face Blur Pro

A professional desktop app for automatically detecting and blurring human faces in videos . Applications: privacy protection, journalism, social media, and surveillance footage.

Built with Python, OpenCV, and CustomTkinter.


WHAT IT DOES

Video files are processed by Video Face Blur Pro frame by frame, all visible human faces are detected utilizing a deep neural network, and Gaussian blur is applied for making them unrecognizable. To reduce flickering, a stabilizer is used and for finding faces at different distances  multi-scale detection is applied.


HOW IT WORKS (THE SCIENCE)

A pre-trained Caffe SSD deep learning model (a convolutional neural network designed for real-time object detection) is included in developing this tool. The model was trained on thousands of face images and runs locally on your machine with no need to internet connection.

The blurring  takes the advantage of a Gaussian kernel. Unlike simple pixelation, Gaussian blur yields smooth, natural-looking results with no sharp artifacts. The knowledge of physics of light and optics has had the main role for developing these models.

Key technologies: Deep Learning, Machine Vision, Convolutional Neural Networks, Optical Image Processing.


COMPETITIVE ADVANTAGE

Most face blur tools require uploading your video to a remote server, which jeopardizes your privacy. They also produce flickering and unstable results.

Video Face Blur Pro:

- Works completely offline — your videos won’t go out of your personal storage
- Uses stabilizer technology to avoid flickering between frames
- Applies multi-scale detection for finding faces in both close-ups and long shots
- Preserves original audio (ffmpeg should be installed for this purpose)
- Provides a clean and user-friendly desktop GUI — no command line needed

Built by a PhD physicist specialized in optics and laser physics. The science behind the blur is as important as the code.


DEPENDENCIES

To run this project, the following steps are needed:

1. Python libraries (install once):
pip install opencv-python customtkinter pillow numpy

2. AI model files (one-time download, place in ~/models/):
- deploy.prototxt (28 KB)
- res10_300x300_ssd_iter_140000.caffemodel (10.7 MB)

Download both from the OpenCV GitHub repository:
https://github.com/opencv/opencv/tree/master/samples/dnn/face_detector

3. ffmpeg (optional if you want to keep the original audio) :
sudo apt install ffmpeg

The program works perfectly without ffmpeg. If ffmpeg is not installed, the output video is saved without audio.


HOW TO RUN

python video_face_blur.py

1. Open a video file
2. Adjust blur strength
3. Adjust the Detection Confidence slider to control the sensitivity level of face detection of the software.
4. Click Start Processing
5. Choose the address to save the output


AUTHOR

Yuseph Alvandi
PhD in Optics and Laser Physics
Python Developer and Image Processing Specialist

GitHub: https://github.com/YusephAlvandi


LICENSE

MIT License