# Drowsiness driving detection system with OpenCV & personalized baseline modeling
***
: In this repository, a program was developed to **identify the driver's drowsiness based on real-time camera image and image processing techniques**, and this program makes warning alarms go off for each level of drowsiness when it detects drowsiness driving.

  
## Description
: Based on the real-time Vision System, drivers' face and eye detection techniques were added, as well as **removing lighting effects** due to the eye detection false positives, drowsiness detection techniques, and **supervised learning algorithms** to identify drowsiness level.
  
The Histogram of Oriented Gradients technology and the learned Face Landmark estimation techniques were used to detect faces and eyes.
 
In order to eliminate the effects of lighting, **the light channels of the original images were separated and reversed, and then composed with the grayscale images of original images**. 
 
Furthermore the concept of **Eye Aspect Ratio was used** to detect drivers' drowsiness. 

Finally, the **KNN algorithm was used** to divide the drivers' level of drowsiness into three stages, and differential alarms go off for each stages.

Through these works, we could research and make technology of intelligent vehicle systems and vision computing, which is gaining much attention recently.
  
    
***Current app path is tested as a modern Python project and now uses MediaPipe Face Mesh for facial landmarks.***

## System diagram
    
Get face images from the camera -> Grayscaling -> Light processing -> HOG & find face -> Face Landmark Estimation -> Detect drowsiness driving. 
   
+ In detail

<img width="771" height="511" alt="Screenshot 2026-02-15 at 7 34 41 PM" src="https://github.com/user-attachments/assets/bfe5933d-8c46-4ffe-b480-3d081a812c07" />

0 : The filming.
  
10 : Lightness preprocessing.
  
100 : Detecting drowsiness.
  
110 : Getting face's image.
  
120 : Finding eyes region.
  
130 : Determining the value of the EAR normally.
  
140 : Determining drowsiness driving.
  
141 : Calculating the value of the EAR.
   
142 : Calculating the amount of time eyes are closed.
    
143 : Calculating the amount of time eyes are opened.
    
144 : Determining the level of the drowsiness.



## Extracting face and eye region
+ Using the **HOG face pattern**, to find the face from the Grayscaled-HOG-input-image. 

<img src="https://user-images.githubusercontent.com/36785390/52613168-3b088480-2ed0-11e9-8651-97afc34f4bae.png" width="60%">
   
+ Use the **Face Landmark Estimation algorithm** to locate the landmarks on the face.
  
<img src="https://user-images.githubusercontent.com/36785390/52613175-3d6ade80-2ed0-11e9-9290-ee5dc2f2d525.png" width="30%">
<img src="https://user-images.githubusercontent.com/36785390/52613176-3f34a200-2ed0-11e9-8f3f-94998fd2ab63.png" width="30%">
  


## Preprocessing
 
+ **Invert the lightness channel** detached from the original image and **composed it with the original grayscale image** to produce a clear image.
  
<img width="653" height="476" alt="Screenshot 2026-02-15 at 3 02 22 PM" src="https://github.com/user-attachments/assets/b8d7a3f5-1f08-4b76-8807-395f6c502cd2" /> 
  
+ Converting color to grayscale using **Luma Coding**


<img src="https://user-images.githubusercontent.com/36785390/52613343-dc8fd600-2ed0-11e9-93f6-e154e20df31d.png" width="35%">
  
<img src="https://user-images.githubusercontent.com/36785390/52613308-bc601700-2ed0-11e9-999e-40a2782932c9.png" width="40%">
  
+ There are many different models in Color Space, the **LAB color space model** is the best way to separate Lightness. [Median filtering](https://en.wikipedia.org/wiki/Median_filter) is applied to convert the value of lightness(L) obtained by using the LAB color space to match the actual lighting conditions because it differs from the actual lighting conditions.
+ The pictures below are the original image, image that separates L channel, image with Median filter applied, and inverted images from left to right. Drowsiness detection method

    
<img src="https://user-images.githubusercontent.com/36785390/52613441-35f80500-2ed1-11e9-9c6c-819b9e92b150.png" width="70%">
   
+ Results of preprocessing
   
<img src="https://user-images.githubusercontent.com/36785390/52613443-385a5f00-2ed1-11e9-94e3-e325b3436041.png" width="20%">
    
     
## Drowsiness detection method
+ Each eye is represented by 6 (x, y)-coordinates
  
<img src="https://user-images.githubusercontent.com/36785390/52702447-83eb3680-2fbf-11e9-985f-f96ec72f5b26.png" width="20%">
   
+ The EAR equation
   
<img src="https://user-images.githubusercontent.com/36785390/52702578-cb71c280-2fbf-11e9-9a06-d4434250d622.png" width ="30%">

+ Calculated EAR
<img src="https://user-images.githubusercontent.com/36785390/52702645-ee9c7200-2fbf-11e9-9757-975fa22da6e1.png" width="60%">

+ The calculated EAR will have a value more than zero when the eyes are open, and a value close to zero when the eyes are closed.
+ This program has **set a 50% value from the average EAR value to the threshold value**. So, 1) measures the average EAR value when the eyes are open, 2) measures the average EAR value when the driver is closing his eyes, and 3) sets the threshold using the above two results.


<img width="762" height="509" alt="Screenshot 2026-02-15 at 7 41 00 PM" src="https://github.com/user-attachments/assets/5df634cd-6cba-45b4-ab9f-b5b944d5be8a" />

  
## Drowsiness level selection
+ Conditions :
  1. 30 FPS
  2. Prescribed speed : 100km/h, Retention distance between vehicles >= 100m
  3. The time which takes a person to push the brakes 0.45 (response time) + 0.2 (brake pushing time) + 0.05 (time to start braking) = 0.7 seconds
  4. The braking distance of a vehicle running at 100 km/h is 56 meters (the driver has 44 meters of free distance)

 <img width="767" height="494" alt="Screenshot 2026-02-15 at 7 44 25 PM" src="https://github.com/user-attachments/assets/2a9d2ae5-264f-405b-ae8d-c29abfc1d3cf" />
  
+ Under the above conditions, the drivers has almost 0.9 seconds of free time (100km/h -> 27m/s == 1.63s of free time. 1.63 - 0.7 = 0.9 s).
+ 30 FPS -> 27 frame = 0.9s.
  + **if EAR < threshold for 27 frame? then going alarm off.**
+ Now I separated the drowsiness phase into three steps.

<img src="https://user-images.githubusercontent.com/36785390/52762348-8058bd80-305a-11e9-9256-905e8de77740.png" width="45%">
  
+ Drowsiness levels are identified by the following conditions.
  1. The first alarm will sound(approximately 0.9 seconds) between level 1 and 2 of the drowsy phase.
  2. If you are dozing (sleeping and waking again and again) in less than 15 seconds, the drowsiness phase starts at level 1 and then the next alarm goes up to 0.
  3. The first alarm is level 2 and the second alarm is level 1 and the third alarm makes level 0 sound when driving drowsy between 15 and 30 seconds.
  4. If you have not been drowsy for more than 30 seconds, set level 2.

   
+ To distinguish drowsiness level, I used K-Nearest Neighbor(KNN) supervised learning algorithm.

. 1. Create arrays with random (x, y)-coordinates.
  
<img src="https://user-images.githubusercontent.com/36785390/52762829-82bc1700-305c-11e9-97cb-b41e35dfb9e6.png" width="30%">
  
  2. Labeling
<img src="https://user-images.githubusercontent.com/36785390/52762830-8485da80-305c-11e9-96db-f24a7a1ebdd6.png" width="40%">
  
  3. Define K value.
<img src="https://user-images.githubusercontent.com/36785390/52762904-e6dedb00-305c-11e9-952c-f201390eb9bd.png" width="50%">
  
  4. Test KNN algorithm.
<img src="https://user-images.githubusercontent.com/36785390/52762907-e8a89e80-305c-11e9-8928-9409bd4eaa7a.png" width="50%">
  
  
## Synthesis
<img src="https://user-images.githubusercontent.com/36785390/52762972-36bda200-305d-11e9-99a6-314dfae8f3c7.png" width="50%">

## Test
+ Before applying preprocessing

[![BeforePreprocessing](https://img.youtube.com/vi/8yLHAP6gmOA/0.jpg)](https://www.youtube.com/watch?v=8yLHAP6gmOA)
+ After applying preprocessing

[![AfterPreprocessing](https://img.youtube.com/vi/7iCVzF3LI6o/0.jpg)](https://www.youtube.com/watch?v=7iCVzF3LI6o)

  
## Execution
+ Install the dependencies:

```bash
pip install -r requirements.txt
```

+ Run the detector from the refactored Python entry point:

```bash
python drowsiness_detector.py --driver-id nishtha
```

+ Optional flags:

```bash
python drowsiness_detector.py --driver-id nishtha --camera-source 0 --frame-width 400 --baseline-seconds 60
```

+ The original notebook is still included as an experiment log, but the main application code now lives in the `driver_drowsiness/` package so it is easier to extend.
+ The detector now uses MediaPipe Face Mesh instead of dlib, which makes setup much easier on modern Macs.
+ The detector supports personalized driver baselines, saves per-driver profiles to `profiles/<driver-id>.json`, and writes JSONL runtime events to `logs/events.jsonl` by default.
+ The live dashboard now combines EAR, MAR, full 3D head pose estimation, fatigue scoring, break recommendation, and session analytics.

## Deployment for Demo / Exam

+ For a local macOS deployment, the repo now includes a one-click launcher and helper scripts.

+ First-time setup:

```bash
./scripts/setup_macos.sh
```

+ Run the detector through the packaged launcher script:

```bash
./scripts/run_detector.sh --driver-id exam_demo --baseline-seconds 20 --recalibrate
```

+ One-click macOS launcher:

```bash
open ./launch_demo.command
```

+ Optional: build a standalone macOS app bundle with PyInstaller:

```bash
./scripts/build_macos_app.sh
```

+ After build, the app bundle will be created at:

```text
dist/Driver Drowsiness Detector.app
```

+ This gives the project a clear deployment path for demonstrations: local environment setup, repeatable run script, and optional desktop app packaging.

## Hosted Deployment

+ For a true non-local deployment, this project now includes a hosted web demo entrypoint at [app.py](/Users/nishthapandey/Desktop/Driver_Drowsiness_Detection/app.py).
+ The hosted version is designed for browser webcam access and is suitable for platforms like Hugging Face Spaces using Gradio.
+ Gradio supports webcam image inputs and streaming from the browser, and Hugging Face Spaces can host Gradio apps directly:
  [Gradio webcam image docs](https://www.gradio.app/main/docs/gradio/image)
  [Gradio streaming inputs guide](https://www.gradio.app/main/guides/streaming-inputs)
  [Hugging Face Gradio Spaces docs](https://huggingface.co/docs/hub/en/spaces-sdks-gradio)

+ To run the hosted entrypoint locally first:

```bash
python app.py
```

+ To deploy on Hugging Face Spaces:

1. Create a new Space and choose `Gradio` as the SDK.
2. Push this repository or upload the project files.
3. Ensure `requirements.txt` is present.
4. Set the app file to `app.py` if prompted.
5. After the Space builds, share the generated public URL.

+ Hugging Face states that each new commit to a Space automatically rebuilds and restarts the app:
  [Spaces overview](https://huggingface.co/docs/hub/spaces-overview)

## Project structure

```text
driver_drowsiness/
  app.py           # Runtime loop and calibration flow
  audio.py         # Alarm playback
  analytics.py     # Session-level summaries and metrics
  baseline.py      # Driver baseline persistence and rolling behavior
  config.py        # Tunable runtime settings and asset paths
  landmarks.py     # MediaPipe eye landmark extraction
  logging_utils.py # JSONL event logging
  metrics.py       # EAR and MAR calculations
  pose.py          # Full 3D head pose estimation using solvePnP
  preprocess.py    # Lighting compensation helpers
  severity.py      # Personalized multi-signal fatigue scoring
  training.py      # Legacy synthetic-KNN experiment
tests/
  test_baseline.py # Personalized baseline tests
  test_metrics.py  # Smoke test for EAR math
```

## Recent improvements

+ The detector can build and reuse a per-driver baseline profile instead of assuming one universal blink pattern.
+ Alarm severity is scored by deviation from that personal baseline, which is a better fit for real-world variation between drivers.
+ MAR-based yawn tracking, full 3D head pose estimation, and a fused fatigue score are now part of the live monitoring path.
+ The app shows a compact dashboard with EAR, MAR, head pose, fatigue score, alert counts, and break recommendations.
+ MediaPipe Face Mesh replaces dlib, which removes the hardest macOS installation blocker in this repo.
+ Alarm and calibration events are logged to JSONL so detector behavior can be reviewed after a run.
  
## References
+ [Machine Learning is Fun! Part 4: Modern Face Recognition with Deep Learning](https://medium.com/@ageitgey/machine-learning-is-fun-part-4-modern-face-recognition-with-deep-learning-c3cffc121d78)
+ [Real-Time Eye Blink Detection using Facial Landmarks](https://vision.fe.uni-lj.si/cvww2016/proceedings/papers/05.pdf)
+ [Eye blink detection with OpenCV, Python, and dlib](https://www.pyimagesearch.com/2017/04/24/eye-blink-detection-opencv-python-dlib/)
+ [dlib install tutorial that I refer to](https://www.pyimagesearch.com/2017/03/27/how-to-install-dlib/)
+ [Histograms of Oriented Gradients for Human Detection](https://lear.inrialpes.fr/people/triggs/pubs/Dalal-cvpr05.pdf)
