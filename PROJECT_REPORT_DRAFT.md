# Project Report Draft

Use this content to fill the template at:
`/Users/nishthapandey/Downloads/PBL-2 OR PBL-4_End Term Report .docx`

Replace the bracketed placeholders with your personal/academic details.

## Cover Page

**A Report**  
**on**  
**Personalized Driver Drowsiness Detection Using Computer Vision and Adaptive Baseline Modeling**

carried out as part of the course **Project Based Learning-2/4**

**Submitted by**  
[Your Name]  
[Your Roll Number / Registration Number]  
IV/VI Semester

in partial fulfilment for the award of the degree  
of  
**BACHELOR OF TECHNOLOGY**  
in  
**Computer Science & Engineering**

Department of Computer Science & Engineering,  
School of Computer Science and Engineering,  
Manipal University Jaipur,  
Jan-May 2026

## Guide Details

**Guide Name:** [Guide Name]  
**Guide Signature (with date):** [To be signed]

## Acknowledgement

This project would not have been completed without the guidance, support, and encouragement of several people who contributed directly and indirectly to its successful completion. I express my sincere gratitude to my internal supervisor, [Guide Name], for continuous guidance, valuable suggestions, and technical feedback throughout the development of this project titled **“Personalized Driver Drowsiness Detection Using Computer Vision and Adaptive Baseline Modeling.”**

I also extend my gratitude to the Head of the Department and all faculty members of the Department of Computer Science and Engineering, Manipal University Jaipur, for providing the academic environment and support necessary to carry out this work. Their encouragement and resources greatly contributed to the quality of this project.

Finally, I would like to thank my classmates, peers, and family members for their motivation, support, and constructive discussions during the course of this work.

**[Your Name]**  
**[Registration Number]**  
Department of Computer Science and Engineering  
School of Computer Science and Engineering  
Date: _____________

## Certificate

This is to certify that the project entitled **“Personalized Driver Drowsiness Detection Using Computer Vision and Adaptive Baseline Modeling”** is a bonafide work carried out as **PBL-2 / PBL-4 End Term Assessment** in partial fulfillment for the award of the degree of **Bachelor of Technology in Computer Science and Engineering**, by **[Your Name]** bearing registration number **[Your Registration Number]**, during the academic semester **IV/VI** of year **2025-2026**.

**Place:** Manipal University Jaipur, Jaipur  
**Name of the project guide:** ________________________  
**Signature of the project guide:** ______________________

## Abstract

Driver drowsiness is one of the major reasons behind road accidents, especially during long-duration driving and monotonous travel conditions. Traditional low-cost drowsiness detection systems generally rely on fixed blink thresholds or common eye closure rules, which are often unreliable because blinking behavior differs significantly across individuals. This project proposes a **personalized driver drowsiness detection system** that learns a driver-specific baseline during an initial calibration phase and then detects fatigue as a deviation from that baseline rather than using a universal threshold.

The proposed system uses **computer vision, facial landmark tracking, Eye Aspect Ratio (EAR) analysis, adaptive baseline modeling, and real-time alert generation**. A webcam captures the driver’s face, MediaPipe extracts eye landmarks, and the system computes blink-related features in real time. These features are compared against a personalized baseline profile to identify abnormal eye behavior that may indicate drowsiness. The system also supports persistent driver profiles, event logging, and multiple alert levels.

The project demonstrates that **personalized modeling is a more justifiable and human-centered alternative** to one-size-fits-all fatigue detection systems. The resulting framework is practical, explainable, low-cost, and suitable as a strong foundation for future research in intelligent driver monitoring systems.

## Table of Contents

Fill this automatically in Word after inserting all section headings.

## List of Figures

Suggested figures to include:

1. Overall system architecture  
2. Eye Aspect Ratio landmark representation  
3. Baseline calibration workflow  
4. Personalized deviation scoring pipeline  
5. Real-time output screenshot of the system  

## List of Tables

Suggested tables to include:

1. Literature review comparison table  
2. Software and hardware requirements  
3. Feature set used in the project  
4. Limitations and future enhancements  

## 1. Introduction

Road safety remains one of the most important challenges in intelligent transportation systems. Driver fatigue and drowsiness reduce reaction speed, attention span, and decision-making capability, which can lead to severe accidents. In recent years, computer vision-based drowsiness detection has emerged as a practical and low-cost approach because it can monitor visual fatigue indicators such as blinking, eye closure duration, and yawning without requiring invasive sensors.

However, a major limitation of many existing approaches is the use of a **common threshold** to classify eye closure or blink behavior for all users. Human eye behavior is not universal; normal blink frequency, closure duration, and resting eye openness vary from person to person. As a result, fixed-threshold systems often produce inaccurate predictions in real-world conditions.

This project addresses that limitation by proposing a **personalized baseline-based drowsiness detection framework**. Instead of assuming the same blinking pattern for every driver, the system first learns the driver’s normal eye behavior and then identifies drowsiness as a deviation from that individual baseline. This improves both the explainability and the practical reliability of the system.

## 2. Motivation

The motivation behind this project comes from a simple but important observation: **there is no universal rule of blinking in nature**. Two different drivers may have completely different normal blink rates, eye openness, and eye closure durations. Therefore, a drowsiness detector based on fixed population thresholds may not be scientifically justified for all users.

This project was motivated by the need to design a more realistic and adaptive monitoring system. By introducing calibration and personalized baseline modeling, the system aims to better reflect real human variability and provide a stronger foundation for future research and real-world driver assistance applications.

## 3. Literature Review

Traditional driver drowsiness detection systems generally follow one of three approaches:

1. **Physiological signal-based approaches**  
   These use EEG, ECG, or heart-rate sensors to monitor fatigue. They can be accurate but are intrusive, expensive, and inconvenient for everyday driving.

2. **Vehicle behavior-based approaches**  
   These analyze steering patterns, lane deviation, or braking behavior. Although useful, they detect fatigue indirectly and are often influenced by road conditions.

3. **Vision-based approaches**  
   These use facial cues such as blink rate, eye closure duration, PERCLOS, yawning, and head pose. They are cost-effective and non-invasive, making them attractive for practical systems.

Most existing vision-based systems use **Eye Aspect Ratio (EAR)** and fixed thresholds to estimate eye closure. While computationally efficient, these systems assume that a single threshold can describe all drivers. More recent studies suggest that personalized calibration or adaptive thresholds are more appropriate because fatigue indicators differ across individuals.

### Literature Review Table

| Study/Approach | Technique Used | Strength | Limitation |
|---|---|---|---|
| Fixed-threshold EAR systems | Eye Aspect Ratio | Simple and real-time | Not personalized |
| PERCLOS-based systems | Eye closure percentage | Strong fatigue cue | Still often threshold-dependent |
| Sensor-based systems | EEG/ECG | High accuracy | Intrusive and expensive |
| Adaptive user-specific systems | Personalized thresholds | Better for individual variation | Requires calibration |
| Proposed work | Personalized baseline + EAR deviation | Explainable, adaptive, practical | Needs larger experimental validation |

## 4. Outcome of Literature Review

The literature review indicates that computer vision is the most suitable domain for practical, low-cost drowsiness monitoring. It also shows that fixed-threshold systems remain common despite the fact that human blink behavior is naturally variable. This establishes a clear research opportunity: building a personalized detection framework that learns a driver’s normal state first and then performs drowsiness detection based on deviation from that learned profile.

## 5. Problem Statement

Most real-time driver drowsiness detection systems rely on common thresholds for blink rate and eye closure, which are not universally valid for all users. This often leads to poor reliability, false alarms, and weak generalization in practical scenarios. Therefore, there is a need to design a real-time, low-cost, non-invasive system that can learn driver-specific normal behavior and detect drowsiness using personalized deviation analysis rather than universal rules.

## 6. Research Objectives

The objectives of this project are:

1. To develop a real-time driver drowsiness detection system using computer vision.
2. To extract eye landmarks and estimate eye closure behavior using Eye Aspect Ratio.
3. To build a driver-specific baseline model through an initial calibration phase.
4. To detect drowsiness by comparing live eye behavior against the personalized baseline.
5. To generate real-time alerts and maintain event logs for later analysis.
6. To create a practical prototype that can be extended into a research-oriented intelligent driver monitoring framework.

## 7. Methodology and Framework

The methodology of the project consists of four major phases:

1. **Frame Acquisition**  
   A webcam captures the live face of the driver in real time.

2. **Facial Landmark Detection**  
   MediaPipe Face Landmarker is used to identify facial landmarks and isolate the eye regions.

3. **Feature Extraction and Baseline Modeling**  
   The system computes Eye Aspect Ratio values and uses them to learn the driver’s normal eye behavior during the calibration phase. The baseline stores mean and standard deviation for open-eye EAR, blink duration, and open-eye intervals.

4. **Deviation-Based Drowsiness Detection**  
   Once the baseline is learned, live eye behavior is compared against it. If significant deviation is detected for sufficient duration, the system classifies the event into different alert levels and triggers an alarm.

This framework improves the reliability of fatigue detection because it adapts to each individual rather than depending on a generalized threshold.

## 8. System Architecture

**Suggested architecture diagram for the report:**

Webcam Input -> Face Landmark Detection -> Eye Landmark Extraction -> EAR Computation -> Personalized Baseline Model -> Rolling Behavior Analysis -> Deviation Scoring -> Alert Generation -> Event Logging

### Architectural Description

The system is composed of modular components:

- **Input Layer:** Captures frames from the webcam.
- **Vision Layer:** Detects face and eye landmarks using MediaPipe.
- **Feature Layer:** Computes EAR and blink-related timing features.
- **Baseline Layer:** Stores the driver’s normal eye behavior profile.
- **Decision Layer:** Compares live behavior against the baseline and computes risk.
- **Output Layer:** Triggers alarms and writes event logs.

This modular structure improves maintainability, interpretability, and future extensibility of the project.

## 9. Algorithms and Techniques Used

### 9.1 MediaPipe Face Landmarker

MediaPipe is used to detect dense facial landmarks efficiently in real time. From the complete facial mesh, only a selected subset of eye landmarks is used for EAR computation. MediaPipe was preferred over dlib because it provides a more modern and Mac-compatible implementation for practical development.

### 9.2 Eye Aspect Ratio (EAR)

EAR is a geometric measure used to estimate whether the eye is open or closed. It is calculated using distances between vertical and horizontal eye landmarks. A lower EAR corresponds to a more closed eye state.

### 9.3 Personalized Baseline Modeling

During the calibration phase, the system records:

- mean open-eye EAR
- standard deviation of open-eye EAR
- blink duration statistics
- open-eye interval statistics

These features define the normal behavior of a specific driver and are saved as a reusable profile.

### 9.4 Deviation-Based Severity Scoring

The system computes a weighted score from:

- EAR drop relative to baseline
- abnormal blink duration
- abnormal open-eye intervals

This deviation score is then converted into alert severity levels.

## 10. Detailed Design Methodologies

The design methodology follows a **modular software architecture**:

- `app.py` handles runtime control and webcam processing.
- `landmarks.py` handles eye landmark extraction.
- `metrics.py` computes EAR.
- `baseline.py` stores baseline and rolling behavior logic.
- `severity.py` converts deviations into alert levels.
- `audio.py` handles alarm playback.
- `logging_utils.py` records runtime events.

This decomposition allows the system to be explained, tested, and enhanced systematically.

## 11. Work Done

The following work has been completed in this project:

1. Studied the limitations of traditional drowsiness detection systems.
2. Designed a personalized baseline-driven detection concept.
3. Implemented real-time face and eye landmark extraction using MediaPipe.
4. Implemented EAR-based live eye monitoring.
5. Added driver-specific calibration and persistent profile storage.
6. Developed deviation-based severity scoring logic.
7. Integrated real-time alarm generation.
8. Added structured event logging for experiment traceability.
9. Migrated the system to a Mac-friendly stack for practical testing.

## 12. Results and Discussion

The project successfully demonstrates a real-time prototype for personalized driver drowsiness detection. The system is capable of capturing live video, extracting eye landmarks, computing EAR values, building driver-specific baselines, and generating alerts when abnormal eye behavior is detected. Event logging and profile persistence further improve reproducibility and analysis.

The most important outcome is conceptual: the system shifts the problem from **fixed-threshold fatigue monitoring** to **personalized deviation analysis**. This makes the framework more realistic and more aligned with actual human variability.

Although the current system works as a strong prototype, it is still in the early research stage. Formal quantitative evaluation with multiple participants, controlled conditions, and comparative accuracy metrics is required before strong scientific claims can be made. Nevertheless, the prototype validates the feasibility of the personalized approach and provides a meaningful base for further research.

## 13. Individual Contribution of Project Members

If this is an individual project:

This project was carried out individually by **[Your Name]**. The work included problem identification, literature review, design of the personalized baseline concept, implementation of the real-time system, migration to MediaPipe, testing, documentation, and preparation of the final report.

If this is a group project, replace this section with member-wise contributions.

## 14. Conclusion and Future Plan

This project presents a real-time driver drowsiness detection framework based on personalized baseline modeling using computer vision. Unlike conventional systems that rely on universal thresholds, the proposed approach learns the normal eye behavior of a specific driver and detects drowsiness through deviation analysis. This improves the conceptual correctness and practical adaptability of the system.

The project demonstrates that personalized monitoring can provide a stronger and more research-worthy foundation for fatigue detection systems. It also establishes a modular architecture that supports future enhancements and experimentation.

### Future Plan

The future scope of the project includes:

1. Integration of **MAR (Mouth Aspect Ratio)** for yawn detection.
2. Inclusion of head pose and gaze estimation.
3. Multi-user data collection and benchmarking.
4. Comparative study with fixed-threshold systems.
5. Statistical evaluation using precision, recall, F1-score, and false alarm rate.
6. Conversion into a publishable research study and deployable driver assistance prototype.

## 15. Outcome

This project has resulted in a working real-time prototype for personalized driver drowsiness detection. The outcome is significant because it demonstrates a shift from rigid threshold-based monitoring toward adaptive, driver-specific modeling. The work is currently suitable as a strong academic project prototype and can be extended into a research paper after structured experimentation, user studies, and quantitative evaluation.

## 16. References

Use numbered references in the final report. Suggested references:

[1] T. Soukupová and J. Čech, “Real-Time Eye Blink Detection using Facial Landmarks,” 21st Computer Vision Winter Workshop, 2016.  
[2] A. Dasgupta et al., literature on driver drowsiness detection using computer vision and behavioral analysis.  
[3] MediaPipe Documentation, Google AI Edge, Face Landmarker.  
[4] Research articles on EAR-based blink analysis and PERCLOS-based driver monitoring.  
[5] Studies on personalized thresholding and adaptive fatigue detection in driver monitoring systems.

## 17. Appendix

Suggested appendix content:

- screenshots of the running system
- sample baseline profile JSON
- sample event log output
- selected code snippets for EAR calculation and baseline scoring
- comparison table of old fixed-threshold vs proposed personalized method

