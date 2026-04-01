from __future__ import annotations

import cv2


def light_removing(frame, blur_kernel: int = 99):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    lightness = lab[:, :, 0]
    median_lightness = cv2.medianBlur(lightness, blur_kernel)
    inverted_lightness = cv2.bitwise_not(median_lightness)
    composed = cv2.addWeighted(gray, 0.75, inverted_lightness, 0.25, 0)
    return lightness, composed
