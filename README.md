# 🎨 Virtual Air Draw (Computer Vision Canvas)

An interactive computer vision application built with **Python**, **OpenCV**, and **MediaPipe**. It tracks your index finger in real-time via your webcam, enabling you to paint, switch colors, erase, and save creations in mid-air using hand gestures.

---

## ✨ Features
* **Real-time Hand Tracking:** Powered by MediaPipe's modern `HandLandmarker` model.
* **Smooth Interpolation:** Features linear interpolation to eliminate jittery finger movements for a fluid drawing experience.
* **Dynamic Color Palette:** Switch between Blue, Green, Red, Yellow, and an Eraser via on-screen air-hovering or instant keyboard shortcuts (`1`-`5`).
* **Session Controls:** Clear canvas (`C`), Save snapshot (`S`), and Quit (`Q`).

---

## 🚀 Getting Started

### Prerequisites
* Python 3.8+ installed on your system.
* A working webcam.

### Installation & Setup

1. Clone the repository:
   ```bash
git clone https://github.com/YOUR-USERNAME/air-draw.git
cd air-draw