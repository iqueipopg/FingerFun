![Mi Banner](Banner_FF2.png)

# Finger Fun 🤖👆

**Finger Fun** is an interactive and fun visual game where players must draw patterns using their fingers or objects in front of a camera. Inspired by the classic game "Simon Says", players are challenged to replicate a sequence of visual patterns (lines, circles, triangles) in the correct order. As the game progresses, the difficulty increases, adding more patterns or speeding up the game. If the player makes a mistake, the sequence resets, and they have to start over.

This project was developed as part of a Computer Vision course (**Visión por Ordenador I**) at the **Universidad Pontificia Comillas, ICAI**, for the **Engineering Mathematics** program.

## 📜 Table of Contents
- [📌 Project Overview](#-project-overview)
- [🛠️ Installation](#️-installation)
- [🎮 How to Play](#-how-to-play)
- [📂 Project Structure](#-project-structure)
- [🖥️ Technologies Used](#-technologies-used)
- [🙌 Credits](#-credits)

## 📌 Project Overview

In **Finger Fun**, the player is tasked with replicating a sequence of geometric patterns that are shown by the system. The game operates as follows:

1. **Calibration**: The system calibrates the camera using a checkerboard pattern to ensure the accurate recognition of patterns.
2. **Pattern Recognition**: A sequence of visual patterns is shown on the screen, which the player must draw in the correct order.
3. **Sequence Validation**: If the player draws the patterns correctly, the game moves to the next level, adding more patterns or speeding up. If the player fails, the sequence resets and they must try again.
4. **Tracking**: The system tracks the movement of the player’s finger or object in real-time using computer vision, ensuring that each drawn pattern is recognized and validated accurately.

Additionally, **Finger Fun** uses computer vision techniques to track the player's finger or object movements, ensuring the accuracy of the drawn patterns.

## 🎮 How to Play

### 🔓 Unlocking the Game
- The player must verify their identity by entering a password.
- Press the `B` key (**Block**) to start the unlocking process.
- Enter the correct recognition pattern to proceed.

### ▶️ Starting the Game
- Once unlocked, press the `A` key to begin the game.
- The game consists of **6 rounds** and the player has **5 lives**.

### 🎨 Gameplay Mechanics
- Each round presents a sequence of geometric shapes.
- The number of shapes in a round corresponds to the current round number (e.g., round 3 has 3 shapes).
- The player must **draw the shapes in the correct order** to advance.
- If the player makes more than **5 mistakes**, the game ends.

### 🕹️ Controls
- The game is played using the **index finger** of the hand.
- The player draws the shape's trajectory in the **OpenCV window**.
- To **erase the trajectory** (like a blackboard), press the `Space` key.
- When satisfied with the drawn shape, press `Enter` to submit and verify it.

### ⏭️ Advancing to the Next Round
- After completing a round, the game **locks the window** and displays the next round’s sequence.

The challenge increases with each round, requiring faster and more precise drawing skills to win!

## 📂 Project Structure
_(Pending details)_

## 🖥️ Technologies Used
_(Pending details)_

## 🙌 Credits
_(Pending details)_
