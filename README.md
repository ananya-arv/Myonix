# Myonix
A wearable rehabilitation sleeve designed to gamify forearm exercises by combining muscle activity sensing (EMG) with cognitive load monitoring (EDA and heart rate). The system adapts exercise difficulty in real-time using lightweight machine learning, providing an AI-driven coach that distinguishes between physical fatigue and mental effort.

**PROTOTYPE IN DEVELOPMENT**

---

## Progress Updates

### 9/19 — Hardware & Signal Integration 🚀

✅ **Integrated EMG + EDA Testing**

- Successfully tested **Electromyography (EMG)** and **Electrodermal Activity (EDA)** sensors with ESP32.  
- During **grip exercises** (wrist muscle activation):  
  - **EDA**: Voltage values decrease as skin/sweat conductance increases with activity.  
  - **EMG**: Clear spikes detected on muscle activation.  
- **LED indicator** lights up on EMG activation.  

📷 _[Insert hardware setup photos here]_  
🎥 _[Insert demo video link here]_  

---

### 📍 EMG Electrode Placement
Proper electrode placement is critical for clean signal capture.  

📷 _![EMG End & Mid Placement](Mid & End Electrode Placement.png)_  
📷 _![EMG Ref Placement](Reference Electrode EMG Placement.png)_ 
📝 _Tip: Document electrode reference (ground), mid, and end positions for consistency across tests._  

---

### 🔜 Next Steps
- **IMU (MPU6050) integration** alongside EMG + EDA to validate correct signal capture.  
- Once all three signals are stable, move on to:  
  - Signal processing (filtering, feature extraction)  
  - Data analysis for activity/stress detection  

---

### 📂 Reference Code
- Main test script: [`emg_eda_test`](./emg_eda_test.ino)  
