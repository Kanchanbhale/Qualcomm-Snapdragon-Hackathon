# Qualcomm-Snapdragon-Hackathon
CampusGuard
Real-Time, On-Device, Multi-Device Agentic AI for Campus Safety

Overview

CampusGuard is a privacy-first, agentic AI system for real-time campus safety monitoring built entirely on Snapdragon-powered devices. Instead of streaming raw video to the cloud, CampusGuard performs on-device intelligence to detect safety-critical events and shares only structured AI events across devices for reasoning and escalation.

The system distributes intelligence across:
Snapdragon 8 Elite (Samsung Galaxy S25) — real-time perception and local decision-making
Snapdragon X Elite (Copilot+ PC) — multi-agent reasoning, orchestration, and visualization
This architecture enables low-latency response, strong privacy guarantees, and scalable multi-device coordination, making it suitable for campuses, public spaces, and smart environments.

Key Capabilities
On-device video inference (no cloud video streaming)

Detection of:
Behavioral anomalies (loitering, abnormal motion)
Abandoned objects
Falls and motionless individuals
Multi-agent reasoning across time and devices
Structured event streaming (JSON, not video)
Real-time alert dashboard on Snapdragon X Elite PC

Why Agentic AI?
CampusGuard is not a single monolithic model. It is a collection of cooperating agents, each with a well-defined role:

Agent	                  Responsibility	                         Device
Perception Agent	  Person, pose, object detection	         S25
Local Temporal Agent	  Short-term behavior reasoning	                 S25
Privacy Guard Agent	  Enforces no raw video transfer	         S25
Event Reasoning Agent	  Aggregates & correlates events	         X Elite PC
Policy Agent              Escalation & confidence logic	                 X Elite PC
Orchestrator Agent	  Multi-device coordination	                 X Elite PC
Human-in-the-loop Agent	  Dashboard & decision support	                 X Elite PC

This distributed intelligence is the core of the system.

Datasets Used
1. ABODA (Abandoned Objects)	Videos	11 video sequences (GitHub)	Each ~2–3 min (per paper) (MDPI) - https://github.com/kevinlin311tw/ABODA?utm_source=chatgpt.com
2. CUHK Avenue (campus anomaly)	Videos	37 clips (16 train + 21 test) (cse.cuhk.edu.hk)	30,652 frames total (15,328 train + 15,324 test) (cse.cuhk.edu.hk) - https://www.cse.cuhk.edu.hk/leojia/projects/detectabnormal/dataset.html?utm_source=chatgpt.com
3. COCO 2017 (general objects + person keypoints)	Images	163,957 images (118,287 train + 5,000 val + 40,670 test) (Dataset Ninja)	Test labels not publicly released (typical COCO eval setup) (GitHub) - https://datasetninja.com/coco-2017?utm_source=chatgpt.com
4. UR Fall Detection	Video + sensors	70 sequences (fenix.ur.edu.pl)	30 falls + 40 ADL (fenix.ur.edu.pl) - https://fenix.ur.edu.pl/mkepski/ds/uf.html?utm_source=chatgpt.com									
CampusGuard focuses on behavior-first anomaly detection, not weapon classification.

Dataset	Purpose
CUHK Avenue	Behavioral anomaly detection
ABODA	Abandoned object reasoning
UR Fall Detection	Fall and safety detection
COCO (pretrained, person class only)	Person detection backbone

All models are executed on-device using pretrained weights optimized for edge inference.

System Architecture
[ S25 Camera Feed ]
        ↓
[ Perception + Temporal Agents ]
        ↓
[ Structured Event Output ]
  (timestamp, type, bbox, confidence)
        ↓
[ Snapdragon X Elite PC ]
        ↓
[ Reasoning + Policy + Orchestration Agents ]
        ↓
[ Alert Dashboard / Timeline / Escalation ]


Only intelligence moves — never raw video.

Software Stack
Snapdragon X Elite (PC)
Frontend: Streamlit (Python)
Backend: Python (x64 for QNN access)
ML Runtime: ONNX Runtime – QNN
SDKs: Nexa SDK
IDE: Visual Studio Code
Snapdragon 8 Elite (S25 Phone)
App: Android (Kotlin)
ML Runtime: ONNX Runtime – Android
Dev Tools: Android SDK
IDE: Android Studio

Pretrained ONNX models for:
Person detection
Pose estimation
Convert models to QNN-compatible format if needed
Validate inference locally on Snapdragon X Elite

Step 3: Android App (S25)
Create Android app in Kotlin
Integrate ONNX Runtime Android
Load models on-device

Implement:
Camera capture
Frame preprocessing
On-device inference

Generate structured event outputs:

{
  "event_type": "abandoned_object",
  "timestamp": "...",
  "confidence": 0.87,
  "bounding_box": [x, y, w, h]
}

Step 4: On-Device Agents (Phone)

Implement:
Perception Agent
Local Temporal Agent (simple sliding window logic)
Privacy Guard Agent (block raw video transmission)

Step 5: Event Streaming
Send structured events from phone → PC
Use lightweight transport (WebSocket / REST)

Step 6: Reasoning Agents (PC)
Implement:
Event aggregation
Temporal correlation
Confidence smoothing
Escalation logic

Example:
“If abandoned_object persists > N seconds with confidence > threshold → alert”

Step 7: Dashboard (PC)
Using Streamlit:
Event timeline
Camera/device IDs
Confidence scores
Bounding box previews

Alert status
Step 8: Demo Scenario

Demonstrate:
Person enters scene
Object left behind
Person exits frame

System detects abandonment
Alert appears on PC dashboard
No video ever leaves the phone
Evaluation Strategy

Latency: on-device inference time

Event accuracy: confidence trends

Privacy: zero raw video transmission

Scalability: multiple phones → one PC

Why Snapdragon
CampusGuard leverages Snapdragon’s strengths:
High-performance on-device AI
QNN-optimized inference
Multi-device AI ecosystems
Privacy-first edge computing

Future Extensions
Multi-camera spatial reasoning
XR-based security visualization
Robotics integration for response
Adaptive policy learning

Final Note
CampusGuard demonstrates that agentic AI does not require the cloud.
By distributing intelligence across Snapdragon-powered devices, the system achieves real-time awareness, strong privacy, and scalable coordination — exactly what modern campuses need.

Final Datasets 
https://universe.roboflow.com/mahad-ahmed/gun-and-knife-detection
https://www.kaggle.com/datasets/raghavnanjappan/weapon-dataset-for-yolov5?utm_source=chatgpt.com

