# AI App Finder

An Android application that allows users to find installed apps by describing their icons using local AI (CLIP/ONNX).

## Features
- **Semantic Search:** Search apps by describing visual elements (e.g., "blue app with bird").
- **Local AI:** Uses ONNX Runtime and CLIP model for privacy and offline support.
- **Material 3 UI:** Modern design with Jetpack Compose.
- **Arabic & English Support:** Full RTL support.

## Technical Stack
- **Language:** Kotlin
- **UI:** Jetpack Compose, Material 3
- **Database:** Room
- **AI Engine:** ONNX Runtime (CLIP Model)
- **Dependency Injection:** Hilt

## Setup
1. Open the project in Android Studio.
2. Place your CLIP ONNX models in `app/src/main/assets/models/`.
    - `visual_encoder.onnx`
    - `text_encoder.onnx`
3. Build and run on an Android device.

## License
Apache License 2.0
