# YOLOX Android / ncnn

This optional demo targets Android API 24+ on **ARMv7 and ARM64**, with CPU
inference and optional Vulkan. It is not an x86 emulator build. A successful
Python installation does not build or validate this application.

## Pinned build baseline

| Component | Version |
| --- | --- |
| JDK | 17 |
| Android Gradle Plugin | 8.13.2 |
| Gradle wrapper | 8.13 (distribution SHA-256 pinned) |
| Android compile/target SDK | 36 |
| SDK Build Tools | 35.0.0 |
| NDK | 28.2.13676358 |
| CMake (Android SDK package) | 3.22.1 |
| ncnn Android Vulkan SDK | 20250916 |

Install the SDK components through Android Studio's SDK Manager or `sdkmanager`:

```shell
sdkmanager "platforms;android-36" "build-tools;35.0.0" "ndk;28.2.13676358" "cmake;3.22.1"
```

Set `ANDROID_HOME` or create an untracked `local.properties` with `sdk.dir`.
Set `JAVA_HOME` to JDK 17. Google/Maven Central replace JCenter. The manifest
declares its launcher activity exported, as required by the modern target SDK.

## Dependencies and model assets

Run the following Python commands **from this directory**. They work on Linux,
macOS and Windows with Python 3.12/3.13 and do not require shell-specific unzip
or checksum utilities. The archives were downloaded and hashed on 2026-10-25;
these digests record tested bytes, not publisher signatures.

Download and verify the pinned SDK before extraction:

```shell
python -c "import pathlib,urllib.request; p=pathlib.Path('.deps'); p.mkdir(exist_ok=True); urllib.request.urlretrieve('https://github.com/Tencent/ncnn/releases/download/20250916/ncnn-20250916-android-vulkan.zip', p/'ncnn.zip')"
python -c "import hashlib,pathlib,zipfile; p=pathlib.Path('.deps/ncnn.zip'); assert hashlib.sha256(p.read_bytes()).hexdigest() == '27e94224e63a74359e4cbf37b9ad86bdeff9d751f737ab7a32971c01ff048ee8'; zipfile.ZipFile(p).extractall('.deps')"
```

CMake requires `.deps/ncnn-20250916-android-vulkan/<ABI>/lib/cmake/ncnn` and
rejects a different SDK version. No developer-specific absolute path is needed.
The JNI code selects CPU/Vulkan before loading each network, matching the
current ncnn API rather than using the removed per-extractor Vulkan setter.

For the historical demo model, download the **matching param/bin pair**:

```shell
python -c "import urllib.request; urllib.request.urlretrieve('https://github.com/Megvii-BaseDetection/storage/releases/download/0.0.1/yolox_s_ncnn.tar.gz', '.deps/model.tar.gz')"
python -c "import hashlib,pathlib,tarfile; p=pathlib.Path('.deps/model.tar.gz'); assert hashlib.sha256(p.read_bytes()).hexdigest() == '3386688304e34761cef43987b4c0f1a789f9e8f625ac4ee304cde222adebd39b'; tarfile.open(p).extractall('.deps/model', filter='data')"
```

Copy **both** `.deps/model/yolox.param` and `.deps/model/yolox.bin` into
`app/src/main/assets`. This intentionally replaces the checked-in legacy param:
it is a different model and must not be paired with the downloaded BIN alone.
The JNI decoder expects 640×640 input, 80 COCO classes, strides 8/16/32 and
`images`/`output` blob names. Asset hashes do not establish accuracy or runtime
compatibility; validate the pair on your device. Do not package arbitrary files
without checking their source and integrity.

## Build and validation limits

Open this directory in an Android Studio version supporting AGP 8.13, or run:

```shell
./gradlew assembleDebug
# Windows: gradlew.bat assembleDebug
```

The wrapper JAR SHA-256 is
`81a82aaea5abcc8ff68b3dfcb58b3c3c429378efd98e7433460610fecd7ae45f`,
verified against Gradle's published checksum. Gradle 8.13 runs with JDK 17 and
resolves AGP 8.13.2 in the Linux orb.

### Optional native validation (2026-10-25, Linux x86-64)

No existing Android SDK or accepted SDK licenses were found. A disposable SDK
root was created at `/tmp/yolox-android-validation/android-sdk`, using Android
Command-line Tools 19.0 from Google's HTTPS repository:

* Archive: `https://dl.google.com/android/repository/commandlinetools-linux-13114758_latest.zip`
* SHA-1: `5fdcc763663eefb86a5b8879697aa6088b041e70`, verified against the
  Linux `cmdline-tools;19.0` entry in
  `https://dl.google.com/android/repository/repository2-3.xml`.
* Observed SHA-256: `7ec965280a073311c339e571cd5de778b9975026cfcbe79f2b1cdcb1e15317ee`.

The first attempt, with closed standard input, correctly stopped at
`LicenceNotAcceptedException` for NDK 28.2.13676358 and produced no APK. After
the user explicitly authorized acceptance of the Android SDK License Agreement,
the four pinned packages were installed using the command below. Only the
applicable `android-sdk-license` was accepted, not preview agreements. Gradle
also installed Platform-Tools 37.0.1 under that already-accepted license.

After extracting the command-line tools into `cmdline-tools/19.0`, commands were
run from this directory:

```shell
export JAVA_HOME=/tmp/yolox-android-validation/jdk-17.0.21+9
export ANDROID_HOME=/tmp/yolox-android-validation/android-sdk
export PATH="$JAVA_HOME/bin:$PATH"
"$ANDROID_HOME/cmdline-tools/19.0/bin/sdkmanager" --sdk_root="$ANDROID_HOME" \
  "platforms;android-36" "build-tools;35.0.0" \
  "ndk;28.2.13676358" "cmake;3.22.1"
# Review and answer the applicable license prompt only when authorized.
# First extract and verify the ncnn SDK/model as described above.
cp app/src/main/assets/yolox.param .deps/original-yolox.param
cp .deps/model/yolox.param .deps/model/yolox.bin app/src/main/assets/
./gradlew assembleDebug --no-daemon --console=plain </dev/null
# Restore the source tree even if the build fails (use a shell trap in scripts).
cp .deps/original-yolox.param app/src/main/assets/yolox.param
rm app/src/main/assets/yolox.bin
./gradlew lintDebug --no-daemon --console=plain </dev/null

APK=app/build/outputs/apk/debug/app-debug.apk
"$ANDROID_HOME/build-tools/35.0.0/zipalign" -c -P 16 -v 4 "$APK"
"$ANDROID_HOME/build-tools/35.0.0/apksigner" verify --verbose "$APK"
"$ANDROID_HOME/build-tools/35.0.0/aapt" dump badging "$APK"
```

Results using the actual pinned SDK/NDK/CMake, not stubs:

```text
assembleDebug: BUILD SUCCESSFUL in 35s
40 actionable tasks: 40 executed
lintDebug: BUILD SUCCESSFUL in 32s
0 errors, 13 warnings
zipalign -c -P 16 -v 4: Verification succesful
apksigner verify: Verifies; APK Signature Scheme v2: true; 1 signer
native-code: 'arm64-v8a' 'armeabi-v7a'
```

Both `buildCMakeDebug[arm64-v8a]` and `buildCMakeDebug[armeabi-v7a]` compiled and
linked successfully. No additional source/build fix was required. Lint warnings
remain for default-locale formatting, framework ExifInterface, redundant label,
available Gradle updates, layout weight, missing icon/content description,
button style and hardcoded text. They were not suppressed. Gradle also reports
deprecated features incompatible with Gradle 9; this build uses pinned 8.13.

The debug APK is `app/build/outputs/apk/debug/app-debug.apk`; tested SHA-256:
`a1e0644becdd3872ff47946425148a53fb8f6ef86af9a78ad6245112e24f5278`.
This is a debug-signed, debuggable test build, not a release artifact. Its
manifest reports minimum SDK 24 and target SDK 36. It contains one native
library per ABI, `libyoloXncnn.so`. Inspection of the extracted libraries with
NDK `toolchains/llvm/prebuilt/linux-x86_64/bin/llvm-readelf -h -l --dynamic
--dyn-syms` confirmed:

| ABI | ELF architecture | All LOAD segment alignments | Uncompressed APK library offset |
| --- | --- | --- | --- |
| arm64-v8a | ELF64 / AArch64 | `0x4000` (16 KB) | 16,384 |
| armeabi-v7a | ELF32 / ARM | `0x1000` (4 KB) | 9,404,416 |

The APK passes 16 KB ZIP alignment, and its 64-bit ELF passes the static 16 KB
segment-alignment check. The 32-bit ELF retains 4 KB segment alignment; do not
describe both ELFs as 16 KB aligned. Both libraries export `JNI_OnLoad`,
`JNI_OnUnload`, and the expected `YOLOXncnn_Init`/`YOLOXncnn_Detect` JNI symbols.
Their dynamic dependencies are only Android system libraries (`libjnigraphics`,
`libdl`, `libandroid`, `liblog`, `libm`, `libc`); no unbundled C++ runtime was
listed.

The packaged matching model assets were verified after APK extraction:

* `yolox.param`: `ee80b3cb89579aedae8ade8b61eacef366b502296acee2dad082174747f25c59`
* `yolox.bin`: `dcf9487b7ff9c5e831eb77b7187f5324a21bcb732e8f8415787cb2603f85e3e8`

The tracked legacy param was restored byte-for-byte after the build, and the
temporary BIN was removed from source assets. A future build must again stage
the matching pair. The SDK/model downloads remain in ignored `.deps/`.

**Compilation, packaging, signing and static alignment checks do not establish
runtime correctness.** Model loading/accuracy, image-picker behavior,
CPU/Vulkan inference, lifecycle behavior and execution on a real 16 KB device
remain unvalidated. No emulator was used. Non-Linux host builds also remain
unverified. Use a real device/connected Android runner for those checks.

## Reference

* [ncnn Android YOLOv5](https://github.com/nihui/ncnn-android-yolov5)
* [AGP 8.13 compatibility](https://developer.android.com/build/releases/past-releases/agp-8-13-0-release-notes)
