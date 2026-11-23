plugins {
    id("com.android.application")
    id("org.jetbrains.kotlin.android")
}

android {
    namespace = "com.wazeomatic.detector"
    compileSdk = 35
    buildToolsVersion = "35.0.0"

    defaultConfig {
        applicationId = "com.wazeomatic.detector"
        minSdk = 29
        targetSdk = 35
        versionCode = 1
        versionName = "0.1"
        ndk {
            // phones + the x86_64 emulator; keeps onnxruntime from bloating the apk
            abiFilters += listOf("arm64-v8a", "x86_64")
        }
    }

    buildTypes {
        release {
            isMinifyEnabled = false
            // -PdebugSignedRelease signs it with the local debug key so it installs for timing on a phone. A
            // debug build is much slower on the device (debuggable apps are not compiled ahead of time), which
            // makes it the wrong build to time (docs/model-research.md)
            if (findProperty("debugSignedRelease") != null) signingConfig = signingConfigs.getByName("debug")
        }
    }
    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_17
        targetCompatibility = JavaVersion.VERSION_17
    }
    kotlinOptions {
        jvmTarget = "17"
    }
    androidResources {
        // onnx files are already dense, compressing them only slows down loading
        noCompress += listOf("onnx")
    }
    testOptions {
        unitTests.isReturnDefaultValues = true
    }
    buildFeatures {
        // BuildConfig.DEBUG gates the test-only INJECT_DETECTION receiver
        buildConfig = true
    }
}

dependencies {
    implementation("androidx.core:core-ktx:1.13.1")
    implementation("androidx.appcompat:appcompat:1.7.0")
    implementation("com.google.android.material:material:1.12.0")
    implementation("androidx.recyclerview:recyclerview:1.3.2")
    implementation("androidx.lifecycle:lifecycle-service:2.8.7")
    implementation("androidx.camera:camera-core:1.4.1")
    implementation("androidx.camera:camera-camera2:1.4.1")
    implementation("androidx.camera:camera-lifecycle:1.4.1")
    implementation("com.microsoft.onnxruntime:onnxruntime-android:1.20.0")

    testImplementation("junit:junit:4.13.2")
    // android.jar only ships stubs of org.json, the real one is needed on the JVM
    testImplementation("org.json:json:20240303")
}

// Copies exported models from <repo>/models into the app assets.
// Override the source file names with e.g. -PcocoModel=coco_yolo11n.int8.onnx
val repoModels = rootProject.file("../../models")
// The open-shape exports (export_onnx.py --dynamic --ship / --coco --dynamic) are preferred: the app runs them at
// the frame's own aspect (640x384 for 16:9 instead of 640x640). Without them the square exports are bundled.
// The hazard model goes int8 when tools/quantize_onnx.py --static has made hazard_yolo11n_dyn.qdq.onnx: about 1.45x
// the frames a second on a mid-range phone with nearly the same reports (docs/model-research.md). COCO stays float: its
// int8 boxes tipped the police classifier on a plain white SUV.
fun preferOpen(open: String, square: String) = if (File(repoModels, open).exists()) open else square
val hazardModel = (findProperty("hazardModel") as String?)
    ?: preferOpen("hazard_yolo11n_dyn.qdq.onnx", preferOpen("hazard_yolo11n_dyn.onnx", "hazard_yolo11n.onnx"))
val cocoModel = (findProperty("cocoModel") as String?) ?: preferOpen("coco_yolo11n_dyn.onnx", "coco_yolo11n.onnx")
// second hazard model fused on the device (ensemble setting); "" (the default since the v3 model)
// ships the single v3 nano. The older 4-class models no longer help it (docs/model-research.md);
// -PhazardModel2=hazard_yolo11s_v3.onnx fuses the v3 small where the phone can afford it.
val hazardModel2 = (findProperty("hazardModel2") as String?) ?: ""
// optional classifiers (tools/train_classifier.py): asset name -> source file in models/. A missing file
// only switches that part off in the app (the status line says so); the build never fails over one.
// Override with e.g. -PvehicleModel=vehicle_cls_v2.onnx, or "" to leave it out.
val classifierModels = linkedMapOf(
    "vehicle_cls" to ((findProperty("vehicleModel") as String?) ?: "vehicle_cls.onnx"),
    // averaged with vehicle_cls on the phone when present (models/README.md)
    "vehicle_cls2" to ((findProperty("vehicleModel2") as String?) ?: "vehicle_cls2.onnx"),
    "weather_cls" to ((findProperty("weatherModel") as String?) ?: "weather_cls.onnx"),
    "surface_cls" to ((findProperty("surfaceModel") as String?) ?: "surface_cls.onnx"),
)

val syncModels by tasks.registering(Copy::class) {
    description = "Copy models/*.onnx and label files into app/src/main/assets"
    doFirst {
        val assets = layout.projectDirectory.dir("src/main/assets")
        if (hazardModel2.isBlank()) {
            // a previous build with a second model leaves its assets behind; Copy never deletes
            assets.file("hazard2.onnx").asFile.delete()
            assets.file("hazard2_labels.txt").asFile.delete()
        }
        for ((asset, src) in classifierModels) {
            val present = src.isNotBlank() && File(repoModels, src).exists() &&
                File(repoModels, src.removeSuffix(".onnx") + ".labels.txt").exists()
            if (!present) {
                if (src.isNotBlank()) logger.lifecycle("syncModels: $src (+ .labels.txt) not in models/, $asset is left out")
                assets.file("$asset.onnx").asFile.delete()
                assets.file("${asset}_labels.txt").asFile.delete()
            }
        }
    }
    for ((asset, src) in classifierModels) {
        if (src.isBlank()) continue
        val labels = src.removeSuffix(".onnx") + ".labels.txt"
        // only copied as a pair: a model without its class names is useless
        if (File(repoModels, src).exists() && File(repoModels, labels).exists()) {
            from(repoModels) {
                include(src)
                rename { "$asset.onnx" }
            }
            from(repoModels) {
                include(labels)
                rename { "${asset}_labels.txt" }
            }
        }
    }
    from(repoModels) {
        include(hazardModel)
        rename { "hazard.onnx" }
    }
    from(repoModels) {
        include(cocoModel)
        rename { "coco.onnx" }
    }
    // optional second hazard model for on-device fusion (-PhazardModel2=hazard_yolo11s_v3.onnx)
    if (hazardModel2.isNotBlank()) {
        from(repoModels) {
            include(hazardModel2)
            rename { "hazard2.onnx" }
        }
        from(repoModels) {
            include(hazardModel2.removeSuffix(".onnx") + ".labels.txt")
            rename { "hazard2_labels.txt" }
        }
    }
    // export_onnx.py writes <model>.labels.txt next to each model
    from(repoModels) {
        include(hazardModel.removeSuffix(".onnx") + ".labels.txt")
        rename { "hazard_labels.txt" }
    }
    from(repoModels) {
        include(cocoModel.removeSuffix(".onnx") + ".labels.txt")
        rename { "coco_labels.txt" }
    }
    into(layout.projectDirectory.dir("src/main/assets"))
}

tasks.named("preBuild") { dependsOn(syncModels) }
