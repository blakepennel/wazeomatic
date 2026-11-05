plugins {
    id("com.android.application")
    id("org.jetbrains.kotlin.android")
}

android {
    namespace = "com.wazeomatic.mockwaze"
    compileSdk = 35

    defaultConfig {
        applicationId = "com.wazeomatic.mockwaze"
        minSdk = 29
        targetSdk = 35
        versionCode = 1
        versionName = "0.1"
    }

    buildTypes {
        release {
            isMinifyEnabled = false
        }
    }
    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_17
        targetCompatibility = JavaVersion.VERSION_17
    }
    kotlinOptions {
        jvmTarget = "17"
    }
}

// No AndroidX on purpose: plain framework views keep the build small and the
// view tree easy to inspect with uiautomator dump.
