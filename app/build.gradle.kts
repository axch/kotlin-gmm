plugins {
    id("org.jetbrains.kotlin.jvm") version "1.9.10"
    // kotlin("jvm") version "1.9.10"
    application
}

dependencies {
    implementation("org.jetbrains.lets-plot:lets-plot-kotlin-jvm:4.4.3")
    implementation("org.jetbrains.lets-plot:lets-plot-image-export:4.0.1")
}

repositories {
    mavenCentral()
}

application {
    // Define the main class for the application.
    mainClass.set("HelloKt")
}
