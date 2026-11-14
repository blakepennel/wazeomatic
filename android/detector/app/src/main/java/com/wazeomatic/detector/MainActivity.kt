package com.wazeomatic.detector

import android.Manifest
import android.app.Activity
import android.app.AlertDialog
import android.content.Intent
import android.content.pm.PackageManager
import android.graphics.Color
import android.os.Bundle
import android.os.PowerManager
import android.text.InputType
import android.view.Gravity
import android.view.View
import android.view.ViewGroup.LayoutParams.MATCH_PARENT
import android.view.ViewGroup.LayoutParams.WRAP_CONTENT
import android.view.WindowInsets
import android.view.WindowManager
import android.widget.ArrayAdapter
import android.widget.Button
import android.widget.CheckBox
import android.widget.EditText
import android.widget.LinearLayout
import android.widget.ListView
import android.widget.RadioButton
import android.widget.RadioGroup
import android.widget.ScrollView
import android.widget.TextView
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale

/**
 * Preview + controls. Launch extras (see README) update the settings and, when "source" or
 * autostart=true is given, start the detector service straight away so adb scripts can
 * drive the app without taps.
 */
class MainActivity : Activity(), PipelineBus.Listener {
    private lateinit var overlay: OverlayView
    private lateinit var statusText: TextView
    private lateinit var statsText: TextView
    private lateinit var batteryHint: LinearLayout
    private lateinit var wazeHint: LinearLayout
    private lateinit var eventsAdapter: ArrayAdapter<String>
    private var pendingStart = false
    private val timeFmt = SimpleDateFormat("HH:mm:ss", Locale.US)

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        window.addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON)
        buildUi()
        PipelineBus.addListener(this)
        handleLaunch(intent)
    }

    override fun onNewIntent(intent: Intent) {
        super.onNewIntent(intent)
        setIntent(intent)
        handleLaunch(intent)
    }

    override fun onResume() {
        super.onResume()
        refreshBatteryHint()
        refreshWazeHint()
    }

    /** Shown while the Waze reader is wanted (setting on) but its accessibility service is switched off. */
    private fun refreshWazeHint() {
        val off = Settings.load(this).wazeReader && !WazeWatcherService.isEnabled(this)
        wazeHint.visibility = if (off) View.VISIBLE else View.GONE
    }

    private fun openAccessibilitySettings() {
        try {
            startActivity(Intent(android.provider.Settings.ACTION_ACCESSIBILITY_SETTINGS))
        } catch (e: Exception) {
            statusText.text = "could not open the accessibility settings: ${e.message}"
        }
    }

    /** Shown while battery optimisation is on for the app; the system may then stop it behind Waze. */
    private fun refreshBatteryHint() {
        val unrestricted = try {
            getSystemService(PowerManager::class.java).isIgnoringBatteryOptimizations(packageName)
        } catch (e: Exception) {
            true
        }
        batteryHint.visibility = if (unrestricted) View.GONE else View.VISIBLE
    }

    private fun openBatterySettings() {
        try {
            startActivity(Intent(android.provider.Settings.ACTION_IGNORE_BATTERY_OPTIMIZATION_SETTINGS))
        } catch (e: Exception) {
            statusText.text = "could not open the battery settings: ${e.message}"
        }
    }

    override fun onDestroy() {
        PipelineBus.removeListener(this)
        super.onDestroy()
    }

    private fun buildUi() {
        val dp = resources.displayMetrics.density
        val root = LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL
            setBackgroundColor(Color.rgb(20, 20, 20))
        }
        statusText = TextView(this).apply {
            setTextColor(Color.WHITE); textSize = 13f
            setPadding((8 * dp).toInt(), (6 * dp).toInt(), (8 * dp).toInt(), 0)
        }
        statsText = TextView(this).apply {
            setTextColor(Color.rgb(120, 230, 120)); textSize = 13f
            setPadding((8 * dp).toInt(), 0, (8 * dp).toInt(), (4 * dp).toInt())
        }
        batteryHint = LinearLayout(this).apply {
            orientation = LinearLayout.HORIZONTAL
            gravity = Gravity.CENTER_VERTICAL
            visibility = View.GONE
            setPadding((8 * dp).toInt(), 0, (8 * dp).toInt(), 0)
            addView(TextView(this@MainActivity).apply {
                text = "Battery is restricted: set it to Unrestricted so detection keeps running behind Waze"
                setTextColor(Color.rgb(255, 190, 80)); textSize = 13f
            }, LinearLayout.LayoutParams(0, WRAP_CONTENT, 1f))
            addView(Button(this@MainActivity).apply {
                text = "Battery settings"
                setOnClickListener { openBatterySettings() }
            }, LinearLayout.LayoutParams(WRAP_CONTENT, WRAP_CONTENT))
        }
        wazeHint = LinearLayout(this).apply {
            orientation = LinearLayout.HORIZONTAL
            gravity = Gravity.CENTER_VERTICAL
            visibility = View.GONE
            setPadding((8 * dp).toInt(), 0, (8 * dp).toInt(), 0)
            addView(TextView(this@MainActivity).apply {
                text = "Waze reader is off: turn on \"Wazeomatic Waze reader\" in Accessibility"
                setTextColor(Color.rgb(255, 190, 80)); textSize = 13f
            }, LinearLayout.LayoutParams(0, WRAP_CONTENT, 1f))
            addView(Button(this@MainActivity).apply {
                text = "Accessibility"
                setOnClickListener { openAccessibilitySettings() }
            }, LinearLayout.LayoutParams(WRAP_CONTENT, WRAP_CONTENT))
        }
        overlay = OverlayView(this)
        val buttons = LinearLayout(this).apply { orientation = LinearLayout.HORIZONTAL; gravity = Gravity.CENTER }
        fun btn(label: String, onClick: () -> Unit) = Button(this).apply {
            text = label
            setOnClickListener { onClick() }
            buttons.addView(this, LinearLayout.LayoutParams(0, WRAP_CONTENT, 1f))
        }
        btn("Start") { requestStart() }
        btn("Stop") { DetectorService.stop(this) }
        btn("Settings") { showSettings() }

        eventsAdapter = ArrayAdapter(this, android.R.layout.simple_list_item_1, ArrayList<String>())
        val list = ListView(this).apply { adapter = eventsAdapter; setBackgroundColor(Color.rgb(36, 36, 36)) }
        PipelineBus.recentEvents.forEach { (ev, dry) -> eventsAdapter.add(eventLine(ev, dry)) }

        root.addView(statusText, LinearLayout.LayoutParams(MATCH_PARENT, WRAP_CONTENT))
        root.addView(statsText, LinearLayout.LayoutParams(MATCH_PARENT, WRAP_CONTENT))
        root.addView(batteryHint, LinearLayout.LayoutParams(MATCH_PARENT, WRAP_CONTENT))
        root.addView(wazeHint, LinearLayout.LayoutParams(MATCH_PARENT, WRAP_CONTENT))
        root.addView(overlay, LinearLayout.LayoutParams(MATCH_PARENT, 0, 3f))
        root.addView(buttons, LinearLayout.LayoutParams(MATCH_PARENT, WRAP_CONTENT))
        root.addView(list, LinearLayout.LayoutParams(MATCH_PARENT, 0, 2f))
        // targetSdk 35 draws edge-to-edge, keep the UI out of the status/navigation bars
        root.setOnApplyWindowInsetsListener { v, insets ->
            val bars = insets.getInsets(WindowInsets.Type.systemBars())
            v.setPadding(bars.left, bars.top, bars.right, bars.bottom)
            insets
        }
        setContentView(root)

        val hasModel = Pipeline.assetExists(this, "hazard.onnx") || Pipeline.assetExists(this, "coco.onnx")
        statusText.text = when {
            PipelineBus.running -> PipelineBus.status
            // the service stopped itself and said why (e.g. the Android 15 time limit)
            PipelineBus.status.startsWith("stopped:") -> PipelineBus.status
            hasModel -> "ready (stopped)"
            else -> "model missing"
        }
        overlay.message = if (hasModel) "press Start" else "model missing - see README"
    }

    private fun handleLaunch(i: Intent?) {
        val extras = i?.extras ?: return
        if (extras.getBoolean("stop", false)) {
            DetectorService.stop(this)
            return
        }
        if (extras.getBoolean("reset_settings", false)) Settings.reset(this)
        val touched = extras.keySet().any { it in Settings.KNOWN_KEYS }
        if (touched) Settings.load(this).withExtras(extras).save(this)
        if (extras.containsKey("source") || extras.getBoolean("autostart", false)) requestStart()
    }

    private fun requestStart() {
        val s = Settings.load(this)
        val needed = ArrayList<String>()
        if (s.source == "camera" && checkSelfPermission(Manifest.permission.CAMERA) != PackageManager.PERMISSION_GRANTED) {
            needed.add(Manifest.permission.CAMERA)
        }
        if (android.os.Build.VERSION.SDK_INT >= 33 &&
            checkSelfPermission(Manifest.permission.POST_NOTIFICATIONS) != PackageManager.PERMISSION_GRANTED
        ) {
            needed.add(Manifest.permission.POST_NOTIFICATIONS)
        }
        // location only when the user switched on "use GPS speed" (camera source; the speed stays on the phone)
        if (s.useGpsSpeed && s.source == "camera" &&
            checkSelfPermission(Manifest.permission.ACCESS_FINE_LOCATION) != PackageManager.PERMISSION_GRANTED
        ) {
            needed.add(Manifest.permission.ACCESS_FINE_LOCATION)
            needed.add(Manifest.permission.ACCESS_COARSE_LOCATION)
        }
        if (needed.isEmpty()) {
            DetectorService.start(this)
        } else {
            pendingStart = true
            requestPermissions(needed.toTypedArray(), REQ_PERMS)
        }
    }

    override fun onRequestPermissionsResult(requestCode: Int, permissions: Array<out String>, grantResults: IntArray) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults)
        if (requestCode == REQ_PERMS && pendingStart) {
            pendingStart = false
            // notifications and location are optional; the service reports a missing camera permission itself
            DetectorService.start(this)
        }
    }

    private fun showSettings() {
        val s = Settings.load(this)
        val dp = resources.displayMetrics.density
        val form = LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL
            setPadding((16 * dp).toInt(), (8 * dp).toInt(), (16 * dp).toInt(), 0)
        }
        fun label(t: String) = form.addView(TextView(this).apply { text = t })
        fun field(t: String, v: String, numeric: Boolean): EditText {
            label(t)
            return EditText(this).apply {
                setText(v)
                inputType = if (numeric) InputType.TYPE_CLASS_NUMBER or InputType.TYPE_NUMBER_FLAG_DECIMAL else InputType.TYPE_CLASS_TEXT
                form.addView(this)
            }
        }
        label("Source")
        val group = RadioGroup(this).apply { orientation = RadioGroup.HORIZONTAL }
        val sources = listOf("camera", "video", "tcp")
        val radios = sources.map { src ->
            RadioButton(this).apply { text = src; id = View.generateViewId(); group.addView(this) }
        }
        radios[sources.indexOf(s.source).coerceAtLeast(0)].isChecked = true
        form.addView(group)
        val conf = field("Confidence threshold", s.confThreshold.toString(), true)
        val cooldown = field("Cooldown per hazard (s)", s.cooldownS.toString(), true)
        val video = field("Video file (in app files dir)", s.video, false)
        val target = field("Target package (empty = implicit broadcast)", s.targetPackage, false)
        val zoom = field("Zoom crop (empty = off, true = default, or x1,y1,x2,y2)", s.zoomCrop, false)
        val mount = field("Orientation (${MountOrientation.MODES.joinToString(", ")})", s.orientation, false)
        val dry = CheckBox(this).apply { text = "Dry run (macro should not submit)"; isChecked = s.dryRun; form.addView(this) }
        val tcp = CheckBox(this).apply { text = "TCP listener on port ${s.tcpPort}"; isChecked = s.tcpEnabled; form.addView(this) }
        val stopped = CheckBox(this).apply { text = "Stopped-vehicle heuristic (experimental)"; isChecked = s.stoppedVehicle; form.addView(this) }
        val wazeReader = CheckBox(this).apply {
            text = "Skip reports Waze already shows (needs the Waze reader in Accessibility)"
            isChecked = s.wazeReader; form.addView(this)
        }
        val vehicleCls = CheckBox(this).apply {
            text = "Vehicle classifier: police / ambulance / fire (needs vehicle_cls.onnx)"
            isChecked = s.vehicleClassifier; form.addView(this)
        }
        val weatherCls = CheckBox(this).apply {
            text = "Weather and road-surface classifiers (need weather_cls / surface_cls.onnx)"
            isChecked = s.weather; form.addView(this)
        }
        val features = field("Scene features (${SceneLabels.FEATURES.joinToString(",")}, or none)", s.sceneFeatures, false)
        val policeSide = CheckBox(this).apply {
            text = "Report police far on the other side as Police > Other side"
            isChecked = s.policeOtherSide; form.addView(this)
        }
        val gps = CheckBox(this).apply {
            text = "Use GPS speed for stopped / moving (asks for location; nothing is sent)"
            isChecked = s.useGpsSpeed; form.addView(this)
        }

        AlertDialog.Builder(this)
            .setTitle("Settings")
            .setView(ScrollView(this).apply { addView(form) })
            .setPositiveButton("Save") { _, _ ->
                val src = sources[radios.indexOfFirst { it.isChecked }.coerceAtLeast(0)]
                s.copy(
                    source = src,
                    confThreshold = conf.text.toString().toFloatOrNull() ?: s.confThreshold,
                    cooldownS = cooldown.text.toString().toFloatOrNull() ?: s.cooldownS,
                    video = video.text.toString().trim().ifEmpty { s.video },
                    targetPackage = target.text.toString().trim(),
                    zoomCrop = zoom.text.toString().trim(),
                    orientation = mount.text.toString().trim().lowercase().takeIf { it in MountOrientation.MODES } ?: s.orientation,
                    dryRun = dry.isChecked,
                    tcpEnabled = tcp.isChecked,
                    stoppedVehicle = stopped.isChecked,
                    wazeReader = wazeReader.isChecked,
                    vehicleClassifier = vehicleCls.isChecked,
                    weather = weatherCls.isChecked,
                    sceneFeatures = features.text.toString().trim().lowercase().replace(" ", ""),
                    policeOtherSide = policeSide.isChecked,
                    useGpsSpeed = gps.isChecked,
                ).save(this)
                refreshWazeHint()
                if (PipelineBus.running) requestStart() // restart with the new settings
            }
            .setNegativeButton("Cancel", null)
            .show()
    }


    private fun eventLine(ev: HazardEvent, dry: Boolean): String =
        String.format(
            Locale.US, "%s  %s  %.2f  %s%s%s",
            timeFmt.format(Date((ev.ts * 1000).toLong())), ev.hazardType, ev.confidence, ev.source,
            ev.label?.let { " ($it)" } ?: "", if (dry) "  [dry]" else ""
        )

    override fun onFrame(r: FrameResult) {
        overlay.show(r.bitmap, r.detections)
        val lat = r.latencyMs.entries.joinToString("  ") { String.format(Locale.US, "%s %.0f ms", it.key, it.value) }
        statsText.text = String.format(Locale.US, "%s   %.1f FPS   frame %d   %d boxes", lat, r.fps, r.frameIndex, r.detections.size)
    }

    override fun onEvent(ev: HazardEvent, dryRun: Boolean) {
        eventsAdapter.insert(eventLine(ev, dryRun), 0)
        while (eventsAdapter.count > 200) eventsAdapter.remove(eventsAdapter.getItem(eventsAdapter.count - 1))
    }

    override fun onStatus(status: String) {
        statusText.text = status
        if (status.startsWith("model missing")) overlay.message = "model missing - see README"
    }

    companion object {
        private const val REQ_PERMS = 1
    }
}
