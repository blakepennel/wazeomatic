package com.wazeomatic.detector

import android.Manifest
import android.content.Intent
import android.content.pm.PackageManager
import android.content.res.ColorStateList
import android.net.Uri
import android.os.Bundle
import android.os.PowerManager
import android.text.SpannableStringBuilder
import android.text.Spanned
import android.text.style.ForegroundColorSpan
import android.view.View
import android.view.WindowManager
import android.widget.TextView
import android.widget.Toast
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AppCompatActivity
import androidx.core.content.IntentCompat
import androidx.core.view.ViewCompat
import androidx.core.view.WindowCompat
import androidx.core.view.WindowInsetsCompat
import androidx.core.widget.TextViewCompat
import androidx.recyclerview.widget.LinearLayoutManager
import androidx.recyclerview.widget.RecyclerView
import com.google.android.material.button.MaterialButton
import com.google.android.material.button.MaterialButtonToggleGroup
import com.google.android.material.dialog.MaterialAlertDialogBuilder
import java.util.Locale

/**
 * Main screen: the analysed frames with their boxes, the state, the last report, the model / health line, a big
 * start / stop button, the source and clip, the dry-run toggle and the recent reports.
 *
 * Launch extras (see README) update the settings and, when "source" or autostart=true is given, start the detector
 * service straight away so adb scripts can drive the app without taps. A video shared to the app or opened with it
 * is copied into the clips folder and selected.
 */
class MainActivity : AppCompatActivity(), PipelineBus.Listener {
    private lateinit var overlay: OverlayView
    private lateinit var statePill: TextView
    private lateinit var fpsPill: TextView
    private lateinit var modePill: TextView
    private lateinit var btnDryRun: MaterialButton
    private lateinit var btnStartStop: MaterialButton
    private lateinit var sourceGroup: MaterialButtonToggleGroup
    private lateinit var clipCard: View
    private lateinit var clipName: TextView
    private lateinit var clipInfo: TextView
    private lateinit var lastReportText: TextView
    private lateinit var lastReportSub: TextView
    private lateinit var lastReportTime: TextView
    private lateinit var healthText: TextView
    private lateinit var batteryHint: View
    private lateinit var wazeHint: View
    private lateinit var reportsEmpty: TextView
    private lateinit var reportsCount: TextView
    private val reports = ReportsAdapter()
    private var updatingUi = false
    private var sawFrame = false

    private val clipPicker = ClipPicker(this) { name -> chooseClip(name) }

    private val permissions = registerForActivityResult(ActivityResultContracts.RequestMultiplePermissions()) {
        // notifications and location are optional; the service reports a missing camera permission itself
        DetectorService.start(this)
    }

    private val settingsScreen = registerForActivityResult(ActivityResultContracts.StartActivityForResult()) { r ->
        if (r.data?.getBooleanExtra(SettingsActivity.EXTRA_CHANGED, false) == true) {
            refreshAll()
            if (PipelineBus.running) {
                Toast.makeText(this, "Restarting with the new settings", Toast.LENGTH_SHORT).show()
                requestStart()
            }
        }
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        WindowCompat.setDecorFitsSystemWindows(window, false)
        window.addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON)
        setContentView(R.layout.activity_main)
        bindViews()
        PipelineBus.addListener(this)
        // a recreated activity (rotation) must not apply the launch extras again: they would restart the detector
        if (savedInstanceState == null) handleLaunch(intent)
        refreshAll()
    }

    override fun onNewIntent(intent: Intent) {
        super.onNewIntent(intent)
        setIntent(intent)
        handleLaunch(intent)
        refreshAll()
    }

    override fun onResume() {
        super.onResume()
        refreshAll()
    }

    override fun onDestroy() {
        PipelineBus.removeListener(this)
        super.onDestroy()
    }

    private fun bindViews() {
        val root = findViewById<View>(R.id.root)
        ViewCompat.setOnApplyWindowInsetsListener(root) { v, insets ->
            val b = insets.getInsets(WindowInsetsCompat.Type.systemBars() or WindowInsetsCompat.Type.displayCutout())
            v.setPadding(b.left, b.top, b.right, b.bottom)
            WindowInsetsCompat.CONSUMED
        }
        overlay = findViewById(R.id.overlay)
        statePill = findViewById(R.id.statePill)
        fpsPill = findViewById(R.id.fpsPill)
        modePill = findViewById(R.id.modePill)
        btnDryRun = findViewById(R.id.btnDryRun)
        btnStartStop = findViewById(R.id.btnStartStop)
        sourceGroup = findViewById(R.id.sourceGroup)
        clipCard = findViewById(R.id.clipCard)
        clipName = findViewById(R.id.clipName)
        clipInfo = findViewById(R.id.clipInfo)
        lastReportText = findViewById(R.id.lastReportText)
        lastReportSub = findViewById(R.id.lastReportSub)
        lastReportTime = findViewById(R.id.lastReportTime)
        healthText = findViewById(R.id.healthText)
        batteryHint = findViewById(R.id.batteryHint)
        wazeHint = findViewById(R.id.wazeHint)
        reportsEmpty = findViewById(R.id.reportsEmpty)
        reportsCount = findViewById(R.id.reportsCount)

        findViewById<RecyclerView>(R.id.reportsList).apply {
            layoutManager = LinearLayoutManager(this@MainActivity)
            adapter = reports
        }
        btnStartStop.setOnClickListener {
            if (PipelineBus.running) DetectorService.stop(this) else requestStart()
        }
        findViewById<View>(R.id.btnSettings).setOnClickListener {
            settingsScreen.launch(Intent(this, SettingsActivity::class.java))
        }
        btnDryRun.setOnClickListener { toggleDryRun() }
        modePill.setOnClickListener { toggleDryRun() }
        sourceGroup.addOnButtonCheckedListener { _, id, checked ->
            if (!checked || updatingUi) return@addOnButtonCheckedListener
            val src = when (id) {
                R.id.btnVideo -> "video"
                R.id.btnTcp -> "tcp"
                else -> "camera"
            }
            changeSource(src)
        }
        val openClips = View.OnClickListener { clipPicker.show(Settings.load(this).video) }
        clipCard.setOnClickListener(openClips)
        findViewById<View>(R.id.btnChooseClip).setOnClickListener(openClips)
        healthText.setOnClickListener {
            healthText.maxLines = if (healthText.maxLines > 3) 3 else 30
        }

        setupHint(batteryHint, R.drawable.ic_battery,
            "Battery is restricted: set it to Unrestricted so detection keeps running behind Waze", "Fix") {
            openSystemScreen(android.provider.Settings.ACTION_IGNORE_BATTERY_OPTIMIZATION_SETTINGS, "battery settings")
        }
        setupHint(wazeHint, R.drawable.ic_eye,
            "Waze reader is off: turn on \"Wazeomatic Waze reader\" in Accessibility to skip alerts Waze already has", "Open") {
            openSystemScreen(android.provider.Settings.ACTION_ACCESSIBILITY_SETTINGS, "accessibility settings")
        }
    }

    private fun setupHint(card: View, icon: Int, text: String, button: String, onClick: () -> Unit) {
        card.findViewById<android.widget.ImageView>(R.id.hintIcon).setImageResource(icon)
        card.findViewById<TextView>(R.id.hintText).text = text
        card.findViewById<MaterialButton>(R.id.hintButton).apply {
            this.text = button
            setOnClickListener { onClick() }
        }
    }

    private fun openSystemScreen(action: String, what: String) {
        try {
            startActivity(Intent(action))
        } catch (e: Exception) {
            Toast.makeText(this, "Could not open the $what: ${e.message}", Toast.LENGTH_LONG).show()
        }
    }

    // ---- actions ----

    private fun save(change: (Settings) -> Settings) {
        change(Settings.load(this)).save(this)
    }

    private fun restartIfRunning(note: String) {
        if (!PipelineBus.running) return
        Toast.makeText(this, note, Toast.LENGTH_SHORT).show()
        requestStart()
    }

    private fun changeSource(src: String) {
        val s = Settings.load(this)
        if (s.source == src) return
        save { it.copy(source = src) }
        refreshAll()
        if (src == "video" && !Clips.resolve(this, s.video).isFile) {
            clipPicker.show(s.video)
            return
        }
        restartIfRunning("Restarting with the ${sourceLabel(src)}")
    }

    private fun chooseClip(name: String) {
        save { it.copy(source = "video", video = name) }
        refreshAll()
        restartIfRunning("Restarting with $name")
    }

    /** Dry run on is one tap; going live (real Waze reports) asks first. */
    private fun toggleDryRun() {
        val s = Settings.load(this)
        if (!s.dryRun) {
            save { it.copy(dryRun = true) }
            refreshAll()
            restartIfRunning("Dry run on: restarting")
            return
        }
        MaterialAlertDialogBuilder(this)
            .setTitle("Send real reports to Waze?")
            .setMessage("With dry run off, the macro submits every report in Waze for real, where other drivers see it.")
            .setNegativeButton("Keep dry run", null)
            .setPositiveButton("Go live") { _, _ ->
                save { it.copy(dryRun = false) }
                refreshAll()
                restartIfRunning("Live reports: restarting")
            }
            .show()
    }

    private fun handleLaunch(i: Intent?) {
        if (i == null) return
        importFromIntent(i)
        val extras = i.extras ?: return
        if (extras.getBoolean("stop", false)) {
            DetectorService.stop(this)
            return
        }
        if (extras.getBoolean("reset_settings", false)) Settings.reset(this)
        val touched = extras.keySet().any { it in Settings.KNOWN_KEYS }
        if (touched) Settings.load(this).withExtras(extras).save(this)
        if (extras.containsKey("source") || extras.getBoolean("autostart", false)) requestStart()
    }

    /** "Share" or "Open with" on a video: copy it into the clips folder and select it. */
    private fun importFromIntent(i: Intent) {
        val uri: Uri? = when (i.action) {
            Intent.ACTION_VIEW -> i.data
            Intent.ACTION_SEND -> IntentCompat.getParcelableExtra(i, Intent.EXTRA_STREAM, Uri::class.java)
            else -> null
        }
        if (uri != null) clipPicker.import(uri)
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
        if (needed.isEmpty()) DetectorService.start(this) else permissions.launch(needed.toTypedArray())
    }

    // ---- what the screen shows ----

    private fun sourceLabel(src: String) = when (src) {
        "video" -> "video clip"
        "tcp" -> "TCP events"
        else -> "camera"
    }

    private fun refreshAll() {
        val s = Settings.load(this)
        updatingUi = true
        sourceGroup.check(
            when (s.source) {
                "video" -> R.id.btnVideo
                "tcp" -> R.id.btnTcp
                else -> R.id.btnCamera
            }
        )
        updatingUi = false
        refreshClip(s)
        refreshDryRun(s)
        refreshState(s)
        refreshHealth()
        refreshHints(s)
        refreshReports()
    }

    private fun refreshClip(s: Settings) {
        clipCard.visibility = if (s.source == "video") View.VISIBLE else View.GONE
        if (s.source != "video") return
        val f = Clips.resolve(this, s.video)
        clipName.text = s.video
        if (!f.isFile) {
            clipInfo.text = "Not in the app's folder: tap to choose a clip"
            clipInfo.setTextColor(getColor(R.color.state_warn))
            return
        }
        clipInfo.setTextColor(getColor(R.color.md_on_surface_variant))
        clipInfo.text = Clips.describe(f, Clips.cachedInfo(f))
        if (Clips.cachedInfo(f) == null) Thread({
            val info = Clips.info(f)
            runOnUiThread { if (!isDestroyed && clipName.text == s.video) clipInfo.text = Clips.describe(f, info) }
        }, "clip-info").start()
    }

    private fun refreshDryRun(s: Settings) {
        val (fg, bg) = if (s.dryRun) R.color.state_warn to R.color.state_warn_container
        else R.color.state_error to R.color.state_error_container
        btnDryRun.text = if (s.dryRun) "Dry run" else "LIVE"
        btnDryRun.setIconResource(if (s.dryRun) R.drawable.ic_shield else R.drawable.ic_send)
        btnDryRun.setTextColor(getColor(fg))
        btnDryRun.iconTint = ColorStateList.valueOf(getColor(fg))
        btnDryRun.backgroundTintList = ColorStateList.valueOf(getColor(bg))
        btnDryRun.contentDescription = if (s.dryRun) "Dry run is on. Tap to send real reports" else "Live: real reports. Tap for dry run"
        modePill.text = if (s.dryRun) "DRY RUN" else "LIVE REPORTS"
        modePill.setTextColor(getColor(fg))
    }

    private enum class State(val label: String, val color: Int) {
        STOPPED("STOPPED", R.color.state_idle),
        STARTING("STARTING", R.color.state_warn),
        RUNNING("RUNNING", R.color.state_running),
        LISTENING("LISTENING", R.color.state_running),
        FINISHED("CLIP ENDED", R.color.state_warn),
        STALLED("STALLED", R.color.state_error),
        PROBLEM("PROBLEM", R.color.state_error),
    }

    private fun currentState(s: Settings): State {
        val st = PipelineBus.status
        if (!PipelineBus.running) return if (st.startsWith("init failed")) State.PROBLEM else State.STOPPED
        return when {
            st.startsWith("init failed") || st.startsWith("model missing") || "camera permission missing" in st ||
                "video not found" in st || "video error" in st || "no video track" in st -> State.PROBLEM
            "stalled" in st -> State.STALLED
            "video finished" in st -> State.FINISHED
            st.startsWith("loading models") -> State.STARTING
            s.source == "tcp" -> State.LISTENING
            !sawFrame -> State.STARTING
            else -> State.RUNNING
        }
    }

    private fun refreshState(s: Settings) {
        val running = PipelineBus.running
        val state = currentState(s)
        statePill.text = state.label
        TextViewCompat.setCompoundDrawableTintList(statePill, ColorStateList.valueOf(getColor(state.color)))
        if (!running) {
            sawFrame = false
            fpsPill.visibility = View.GONE
        }
        btnStartStop.text = if (running) "Stop" else "Start"
        btnStartStop.setIconResource(if (running) R.drawable.ic_stop else R.drawable.ic_play)
        btnStartStop.backgroundTintList = ColorStateList.valueOf(getColor(if (running) R.color.stop_button else R.color.start_button))
        btnStartStop.setTextColor(getColor(R.color.on_action_button))
        btnStartStop.iconTint = ColorStateList.valueOf(getColor(R.color.on_action_button))

        val hasModel = Pipeline.assetExists(this, "hazard.onnx") || Pipeline.assetExists(this, "coco.onnx")
        when {
            !hasModel && s.source != "tcp" -> { overlay.message = "Model missing"; overlay.hint = "Build with the models in models/ (see README)" }
            !running -> {
                overlay.clear()
                overlay.message = "Press Start"
                overlay.hint = when (s.source) {
                    "video" -> "Video clip: ${s.video}"
                    "tcp" -> "TCP events on port ${s.tcpPort}"
                    else -> "Camera, ${s.maxFps.toString().removeSuffix(".0")} fps cap"
                }
            }
            s.source == "tcp" -> { overlay.message = "Listening for TCP events"; overlay.hint = "Port ${s.tcpPort}, no detector runs" }
            state == State.PROBLEM -> { overlay.message = "Not running"; overlay.hint = "See the health line below" }
            else -> { overlay.message = "Starting…"; overlay.hint = "Loading the models" }
        }
    }

    private val warnWord = Regex("\\b(restricted|stalled|missing|failed|error|not found|portrait|no location permission)\\b")

    /** The service's status line ("models | tcp | video | battery") with the warnings in amber. */
    private fun refreshHealth() {
        val st = PipelineBus.status
        val parts: List<String> = if (PipelineBus.running || st.startsWith("stopped:") || st.startsWith("init failed")) {
            st.split(" | ").map { it.trim() }.filter { it.isNotEmpty() }
        } else {
            modelSummary()
        }
        val b = SpannableStringBuilder()
        val warn = getColor(R.color.state_warn)
        for ((i, p) in parts.withIndex()) {
            if (i > 0) b.append("  ·  ")
            val start = b.length
            b.append(p)
            if (warnWord.containsMatchIn(p)) b.setSpan(ForegroundColorSpan(warn), start, b.length, Spanned.SPAN_EXCLUSIVE_EXCLUSIVE)
        }
        healthText.text = b
    }

    /** Before a start: which models this build carries. */
    private fun modelSummary(): List<String> {
        val models = listOf(
            "hazard.onnx" to "hazard", "hazard2.onnx" to "hazard 2", "coco.onnx" to "coco",
            "vehicle_cls.onnx" to "vehicle", "vehicle_cls2.onnx" to "vehicle 2",
            "weather_cls.onnx" to "weather", "surface_cls.onnx" to "surface",
        )
        val (have, missing) = models.partition { Pipeline.assetExists(this, it.first) }
        val out = ArrayList<String>()
        out.add(if (have.isEmpty()) "no model bundled" else "models: " + have.joinToString(", ") { it.second })
        val optional = missing.filter { it.second in setOf("vehicle", "weather", "surface") }
        if (optional.isNotEmpty()) out.add(optional.joinToString(", ") { it.second } + " missing")
        out.add(if (batteryUnrestricted()) "battery: unrestricted" else "battery: restricted")
        return out
    }

    private fun batteryUnrestricted(): Boolean = try {
        getSystemService(PowerManager::class.java).isIgnoringBatteryOptimizations(packageName)
    } catch (e: Exception) {
        true
    }

    private fun refreshHints(s: Settings) {
        batteryHint.visibility = if (batteryUnrestricted()) View.GONE else View.VISIBLE
        val readerOff = s.wazeReader && !WazeWatcherService.isEnabled(this)
        wazeHint.visibility = if (readerOff) View.VISIBLE else View.GONE
    }

    private fun refreshReports() {
        val list = PipelineBus.recentEvents.toList()
        reports.submit(list)
        reportsEmpty.visibility = if (list.isEmpty()) View.VISIBLE else View.GONE
        reportsCount.text = if (list.isEmpty()) "" else "${list.size} this session"
        val last = list.firstOrNull()
        if (last == null) {
            lastReportText.text = "None yet"
            lastReportSub.visibility = View.GONE
            lastReportTime.text = ""
            return
        }
        val (ev, dry) = last
        lastReportText.text = ReportFormat.title(ev)
        lastReportSub.visibility = View.VISIBLE
        lastReportSub.text = ReportFormat.detail(ev, withTime = false) + if (dry) "  ·  dry run" else "  ·  sent"
        lastReportTime.text = ReportFormat.time(ev)
    }

    // ---- PipelineBus ----

    override fun onFrame(r: FrameResult) {
        if (!PipelineBus.running) return
        overlay.show(r.bitmap, r.detections)
        val ms = r.latencyMs["infer"] ?: r.latencyMs["wall"]
        fpsPill.text = String.format(Locale.US, "%.1f fps", r.fps) + (ms?.let { String.format(Locale.US, "  ·  %.0f ms", it) } ?: "")
        fpsPill.visibility = View.VISIBLE
        if (!sawFrame) {
            sawFrame = true
            refreshState(Settings.load(this))
        }
    }

    override fun onEvent(ev: HazardEvent, dryRun: Boolean) {
        refreshReports()
    }

    override fun onStatus(status: String) {
        val s = Settings.load(this)
        refreshState(s)
        refreshHealth()
    }
}
