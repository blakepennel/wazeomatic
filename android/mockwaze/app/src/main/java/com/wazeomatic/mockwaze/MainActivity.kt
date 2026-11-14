package com.wazeomatic.mockwaze

import android.animation.ValueAnimator
import android.app.Activity
import android.content.BroadcastReceiver
import android.content.Context
import android.content.Intent
import android.content.IntentFilter
import android.graphics.Color
import android.graphics.Typeface
import android.graphics.drawable.GradientDrawable
import android.os.Build
import android.os.Bundle
import android.os.Handler
import android.os.Looper
import android.os.SystemClock
import android.view.Gravity
import android.view.InputDevice
import android.view.MotionEvent
import android.view.View
import android.view.WindowInsets
import android.view.WindowManager
import android.view.animation.LinearInterpolator
import android.widget.FrameLayout
import android.widget.ImageView
import android.widget.LinearLayout
import android.widget.TextView

/**
 * Main screen of the mock: Waze-style idle map, menu button, speedometer, "Where to?" bar and
 * the round report button. The report flow is a bottom sheet with two pages, laid out like the
 * 2024 Waze app:
 *   GRID    "What do you see?"  3 x 4 round tiles
 *   CONFIRM "Report a hazard"   the category's tiles (category itself preselected), Cancel / Report
 * The confirm page sends by itself after auto_send_ms, like Waze does, unless Cancel is tapped.
 * Every view is built in code so the tap targets are easy to find by text/description
 * (MacroDroid "UI interaction" or uiautomator) and by coordinates.
 */
class MainActivity : Activity() {

    enum class Page { NONE, GRID, CONFIRM }

    private companion object {
        /** How long a macro step waits for the app to be resumed and focused before it gives up. */
        const val OBSTRUCTED_TIMEOUT_MS = 2000L
        const val OBSTRUCTED_POLL_MS = 100L
    }

    private val handler = Handler(Looper.getMainLooper())
    private lateinit var root: FrameLayout
    private lateinit var reportButton: ImageView
    private lateinit var menuButton: ImageView
    private lateinit var musicButton: ImageView
    private lateinit var bottomBar: LinearLayout
    private lateinit var scrim: View
    private lateinit var sheet: LinearLayout
    private lateinit var sheetContent: FrameLayout
    private lateinit var sheetHandle: View
    private lateinit var toastView: LinearLayout
    private lateinit var speed: SpeedometerView
    private lateinit var badge: TextView
    private lateinit var alertCard: AlertCard

    var page = Page.NONE
        private set
    private var confirmPath: MenuPath? = null
    /** Tap targets on the current sheet page, by name (tile:Hazard, sub:Pothole, confirm:report ...). */
    private val targets = HashMap<String, View>()
    /** Input is ignored while the sheet is still moving, like a real app mid-animation. */
    private var animEndAt = 0L
    private var macroJob: MacroJob? = null
    private var bottomInset = 0
    private var topInset = 0
    private var countdown: ValueAnimator? = null
    private val hideToast = Runnable { toastView.visibility = View.GONE }

    private val controlReceiver = object : BroadcastReceiver() {
        override fun onReceive(context: Context, intent: Intent) {
            when (intent.action) {
                AlertSpec.ACTION_SHOW -> showAlert(intent)
                AlertSpec.ACTION_CLEAR -> clearAlerts()
                else -> if (Config.applyExtras(context, intent)) AutoMacroService.sync(context)
            }
        }
    }

    // ---------------------------------------------------------------- lifecycle

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        window.addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON)
        goEdgeToEdge()
        buildUi()
        setContentView(root)
        if (savedInstanceState == null) Config.applyExtras(this, intent)
        AutoMacroService.sync(this)
    }

    override fun onNewIntent(intent: Intent) {
        super.onNewIntent(intent)
        setIntent(intent)
        Config.applyExtras(this, intent)
        AutoMacroService.sync(this)
    }

    override fun onStart() {
        super.onStart()
        val f = IntentFilter(Config.ACTION_SET_TIMING).apply {
            addAction(AlertSpec.ACTION_SHOW)
            addAction(AlertSpec.ACTION_CLEAR)
        }
        if (Build.VERSION.SDK_INT >= 33) registerReceiver(controlReceiver, f, Context.RECEIVER_EXPORTED)
        else registerReceiver(controlReceiver, f)
    }

    override fun onStop() {
        unregisterReceiver(controlReceiver)
        // an alert card is part of the map screen; it does not outlive it
        if (alertCard.current != null) alertCard.dismiss("cleared")
        super.onStop()
    }

    override fun onResume() {
        super.onResume()
        MacroEngine.onActivityResumed(this)
    }

    override fun onPause() {
        MacroEngine.onActivityPaused(this)
        super.onPause()
    }

    @Deprecated("Deprecated in Java")
    override fun onBackPressed() {
        when (page) {
            Page.NONE -> @Suppress("DEPRECATION") super.onBackPressed()
            Page.GRID -> closeSheet()
            Page.CONFIRM -> dismissConfirm("close")
        }
    }

    // ---------------------------------------------------------------- layout

    @Suppress("DEPRECATION")
    private fun goEdgeToEdge() {
        // targetSdk 35 is edge-to-edge anyway on Android 15; do the same on older versions
        // so the layout (and therefore tap coordinates) is identical everywhere.
        if (Build.VERSION.SDK_INT >= 30) {
            window.setDecorFitsSystemWindows(false)
        } else {
            window.decorView.systemUiVisibility = View.SYSTEM_UI_FLAG_LAYOUT_STABLE or
                View.SYSTEM_UI_FLAG_LAYOUT_HIDE_NAVIGATION or View.SYSTEM_UI_FLAG_LAYOUT_FULLSCREEN
        }
        window.statusBarColor = Color.TRANSPARENT
        window.navigationBarColor = Color.TRANSPARENT
    }

    private fun buildUi() {
        root = FrameLayout(this)
        root.addView(FakeMapView(this), FrameLayout.LayoutParams(-1, -1))

        // top-left: white rounded square with the hamburger menu (opens the mock's settings)
        menuButton = ImageView(this).apply {
            setImageResource(R.drawable.ic_menu)
            contentDescription = "Menu"
            background = roundRect(Color.WHITE, dpf(16f))
            elevation = dpf(4f)
            setPadding(dp(11), dp(11), dp(11), dp(11))
            setOnClickListener { startActivity(Intent(this@MainActivity, SettingsActivity::class.java)) }
        }
        root.addView(menuButton, FrameLayout.LayoutParams(dp(51), dp(51), Gravity.TOP or Gravity.START))

        // top-right: pink music button (decoration only)
        musicButton = ImageView(this).apply {
            setImageResource(R.drawable.ic_music)
            contentDescription = "Music"
            background = oval(Palette.PINK)
            elevation = dpf(4f)
            setPadding(dp(9), dp(9), dp(9), dp(9))
        }
        root.addView(musicButton, FrameLayout.LayoutParams(dp(40), dp(40), Gravity.TOP or Gravity.END))

        badge = TextView(this).apply {
            text = "MOCK WAZE · nothing is sent"
            setTextColor(Color.WHITE)
            textSize = 10f
            setPadding(dp(8), dp(3), dp(8), dp(3))
            background = roundRect(0xB3202124.toInt(), dpf(8f))
        }
        root.addView(badge, FrameLayout.LayoutParams(-2, -2, Gravity.TOP or Gravity.CENTER_HORIZONTAL))

        // alert card of another driver's report (SHOW_ALERT); added before the sheet so the sheet covers it
        alertCard = AlertCard(this) { spec, why ->
            ReportLog.alert(this, if (why == "cleared") "cleared" else "dismissed", spec, why)
        }
        root.addView(alertCard, FrameLayout.LayoutParams(-1, -2, Gravity.TOP))

        // bottom: white card with the "Where to?" search field
        bottomBar = LinearLayout(this).apply {
            orientation = LinearLayout.HORIZONTAL
            gravity = Gravity.CENTER_VERTICAL
            background = GradientDrawable().apply {
                setColor(Color.WHITE)
                cornerRadii = floatArrayOf(dpf(24f), dpf(24f), dpf(24f), dpf(24f), 0f, 0f, 0f, 0f)
            }
            elevation = dpf(10f)
        }
        val search = LinearLayout(this).apply {
            orientation = LinearLayout.HORIZONTAL
            gravity = Gravity.CENTER_VERTICAL
            setPadding(dp(16), 0, dp(16), 0)
            background = roundRect(0xFFF1F2F4.toInt(), dpf(26f))
        }
        search.addView(ImageView(this).apply { setImageResource(R.drawable.ic_search) }, LinearLayout.LayoutParams(dp(22), dp(22)))
        search.addView(TextView(this).apply {
            text = "Where to?"
            textSize = 17f
            setTextColor(0xFF5F6368.toInt())
            setPadding(dp(12), 0, 0, 0)
        }, LinearLayout.LayoutParams(0, -2, 1f))
        bottomBar.addView(search, LinearLayout.LayoutParams(0, dp(52), 1f))
        root.addView(bottomBar, FrameLayout.LayoutParams(-1, -2, Gravity.BOTTOM))

        speed = SpeedometerView(this)
        root.addView(speed, FrameLayout.LayoutParams(dpf(63.5f).toInt(), dpf(63.5f).toInt(), Gravity.BOTTOM or Gravity.START))

        // round white report button with the yellow "+" triangle
        reportButton = ImageView(this).apply {
            setImageResource(R.drawable.ic_report_fab)
            contentDescription = "Report button"
            background = oval(Color.WHITE)
            elevation = dpf(8f)
            setPadding(dp(12), dp(12), dp(12), dp(12))
            setOnClickListener { if (!animating()) openSheet() }
        }
        root.addView(reportButton, FrameLayout.LayoutParams(dp(68), dp(68), Gravity.BOTTOM or Gravity.END))

        scrim = View(this).apply {
            setBackgroundColor(Color.TRANSPARENT) // Waze does not dim the map behind the sheet
            visibility = View.GONE
            setOnClickListener {
                if (animating()) return@setOnClickListener
                when (page) {
                    Page.GRID -> closeSheet()
                    Page.CONFIRM -> dismissConfirm("close")
                    Page.NONE -> {}
                }
            }
        }
        root.addView(scrim, FrameLayout.LayoutParams(-1, -1))

        sheet = LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL
            isClickable = true // swallow taps so they do not reach the scrim
            background = GradientDrawable().apply {
                setColor(Color.WHITE)
                cornerRadii = floatArrayOf(dpf(28f), dpf(28f), dpf(28f), dpf(28f), 0f, 0f, 0f, 0f)
            }
            elevation = dpf(16f)
            visibility = View.INVISIBLE
        }
        val handle = View(this).apply { background = roundRect(0xFFC4C7CC.toInt(), dpf(2f)) }
        sheet.addView(handle, LinearLayout.LayoutParams(dp(42), dp(4)).apply {
            gravity = Gravity.CENTER_HORIZONTAL; topMargin = dp(10); bottomMargin = 0
        })
        sheetHandle = handle
        sheetContent = FrameLayout(this)
        sheet.addView(sheetContent, LinearLayout.LayoutParams(-1, -2))
        root.addView(sheet, FrameLayout.LayoutParams(-1, -2, Gravity.BOTTOM))

        // "Thanks! You helped nearby Wazers" card
        toastView = LinearLayout(this).apply {
            orientation = LinearLayout.HORIZONTAL
            gravity = Gravity.CENTER_VERTICAL
            contentDescription = "Thanks!"
            setPadding(dp(14), dp(12), dp(20), dp(12))
            background = roundRect(Color.WHITE, dpf(16f))
            elevation = dpf(20f)
            visibility = View.GONE
        }
        toastView.addView(ImageView(this).apply { setImageResource(R.drawable.ic_check_green) }, LinearLayout.LayoutParams(dp(28), dp(28)))
        val toastText = LinearLayout(this).apply { orientation = LinearLayout.VERTICAL; setPadding(dp(12), 0, 0, 0) }
        toastText.addView(TextView(this).apply {
            text = "Thanks!"
            textSize = 16f
            typeface = Typeface.DEFAULT_BOLD
            setTextColor(Palette.INK)
        })
        toastText.addView(TextView(this).apply {
            text = "You helped nearby Wazers"
            textSize = 14f
            setTextColor(0xFF3C4043.toInt())
        })
        toastView.addView(toastText)
        root.addView(toastView, FrameLayout.LayoutParams(-2, -2, Gravity.BOTTOM or Gravity.START))

        root.setOnApplyWindowInsetsListener { _, insets ->
            if (Build.VERSION.SDK_INT >= 30) {
                val i = insets.getInsets(WindowInsets.Type.systemBars())
                topInset = i.top; bottomInset = i.bottom
            } else {
                @Suppress("DEPRECATION")
                topInset = insets.systemWindowInsetTop
                @Suppress("DEPRECATION")
                bottomInset = insets.systemWindowInsetBottom
            }
            applyInsets()
            insets
        }
        applyInsets()
    }

    private fun applyInsets() {
        bottomBar.setPadding(dp(16), dp(12), dp(16), dp(12) + bottomInset)
        sheet.setPadding(0, 0, 0, bottomInset)
        val barH = dp(76) + bottomInset
        (reportButton.layoutParams as FrameLayout.LayoutParams).apply {
            marginEnd = dp(12); bottomMargin = barH + dp(14)
        }
        (speed.layoutParams as FrameLayout.LayoutParams).apply {
            marginStart = dpf(11.6f).toInt(); bottomMargin = barH + dp(14)
        }
        (menuButton.layoutParams as FrameLayout.LayoutParams).apply {
            marginStart = dpf(14.8f).toInt(); topMargin = topInset + dp(16)
        }
        (musicButton.layoutParams as FrameLayout.LayoutParams).apply {
            marginEnd = dpf(11.6f).toInt(); topMargin = topInset + dp(16)
        }
        (badge.layoutParams as FrameLayout.LayoutParams).topMargin = topInset + dp(22)
        // below the 51 dp menu button with a gap, far above the report button and the sheet
        (alertCard.layoutParams as FrameLayout.LayoutParams).apply {
            topMargin = topInset + dp(84); marginStart = dp(14); marginEnd = dp(14)
        }
        (toastView.layoutParams as FrameLayout.LayoutParams).apply {
            marginStart = dp(16); bottomMargin = barH + dp(12)
        }
        root.requestLayout()
    }

    // ---------------------------------------------------------------- alert cards

    private fun showAlert(intent: Intent) {
        val spec = AlertSpec.fromIntent(intent)
        if (spec == null) {
            @Suppress("DEPRECATION")
            ReportLog.alert(this, "ignored", null, "unknown alert_type '${intent.extras?.get("alert_type")}'")
            return
        }
        alertCard.show(spec)
        ReportLog.alert(this, "shown", spec)
    }

    private fun clearAlerts() {
        if (alertCard.current != null) alertCard.dismiss("cleared")
        else ReportLog.alert(this, "cleared", null, "no alert on screen")
    }

    // ---------------------------------------------------------------- sheet

    private fun animating() = SystemClock.uptimeMillis() < animEndAt

    private fun markAnim(ms: Long) {
        animEndAt = SystemClock.uptimeMillis() + ms
    }

    private fun openSheet() {
        if (page != Page.NONE) return
        val ms = Config.int(this, "sheet_ms").toLong()
        showGrid(fade = false)
        scrim.alpha = 0f
        scrim.visibility = View.VISIBLE
        scrim.animate().alpha(1f).setDuration(ms).start()
        sheet.visibility = View.VISIBLE
        sheet.translationY = root.height.toFloat()
        sheet.animate().translationY(0f).setDuration(ms).start()
        markAnim(ms)
    }

    private fun closeSheet() {
        if (page == Page.NONE) return
        stopCountdown()
        val ms = Config.int(this, "sheet_ms").toLong()
        page = Page.NONE
        confirmPath = null
        targets.clear()
        scrim.animate().alpha(0f).setDuration(ms).withEndAction { scrim.visibility = View.GONE }.start()
        sheet.animate().translationY(sheet.height.toFloat()).setDuration(ms).withEndAction {
            if (page == Page.NONE) {
                sheet.visibility = View.INVISIBLE
                sheetContent.removeAllViews()
                (sheetContent.layoutParams as LinearLayout.LayoutParams).height = -2
            }
        }.start()
        markAnim(ms)
    }

    private fun setPage(p: Page, content: View, newTargets: Map<String, View>, fade: Boolean = true) {
        stopCountdown()
        sheetContent.removeAllViews()
        sheetContent.addView(content, FrameLayout.LayoutParams(-1, -2))
        targets.clear()
        targets.putAll(newTargets)
        page = p
        val ms = Config.int(this, "page_ms").toLong()
        if (fade && ms > 0) {
            content.alpha = 0f
            content.animate().alpha(1f).setDuration(ms).start()
            markAnim(ms)
        }
    }

    private fun header(text: String, t: HashMap<String, View>, withHandle: Boolean, onClose: () -> Unit): View {
        sheetHandle.visibility = if (withHandle) View.VISIBLE else View.GONE
        val h = LinearLayout(this).apply {
            orientation = LinearLayout.HORIZONTAL
            gravity = Gravity.CENTER_VERTICAL
            setPadding(dp(16), if (withHandle) 0 else dp(14), dp(3), dp(3))
        }
        h.addView(TextView(this).apply {
            this.text = text
            textSize = 22f
            typeface = Typeface.DEFAULT_BOLD
            setTextColor(Palette.INK)
        }, LinearLayout.LayoutParams(0, -2, 1f))
        val close = ImageView(this).apply {
            setImageResource(R.drawable.ic_close)
            contentDescription = "Close"
            setPadding(dp(8), dp(8), dp(8), dp(8)) // 28 dp drawable -> 12 dp glyph, like Waze
            setOnClickListener { if (!animating()) onClose() }
        }
        t["close"] = close
        h.addView(close, LinearLayout.LayoutParams(dp(44), dp(44)))
        return h
    }

    private fun grid(options: List<Option>, keyPrefix: String, t: HashMap<String, View>, onTap: (Option, Tile) -> Unit): LinearLayout {
        val col = LinearLayout(this).apply { orientation = LinearLayout.VERTICAL }
        for (row in options.chunked(3)) {
            val r = LinearLayout(this).apply { orientation = LinearLayout.HORIZONTAL }
            for (opt in row) {
                val tile = Tile(this, opt.label, opt.icon)
                tile.setOnClickListener { if (!animating()) onTap(opt, tile) }
                t["$keyPrefix${opt.label}"] = tile
                r.addView(tile, LinearLayout.LayoutParams(0, -2, 1f))
            }
            repeat(3 - row.size) { r.addView(View(this), LinearLayout.LayoutParams(0, 1, 1f)) }
            col.addView(r, LinearLayout.LayoutParams(-1, -2))
        }
        return col
    }

    private fun showGrid(fade: Boolean = true) {
        val t = HashMap<String, View>()
        // Waze shows three rows of tiles and the fourth row needs a scroll; the sheet is 424 dp
        // tall above the navigation bar (handle area included)
        val col = LinearLayout(this).apply { orientation = LinearLayout.VERTICAL }
        col.addView(header("What do you see?", t, withHandle = true) { closeSheet() }, LinearLayout.LayoutParams(-1, -2))
        val tiles = grid(WazeMenu.categories.map { Option(it.label, it.icon) }, "tile:", t) { opt, _ ->
            WazeMenu.category(opt.label)?.let { showConfirm(MenuPath(it.label)) }
        }
        tiles.setPadding(dpf(14.6f).toInt(), 0, dpf(14.6f).toInt(), 0)
        val scroll = android.widget.ScrollView(this).apply {
            isVerticalScrollBarEnabled = false
            addView(tiles, FrameLayout.LayoutParams(-1, -2))
        }
        col.addView(scroll, LinearLayout.LayoutParams(-1, 0, 1f))
        setPage(Page.GRID, col, t, fade)
        (sheetContent.layoutParams as LinearLayout.LayoutParams).height = dp(424) - dp(14)
        sheetContent.requestLayout()
    }

    private fun showConfirm(path: MenuPath) {
        val t = HashMap<String, View>()
        confirmPath = path
        val cat = WazeMenu.category(path.category)!!
        (sheetContent.layoutParams as LinearLayout.LayoutParams).height = -2
        val col = LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL
            setPadding(0, 0, 0, dp(15))
        }
        col.addView(header(cat.sheetTitle, t, withHandle = false) { dismissConfirm("close") }, LinearLayout.LayoutParams(-1, -2))

        lateinit var refresh: () -> Unit
        val tiles = grid(cat.options, "sub:", t) { opt, _ ->
            val cur = confirmPath ?: return@grid
            confirmPath = MenuPath(cur.category, if (opt.label == cat.label) "" else opt.label)
            refresh()
        }
        refresh = {
            val p = confirmPath
            for (opt in cat.options) {
                val chosen = p != null && (if (opt.label == cat.label) p.item.isEmpty() else p.item == opt.label)
                (t["sub:${opt.label}"] as? Tile)?.setChosen(chosen)
            }
        }
        refresh()
        tiles.setPadding(dpf(14.6f).toInt(), 0, dpf(14.6f).toInt(), 0)
        col.addView(tiles, LinearLayout.LayoutParams(-1, -2))

        // Cancel 187 dp, 21.5 dp gap, Report 173 dp, both 44 dp tall, 15 dp side margins
        val buttons = LinearLayout(this).apply {
            orientation = LinearLayout.HORIZONTAL
            setPadding(dp(15), dp(21), dp(15), 0)
        }
        val cancel = TextView(this).apply {
            text = "Cancel"
            contentDescription = "Cancel"
            textSize = 17f
            typeface = Typeface.DEFAULT_BOLD
            gravity = Gravity.CENTER
            setTextColor(Palette.SELECT)
            background = roundRect(Palette.CANCEL_BG, dpf(22f))
            setOnClickListener { if (!animating()) onCancel("button") }
        }
        val report = ReportButton(this).apply {
            setOnClickListener { if (!animating()) onSend("button") }
        }
        buttons.addView(cancel, LinearLayout.LayoutParams(0, dp(44), 187.4f).apply { marginEnd = dpf(10.75f).toInt() })
        buttons.addView(report, LinearLayout.LayoutParams(0, dp(44), 173.1f).apply { marginStart = dpf(10.75f).toInt() })
        col.addView(buttons, LinearLayout.LayoutParams(-1, -2))
        t["confirm:report"] = report
        t["confirm:cancel"] = cancel
        setPage(Page.CONFIRM, col, t)
        startCountdown(report)
    }

    /** Waze sends the report by itself a few seconds after the confirm sheet opens. */
    private fun startCountdown(button: ReportButton) {
        val ms = Config.int(this, "auto_send_ms").toLong()
        if (ms <= 0) return
        countdown = ValueAnimator.ofFloat(0f, 1f).apply {
            duration = ms
            interpolator = LinearInterpolator()
            addUpdateListener { button.progress = it.animatedValue as Float }
            doOnEndOnce { if (page == Page.CONFIRM) onSend("timeout") }
            start()
        }
    }

    private fun ValueAnimator.doOnEndOnce(block: () -> Unit) {
        addListener(object : android.animation.AnimatorListenerAdapter() {
            private var cancelled = false
            override fun onAnimationCancel(animation: android.animation.Animator) { cancelled = true }
            override fun onAnimationEnd(animation: android.animation.Animator) { if (!cancelled) block() }
        })
    }

    private fun stopCountdown() {
        countdown?.cancel()
        countdown = null
    }

    private fun via() = if (macroJob != null) "auto_macro" else "touch"

    private fun onSend(trigger: String) {
        val p = confirmPath ?: return
        stopCountdown()
        ReportLog.report(this, p, "send", via(), macroJob, trigger)
        closeSheet()
        toastView.visibility = View.VISIBLE
        handler.removeCallbacks(hideToast)
        handler.postDelayed(hideToast, Config.int(this, "toast_ms").toLong())
        // a report that went out by itself while the macro was still on the sheet is a macro bug
        macroJob?.takeIf { trigger == "timeout" }?.let { failMacro(it, "auto send fired after auto_send_ms before the macro tapped") }
    }

    private fun onCancel(trigger: String) {
        val p = confirmPath ?: return
        stopCountdown()
        // Reaching the confirm sheet and backing out = dry-run rehearsal, nothing is reported.
        ReportLog.report(this, p, "cancel", via(), macroJob, trigger)
        closeSheet()
    }

    /** X, back or scrim on the confirm sheet: same as Cancel, logged with its own trigger. */
    private fun dismissConfirm(trigger: String) = onCancel(trigger)

    // ---------------------------------------------------------------- auto macro

    /**
     * Replays the MacroDroid macro: tap, wait, check, tap, ... with the configured waits.
     * Taps are touch events at the centre of the target view, dispatched in-process to this
     * window's decor view. A tap during an animation or on a view covered by another view of
     * this window misses like a coordinate tap would. A window on top of the app (system
     * dialog, another activity) does not intercept such an event, whereas a MacroDroid tap at
     * screen coordinates would land on that window. So every step first requires that the
     * activity is resumed and has window focus (see whenUnobstructed) and the macro ends with
     * outcome "obstructed" when it does not.
     */
    fun runMacro(job: MacroJob) {
        macroJob = job
        val cat = WazeMenu.category(job.path.category)
        if (cat == null) {
            endMacro(job, "failed", "unknown category ${job.path.category}")
            return
        }
        val steps = ArrayList<Pair<String, Page>>()
        steps += "report_button" to Page.GRID
        steps += "tile:${cat.label}" to Page.CONFIRM
        if (job.path.item.isNotEmpty()) steps += "sub:${job.path.item}" to Page.CONFIRM
        steps += (if (job.dryRun) "confirm:cancel" else "confirm:report") to Page.NONE
        if (page != Page.NONE) closeSheet() // macro starts from the map screen
        handler.postDelayed({ step(job, steps, 0) }, Config.int(this, "macro_start_delay_ms").toLong())
    }

    private fun step(job: MacroJob, steps: List<Pair<String, Page>>, i: Int) {
        if (isFinishing || isDestroyed) { endMacro(job, "failed", "activity closed"); return }
        if (macroJob !== job) return // already ended (for example by an auto send)
        if (i >= steps.size) {
            endMacro(job, if (job.dryRun) "dry_run_cancelled" else "sent")
            return
        }
        val (key, expect) = steps[i]
        val wait = Config.int(this, "macro_wait_ms").toLong()
        val before = if (key == "confirm:cancel") Config.int(this, "dry_run_hold_ms").toLong() else 0L
        handler.postDelayed({
            whenUnobstructed(job, key, SystemClock.uptimeMillis() + OBSTRUCTED_TIMEOUT_MS) {
                val v = if (key == "report_button") reportButton else targets[key]
                if (v == null || !v.isShown) {
                    failMacro(job, "'$key' not on screen (page=$page)")
                    return@whenUnobstructed
                }
                tap(v)
                handler.postDelayed({
                    if (macroJob !== job) return@postDelayed
                    if (page != expect) failMacro(job, "after tapping '$key' expected $expect but page=$page")
                    else if (key.startsWith("sub:") && confirmPath?.item != key.removePrefix("sub:"))
                        failMacro(job, "tapped '$key' but selected tile is '${confirmPath?.item}'")
                    else step(job, steps, i + 1)
                }, wait)
            }
        }, before)
    }

    /**
     * Runs [action] once this activity is resumed and its window has input focus, i.e. a tap
     * at screen coordinates would reach it. Polls for up to OBSTRUCTED_TIMEOUT_MS (focus
     * arrives a moment after onResume), then ends the macro with outcome "obstructed".
     */
    private fun whenUnobstructed(job: MacroJob, key: String, deadline: Long, action: () -> Unit) {
        if (macroJob !== job) return
        if (isFinishing || isDestroyed) { endMacro(job, "failed", "activity closed"); return }
        val resumed = MacroEngine.foreground === this
        val focused = hasWindowFocus()
        if (resumed && focused) {
            action()
            return
        }
        if (SystemClock.uptimeMillis() >= deadline) {
            if (page != Page.NONE) closeSheet()
            endMacro(job, "obstructed", "step '$key': " +
                (if (!resumed) "activity not resumed" else "window has no focus (another window is on top)") +
                " for $OBSTRUCTED_TIMEOUT_MS ms, tap not sent")
            return
        }
        handler.postDelayed({ whenUnobstructed(job, key, deadline, action) }, OBSTRUCTED_POLL_MS)
    }

    private fun tap(v: View) {
        val loc = IntArray(2)
        v.getLocationInWindow(loc)
        val x = loc[0] + v.width / 2f
        val y = loc[1] + v.height / 2f
        val t = SystemClock.uptimeMillis()
        val down = MotionEvent.obtain(t, t, MotionEvent.ACTION_DOWN, x, y, 0)
        down.source = InputDevice.SOURCE_TOUCHSCREEN
        window.decorView.dispatchTouchEvent(down)
        down.recycle()
        val up = MotionEvent.obtain(t, t + 50, MotionEvent.ACTION_UP, x, y, 0)
        up.source = InputDevice.SOURCE_TOUCHSCREEN
        window.decorView.dispatchTouchEvent(up)
        up.recycle()
    }

    private fun failMacro(job: MacroJob, detail: String) {
        if (page != Page.NONE) closeSheet()
        endMacro(job, "failed", detail)
    }

    private fun endMacro(job: MacroJob, outcome: String, detail: String = "") {
        if (macroJob === job) macroJob = null
        MacroEngine.finished(this, job, outcome, detail)
    }
}
