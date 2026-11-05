package com.wazeomatic.mockwaze

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
import android.widget.FrameLayout
import android.widget.ImageView
import android.widget.LinearLayout
import android.widget.ScrollView
import android.widget.TextView

/**
 * Main screen of the mock: fake map, bottom bar and the orange report button.
 * The report flow lives in a bottom sheet with two pages: GRID -> CONFIRM. For Hazard the
 * confirm sheet also holds the subtype tiles (flat list, Hazard itself preselected).
 * Every view is built in code so the tap targets are easy to find by text/description
 * (MacroDroid "UI interaction" or uiautomator) and by coordinates.
 */
class MainActivity : Activity() {

    enum class Page { NONE, GRID, CONFIRM }

    private val handler = Handler(Looper.getMainLooper())
    private lateinit var root: FrameLayout
    private lateinit var reportButton: ImageView
    private lateinit var settingsButton: TextView
    private lateinit var bottomBar: LinearLayout
    private lateinit var scrim: View
    private lateinit var sheet: LinearLayout
    private lateinit var sheetContent: FrameLayout
    private lateinit var toastView: TextView
    private lateinit var speed: TextView
    private lateinit var badge: TextView

    var page = Page.NONE
        private set
    private var confirmPath: MenuPath? = null
    /** Tap targets on the current sheet page, by name (tile:Hazard, sub:Pothole, confirm:send ...). */
    private val targets = HashMap<String, View>()
    /** Input is ignored while the sheet is still moving, like a real app mid-animation. */
    private var animEndAt = 0L
    private var macroJob: MacroJob? = null
    private var bottomInset = 0
    private var topInset = 0
    private val hideToast = Runnable { toastView.visibility = View.GONE }

    private val controlReceiver = object : BroadcastReceiver() {
        override fun onReceive(context: Context, intent: Intent) {
            if (Config.applyExtras(context, intent)) AutoMacroService.sync(context)
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
        val f = IntentFilter(Config.ACTION_SET_TIMING)
        if (Build.VERSION.SDK_INT >= 33) registerReceiver(controlReceiver, f, Context.RECEIVER_EXPORTED)
        else registerReceiver(controlReceiver, f)
    }

    override fun onStop() {
        unregisterReceiver(controlReceiver)
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
            Page.CONFIRM -> showGrid()
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

        settingsButton = TextView(this).apply {
            text = "Settings"
            contentDescription = "Settings"
            setTextColor(0xFF1F2937.toInt())
            textSize = 14f
            setPadding(dp(14), dp(8), dp(14), dp(8))
            background = roundRect(Color.WHITE, dpf(18f))
            elevation = dpf(4f)
            setOnClickListener { startActivity(Intent(this@MainActivity, SettingsActivity::class.java)) }
        }
        root.addView(settingsButton, FrameLayout.LayoutParams(-2, -2, Gravity.TOP or Gravity.START))

        badge = TextView(this).apply {
            text = "MOCK WAZE - test build, nothing is sent"
            setTextColor(Color.WHITE)
            textSize = 11f
            setPadding(dp(10), dp(4), dp(10), dp(4))
            background = roundRect(0xCC111827.toInt(), dpf(10f))
        }
        root.addView(badge, FrameLayout.LayoutParams(-2, -2, Gravity.TOP or Gravity.CENTER_HORIZONTAL))

        // bottom bar: "Where to?" search pill like the Waze home screen
        bottomBar = LinearLayout(this).apply {
            orientation = LinearLayout.HORIZONTAL
            gravity = Gravity.CENTER_VERTICAL
            background = GradientDrawable().apply {
                setColor(Color.WHITE)
                cornerRadii = floatArrayOf(dpf(20f), dpf(20f), dpf(20f), dpf(20f), 0f, 0f, 0f, 0f)
            }
            elevation = dpf(8f)
            setPadding(dp(16), dp(12), dp(16), dp(12))
        }
        val search = TextView(this).apply {
            text = "Where to?"
            textSize = 17f
            setTextColor(0xFF4B5563.toInt())
            setPadding(dp(16), 0, dp(16), 0)
            gravity = Gravity.CENTER_VERTICAL
            background = roundRect(0xFFF1F3F5.toInt(), dpf(24f))
        }
        bottomBar.addView(search, LinearLayout.LayoutParams(0, dp(48), 1f))
        val myWaze = TextView(this).apply {
            text = "My Waze"
            textSize = 13f
            setTextColor(0xFF2F80ED.toInt())
            typeface = Typeface.DEFAULT_BOLD
            setPadding(dp(14), 0, 0, 0)
        }
        bottomBar.addView(myWaze, LinearLayout.LayoutParams(-2, -2))
        root.addView(bottomBar, FrameLayout.LayoutParams(-1, -2, Gravity.BOTTOM))

        speed = TextView(this).apply {
            text = "0\nkm/h"
            gravity = Gravity.CENTER
            textSize = 12f
            setTextColor(0xFF111827.toInt())
            background = oval(Color.WHITE)
            elevation = dpf(6f)
        }
        root.addView(speed, FrameLayout.LayoutParams(dp(58), dp(58), Gravity.BOTTOM or Gravity.START))

        reportButton = ImageView(this).apply {
            setImageResource(R.drawable.ic_report)
            contentDescription = "Report"
            background = oval(0xFFFF9500.toInt())
            elevation = dpf(8f)
            setPadding(dp(16), dp(16), dp(16), dp(16))
            setOnClickListener { if (!animating()) openSheet() }
        }
        root.addView(reportButton, FrameLayout.LayoutParams(dp(64), dp(64), Gravity.BOTTOM or Gravity.END))

        scrim = View(this).apply {
            setBackgroundColor(0x66000000)
            visibility = View.GONE
            setOnClickListener { if (!animating() && page != Page.NONE) closeSheet() }
        }
        root.addView(scrim, FrameLayout.LayoutParams(-1, -1))

        sheet = LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL
            isClickable = true // swallow taps so they do not reach the scrim
            background = GradientDrawable().apply {
                setColor(Color.WHITE)
                cornerRadii = floatArrayOf(dpf(22f), dpf(22f), dpf(22f), dpf(22f), 0f, 0f, 0f, 0f)
            }
            elevation = dpf(16f)
            visibility = View.INVISIBLE
        }
        val handle = View(this).apply { background = roundRect(0xFFD1D5DB.toInt(), dpf(2f)) }
        sheet.addView(handle, LinearLayout.LayoutParams(dp(40), dp(4)).apply {
            gravity = Gravity.CENTER_HORIZONTAL; topMargin = dp(10); bottomMargin = dp(4)
        })
        sheetContent = FrameLayout(this)
        sheet.addView(sheetContent, LinearLayout.LayoutParams(-1, -2))
        root.addView(sheet, FrameLayout.LayoutParams(-1, -2, Gravity.BOTTOM))

        toastView = TextView(this).apply {
            text = "Thanks!"
            contentDescription = "Thanks!"
            setTextColor(Color.WHITE)
            textSize = 16f
            typeface = Typeface.DEFAULT_BOLD
            setPadding(dp(22), dp(12), dp(22), dp(12))
            background = roundRect(0xEE111827.toInt(), dpf(24f))
            elevation = dpf(20f)
            visibility = View.GONE
        }
        root.addView(toastView, FrameLayout.LayoutParams(-2, -2, Gravity.BOTTOM or Gravity.CENTER_HORIZONTAL))

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
        val barH = dp(72) + bottomInset
        (reportButton.layoutParams as FrameLayout.LayoutParams).apply {
            marginEnd = dp(16); bottomMargin = barH + dp(16)
        }
        (speed.layoutParams as FrameLayout.LayoutParams).apply {
            marginStart = dp(16); bottomMargin = barH + dp(18)
        }
        (settingsButton.layoutParams as FrameLayout.LayoutParams).apply {
            marginStart = dp(16); topMargin = topInset + dp(12)
        }
        (badge.layoutParams as FrameLayout.LayoutParams).topMargin = topInset + dp(60)
        (toastView.layoutParams as FrameLayout.LayoutParams).bottomMargin = barH + dp(100)
        root.requestLayout()
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
        val ms = Config.int(this, "sheet_ms").toLong()
        page = Page.NONE
        confirmPath = null
        targets.clear()
        scrim.animate().alpha(0f).setDuration(ms).withEndAction { scrim.visibility = View.GONE }.start()
        sheet.animate().translationY(sheet.height.toFloat()).setDuration(ms).withEndAction {
            if (page == Page.NONE) {
                sheet.visibility = View.INVISIBLE
                sheetContent.removeAllViews()
            }
        }.start()
        markAnim(ms)
    }

    private fun setPage(p: Page, content: View, newTargets: Map<String, View>, fade: Boolean = true) {
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

    private fun title(text: String, size: Float = 20f) = TextView(this).apply {
        this.text = text
        textSize = size
        typeface = Typeface.DEFAULT_BOLD
        setTextColor(0xFF111827.toInt())
    }

    private fun showGrid(fade: Boolean = true) {
        val t = HashMap<String, View>()
        val col = LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL
            setPadding(dp(12), dp(4), dp(12), dp(12))
        }
        col.addView(title("Report").apply { setPadding(dp(8), dp(4), 0, dp(8)) })
        for (row in WazeMenu.categories.chunked(3)) {
            val r = LinearLayout(this).apply { orientation = LinearLayout.HORIZONTAL }
            for (cat in row) {
                val tile = makeTile(this, cat.label, cat.color, cat.glyph)
                tile.setOnClickListener { if (!animating()) onCategory(cat) }
                t["tile:${cat.label}"] = tile
                r.addView(tile, LinearLayout.LayoutParams(0, -2, 1f))
            }
            col.addView(r)
        }
        setPage(Page.GRID, col, t, fade)
    }

    private fun onCategory(cat: Category) {
        showConfirm(MenuPath(cat.label))
    }

    private fun header(text: String, t: HashMap<String, View>, onBack: () -> Unit): View {
        val h = LinearLayout(this).apply {
            orientation = LinearLayout.HORIZONTAL
            gravity = Gravity.CENTER_VERTICAL
            setPadding(dp(4), 0, dp(8), dp(4))
        }
        val back = TextView(this).apply {
            this.text = "‹"
            contentDescription = "Back"
            textSize = 30f
            setTextColor(0xFF374151.toInt())
            gravity = Gravity.CENTER
            setOnClickListener { if (!animating()) onBack() }
        }
        t["back"] = back
        h.addView(back, LinearLayout.LayoutParams(dp(48), dp(48)))
        h.addView(title(text))
        return h
    }

    private fun showConfirm(path: MenuPath) {
        val t = HashMap<String, View>()
        confirmPath = path
        val cat = WazeMenu.category(path.category)!!
        val col = LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL
            gravity = Gravity.CENTER_HORIZONTAL
            setPadding(dp(16), 0, dp(16), dp(16))
        }
        val hdr = header(cat.label, t) { showGrid() }
        col.addView(hdr, LinearLayout.LayoutParams(-1, -2))
        val icon = TextView(this).apply {
            text = cat.glyph
            textSize = 34f
            typeface = Typeface.DEFAULT_BOLD
            gravity = Gravity.CENTER
            setTextColor(Color.WHITE)
            background = oval(cat.color)
        }
        col.addView(icon, LinearLayout.LayoutParams(dp(80), dp(80)).apply { topMargin = dp(4) })
        val selected = TextView(this).apply {
            textSize = 18f
            setTextColor(0xFF111827.toInt())
            gravity = Gravity.CENTER
            setPadding(0, dp(10), 0, dp(10))
        }
        fun refreshSelected() {
            val p = confirmPath ?: return
            selected.text = if (p.item.isNotEmpty()) p.item else cat.label
            for (sub in WazeMenu.hazardSubtypes) {
                t["sub:$sub"]?.isSelected = sub == p.item
                (t["sub:$sub"] as? LinearLayout)?.getChildAt(0)?.alpha = if (p.item.isEmpty() || sub == p.item) 1f else 0.4f
            }
        }
        col.addView(selected, LinearLayout.LayoutParams(-1, -2))
        if (cat.hasSubmenu) {
            // Flat subtype list; tapping one selects it (tap again to go back to plain Hazard).
            for (row in WazeMenu.hazardSubtypes.chunked(3)) {
                val r = LinearLayout(this).apply { orientation = LinearLayout.HORIZONTAL }
                for (sub in row) {
                    val tile = makeTile(this, sub, cat.color, sub.take(1), iconDp = 50)
                    tile.setOnClickListener {
                        if (!animating()) {
                            val cur = confirmPath ?: return@setOnClickListener
                            confirmPath = MenuPath(cur.category, if (cur.item == sub) "" else sub)
                            refreshSelected()
                        }
                    }
                    t["sub:$sub"] = tile
                    r.addView(tile, LinearLayout.LayoutParams(0, -2, 1f))
                }
                repeat(3 - row.size) { r.addView(android.view.View(this), LinearLayout.LayoutParams(0, 1, 1f)) }
                col.addView(r, LinearLayout.LayoutParams(-1, -2))
            }
        }
        refreshSelected()
        col.addView(android.view.View(this), LinearLayout.LayoutParams(1, dp(14)))
        val buttons = LinearLayout(this).apply { orientation = LinearLayout.HORIZONTAL }
        val cancel = TextView(this).apply {
            text = "Cancel"
            contentDescription = "Cancel"
            textSize = 17f
            gravity = Gravity.CENTER
            setTextColor(0xFF374151.toInt())
            background = roundRect(0xFFE5E7EB.toInt(), dpf(26f))
            setOnClickListener { if (!animating()) onCancel() }
        }
        val send = TextView(this).apply {
            text = "Send"
            contentDescription = "Send"
            textSize = 17f
            typeface = Typeface.DEFAULT_BOLD
            gravity = Gravity.CENTER
            setTextColor(Color.WHITE)
            background = roundRect(0xFF2F80ED.toInt(), dpf(26f))
            setOnClickListener { if (!animating()) onSend() }
        }
        buttons.addView(cancel, LinearLayout.LayoutParams(0, dp(52), 1f).apply { marginEnd = dp(8) })
        buttons.addView(send, LinearLayout.LayoutParams(0, dp(52), 1.4f).apply { marginStart = dp(8) })
        col.addView(buttons, LinearLayout.LayoutParams(-1, -2))
        t["confirm:send"] = send
        t["confirm:cancel"] = cancel
        setPage(Page.CONFIRM, col, t)
    }

    private fun via() = if (macroJob != null) "auto_macro" else "touch"

    private fun onSend() {
        val p = confirmPath ?: return
        ReportLog.report(this, p, "send", via(), macroJob)
        closeSheet()
        toastView.visibility = View.VISIBLE
        handler.removeCallbacks(hideToast)
        handler.postDelayed(hideToast, Config.int(this, "toast_ms").toLong())
    }

    private fun onCancel() {
        val p = confirmPath ?: return
        // Reaching the confirm sheet and backing out = dry-run rehearsal, nothing is reported.
        ReportLog.report(this, p, "cancel", via(), macroJob)
        closeSheet()
    }

    // ---------------------------------------------------------------- auto macro

    /**
     * Replays the MacroDroid macro: tap, wait, check, tap, ... with the configured waits.
     * Taps are real touch events at the centre of the target view, dispatched through the
     * window, so a tap during an animation or on a covered view misses just like it would
     * for a coordinate tap.
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
        steps += (if (job.dryRun) "confirm:cancel" else "confirm:send") to Page.NONE
        if (page != Page.NONE) closeSheet() // macro starts from the map screen
        handler.postDelayed({ step(job, steps, 0) }, Config.int(this, "macro_start_delay_ms").toLong())
    }

    private fun step(job: MacroJob, steps: List<Pair<String, Page>>, i: Int) {
        if (isFinishing || isDestroyed) { endMacro(job, "failed", "activity closed"); return }
        if (i >= steps.size) {
            endMacro(job, if (job.dryRun) "dry_run_cancelled" else "sent")
            return
        }
        val (key, expect) = steps[i]
        val wait = Config.int(this, "macro_wait_ms").toLong()
        val before = if (key == "confirm:cancel") Config.int(this, "dry_run_hold_ms").toLong() else 0L
        handler.postDelayed({
            val v = if (key == "report_button") reportButton else targets[key]
            if (v == null || !v.isShown) {
                failMacro(job, "'$key' not on screen (page=$page)")
                return@postDelayed
            }
            tap(v)
            handler.postDelayed({
                if (page != expect) failMacro(job, "after tapping '$key' expected $expect but page=$page")
                else if (key.startsWith("sub:") && confirmPath?.item != key.removePrefix("sub:"))
                    failMacro(job, "tapped '$key' but selected subtype is '${confirmPath?.item}'")
                else step(job, steps, i + 1)
            }, wait)
        }, before)
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
