package com.wazeomatic.mockwaze

import android.content.Context
import android.content.Intent
import android.graphics.Typeface
import android.os.Handler
import android.os.Looper
import android.view.Gravity
import android.view.View
import android.widget.ImageView
import android.widget.LinearLayout
import android.widget.TextView

/** One alert card request (extras of SHOW_ALERT). */
data class AlertSpec(
    val alertType: String,
    val title: String,
    val distance: String,
    val minutesAgo: Int,
    val stillThere: Boolean,
    val durationMs: Long,
) {
    companion object {
        const val ACTION_SHOW = "com.wazeomatic.mockwaze.SHOW_ALERT"
        const val ACTION_CLEAR = "com.wazeomatic.mockwaze.CLEAR_ALERTS"
        const val DEFAULT_DURATION_MS = 8000L

        /** null when alert_type is missing or not a schema.py key. */
        @Suppress("DEPRECATION")
        fun fromIntent(intent: Intent): AlertSpec? {
            val ex = intent.extras ?: return null
            val type = ex.get("alert_type")?.toString()?.trim() ?: return null
            val title = WazeMenu.alertTitles[type] ?: return null
            fun int(k: String): Long? = when (val v = ex.get(k)) {
                is Int -> v.toLong()
                is Long -> v
                is String -> v.trim().toLongOrNull()
                else -> null
            }
            return AlertSpec(
                alertType = type,
                title = title,
                distance = ex.get("distance")?.toString()?.trim().orEmpty(),
                minutesAgo = (int("minutes_ago") ?: 0L).toInt().coerceAtLeast(0),
                stillThere = Config.parseBool(ex.get("still_there")) ?: false,
                durationMs = (int("duration_ms") ?: DEFAULT_DURATION_MS).coerceAtLeast(0L),
            )
        }
    }
}

/**
 * Waze-style card of a report another driver filed: icon, title, distance, "Reported N min ago" and,
 * for a "Still there?" prompt, Yes / No buttons that only dismiss the card. Every text is a TextView so
 * the accessibility tree (uiautomator, the detector's Waze reader) sees it. It sits below the top-left
 * menu button, well above the report button and the bottom sheet, and is drawn under the sheet.
 */
class AlertCard(ctx: Context, private val onDismiss: (AlertSpec, String) -> Unit) : LinearLayout(ctx) {
    private val handler = Handler(Looper.getMainLooper())
    private val icon = ImageView(ctx).apply { importantForAccessibility = View.IMPORTANT_FOR_ACCESSIBILITY_NO }
    private val title = text(ctx, 18f, bold = true, color = Palette.INK)
    private val distance = text(ctx, 15f, bold = false, color = 0xFF3C4043.toInt())
    private val reported = text(ctx, 14f, bold = false, color = 0xFF5F6368.toInt())
    private val stillRow = LinearLayout(ctx).apply { orientation = HORIZONTAL; gravity = Gravity.CENTER_VERTICAL }
    private val stillText = text(ctx, 15f, bold = true, color = Palette.INK).apply { text = "Still there?" }
    private val yes = button(ctx, "Yes")
    private val no = button(ctx, "No")
    private val hide = Runnable { current?.let { dismiss("timeout") } }

    var current: AlertSpec? = null
        private set

    init {
        orientation = HORIZONTAL
        gravity = Gravity.TOP
        background = roundRect(android.graphics.Color.WHITE, ctx.dpf(16f))
        elevation = ctx.dpf(6f)
        setPadding(ctx.dp(14), ctx.dp(12), ctx.dp(14), ctx.dp(12))
        visibility = View.GONE
        isClickable = true   // taps on the card do not fall through to the map
        addView(icon, LayoutParams(ctx.dp(40), ctx.dp(40)).apply { marginEnd = ctx.dp(12) })
        val col = LinearLayout(ctx).apply { orientation = VERTICAL }
        col.addView(title)
        col.addView(distance)
        col.addView(reported)
        stillRow.addView(stillText, LayoutParams(0, -2, 1f))
        stillRow.addView(yes, LayoutParams(ctx.dp(64), ctx.dp(36)).apply { marginEnd = ctx.dp(8) })
        stillRow.addView(no, LayoutParams(ctx.dp(64), ctx.dp(36)))
        col.addView(stillRow, LayoutParams(-1, -2))
        addView(col, LayoutParams(0, -2, 1f))
        yes.setOnClickListener { dismiss("yes") }
        no.setOnClickListener { dismiss("no") }
    }

    fun show(spec: AlertSpec) {
        handler.removeCallbacks(hide)
        current = spec
        icon.setImageResource(WazeMenu.iconFor(spec.alertType))
        title.text = spec.title
        distance.text = spec.distance
        distance.visibility = if (spec.distance.isEmpty()) View.GONE else View.VISIBLE
        reported.text = when (spec.minutesAgo) {
            0 -> "Reported just now"
            else -> "Reported ${spec.minutesAgo} min ago"
        }
        stillRow.visibility = if (spec.stillThere) View.VISIBLE else View.GONE
        visibility = View.VISIBLE
        if (spec.durationMs > 0) handler.postDelayed(hide, spec.durationMs)
    }

    /** Hides the card; [why] is yes / no / timeout / cleared. */
    fun dismiss(why: String) {
        handler.removeCallbacks(hide)
        val spec = current ?: return
        current = null
        visibility = View.GONE
        onDismiss(spec, why)
    }

    private companion object {
        fun text(ctx: Context, size: Float, bold: Boolean, color: Int) = TextView(ctx).apply {
            textSize = size
            if (bold) typeface = Typeface.DEFAULT_BOLD
            setTextColor(color)
        }

        fun button(ctx: Context, label: String) = TextView(ctx).apply {
            text = label
            contentDescription = label
            textSize = 15f
            typeface = Typeface.DEFAULT_BOLD
            gravity = Gravity.CENTER
            setTextColor(Palette.SELECT)
            background = roundRect(Palette.CANCEL_BG, ctx.dpf(18f))
        }
    }
}
