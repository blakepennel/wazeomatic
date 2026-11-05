package com.wazeomatic.mockwaze

import android.content.Context
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.graphics.Path
import android.graphics.Typeface
import android.graphics.drawable.GradientDrawable
import android.view.Gravity
import android.view.View
import android.widget.LinearLayout
import android.widget.TextView

fun Context.dp(v: Int): Int = (v * resources.displayMetrics.density + 0.5f).toInt()
fun Context.dpf(v: Float): Float = v * resources.displayMetrics.density

fun roundRect(color: Int, radiusPx: Float) = GradientDrawable().apply {
    setColor(color)
    cornerRadius = radiusPx
}

fun oval(color: Int) = GradientDrawable().apply {
    shape = GradientDrawable.OVAL
    setColor(color)
}

/** Round coloured icon with a label underneath, like the Waze report tiles. */
fun makeTile(ctx: Context, label: String, color: Int, glyph: String, iconDp: Int = 60): LinearLayout {
    val tile = LinearLayout(ctx).apply {
        orientation = LinearLayout.VERTICAL
        gravity = Gravity.CENTER_HORIZONTAL
        setPadding(ctx.dp(4), ctx.dp(8), ctx.dp(4), ctx.dp(8))
        isClickable = true
        isFocusable = true
        contentDescription = label
        val outValue = android.util.TypedValue()
        ctx.theme.resolveAttribute(android.R.attr.selectableItemBackground, outValue, true)
        setBackgroundResource(outValue.resourceId)
    }
    val icon = TextView(ctx).apply {
        text = glyph
        gravity = Gravity.CENTER
        setTextColor(Color.WHITE)
        textSize = 22f
        typeface = Typeface.DEFAULT_BOLD
        background = oval(color)
        importantForAccessibility = View.IMPORTANT_FOR_ACCESSIBILITY_NO
    }
    tile.addView(icon, LinearLayout.LayoutParams(ctx.dp(iconDp), ctx.dp(iconDp)))
    val text = TextView(ctx).apply {
        this.text = label
        gravity = Gravity.CENTER
        setTextColor(0xFF1F2937.toInt())
        textSize = 13f
        maxLines = 2
    }
    tile.addView(text, LinearLayout.LayoutParams(LinearLayout.LayoutParams.MATCH_PARENT,
        LinearLayout.LayoutParams.WRAP_CONTENT).apply { topMargin = ctx.dp(6) })
    return tile
}

/** Static fake map: a few streets, a park, a river, the blue route and the car arrow. */
class FakeMapView(ctx: Context) : View(ctx) {
    private val bg = Paint().apply { color = 0xFFEEF0EA.toInt() }
    private val park = Paint(Paint.ANTI_ALIAS_FLAG).apply { color = 0xFFCDE8C4.toInt() }
    private val water = Paint(Paint.ANTI_ALIAS_FLAG).apply { color = 0xFFAED8F2.toInt() }
    private val block = Paint(Paint.ANTI_ALIAS_FLAG).apply { color = 0xFFE3E4DE.toInt() }
    private val roadEdge = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        color = 0xFFD0D2CC.toInt(); style = Paint.Style.STROKE; strokeCap = Paint.Cap.ROUND
    }
    private val road = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        color = Color.WHITE; style = Paint.Style.STROKE; strokeCap = Paint.Cap.ROUND
    }
    private val highway = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        color = 0xFFFFE08A.toInt(); style = Paint.Style.STROKE; strokeCap = Paint.Cap.ROUND
    }
    private val route = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        color = 0xFF2F80ED.toInt(); style = Paint.Style.STROKE; strokeCap = Paint.Cap.ROUND
        strokeJoin = Paint.Join.ROUND
    }
    private val label = Paint(Paint.ANTI_ALIAS_FLAG).apply { color = 0xFF6B7280.toInt() }
    private val car = Paint(Paint.ANTI_ALIAS_FLAG).apply { color = 0xFF2F80ED.toInt() }
    private val carRing = Paint(Paint.ANTI_ALIAS_FLAG).apply { color = Color.WHITE }

    init {
        contentDescription = "Map"
    }

    override fun onDraw(c: Canvas) {
        val w = width.toFloat()
        val h = height.toFloat()
        c.drawRect(0f, 0f, w, h, bg)
        // blocks
        for (i in 0 until 6) for (j in 0 until 10) {
            val x = i * w / 5f - w * 0.05f
            val y = j * h / 9f - h * 0.03f
            c.drawRoundRect(x + 14, y + 14, x + w / 5f - 14, y + h / 9f - 14, 10f, 10f, block)
        }
        c.drawRoundRect(w * 0.05f, h * 0.12f, w * 0.38f, h * 0.27f, 30f, 30f, park)
        c.drawRoundRect(w * 0.62f, h * 0.55f, w * 0.95f, h * 0.68f, 30f, 30f, park)
        val river = Path().apply {
            moveTo(0f, h * 0.36f)
            cubicTo(w * 0.3f, h * 0.30f, w * 0.6f, h * 0.46f, w, h * 0.40f)
            lineTo(w, h * 0.44f)
            cubicTo(w * 0.6f, h * 0.50f, w * 0.3f, h * 0.34f, 0f, h * 0.40f)
            close()
        }
        c.drawPath(river, water)

        val minor = context.dpf(9f)
        roadEdge.strokeWidth = minor + context.dpf(2f)
        road.strokeWidth = minor
        for (i in 1 until 5) {
            val x = i * w / 5f - w * 0.05f
            c.drawLine(x, 0f, x, h, roadEdge); c.drawLine(x, 0f, x, h, road)
        }
        for (j in 1 until 9) {
            val y = j * h / 9f - h * 0.03f
            c.drawLine(0f, y, w, y, roadEdge); c.drawLine(0f, y, w, y, road)
        }
        highway.strokeWidth = context.dpf(14f)
        roadEdge.strokeWidth = context.dpf(17f)
        c.drawLine(0f, h * 0.85f, w, h * 0.15f, roadEdge)
        c.drawLine(0f, h * 0.85f, w, h * 0.15f, highway)

        route.strokeWidth = context.dpf(8f)
        val cx = w * 0.55f
        val cy = h * 0.62f
        val r = Path().apply {
            moveTo(cx, cy)
            lineTo(cx, 4 * h / 9f - h * 0.03f)
            lineTo(4 * w / 5f - w * 0.05f, 4 * h / 9f - h * 0.03f)
            lineTo(4 * w / 5f - w * 0.05f, 0f)
        }
        c.drawPath(r, route)

        label.textSize = context.dpf(12f)
        c.drawText("Riverside Park", w * 0.09f, h * 0.20f, label)
        c.drawText("Main St", w * 0.62f, 3 * h / 9f - h * 0.03f - context.dpf(8f), label)

        val rad = context.dpf(15f)
        c.drawCircle(cx, cy, rad + context.dpf(4f), carRing)
        c.drawCircle(cx, cy, rad, car)
        val arrow = Path().apply {
            moveTo(cx, cy - rad * 0.6f)
            lineTo(cx + rad * 0.45f, cy + rad * 0.45f)
            lineTo(cx, cy + rad * 0.2f)
            lineTo(cx - rad * 0.45f, cy + rad * 0.45f)
            close()
        }
        c.drawPath(arrow, carRing)
    }
}
