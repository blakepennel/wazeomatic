package com.wazeomatic.mockwaze

import android.content.Context
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.DashPathEffect
import android.graphics.Paint
import android.graphics.Path
import android.graphics.RectF
import android.graphics.Typeface
import android.graphics.drawable.GradientDrawable
import android.view.Gravity
import android.view.View
import android.widget.FrameLayout
import android.widget.ImageView
import android.widget.LinearLayout
import android.widget.TextView

fun Context.dp(v: Int): Int = (v * resources.displayMetrics.density + 0.5f).toInt()
fun Context.dpf(v: Float): Float = v * resources.displayMetrics.density

/** Waze palette (sampled from the 2024 app screenshots). */
object Palette {
    const val INK = 0xFF202124.toInt()          // titles and labels
    const val BLUE = 0xFF0A99FF.toInt()         // primary buttons, selection ring
    const val BLUE_DARK = 0xFF0071C9.toInt()    // auto-send progress fill
    const val TEAL = 0xFF10B3CC.toInt()         // car arrow
    const val TILE_BG = 0xFFF1F2F4.toInt()      // round tile background
    const val CANCEL_BG = 0xFFEDF0F4.toInt()
    const val MAP_BG = 0xFFF8F8F6.toInt()
    const val ROAD = 0xFFD8DBD3.toInt()
    const val ROAD_MAJOR = 0xFFCFD3CA.toInt()
    const val PINK = 0xFFFF3D9A.toInt()
    const val SPEED_BG = 0xFF2C333C.toInt()
}

fun roundRect(color: Int, radiusPx: Float) = GradientDrawable().apply {
    setColor(color)
    cornerRadius = radiusPx
}

fun oval(color: Int) = GradientDrawable().apply {
    shape = GradientDrawable.OVAL
    setColor(color)
}

/**
 * Round light-grey tile with an icon and a label underneath, like the Waze report tiles.
 * Selected tiles get the blue ring and the blue check badge in the top-right corner.
 */
class Tile(ctx: Context, val label: String, icon: Int, circleDp: Int = 72, iconDp: Int = 44) : LinearLayout(ctx) {
    private val ring: View
    private val badge: ImageView

    init {
        orientation = VERTICAL
        gravity = Gravity.CENTER_HORIZONTAL
        setPadding(ctx.dp(2), ctx.dp(6), ctx.dp(2), ctx.dp(6))
        isClickable = true
        isFocusable = true
        contentDescription = label
        // no ripple: Waze's tiles only show the blue ring, and the borderless ripple would
        // draw a grey disc the size of the whole tile

        val holder = FrameLayout(ctx)
        val circle = View(ctx).apply { background = oval(Palette.TILE_BG) }
        holder.addView(circle, FrameLayout.LayoutParams(ctx.dp(circleDp), ctx.dp(circleDp), Gravity.CENTER))
        val img = ImageView(ctx).apply {
            setImageResource(icon)
            importantForAccessibility = View.IMPORTANT_FOR_ACCESSIBILITY_NO
        }
        holder.addView(img, FrameLayout.LayoutParams(ctx.dp(iconDp), ctx.dp(iconDp), Gravity.CENTER))
        ring = View(ctx).apply {
            background = GradientDrawable().apply {
                shape = GradientDrawable.OVAL
                setColor(Color.TRANSPARENT)
                setStroke(ctx.dp(2), Palette.BLUE)
            }
            visibility = View.INVISIBLE
        }
        val ringSize = ctx.dp(circleDp + 8)
        holder.addView(ring, FrameLayout.LayoutParams(ringSize, ringSize, Gravity.CENTER))
        badge = ImageView(ctx).apply {
            setImageResource(R.drawable.ic_check)
            background = oval(Palette.BLUE)
            setPadding(ctx.dp(4), ctx.dp(4), ctx.dp(4), ctx.dp(4))
            visibility = View.INVISIBLE
            importantForAccessibility = View.IMPORTANT_FOR_ACCESSIBILITY_NO
        }
        holder.addView(badge, FrameLayout.LayoutParams(ctx.dp(22), ctx.dp(22), Gravity.TOP or Gravity.END).apply {
            topMargin = ctx.dp(2); marginEnd = ctx.dp(2)
        })
        addView(holder, LayoutParams(ringSize, ringSize))

        val text = TextView(ctx).apply {
            this.text = label
            gravity = Gravity.CENTER
            setTextColor(Palette.INK)
            textSize = 15f
            maxLines = 2
        }
        addView(text, LayoutParams(LayoutParams.MATCH_PARENT, LayoutParams.WRAP_CONTENT).apply { topMargin = ctx.dp(6) })
    }

    fun setChosen(chosen: Boolean) {
        isSelected = chosen
        ring.visibility = if (chosen) View.VISIBLE else View.INVISIBLE
        badge.visibility = if (chosen) View.VISIBLE else View.INVISIBLE
    }
}

/**
 * The blue "Report" button of the confirm sheet. Waze sends the report by itself after a few
 * seconds; while it waits, the left part of the button fills with a darker blue. progress is
 * 0..1 of that countdown.
 */
class ReportButton(ctx: Context) : TextView(ctx) {
    private val base = Paint(Paint.ANTI_ALIAS_FLAG).apply { color = Palette.BLUE }
    private val fill = Paint(Paint.ANTI_ALIAS_FLAG).apply { color = Palette.BLUE_DARK }
    private val clip = Path()
    private val rect = RectF()
    var progress = 0f
        set(v) { field = v.coerceIn(0f, 1f); invalidate() }

    init {
        text = "Report"
        contentDescription = "Report"
        textSize = 17f
        typeface = Typeface.DEFAULT_BOLD
        gravity = Gravity.CENTER
        setTextColor(Color.WHITE)
        isClickable = true
        isFocusable = true
    }

    override fun onDraw(c: Canvas) {
        val r = height / 2f
        rect.set(0f, 0f, width.toFloat(), height.toFloat())
        clip.reset()
        clip.addRoundRect(rect, r, r, Path.Direction.CW)
        c.save()
        c.clipPath(clip)
        c.drawRect(rect, base)
        if (progress > 0f) c.drawRect(0f, 0f, width * progress, height.toFloat(), fill)
        c.restore()
        super.onDraw(c)
    }
}

/** Dark round speedometer, bottom-left of the Waze map. */
class SpeedometerView(ctx: Context) : View(ctx) {
    private val bg = Paint(Paint.ANTI_ALIAS_FLAG).apply { color = Palette.SPEED_BG }
    private val ticks = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        color = 0xFF8A9099.toInt(); style = Paint.Style.STROKE
    }
    private val big = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        color = Color.WHITE; typeface = Typeface.DEFAULT_BOLD; textAlign = Paint.Align.CENTER
    }
    private val small = Paint(Paint.ANTI_ALIAS_FLAG).apply { color = Color.WHITE; textAlign = Paint.Align.CENTER }

    init {
        contentDescription = "0 MPH"
        elevation = ctx.dpf(6f)
        outlineProvider = null
    }

    override fun onDraw(c: Canvas) {
        val cx = width / 2f
        val cy = height / 2f
        val r = width / 2f
        c.drawCircle(cx, cy, r, bg)
        ticks.strokeWidth = context.dpf(3f)
        ticks.pathEffect = DashPathEffect(floatArrayOf(context.dpf(2f), context.dpf(3.2f)), 0f)
        c.drawCircle(cx, cy, r - context.dpf(6f), ticks)
        big.textSize = context.dpf(26f)
        small.textSize = context.dpf(11f)
        c.drawText("0", cx, cy + context.dpf(4f), big)
        c.drawText("MPH", cx, cy + context.dpf(19f), small)
    }
}

/**
 * Static look-alike of the idle Waze map: near-white background, soft grey roads without
 * outlines, blurred-looking street names along the roads and the teal arrow in a grey disc.
 */
class FakeMapView(ctx: Context) : View(ctx) {
    private val bg = Paint().apply { color = Palette.MAP_BG }
    private val road = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        color = Palette.ROAD; style = Paint.Style.STROKE; strokeCap = Paint.Cap.ROUND
    }
    private val major = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        color = Palette.ROAD_MAJOR; style = Paint.Style.STROKE; strokeCap = Paint.Cap.ROUND
    }
    private val label = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        color = 0xFF6F7378.toInt(); textAlign = Paint.Align.CENTER
    }
    private val disc = Paint(Paint.ANTI_ALIAS_FLAG).apply { color = 0xFFD5DDE6.toInt() }
    private val arrow = Paint(Paint.ANTI_ALIAS_FLAG).apply { color = Palette.TEAL }

    init {
        contentDescription = "Map"
    }

    private fun roadLabel(c: Canvas, text: String, x: Float, y: Float, angle: Float) {
        c.save()
        c.rotate(angle, x, y)
        c.drawText(text, x, y + label.textSize / 3f, label)
        c.restore()
    }

    override fun onDraw(c: Canvas) {
        val w = width.toFloat()
        val h = height.toFloat()
        c.drawRect(0f, 0f, w, h, bg)
        road.strokeWidth = context.dpf(9f)
        major.strokeWidth = context.dpf(12f)

        // a loose grid like a residential area: two verticals, a few horizontals, one curve
        val v1 = w * 0.23f
        val v2 = w * 0.62f
        c.drawLine(v1, h * 0.28f, v1 + w * 0.02f, h, road)
        c.drawLine(v2, 0f, v2 - w * 0.01f, h, major)
        c.drawLine(0f, h * 0.10f, w, h * 0.12f, road)
        val curve = Path().apply {
            moveTo(0f, h * 0.29f)
            quadTo(w * 0.5f, h * 0.30f, w, h * 0.265f)
        }
        c.drawPath(curve, major)
        c.drawLine(v2, h * 0.55f, w * 0.80f, h * 0.55f, road)
        val hook = Path().apply {
            moveTo(w * 0.80f, h * 0.55f)
            quadTo(w * 0.86f, h * 0.55f, w * 0.84f, h * 0.585f)
        }
        c.drawPath(hook, road)
        c.drawLine(0f, h * 0.62f, v1 + w * 0.02f, h * 0.62f, road)
        c.drawLine(v1 + w * 0.02f, h * 0.84f, w, h * 0.86f, road)

        label.textSize = context.dpf(11.5f)
        roadLabel(c, "N Mississippi St", v2 - w * 0.005f, h * 0.18f, -90f)
        roadLabel(c, "Crestline Dr", v1 + w * 0.005f, h * 0.47f, -90f)
        roadLabel(c, "W 9th St", w * 0.50f, h * 0.283f, 0f)
        roadLabel(c, "Stratford Rd", w * 0.72f, h * 0.44f, 0f)
        roadLabel(c, "W 15th St", w * 0.40f, h * 0.85f, 1f)

        // car arrow
        val cx = w * 0.5f
        val cy = h * 0.55f
        val r = context.dpf(27f)
        c.drawCircle(cx, cy, r, disc)
        val a = Path().apply {
            moveTo(cx, cy - r * 0.62f)
            lineTo(cx + r * 0.45f, cy + r * 0.42f)
            lineTo(cx, cy + r * 0.18f)
            lineTo(cx - r * 0.45f, cy + r * 0.42f)
            close()
        }
        c.drawPath(a, arrow)
    }
}
