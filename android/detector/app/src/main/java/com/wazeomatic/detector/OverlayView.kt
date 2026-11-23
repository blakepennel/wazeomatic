package com.wazeomatic.detector

import android.content.Context
import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.graphics.RectF
import android.graphics.Typeface
import android.util.AttributeSet
import android.view.View
import java.util.Locale

/** Shows the last analysed frame (fit-center) with detection boxes on top, or a message while there is none. */
class OverlayView @JvmOverloads constructor(ctx: Context, attrs: AttributeSet? = null) : View(ctx, attrs) {
    private var frame: Bitmap? = null
    private var dets: List<Detection> = emptyList()
    var message: String = ""
        set(v) { field = v; invalidate() }
    var hint: String = ""
        set(v) { field = v; invalidate() }

    private val dp = resources.displayMetrics.density
    private val box = Paint().apply { style = Paint.Style.STROKE; strokeWidth = 2.5f * dp; isAntiAlias = true }
    private val text = Paint().apply {
        color = Color.WHITE; textSize = 14f * dp; isAntiAlias = true; typeface = Typeface.DEFAULT_BOLD
    }
    private val textBg = Paint().apply { style = Paint.Style.FILL; isAntiAlias = true }
    private val msgPaint = Paint().apply {
        color = Color.WHITE; textSize = 22f * dp; isAntiAlias = true; textAlign = Paint.Align.CENTER; typeface = Typeface.DEFAULT_BOLD
    }
    private val hintPaint = Paint().apply {
        color = Color.rgb(170, 180, 178); textSize = 15f * dp; isAntiAlias = true; textAlign = Paint.Align.CENTER
    }
    private val bmpPaint = Paint(Paint.FILTER_BITMAP_FLAG)
    private val dst = RectF()
    private val tag = RectF()

    fun show(bmp: Bitmap, detections: List<Detection>) {
        frame = bmp
        dets = detections
        invalidate()
    }

    /** Back to the message (the detector stopped). */
    fun clear() {
        frame = null
        dets = emptyList()
        invalidate()
    }

    private fun colorFor(d: Detection): Int = when {
        d.label.startsWith("police") -> Color.rgb(90, 150, 255)
        d.model == "hazard" -> Color.rgb(255, 160, 40)
        else -> Color.rgb(60, 210, 200)
    }

    override fun onDraw(c: Canvas) {
        c.drawColor(Color.BLACK)
        val b = frame
        if (b == null) {
            if (message.isNotEmpty()) c.drawText(message, width / 2f, height / 2f, msgPaint)
            if (hint.isNotEmpty()) c.drawText(hint, width / 2f, height / 2f + 30f * dp, hintPaint)
            return
        }
        val s = minOf(width.toFloat() / b.width, height.toFloat() / b.height)
        val w = b.width * s
        val h = b.height * s
        val ox = (width - w) / 2
        val oy = (height - h) / 2
        dst.set(ox, oy, ox + w, oy + h)
        c.drawBitmap(b, null, dst, bmpPaint)
        val pad = 5f * dp
        val th = text.textSize + 2 * pad
        for (d in dets) {
            val col = colorFor(d)
            box.color = col
            textBg.color = col
            val l = ox + (d.bbox.cx - d.bbox.w / 2) * w
            val t = oy + (d.bbox.cy - d.bbox.h / 2) * h
            val r = ox + (d.bbox.cx + d.bbox.w / 2) * w
            val btm = oy + (d.bbox.cy + d.bbox.h / 2) * h
            c.drawRoundRect(l, t, r, btm, 4f * dp, 4f * dp, box)
            val label = String.format(Locale.US, "%s %.0f%%", d.label.replace('_', ' '), d.confidence * 100)
            val tw = text.measureText(label)
            val ty = maxOf(t, oy + th)
            tag.set(l, ty - th, l + tw + 2 * pad, ty)
            c.drawRoundRect(tag, 4f * dp, 4f * dp, textBg)
            c.drawText(label, l + pad, ty - pad - text.descent() / 2, text)
        }
    }
}
