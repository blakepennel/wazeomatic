package com.wazeomatic.detector

import kotlin.math.ceil
import kotlin.math.floor
import kotlin.math.max
import kotlin.math.min

// Pure-Kotlin image helpers for the scene layer and the classifiers, written to give the same numbers as the
// OpenCV calls in wazeomatic/scene.py and classifiers.py (OpenCV 5, the version in the venv): BGR2GRAY (fixed point), BGR2HSV for 8-bit images
// (OpenCV's division tables), INTER_AREA downscaling (fractional box weights) and INTER_LINEAR upscaling
// (half-pixel centres). No Android imports, so the JVM tests run them.

/** An RGB image as packed 0xAARRGGBB ints, row major (what Bitmap.getPixels returns). */
class RgbImage(val width: Int, val height: Int, val argb: IntArray) {
    init {
        require(width >= 0 && height >= 0 && argb.size >= width * height) { "pixel array too small for ${width}x$height" }
    }

    fun r(x: Int, y: Int) = (argb[y * width + x] shr 16) and 0xff
    fun g(x: Int, y: Int) = (argb[y * width + x] shr 8) and 0xff
    fun b(x: Int, y: Int) = argb[y * width + x] and 0xff

    /** Copy of the pixel rectangle [x1, x2) x [y1, y2), clamped to the image (numpy slicing). */
    fun crop(x1: Int, y1: Int, x2: Int, y2: Int): RgbImage {
        val ax1 = x1.coerceIn(0, width); val ax2 = x2.coerceIn(ax1, width)
        val ay1 = y1.coerceIn(0, height); val ay2 = y2.coerceIn(ay1, height)
        val w = ax2 - ax1
        val h = ay2 - ay1
        val out = IntArray(w * h)
        for (y in 0 until h) System.arraycopy(argb, (ay1 + y) * width + ax1, out, y * w, w)
        return RgbImage(w, h, out)
    }

    /** Fill the rectangle [x1, x2) x [y1, y2) with one colour (test images). */
    fun fill(x1: Int, y1: Int, x2: Int, y2: Int, r: Int, g: Int, b: Int): RgbImage {
        val p = pack(r, g, b)
        for (y in max(0, y1) until min(height, y2)) for (x in max(0, x1) until min(width, x2)) argb[y * width + x] = p
        return this
    }

    fun copy(): RgbImage = RgbImage(width, height, argb.copyOf(width * height))

    companion object {
        fun filled(w: Int, h: Int, r: Int, g: Int, b: Int): RgbImage = RgbImage(w, h, IntArray(w * h) { pack(r, g, b) })

        fun pack(r: Int, g: Int, b: Int): Int = (0xff shl 24) or (r shl 16) or (g shl 8) or b
    }
}

object ImageOps {
    private const val HSV_SHIFT = 12
    private val SDIV = IntArray(256) { if (it == 0) 0 else Math.rint((255 shl HSV_SHIFT) / it.toDouble()).toInt() }
    private val HDIV180 = IntArray(256) { if (it == 0) 0 else Math.rint((180 shl HSV_SHIFT) / (6.0 * it)).toInt() }

    /** cv2.COLOR_BGR2GRAY on 8-bit as OpenCV 5 computes it: (9798 R + 19235 G + 3735 B + 2^14) >> 15 (equal for all 2^24 colours). */
    fun grey(p: Int): Int {
        val r = (p shr 16) and 0xff
        val g = (p shr 8) and 0xff
        val b = p and 0xff
        return (r * 9798 + g * 19235 + b * 3735 + (1 shl 14)) shr 15
    }

    /**
     * cv2.COLOR_BGR2HSV for one 8-bit pixel (H in [0, 180), S and V in [0, 255]), with OpenCV's fixed-point
     * division tables so the values match bit for bit. Packed as (h shl 16) or (s shl 8) or v.
     */
    fun hsv(p: Int): Int {
        val r = (p shr 16) and 0xff
        val g = (p shr 8) and 0xff
        val b = p and 0xff
        val v = max(r, max(g, b))
        val diff = v - min(r, min(g, b))
        val half = 1 shl (HSV_SHIFT - 1)
        val s = (diff * SDIV[v] + half) shr HSV_SHIFT
        val raw = when (v) {
            r -> g - b
            g -> b - r + 2 * diff
            else -> r - g + 4 * diff
        }
        var h = (raw * HDIV180[diff] + half) shr HSV_SHIFT
        if (h < 0) h += 180
        return (h shl 16) or (s shl 8) or v
    }

    /**
     * INTER_AREA weights along one axis for a downscale from [src] to [dst] samples: for every output index the
     * source indices it covers and the covered share of each (summing to 1).
     */
    class AxisWeights(val idx: Array<IntArray>, val w: Array<FloatArray>)

    fun areaWeights(src: Int, dst: Int): AxisWeights {
        val scale = src.toDouble() / dst
        val idx = arrayOfNulls<IntArray>(dst)
        val ws = arrayOfNulls<FloatArray>(dst)
        for (d in 0 until dst) {
            val f0 = d * scale
            val f1 = min((d + 1) * scale, src.toDouble())
            val i0 = floor(f0).toInt()
            val i1 = min(src, ceil(f1 - 1e-9).toInt())
            val ii = ArrayList<Int>(i1 - i0 + 1)
            val ww = ArrayList<Float>(i1 - i0 + 1)
            for (i in i0 until max(i1, i0 + 1)) {
                val a = max(f0, i.toDouble())
                val b = min(f1, (i + 1).toDouble())
                if (b - a > 1e-9) { ii.add(min(i, src - 1)); ww.add(((b - a) / (f1 - f0)).toFloat()) }
            }
            idx[d] = ii.toIntArray()
            ws[d] = ww.toFloatArray()
        }
        @Suppress("UNCHECKED_CAST")
        return AxisWeights(idx as Array<IntArray>, ws as Array<FloatArray>)
    }

    /**
     * Area-average to dw x dh, reading source sample (x, y) through [get]. Separable: rows first (into a
     * dw x h buffer), then columns, so the cost is about (w + dw) * h no matter how large the source is.
     */
    private inline fun areaResize(w: Int, h: Int, dw: Int, dh: Int, get: (Int) -> Float): FloatArray {
        if (dw in 1 until w && dh in 1 until h && w % dw == 0 && h % dh == 0) {
            // whole-number ratio (1920x1080 or 1280x720 to 640x360): INTER_AREA is the plain block mean, and one
            // pass over the source without the weight tables is several times faster on a phone
            val fx = w / dw
            val fy = h / dh
            val out = FloatArray(dw * dh)
            for (y in 0 until h) {
                val row = y * w
                val o = (y / fy) * dw
                var i = row
                for (x in 0 until dw) {
                    var s = 0f
                    for (k in 0 until fx) s += get(i + k)
                    out[o + x] += s
                    i += fx
                }
            }
            val inv = 1f / (fx * fy)
            for (k in out.indices) out[k] *= inv
            return out
        }
        val wx = areaWeights(w, dw)
        val wy = areaWeights(h, dh)
        val tmp = FloatArray(dw * h)
        for (y in 0 until h) {
            val row = y * w
            for (x in 0 until dw) {
                val ix = wx.idx[x]
                val fx = wx.w[x]
                var s = 0f
                for (k in ix.indices) s += get(row + ix[k]) * fx[k]
                tmp[y * dw + x] = s
            }
        }
        val out = FloatArray(dw * dh)
        for (y in 0 until dh) {
            val iy = wy.idx[y]
            val fy = wy.w[y]
            for (x in 0 until dw) {
                var s = 0f
                for (k in iy.indices) s += tmp[iy[k] * dw + x] * fy[k]
                out[y * dw + x] = s
            }
        }
        return out
    }

    /** Area-average a single float channel (w x h, row major) down to dw x dh. */
    fun areaResize(src: FloatArray, w: Int, h: Int, dw: Int, dh: Int): FloatArray =
        if (dw == w && dh == h) src.copyOf(w * h) else areaResize(w, h, dw, dh) { src[it] }

    /** 8-bit grey of [img] as floats, area-averaged to dw x dh: EgoMotion.grey_small. */
    fun greySmall(img: RgbImage, dw: Int, dh: Int): FloatArray {
        val px = img.argb
        return areaResize(img.width, img.height, dw, dh) { grey(px[it]).toFloat() }
    }

    /**
     * cv2.resize of an 8-bit RGB image: INTER_AREA when [area] (the default when shrinking), INTER_LINEAR
     * otherwise; results rounded to 8 bits like OpenCV's saturate_cast.
     */
    fun resize(img: RgbImage, dw: Int, dh: Int, area: Boolean = dw <= img.width && dh <= img.height): RgbImage {
        if (dw == img.width && dh == img.height) return img.copy()
        val px = img.argb
        val w = img.width
        val h = img.height
        val rc: FloatArray
        val gc: FloatArray
        val bc: FloatArray
        if (area) {
            rc = areaResize(w, h, dw, dh) { ((px[it] shr 16) and 0xff).toFloat() }
            gc = areaResize(w, h, dw, dh) { ((px[it] shr 8) and 0xff).toFloat() }
            bc = areaResize(w, h, dw, dh) { (px[it] and 0xff).toFloat() }
        } else {
            rc = bilinear(w, h, dw, dh) { ((px[it] shr 16) and 0xff).toFloat() }
            gc = bilinear(w, h, dw, dh) { ((px[it] shr 8) and 0xff).toFloat() }
            bc = bilinear(w, h, dw, dh) { (px[it] and 0xff).toFloat() }
        }
        val out = IntArray(dw * dh)
        for (i in out.indices) out[i] = RgbImage.pack(sat(rc[i]), sat(gc[i]), sat(bc[i]))
        return RgbImage(dw, dh, out)
    }

    private fun sat(v: Float): Int = Math.rint(v.toDouble()).toInt().coerceIn(0, 255)

    /** INTER_LINEAR with half-pixel centres, edges clamped. */
    private inline fun bilinear(w: Int, h: Int, dw: Int, dh: Int, get: (Int) -> Float): FloatArray {
        val out = FloatArray(dw * dh)
        val sx = w.toDouble() / dw
        val sy = h.toDouble() / dh
        for (y in 0 until dh) {
            val fy = ((y + 0.5) * sy - 0.5).coerceIn(0.0, (h - 1).toDouble())
            val y0 = floor(fy).toInt()
            val y1 = min(y0 + 1, h - 1)
            val ay = (fy - y0).toFloat()
            for (x in 0 until dw) {
                val fx = ((x + 0.5) * sx - 0.5).coerceIn(0.0, (w - 1).toDouble())
                val x0 = floor(fx).toInt()
                val x1 = min(x0 + 1, w - 1)
                val ax = (fx - x0).toFloat()
                val top = get(y0 * w + x0) * (1 - ax) + get(y0 * w + x1) * ax
                val bot = get(y1 * w + x0) * (1 - ax) + get(y1 * w + x1) * ax
                out[y * dw + x] = top * (1 - ay) + bot * ay
            }
        }
        return out
    }

    /**
     * classifiers.square_crop: the normalised box grown by [pad] on each side, clamped to the frame, letterboxed
     * into size x size (fill 114, centred; INTER_AREA when shrinking, INTER_LINEAR when growing).
     */
    fun squareCrop(img: RgbImage, bbox: BBox, pad: Double = 0.12, size: Int = 128, fill: Int = 114): RgbImage {
        val w = img.width
        val h = img.height
        val cx = bbox.cx.toDouble()
        val cy = bbox.cy.toDouble()
        val bw = bbox.w * (1 + 2 * pad)
        val bh = bbox.h * (1 + 2 * pad)
        val x1 = max(0.0, (cx - bw / 2) * w).toInt()
        val y1 = max(0.0, (cy - bh / 2) * h).toInt()
        val x2 = min(w.toDouble(), (cx + bw / 2) * w).toInt()
        val y2 = min(h.toDouble(), (cy + bh / 2) * h).toInt()
        // at least one pixel, and inside the frame
        val sx1 = min(x1, w - 1)
        val sy1 = min(y1, h - 1)
        val crop = img.crop(sx1, sy1, max(x2, sx1 + 1), max(y2, sy1 + 1))
        return letterbox(crop, size, fill)
    }

    /** Letterbox [img] into size x size (fill, centred), the second half of square_crop. */
    fun letterbox(img: RgbImage, size: Int, fill: Int = 114): RgbImage {
        val s = size.toDouble() / max(img.width, img.height)
        val nw = max(1, Math.rint(img.width * s).toInt())        // Python round(): half to even
        val nh = max(1, Math.rint(img.height * s).toInt())
        val resized = resize(img, nw, nh, area = s < 1)
        val out = IntArray(size * size) { RgbImage.pack(fill, fill, fill) }
        val px = (size - nw) / 2
        val py = (size - nh) / 2
        for (y in 0 until nh) System.arraycopy(resized.argb, y * nw, out, (py + y) * size + px, nw)
        return RgbImage(size, size, out)
    }

    /** RGB [0, 1] planes (C, H, W) of [img] written into [dst] at [offset]. */
    fun toChw(img: RgbImage, dst: FloatArray, offset: Int) {
        val plane = img.width * img.height
        for (i in 0 until plane) {
            val p = img.argb[i]
            dst[offset + i] = ((p shr 16) and 0xff) / 255f
            dst[offset + plane + i] = ((p shr 8) and 0xff) / 255f
            dst[offset + 2 * plane + i] = (p and 0xff) / 255f
        }
    }
}
