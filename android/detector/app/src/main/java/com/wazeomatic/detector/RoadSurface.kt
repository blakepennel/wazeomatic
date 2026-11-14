package com.wazeomatic.detector

import kotlin.math.max
import kotlin.math.min

/**
 * Road-surface classifier (dry, wet, water, slush, snow, ice) on the patch of road ahead. Twin of
 * classifiers.RoadSurface: the patch moves up when the learnt hood zone is taller than the default, so it never
 * looks at the dashboard, and class names come back prefixed "surface_".
 */
class RoadSurface(
    val clf: ImageClassifier,
    val everyS: Double = 0.5,
    val patch: DoubleArray = ROAD_PATCH,
) : ScenePart {
    /** Learnt hood zone (fraction of the height); set through [setHoodFrac]. */
    @Volatile var hood = 0.0
        private set
    private var lastT = -1e9
    var last: Map<String, Float> = emptyMap()
        private set
    override var lastMs = 0.0
        private set

    override fun setHoodFrac(frac: Double) {
        hood = frac
    }

    /** x1, y1, x2, y2 in pixels. */
    fun region(h: Int, w: Int): IntArray {
        val (x1, y1, x2, y2) = patch.toList()
        val bottom = min(y2, 1.0 - max(hood, 0.10) - 0.02)
        val top = max(0.45, bottom - (y2 - y1))
        return intArrayOf((x1 * w).toInt(), (top * h).toInt(), (x2 * w).toInt(), (bottom * h).toInt())
    }

    override fun invoke(image: RgbImage, t: Double): Map<String, Float>? {
        if (t - lastT < everyS) {
            lastMs = 0.0
            return null
        }
        lastT = t
        val r = region(image.height, image.width)
        val s = clf.size
        val p = clf.predict(listOf(ImageOps.resize(image.crop(r[0], r[1], r[2], r[3]), s, s, area = true)))[0]
        lastMs = clf.lastMs
        last = clf.labels.withIndex().filter { it.index < p.size }.associate { "surface_${it.value}" to p[it.index] }
        return last
    }

    override fun close() = clf.close()

    companion object {
        /** x1, y1, x2, y2 as fractions of the frame: centre of the lane, just above the hood zone. */
        val ROAD_PATCH = doubleArrayOf(0.40, 0.60, 0.60, 0.86)
    }
}
