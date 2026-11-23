package com.wazeomatic.detector

import kotlin.math.abs
import kotlin.math.exp
import kotlin.math.floor
import kotlin.math.hypot
import kotlin.math.ln
import kotlin.math.max
import kotlin.math.min
import kotlin.math.sqrt

// Twin of wazeomatic/motion.py: is a police car standing still? Waze's Police report is for a police car that
// stands, so police boxes of cars read as driving (or not read yet) are relabelled before the report manager.
// Corners and optical flow are computed here in plain Kotlin (the app has no OpenCV): Shi-Tomasi corners and
// pyramidal Lucas-Kanade with a forward-backward check, the same method as cv2.goodFeaturesToTrack and
// cv2.calcOpticalFlowPyrLK, so the numbers are close to the desktop's but not identical.

/** Defaults are motion.MotionConfig's. */
data class MotionConfig(
    val width: Int = 640,
    val horizonY: Double = 0.50,
    val driveSide: String = "right",
    val trackMinConf: Double = 0.3,
    val linkIou: Double = 0.3,
    val trackGapS: Double = 1.5,
    val minMeasures: Int = 3,
    val window: Int = 12,
    val roadCorners: Int = 250,
    val carCorners: Int = 40,
    val minRoadPairs: Int = 20,
    val horizonTol: Double = 0.12,           // 0 turns the horizon check off
    val minCarPoints: Int = 8,
    val roadLk: IntArray = intArrayOf(21, 4),
    val carLk: IntArray = intArrayOf(15, 3),
    val minExpansion: Double = 0.02,
    val oncomingX: Double = 0.4,
    val residOncoming: Double = 0.3,
    val residOther: Double = 0.6,
    val minSpreadRatio: Double = 0.3,
    val stillSpanS: Double = 1.0,
    val stillShift: Double = 0.1,
    val stillGrowth: Double = 0.1,
)

/** An 8-bit grey image as floats, row-major. */
class GreyImage(val w: Int, val h: Int, val px: FloatArray) {
    operator fun get(x: Int, y: Int) = px[y * w + x]

    /** Bilinear sample; coordinates are clamped to the image. */
    fun sample(x: Double, y: Double): Double {
        val xc = x.coerceIn(0.0, w - 1.0)
        val yc = y.coerceIn(0.0, h - 1.0)
        val x0 = min(floor(xc).toInt(), w - 2).coerceAtLeast(0)
        val y0 = min(floor(yc).toInt(), h - 2).coerceAtLeast(0)
        val fx = xc - x0
        val fy = yc - y0
        val i = y0 * w + x0
        val a = px[i] + (px[i + 1] - px[i]) * fx
        val b = px[i + w] + (px[i + w + 1] - px[i + w]) * fx
        return a + (b - a) * fy
    }

    /** Half-size image, each pixel the mean of a 2 x 2 block. */
    fun down(): GreyImage {
        val dw = w / 2
        val dh = h / 2
        val out = FloatArray(dw * dh)
        for (y in 0 until dh) for (x in 0 until dw) {
            val i = 2 * y * w + 2 * x
            out[y * dw + x] = (px[i] + px[i + 1] + px[i + w] + px[i + w + 1]) / 4f
        }
        return GreyImage(dw, dh, out)
    }
}

object Flow {
    /**
     * Shi-Tomasi corners: the smaller eigenvalue of the gradient structure tensor over 3 x 3 blocks, local
     * maxima at least [quality] of the strongest, strongest first, no two closer than [minDist] px, only where
     * [mask] is true (null = everywhere). As cv2.goodFeaturesToTrack(img, maxCorners, quality, minDist, mask).
     */
    fun corners(img: GreyImage, maxCorners: Int, quality: Double, minDist: Double, mask: BooleanArray?): List<DoubleArray> {
        val w = img.w
        val h = img.h
        if (w < 5 || h < 5) return emptyList()
        val gx = FloatArray(w * h)
        val gy = FloatArray(w * h)
        for (y in 1 until h - 1) for (x in 1 until w - 1) {       // Sobel 3 x 3
            val i = y * w + x
            val p = img.px
            gx[i] = (p[i - w + 1] + 2 * p[i + 1] + p[i + w + 1]) - (p[i - w - 1] + 2 * p[i - 1] + p[i + w - 1])
            gy[i] = (p[i + w - 1] + 2 * p[i + w] + p[i + w + 1]) - (p[i - w - 1] + 2 * p[i - w] + p[i - w + 1])
        }
        val eig = FloatArray(w * h)
        var best = 0f
        for (y in 2 until h - 2) for (x in 2 until w - 2) {
            val i = y * w + x
            if (mask != null && !mask[i]) continue
            var a = 0f; var b = 0f; var c = 0f
            for (dy in -1..1) for (dx in -1..1) {
                val j = i + dy * w + dx
                a += gx[j] * gx[j]; b += gx[j] * gy[j]; c += gy[j] * gy[j]
            }
            val e = ((a + c) - sqrt(((a - c) * (a - c) + 4 * b * b).toDouble()).toFloat()) / 2f
            eig[i] = e
            if (e > best) best = e
        }
        if (best <= 0f) return emptyList()
        val thr = (best * quality).toFloat()
        val cand = ArrayList<Int>()
        for (y in 2 until h - 2) for (x in 2 until w - 2) {
            val i = y * w + x
            val e = eig[i]
            if (e < thr || e <= 0f) continue
            var isMax = true
            loop@ for (dy in -1..1) for (dx in -1..1) {
                if ((dx != 0 || dy != 0) && eig[i + dy * w + dx] > e) { isMax = false; break@loop }
            }
            if (isMax) cand.add(i)
        }
        cand.sortWith(compareByDescending<Int> { eig[it] }.thenBy { it })
        val cell = max(1.0, minDist)
        val gw = (w / cell).toInt() + 1
        val gh = (h / cell).toInt() + 1
        val grid = Array(gw * gh) { ArrayList<DoubleArray>(2) }
        val out = ArrayList<DoubleArray>()
        val d2 = minDist * minDist
        for (i in cand) {
            val x = (i % w).toDouble()
            val y = (i / w).toDouble()
            val cx = (x / cell).toInt()
            val cy = (y / cell).toInt()
            var ok = true
            loop@ for (yy in max(0, cy - 1)..min(gh - 1, cy + 1)) for (xx in max(0, cx - 1)..min(gw - 1, cx + 1)) {
                for (p in grid[yy * gw + xx]) {
                    if ((p[0] - x) * (p[0] - x) + (p[1] - y) * (p[1] - y) < d2) { ok = false; break@loop }
                }
            }
            if (!ok) continue
            val pt = doubleArrayOf(x, y)
            grid[cy * gw + cx].add(pt)
            out.add(pt)
            if (out.size >= maxCorners) break
        }
        return out
    }

    fun pyramid(img: GreyImage, levels: Int): List<GreyImage> {
        val out = arrayListOf(img)
        while (out.size <= levels && out.last().w >= 16 && out.last().h >= 16) out.add(out.last().down())
        return out
    }

    /** One point through pyramidal Lucas-Kanade from [a] to [b] (pyramids); null when it is lost. */
    fun lk(a: List<GreyImage>, b: List<GreyImage>, x: Double, y: Double, win: Int): DoubleArray? {
        val half = win / 2
        val n = (2 * half + 1) * (2 * half + 1)
        val iw = DoubleArray(n); val ix = DoubleArray(n); val iy = DoubleArray(n)
        var gx = 0.0
        var gy = 0.0
        val top = min(a.size, b.size) - 1
        for (lv in top downTo 0) {
            val ia = a[lv]
            val jb = b[lv]
            val s = 1.0 / (1 shl lv)
            val px = x * s
            val py = y * s
            var k = 0
            var g11 = 0.0; var g12 = 0.0; var g22 = 0.0
            for (dy in -half..half) for (dx in -half..half) {
                val sx = px + dx
                val sy = py + dy
                iw[k] = ia.sample(sx, sy)
                ix[k] = (ia.sample(sx + 1, sy) - ia.sample(sx - 1, sy)) / 2
                iy[k] = (ia.sample(sx, sy + 1) - ia.sample(sx, sy - 1)) / 2
                g11 += ix[k] * ix[k]; g12 += ix[k] * iy[k]; g22 += iy[k] * iy[k]
                k++
            }
            val det = g11 * g22 - g12 * g12
            val minEig = ((g11 + g22) - sqrt((g11 - g22) * (g11 - g22) + 4 * g12 * g12)) / 2 / n
            if (minEig < 1e-4 || abs(det) < 1e-12) return null
            var vx = 0.0
            var vy = 0.0
            for (iter in 0 until 30) {
                var b1 = 0.0; var b2 = 0.0
                k = 0
                for (dy in -half..half) for (dx in -half..half) {
                    val it = jb.sample(px + gx + vx + dx, py + gy + vy + dy) - iw[k]
                    b1 += it * ix[k]; b2 += it * iy[k]
                    k++
                }
                val ddx = -(g22 * b1 - g12 * b2) / det
                val ddy = -(-g12 * b1 + g11 * b2) / det
                vx += ddx
                vy += ddy
                if (ddx * ddx + ddy * ddy < 1e-4) break
            }
            if (lv > 0) { gx = 2 * (gx + vx); gy = 2 * (gy + vy) } else { gx += vx; gy += vy }
        }
        val nx = x + gx
        val ny = y + gy
        val w0 = a[0].w
        val h0 = a[0].h
        if (nx < 0 || ny < 0 || nx > w0 - 1 || ny > h0 - 1) return null
        return doubleArrayOf(nx, ny)
    }

    /** Points tracked from [a] to [b] with a forward-backward check (< [fbPx]); returns matched (from, to). */
    fun track(a: List<GreyImage>, b: List<GreyImage>, pts: List<DoubleArray>, win: Int,
              fbPx: Double = 1.0): Pair<List<DoubleArray>, List<DoubleArray>> {
        val from = ArrayList<DoubleArray>()
        val to = ArrayList<DoubleArray>()
        for (p in pts) {
            val q = lk(a, b, p[0], p[1], win) ?: continue
            val r = lk(b, a, q[0], q[1], win) ?: continue
            if (hypot(r[0] - p[0], r[1] - p[1]) < fbPx) { from.add(p); to.add(q) }
        }
        return from to to
    }

    fun median(xs: DoubleArray): Double {
        val s = xs.sortedArray()
        val n = s.size
        return if (n % 2 == 1) s[n / 2] else (s[n / 2 - 1] + s[n / 2]) / 2.0
    }

    /** numpy.percentile with linear interpolation. */
    fun percentile(xs: DoubleArray, q: Double): Double {
        val s = xs.sortedArray()
        val pos = q / 100.0 * (s.size - 1)
        val lo = floor(pos).toInt()
        val hi = min(lo + 1, s.size - 1)
        return s[lo] + (s[hi] - s[lo]) * (pos - lo)
    }

    /** Least squares of r = A * y + B. */
    private fun fitLine(y: DoubleArray, r: DoubleArray, keep: BooleanArray?): DoubleArray {
        var n = 0.0; var sy = 0.0; var sr = 0.0; var syy = 0.0; var syr = 0.0
        for (i in y.indices) {
            if (keep != null && !keep[i]) continue
            n += 1; sy += y[i]; sr += r[i]; syy += y[i] * y[i]; syr += y[i] * r[i]
        }
        val d = n * syy - sy * sy
        if (abs(d) < 1e-12) return doubleArrayOf(0.0, if (n > 0) sr / n else 0.0)
        val a = (n * syr - sy * sr) / d
        return doubleArrayOf(a, (sr - a * sy) / n)
    }

    /** motion.road_pair_rate: horizontal log-spread rate of road point pairs on nearly one row, as A * y + B. */
    fun roadPairRate(a: List<DoubleArray>, b: List<DoubleArray>, h: Int, dt: Double, minPairs: Int = 20): DoubleArray? {
        val ys = ArrayList<Double>()
        val rs = ArrayList<Double>()
        for (i in a.indices) for (j in i + 1 until a.size) {
            if (abs(a[i][1] - a[j][1]) / h >= 0.015) continue
            val da = abs(a[i][0] - a[j][0])
            if (da <= 15) continue
            val db = abs(b[i][0] - b[j][0])
            if (db <= 1e-3) continue
            ys.add((a[i][1] + a[j][1]) / 2 / h)
            rs.add(ln(db / da) / dt)
        }
        if (ys.size < minPairs) return null
        val y = ys.toDoubleArray()
        val r = rs.toDoubleArray()
        var sol = fitLine(y, r, null)
        repeat(2) {
            val res = DoubleArray(y.size) { abs(sol[0] * y[it] + sol[1] - r[it]) }
            val cut = percentile(res, 60.0)
            sol = fitLine(y, r, BooleanArray(y.size) { res[it] <= cut })
        }
        return sol
    }

    /** motion._spread: median log change of the horizontal distance between point pairs. */
    fun spread(a: List<DoubleArray>, b: List<DoubleArray>, minDx: Double = 8.0, minPairs: Int = 10): Double? {
        val v = ArrayList<Double>()
        for (i in a.indices) for (j in i + 1 until a.size) {
            val da = abs(a[i][0] - a[j][0])
            val db = abs(b[i][0] - b[j][0])
            if (da > minDx && db > 1e-3) v.add(ln(db / da))
        }
        return if (v.size < minPairs) null else median(v.toDoubleArray())
    }
}

class PoliceMotion(val cfg: MotionConfig = MotionConfig()) {
    class Track(t: Double, bbox: BBox) {
        val boxes = ArrayDeque<Triple<Double, BBox, Boolean>>()     // (t, box, we stood)
        val resid = ArrayDeque<DoubleArray>()                       // (residual, centre x, spread ratio)
        val still = ArrayDeque<Boolean>()
        var state = "pending"
        init { boxes.addLast(Triple(t, bbox, false)) }
        val lastT get() = boxes.last().first
        val lastBox get() = boxes.last().second
    }

    private var prev: List<GreyImage>? = null
    private var prevT = 0.0
    private val tracks = ArrayList<Track>()
    var lastTracks: List<Track> = emptyList()
        private set

    /** [dets] with the police boxes of pending and moving tracks relabelled (_pending / _moving). */
    fun update(t: Double, img: RgbImage, dets: List<Detection>, egoState: String, hoodFrac: Double = 0.0): List<Detection> {
        val h = max(16, Math.round(img.height.toDouble() * cfg.width / img.width).toInt())
        val g = GreyImage(cfg.width, h, ImageOps.greySmall(img, cfg.width, h))
        val levels = max(cfg.roadLk[1], cfg.carLk[1])
        val pyr = Flow.pyramid(g, levels)
        val prevPyr = prev
        val pt = prevT
        prev = pyr
        prevT = t
        tracks.removeAll { t - it.lastT > cfg.trackGapS }

        val police = dets.filter { it.label == POLICE && it.confidence >= cfg.trackMinConf }
        if (police.isEmpty()) {
            lastTracks = emptyList()
            return dets
        }
        val stationary = egoState == "stationary"
        val dt = t - pt
        val follow = HashMap<Track, Pair<List<DoubleArray>, List<DoubleArray>>>()
        val pred = HashMap<Track, BBox>()
        for (tr in tracks) {
            var box = tr.lastBox
            if (prevPyr != null && !stationary && dt > 0 && abs(tr.lastT - pt) < 1e-6) {
                val f = follow(prevPyr, pyr, box)
                if (f != null) {
                    follow[tr] = f
                    val dx = Flow.median(DoubleArray(f.first.size) { f.second[it][0] - f.first[it][0] })
                    val dy = Flow.median(DoubleArray(f.first.size) { f.second[it][1] - f.first[it][1] })
                    val sc = exp(Flow.spread(f.first, f.second) ?: 0.0)
                    box = BBox((box.cx + dx / g.w).toFloat(), (box.cy + dy / g.h).toFloat(),
                        (box.w * sc).toFloat(), (box.h * sc).toFloat())
                }
            }
            pred[tr] = box
        }
        data class Cand(val iou: Double, val k: Int, val n: Int)
        val cands = ArrayList<Cand>()
        for ((k, d) in police.withIndex()) for ((n, tr) in tracks.withIndex()) {
            if (tr.lastT < t) cands.add(Cand(boxIou(pred[tr]!!, d.bbox), k, n))
        }
        // as Python's sorted(reverse=True) on (iou, k, n) tuples
        cands.sortWith(compareByDescending<Cand> { it.iou }.thenByDescending { it.k }.thenByDescending { it.n })
        val assigned = HashMap<Int, Track>()
        val taken = HashSet<Int>()
        for (c in cands) {
            if (c.iou < cfg.linkIou) break
            if (c.k in assigned || c.n in taken) continue
            assigned[c.k] = tracks[c.n]
            taken.add(c.n)
        }
        var road: DoubleArray? = null
        var roadTried = false
        for ((k, d) in police.withIndex()) {
            val tr = assigned[k]
            if (tr == null) {
                val nt = Track(t, d.bbox)
                tracks.add(nt)
                assigned[k] = nt
                continue
            }
            val prevBox = tr.lastBox
            tr.boxes.addLast(Triple(t, d.bbox, stationary))
            while (tr.boxes.size > 64) tr.boxes.removeFirst()
            if (stationary) stillCheck(tr, t)
            else {
                val f = follow[tr]
                if (f != null && prevPyr != null) {
                    if (!roadTried) { road = road(prevPyr, pyr, dets, dt, hoodFrac); roadTried = true }
                    val r = road
                    if (r != null) {
                        val m = measure(f.first, f.second, prevBox, r, dt, g.h, g.w)
                        if (m != null) {
                            tr.resid.addLast(doubleArrayOf(m[0], prevBox.cx.toDouble(), m[1]))
                            while (tr.resid.size > cfg.window) tr.resid.removeFirst()
                        }
                    }
                }
            }
            decide(tr)
        }
        lastTracks = police.indices.map { assigned[it]!! }
        val held = HashMap<BBox, String>()
        for ((k, d) in police.withIndex()) if (assigned[k]!!.state != "stopped") held[d.bbox] = assigned[k]!!.state
        // identity, not equality: two equal detections are still two boxes
        val ids = java.util.IdentityHashMap<Detection, Int>()
        for ((k, d) in police.withIndex()) ids[d] = k
        return dets.map { d ->
            val k = ids[d]
            val state = if (k != null) assigned[k]!!.state else if (d.label == OTHER_SIDE) held[d.bbox] else null
            if (state == null || state == "stopped") d
            else d.copy(label = d.label + if (state == "moving") "_moving" else "_pending")
        }
    }

    private fun stillCheck(tr: Track, t: Double) {
        val old = tr.boxes.filter { t - it.first >= cfg.stillSpanS * 0.8 && it.third }
        if (old.isEmpty()) return
        val b0 = old.last().second
        val b1 = tr.boxes.last().second
        val shift = abs(b1.cx - b0.cx) / max(1e-6, (b0.w + b1.w) / 2.0)
        val growth = abs(ln(max(1e-9, b1.w.toDouble() * b1.h) / max(1e-9, b0.w.toDouble() * b0.h))) / 2
        tr.still.addLast(shift < cfg.stillShift && growth < cfg.stillGrowth)
        while (tr.still.size > cfg.window) tr.still.removeFirst()
    }

    private fun decide(tr: Track) {
        val nm = tr.resid.size
        val ns = tr.still.size
        if (max(nm, ns) < cfg.minMeasures) return
        if (nm >= ns) {
            val r = Flow.median(DoubleArray(nm) { tr.resid[it][0] })
            val x = Flow.median(DoubleArray(nm) { tr.resid[it][1] })
            val spread = Flow.median(DoubleArray(nm) { tr.resid[it][2] })
            val oncoming = if (cfg.driveSide == "right") x < cfg.oncomingX else x > 1 - cfg.oncomingX
            val moving = r > (if (oncoming) cfg.residOncoming else cfg.residOther) || spread < cfg.minSpreadRatio
            tr.state = if (moving) "moving" else "stopped"
        } else {
            tr.state = if (tr.still.count { it } * 2 >= ns) "stopped" else "moving"
        }
    }

    /** Road expansion fit and camera rotation shift: (A, B, shift x px, shift y px), or null. */
    private fun road(a: List<GreyImage>, b: List<GreyImage>, dets: List<Detection>, dt: Double, hoodFrac: Double): DoubleArray? {
        val img = a[0]
        val w = img.w
        val h = img.h
        val y1 = ((cfg.horizonY + 0.06) * h).toInt()
        val y2 = ((0.97 - hoodFrac) * h).toInt()
        if (y2 - y1 < 10) return null
        val mask = BooleanArray(w * h)
        for (y in max(0, y1) until min(h, y2)) for (x in (0.1 * w).toInt() until (0.9 * w).toInt()) mask[y * w + x] = true
        for (d in dets) {
            if (d.label !in MASK_LABELS) continue
            val bx1 = max(0, ((d.bbox.cx - d.bbox.w / 2.0) * w).toInt())
            val bx2 = min(w, ((d.bbox.cx + d.bbox.w / 2.0) * w).toInt() + 1)
            val by1 = max(0, ((d.bbox.cy - d.bbox.h / 2.0) * h).toInt())
            val by2 = min(h, ((d.bbox.cy + d.bbox.h / 2.0) * h).toInt() + 1)
            for (y in by1 until by2) for (x in bx1 until bx2) mask[y * w + x] = false
        }
        val pts = Flow.corners(img, cfg.roadCorners, 0.005, 6.0, mask)
        val (pa, pb) = Flow.track(a.take(cfg.roadLk[1] + 1), b.take(cfg.roadLk[1] + 1), pts, cfg.roadLk[0])
        if (pa.size < 15) return null
        val fit = Flow.roadPairRate(pa, pb, h, dt, cfg.minRoadPairs) ?: return null
        // a flat road's spread rate is zero at the horizon; a fit that puts that row far off is not the road
        if (cfg.horizonTol > 0 && (!(fit[0] > 0) || abs(-fit[1] / fit[0] - cfg.horizonY) > cfg.horizonTol)) return null
        val fx = w / 2.0
        val fy = cfg.horizonY * h
        val rx = DoubleArray(pa.size)
        val ry = DoubleArray(pa.size)
        for (i in pa.indices) {
            val e = fit[0] * pa[i][1] / h + fit[1]
            rx[i] = pb[i][0] - pa[i][0] - (pa[i][0] - fx) * e * dt
            ry[i] = pb[i][1] - pa[i][1] - (pa[i][1] - fy) * e * dt
        }
        return doubleArrayOf(fit[0], fit[1], Flow.median(rx), Flow.median(ry))
    }

    /** The car's corners (inner 80 % of its box) followed from [a] into [b]. */
    private fun follow(a: List<GreyImage>, b: List<GreyImage>, box: BBox): Pair<List<DoubleArray>, List<DoubleArray>>? {
        val img = a[0]
        val w = img.w
        val h = img.h
        if (box.w * w < 24 || box.h * h < 16) return null
        val mask = BooleanArray(w * h)
        val y1 = max(0, ((box.cy - 0.4 * box.h) * h).toInt())
        val y2 = min(h, ((box.cy + 0.4 * box.h) * h).toInt())
        val x1 = max(0, ((box.cx - 0.4 * box.w) * w).toInt())
        val x2 = min(w, ((box.cx + 0.4 * box.w) * w).toInt())
        for (y in y1 until y2) for (x in x1 until x2) mask[y * w + x] = true
        val pts = Flow.corners(img, cfg.carCorners, 0.01, 3.0, mask)
        if (pts.size < cfg.minCarPoints) return null
        val r = Flow.track(a.take(cfg.carLk[1] + 1), b.take(cfg.carLk[1] + 1), pts, cfg.carLk[0])
        return if (r.first.size >= cfg.minCarPoints) r else null
    }

    /** (flow residual, spread ratio) of a car whose corners moved from [pa] to [pb]. */
    private fun measure(pa: List<DoubleArray>, pb: List<DoubleArray>, box: BBox, road: DoubleArray, dt: Double,
                        h: Int, w: Int): DoubleArray? {
        val e = road[0] * (box.cy + box.h / 2.0) + road[1]
        if (!(e > cfg.minExpansion)) return null
        val cx = Flow.median(DoubleArray(pa.size) { pa[it][0] })
        val cy = Flow.median(DoubleArray(pa.size) { pa[it][1] })
        val px = (cx - w / 2.0) * e * dt
        val py = (cy - cfg.horizonY * h) * e * dt
        val ax = Flow.median(DoubleArray(pa.size) { pb[it][0] - pa[it][0] }) - road[2]
        val ay = Flow.median(DoubleArray(pa.size) { pb[it][1] - pa[it][1] }) - road[3]
        val resid = hypot(ax - px, ay - py) / (hypot(px, py) + 2.0)
        val s = Flow.spread(pa, pb) ?: return null
        return doubleArrayOf(resid, s / dt / e)
    }

    companion object {
        const val POLICE = "police_vehicle"
        const val OTHER_SIDE = "scene_police_other_side"
        val MASK_LABELS = setOf("car", "truck", "bus", "person", "police_vehicle", "ambulance", "fire_truck",
            "bicycle", "motorcycle")
    }
}
