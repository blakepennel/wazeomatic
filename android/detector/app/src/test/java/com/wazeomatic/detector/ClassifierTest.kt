package com.wazeomatic.detector

import org.junit.Assert.assertArrayEquals
import org.junit.Assert.assertEquals
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Test
import kotlin.math.abs

/**
 * Image helpers against numbers from OpenCV 5.0 (cv2.cvtColor / cv2.resize / classifiers.square_crop on the same
 * synthetic image), and the second stage and scene classifiers with a fake model.
 */
class ClassifierTest {
    /** 37 x 23, r = (37x + 11y) % 256, g = (7x + 53y) % 256, b = (x^2 + 3y) % 256. */
    private fun pattern(): RgbImage {
        val w = 37
        val h = 23
        return RgbImage(w, h, IntArray(w * h) { i ->
            val x = i % w
            val y = i / w
            RgbImage.pack((x * 37 + y * 11) % 256, (x * 7 + y * 53) % 256, (x * x + y * 3) % 256)
        })
    }

    private fun sums(img: RgbImage): LongArray {
        val s = LongArray(3)
        for (p in img.argb) { s[0] += ((p shr 16) and 0xff).toLong(); s[1] += ((p shr 8) and 0xff).toLong(); s[2] += (p and 0xff).toLong() }
        return s
    }

    private fun assertSumsNear(expected: LongArray, actual: LongArray, tol: Long) {
        for (i in 0..2) assertTrue("channel $i: ${actual.toList()} vs ${expected.toList()}", abs(expected[i] - actual[i]) <= tol)
    }

    @Test
    fun hsvAndGreyMatchOpenCv() {
        val img = pattern()
        var hs = 0L; var ss = 0L; var vs = 0L; var gs = 0L
        for (p in img.argb) {
            val v = ImageOps.hsv(p)
            hs += v shr 16; ss += (v shr 8) and 0xff; vs += v and 0xff
            gs += ImageOps.grey(p)
        }
        assertEquals(listOf(69170L, 150686L, 159665L), listOf(hs, ss, vs))
        assertEquals(106786L, gs)
        fun hsvAt(x: Int, y: Int) = ImageOps.hsv(img.argb[y * 37 + x]).let { listOf(it shr 16, (it shr 8) and 0xff, it and 0xff) }
        assertEquals(listOf(120, 24, 64), hsvAt(7, 5))
        assertEquals(listOf(82, 203, 246), hsvAt(30, 20))
    }

    @Test
    fun areaResizeMatchesOpenCv() {
        val small = ImageOps.resize(pattern(), 10, 7)
        assertSumsNear(longArrayOf(8914, 8905, 7827), sums(small), 35)       // rounding differences only
        val p = small.argb[3 * 10 + 4]
        val px = intArrayOf((p shr 16) and 0xff, (p shr 8) and 0xff, p and 0xff)
        for ((e, a) in intArrayOf(171, 173, 77).zip(px)) assertTrue("$e vs $a", abs(e - a) <= 1)
    }

    @Test
    fun greySmallMatchesEgoMotion() {
        // the pattern blown up 4x (nearest), then box-averaged to 80 x 45 as EgoMotion.grey_small does
        val src = pattern()
        val big = RgbImage(148, 92, IntArray(148 * 92) { i -> src.argb[(i / 148 / 4) * 37 + (i % 148) / 4] })
        val g = ImageOps.greySmall(big, 80, 45)
        assertEquals(451738.625, g.sum().toDouble(), 2.0)
        assertEquals(93.0, g[10 * 80 + 20].toDouble(), 0.01)
    }

    @Test
    fun squareCropMatchesPython() {
        val down = ImageOps.squareCrop(pattern(), BBox(0.45f, 0.55f, 0.4f, 0.6f), size = 16)
        assertEquals(16, down.width)
        assertSumsNear(longArrayOf(33206, 31723, 32617), sums(down), 256)
        val up = ImageOps.squareCrop(pattern(), BBox(0.5f, 0.5f, 0.2f, 0.2f), size = 32)
        assertSumsNear(longArrayOf(124728, 112483, 130558), sums(up), 1024)  // OpenCV's bilinear is fixed point
        // a crop is letterboxed with 114 grey bars on the short side
        val wide = ImageOps.squareCrop(RgbImage.filled(200, 50, 10, 20, 30), BBox(0.5f, 0.5f, 0.5f, 0.2f), size = 32)
        assertEquals(RgbImage.pack(114, 114, 114), wide.argb[0])
        assertEquals(RgbImage.pack(10, 20, 30), wide.argb[16 * 32 + 16])
    }

    @Test
    fun softmaxOnlyForLogits() {
        val probs = arrayOf(floatArrayOf(0.7f, 0.2f, 0.1f))
        assertTrue(OnnxClassifier.softmaxIfNeeded(probs) === probs)
        val sm = OnnxClassifier.softmaxIfNeeded(arrayOf(floatArrayOf(2f, 1f, -1f), floatArrayOf(0f, 0f, 0f)))
        assertEquals(1.0, sm[0].sum().toDouble(), 1e-5)
        assertTrue(sm[0][0] > sm[0][1] && sm[0][1] > sm[0][2])
        assertArrayEquals(floatArrayOf(1 / 3f, 1 / 3f, 1 / 3f), sm[1], 1e-6f)
        assertEquals(listOf("civilian", "police"), OnnxClassifier.parseLabels("civilian\r\n\npolice\n"))
    }

    private class FakeClassifier(override val labels: List<String>, override val size: Int = 64, val row: (Int) -> FloatArray) : ImageClassifier {
        val seen = ArrayList<RgbImage>()
        override val lastMs = 1.5
        override fun predict(crops: List<RgbImage>): Array<FloatArray> {
            seen += crops
            return Array(crops.size) { row(seen.size - crops.size + it) }
        }
    }

    @Test
    fun vehicleCandidatesAreTheLargestConfidentVehicles() {
        val fake = FakeClassifier(listOf("civilian", "police", "ambulance", "fire")) { floatArrayOf(0.1f, 0.8f, 0.06f, 0.04f) }
        val vc = VehicleClassifier(fake)
        val dets = listOf(
            Detection("car", 0.9f, BBox(0.5f, 0.5f, 0.10f, 0.10f), "coco"),
            Detection("truck", 0.9f, BBox(0.5f, 0.5f, 0.30f, 0.30f), "coco"),
            Detection("bus", 0.3f, BBox(0.5f, 0.5f, 0.50f, 0.50f), "coco"),      // below min_vehicle_conf
            Detection("car", 0.9f, BBox(0.5f, 0.5f, 0.02f, 0.40f), "coco"),      // 25.6 px wide at 1280: too thin
            Detection("person", 0.9f, BBox(0.5f, 0.5f, 0.40f, 0.40f), "coco"),
            Detection("car", 0.9f, BBox(0.2f, 0.5f, 0.20f, 0.20f), "coco"),
            Detection("car", 0.9f, BBox(0.8f, 0.5f, 0.15f, 0.15f), "coco"),
            Detection("car", 0.9f, BBox(0.3f, 0.3f, 0.05f, 0.05f), "coco"),
        )
        val cand = vc.candidates(1280, 720, dets)
        assertEquals(listOf(0.30f, 0.20f, 0.15f, 0.10f), cand.map { it.bbox.w })      // largest four
        val out = vc(RgbImage.filled(1280, 720, 90, 90, 90), dets)
        assertEquals(4, fake.seen.size)
        assertTrue(fake.seen.all { it.width == 64 && it.height == 64 })
        // police 0.8 and ambulance 0.06 per crop; fire 0.04 is under 0.05
        assertEquals(List(4) { listOf("police_vehicle", "ambulance") }.flatten(), out.map { it.label })
        assertTrue(out.all { it.model == "vehicle" })
        assertEquals(cand[0].bbox, out[0].bbox)
        assertEquals(1.5, vc.lastMs, 0.0)
        assertTrue(vc(RgbImage.filled(64, 64, 0, 0, 0), dets.filter { it.label == "person" }).isEmpty())
    }

    @Test
    fun sceneClassifiersRunEveryHalfSecondAndPrefixSurface() {
        val weather = SceneClassifier(FakeClassifier(listOf("clear", "fog"), 32) { floatArrayOf(0.2f, 0.8f) })
        val surface = RoadSurface(FakeClassifier(listOf("dry", "wet"), 32) { floatArrayOf(0.4f, 0.6f) })
        val both = SceneClassifiers(listOf(weather, null, surface))
        val img = RgbImage.filled(320, 180, 50, 50, 50)
        assertEquals(mapOf("clear" to 0.2f, "fog" to 0.8f, "surface_dry" to 0.4f, "surface_wet" to 0.6f), both(img, 0.0))
        assertNull(both(img, 0.3))
        assertTrue(both(img, 0.5) != null)
    }

    @Test
    fun roadPatchMovesAboveTheHood() {
        val rs = RoadSurface(FakeClassifier(listOf("dry"), 32) { floatArrayOf(1f) })
        assertArrayEquals(intArrayOf(512, 432, 768, 619), rs.region(720, 1280))        // the default patch, 0.60 to 0.86
        rs.setHoodFrac(0.30)
        // bottom 0.68, top max(0.45, 0.68 - 0.26) = 0.45
        assertArrayEquals(intArrayOf(512, 324, 768, 489), rs.region(720, 1280))
    }

    /** 320 x 180, r = (37x + 11y) % 256, g = (7x + 53y) % 256, b = (x^2 + 3y) % 256. */
    private fun bigPattern(): RgbImage = RgbImage(320, 180, IntArray(320 * 180) { i ->
        val x = i % 320
        val y = i / 320
        RgbImage.pack((x * 37 + y * 11) % 256, (x * 7 + y * 53) % 256, (x * x + y * 3) % 256)
    })

    @Test
    fun topViewWarpMatchesOpenCv() {
        // classifiers.topview_quad(320, 180, 70, 1.3, 0.5, 7.5, 3.6) and cv2.warpPerspective to 32 x 32
        val q = RoadSurface.topviewQuad(320, 180, 70.0, 1.3, 0.5, 7.5, 3.6)
        val ref = arrayOf(doubleArrayOf(115.773, 121.941), doubleArrayOf(204.227, 121.941),
            doubleArrayOf(232.159, 142.115), doubleArrayOf(87.841, 142.115))
        for (i in 0..3) assertArrayEquals(ref[i], q[i], 1e-3)
        val out = RoadSurface.warpToSquare(bigPattern(), q, 32)
        assertEquals(32, out.width)
        // OpenCV interpolates on a 1/32 pixel grid, so compare sums, not single pixels
        assertSumsNear(longArrayOf(129852, 131239, 130851), sums(out), 700)
    }

    @Test
    fun topViewSkipsACoveredRoadAndTheHood() {
        val fake = FakeClassifier(listOf("damp", "dry", "slush", "snow", "wet"), 32) { floatArrayOf(0f, 1f, 0f, 0f, 0f) }
        val rs = RoadSurface(fake)
        assertTrue(rs.isTopView)
        rs.topview = doubleArrayOf(70.0, 1.3, 0.5)
        val img = RgbImage.filled(1280, 720, 90, 90, 90)
        // a car right ahead covers the square 15 m ahead (rows ~0.59-0.63, columns ~0.42-0.58)
        val car = Detection("car", 0.9f, BBox(0.5f, 0.6f, 0.3f, 0.2f), model = "coco")
        assertNull(rs(img, 0.0, listOf(car)))
        assertEquals(1, rs.skipped)
        // a car in the next lane does not
        val side = Detection("car", 0.9f, BBox(0.85f, 0.6f, 0.2f, 0.2f), model = "coco")
        assertEquals(1f, rs(img, 0.1, listOf(side))!!["surface_dry"])
        assertTrue(fake.seen.last().width == 32)
        // a mount so low on the frame that the square falls into the hood zone is skipped too
        rs.topview = doubleArrayOf(70.0, 1.3, 0.85)
        assertNull(rs(img, 1.0, emptyList()))
        assertEquals(2, rs.skipped)
    }

    @Test
    fun cameraGeometrySettingReachesTheSceneRules() {
        val g = Settings(cameraGeometry = "47.2,1.44,0.53").sceneConfigOrNull()!!
        assertEquals(47.2, g.hfovDeg, 0.0)
        assertEquals(1.44, g.camHeightM, 0.0)
        assertEquals(0.53, g.horizonY, 0.0)
        // malformed or out of range -> the default 70,1.3,0.5
        assertArrayEquals(doubleArrayOf(70.0, 1.3, 0.5), Settings(cameraGeometry = "70,1.3").geometry, 0.0)
        assertArrayEquals(doubleArrayOf(70.0, 1.3, 0.5), Settings(cameraGeometry = "70,13,0.5").geometry, 0.0)
    }
}
