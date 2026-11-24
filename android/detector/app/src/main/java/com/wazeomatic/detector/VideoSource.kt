package com.wazeomatic.detector

import android.graphics.Bitmap
import android.media.Image
import android.media.MediaCodec
import android.media.MediaCodecInfo
import android.media.MediaExtractor
import android.media.MediaFormat
import android.util.Log
import java.io.File

/**
 * Decodes an MP4 with MediaExtractor + MediaCodec (ByteBuffer/Image output, no surface) and
 * hands every [frameStep]-th frame to [onFrame] on a delivery thread, with its position in the clip in
 * seconds (growing across loops, so time never runs backwards). Runs as fast as inference allows, which is
 * what we want for repeatable tests; the scene rules run on the clip time, as they do on the desktop.
 * The decoder works one frame ahead of [onFrame] (a queue of one), so decoding and the YUV conversion overlap
 * with inference instead of adding to it, as a camera's frames would.
 *
 * With [realtime] the clip plays at its own speed instead, like a camera: the decoder keeps to the wall clock and
 * only converts the frame the pipeline will take next (when it is waiting, or due to be free by the time the
 * conversion is done; at most [maxFps] a second), so a slow phone skips frames and always works on a fresh one
 * instead of falling behind ([frameStep] is not used). The third argument of
 * [onFrame] is then the frame's due time on the elapsedRealtimeNanos clock (0 otherwise), for the lag log.
 */
class VideoSource(
    private val file: File,
    private val frameStep: Int,
    private val loop: Boolean,
    private val onFrame: (Bitmap, Double, Long) -> Unit,
    private val onStatus: (String) -> Unit,
    private val realtime: Boolean = false,
    maxFps: Float = 6f,
) {
    @Volatile private var running = false
    private val throttle = FrameThrottle((1e9 / maxFps.coerceIn(0.5f, 30f)).toLong())
    // realtime: the consumer is waiting; when it last took a frame; smoothed busy time a frame and conversion time
    @Volatile private var idle = false
    @Volatile private var lastTakeNs = 0L
    @Volatile private var busyNs = 0.0
    private var convertNs = 0.0
    private var thread: Thread? = null
    private var consumer: Thread? = null
    private val queue = java.util.concurrent.ArrayBlockingQueue<Triple<Bitmap, Double, Long>>(1)

    fun start() {
        running = true
        queue.clear()
        consumer = Thread({
            while (running) {
                if (!idle && lastTakeNs != 0L) {
                    // busy time of the frame just done: from taking it to asking for the next
                    val b = (android.os.SystemClock.elapsedRealtimeNanos() - lastTakeNs).toDouble()
                    busyNs = if (busyNs == 0.0) b else busyNs * 0.8 + b * 0.2
                }
                idle = true
                val (bmp, ts, due) = queue.poll(100, java.util.concurrent.TimeUnit.MILLISECONDS) ?: continue
                idle = false
                lastTakeNs = android.os.SystemClock.elapsedRealtimeNanos()
                try {
                    onFrame(bmp, ts, due)
                } catch (e: Exception) {
                    Log.e(TAG, "frame handler failed", e)
                }
            }
        }, "video-frames").apply { start() }
        thread = Thread({
            try {
                decodeLoop()
            } catch (e: Exception) {
                Log.e(TAG, "video decode failed", e)
                onStatus("video error: ${e.message}")
            }
        }, "video-decode").apply { start() }
    }

    fun stop() {
        running = false
        thread?.join(3000)
        consumer?.join(3000)
    }

    /** Waits while the previous frame is still queued; false when stopped meanwhile. */
    private fun deliver(bmp: Bitmap, ts: Double, due: Long = 0L): Boolean {
        while (running) if (queue.offer(Triple(bmp, ts, due), 100, java.util.concurrent.TimeUnit.MILLISECONDS)) return true
        return false
    }

    private fun decodeLoop() {
        if (!file.exists()) {
            onStatus("video not found: ${file.absolutePath}")
            return
        }
        val ex = MediaExtractor()
        ex.setDataSource(file.absolutePath)
        var track = -1
        var format: MediaFormat? = null
        for (i in 0 until ex.trackCount) {
            val f = ex.getTrackFormat(i)
            if (f.getString(MediaFormat.KEY_MIME)?.startsWith("video/") == true) {
                track = i
                format = f
                break
            }
        }
        if (track < 0 || format == null) {
            onStatus("no video track in ${file.name}")
            ex.release()
            return
        }
        ex.selectTrack(track)
        val mime = format.getString(MediaFormat.KEY_MIME)!!
        format.setInteger(MediaFormat.KEY_COLOR_FORMAT, MediaCodecInfo.CodecCapabilities.COLOR_FormatYUV420Flexible)
        val codec = MediaCodec.createDecoderByType(mime)
        codec.configure(format, null, null, 0)
        codec.start()
        Log.i(TAG, "video ${file.name}: $mime ${format.getInteger(MediaFormat.KEY_WIDTH)}x${format.getInteger(MediaFormat.KEY_HEIGHT)} " +
            (if (realtime) "real time" else "step=$frameStep") + " loop=$loop")
        onStatus("video ${file.name}")

        val info = MediaCodec.BufferInfo()
        var inputDone = false
        var decoded = 0L
        var pixels = IntArray(0)
        var loopOffsetUs = 0L       // clip time of the loops already played
        var lastPtsUs = 0L
        var startWallNs = 0L        // realtime: wall clock and clip time of the first frame
        var startClipUs = 0L
        var taken = 0L
        try {
            while (running) {
                if (!inputDone) {
                    val inIdx = codec.dequeueInputBuffer(10_000)
                    if (inIdx >= 0) {
                        val buf = codec.getInputBuffer(inIdx)!!
                        val n = ex.readSampleData(buf, 0)
                        if (n < 0) {
                            codec.queueInputBuffer(inIdx, 0, 0, 0, MediaCodec.BUFFER_FLAG_END_OF_STREAM)
                            inputDone = true
                        } else {
                            codec.queueInputBuffer(inIdx, 0, n, ex.sampleTime, 0)
                            ex.advance()
                        }
                    }
                }
                val outIdx = codec.dequeueOutputBuffer(info, 10_000)
                if (outIdx < 0) continue
                var bmp: Bitmap? = null
                var dueNs = 0L
                val take = if (!realtime) decoded % frameStep == 0L else if (info.size <= 0) false else {
                    val clipUs = loopOffsetUs + info.presentationTimeUs
                    if (startWallNs == 0L) {
                        startWallNs = android.os.SystemClock.elapsedRealtimeNanos()
                        startClipUs = clipUs
                    }
                    dueNs = startWallNs + (clipUs - startClipUs) * 1000
                    // ahead of the clock: hold the frame until its time, as a camera would deliver it
                    val early = dueNs - android.os.SystemClock.elapsedRealtimeNanos()
                    if (early > 0) Thread.sleep(early / 1_000_000, (early % 1_000_000).toInt())
                    // convert the frame the pipeline will take next: at once when it is waiting, or when it is
                    // expected to be free by the time the conversion is done (its smoothed busy time). Earlier frames
                    // are skipped unconverted, so it always works on a fresh frame without waiting for one.
                    val now = android.os.SystemClock.elapsedRealtimeNanos()
                    val soon = queue.isEmpty() && busyNs > 0 && now - lastTakeNs >= busyNs - convertNs - FRAME_SLACK_NS
                    (idle || soon) && throttle.accept(dueNs)
                }
                if (info.size > 0 && take) {
                    val img = codec.getOutputImage(outIdx)
                    if (img != null) {
                        val c0 = System.nanoTime()
                        val r = img.cropRect
                        if (pixels.size != r.width() * r.height()) pixels = IntArray(r.width() * r.height())
                        yuvToArgb(img, pixels)
                        bmp = Bitmap.createBitmap(pixels, r.width(), r.height(), Bitmap.Config.ARGB_8888)
                        img.close()
                        val c = (System.nanoTime() - c0).toDouble()
                        convertNs = if (convertNs == 0.0) c else convertNs * 0.8 + c * 0.2
                    }
                }
                if (info.size > 0) {
                    decoded++
                    lastPtsUs = info.presentationTimeUs
                }
                val ts = (loopOffsetUs + info.presentationTimeUs) / 1e6
                codec.releaseOutputBuffer(outIdx, false)
                bmp?.let { taken++; deliver(it, ts, dueNs) }
                if (info.flags and MediaCodec.BUFFER_FLAG_END_OF_STREAM != 0) {
                    val note = if (realtime) "$decoded frames, $taken in real time" else "$decoded frames"
                    if (!loop) {
                        onStatus("video finished ($note)")
                        break
                    }
                    Log.i(TAG, "video looped after $note")
                    loopOffsetUs += lastPtsUs + 33_333      // about one frame after the last one
                    ex.seekTo(0, MediaExtractor.SEEK_TO_PREVIOUS_SYNC)
                    codec.flush()
                    inputDone = false
                }
            }
        } finally {
            try { codec.stop() } catch (_: Exception) {}
            codec.release()
            ex.release()
        }
    }

    companion object {
        private const val TAG = "Wazeomatic"
        // realtime: start the conversion this much early, about half a frame of a 30 fps clip
        private const val FRAME_SLACK_NS = 15_000_000.0

        /** YUV_420_888 (any plane layout) -> ARGB, BT.601 limited range. */
        fun yuvToArgb(img: Image, out: IntArray) {
            val r = img.cropRect
            val w = r.width()
            val h = r.height()
            val planes = img.planes
            val yArr = ByteArray(planes[0].buffer.remaining()).also { planes[0].buffer.get(it) }
            val uArr = ByteArray(planes[1].buffer.remaining()).also { planes[1].buffer.get(it) }
            val vArr = ByteArray(planes[2].buffer.remaining()).also { planes[2].buffer.get(it) }
            val yRs = planes[0].rowStride
            val yPs = planes[0].pixelStride
            val uRs = planes[1].rowStride
            val uPs = planes[1].pixelStride
            val vRs = planes[2].rowStride
            val vPs = planes[2].pixelStride
            var o = 0
            for (y in 0 until h) {
                val sy = y + r.top
                val yRow = sy * yRs
                val uRow = (sy shr 1) * uRs
                val vRow = (sy shr 1) * vRs
                for (x in 0 until w) {
                    val sx = x + r.left
                    val yy = ((yArr[yRow + sx * yPs].toInt() and 0xff) - 16).coerceAtLeast(0) * 1192
                    val u = (uArr[uRow + (sx shr 1) * uPs].toInt() and 0xff) - 128
                    val v = (vArr[vRow + (sx shr 1) * vPs].toInt() and 0xff) - 128
                    val rr = ((yy + 1634 * v) shr 10).coerceIn(0, 255)
                    val gg = ((yy - 833 * v - 400 * u) shr 10).coerceIn(0, 255)
                    val bb = ((yy + 2066 * u) shr 10).coerceIn(0, 255)
                    out[o++] = (0xff shl 24) or (rr shl 16) or (gg shl 8) or bb
                }
            }
        }
    }
}
