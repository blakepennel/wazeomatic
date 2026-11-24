package com.wazeomatic.detector

import android.content.Context
import android.graphics.Bitmap
import android.media.MediaMetadataRetriever
import android.net.Uri
import android.provider.OpenableColumns
import android.util.Log
import android.util.LruCache
import java.io.File
import java.util.Locale

/**
 * The video clips in the app's files folder (where `adb push` puts them and where the video source reads them),
 * and copying a clip in from a content URI (the document picker, "share" or "open with").
 */
object Clips {
    private const val TAG = "Wazeomatic"
    val VIDEO_EXT = setOf("mp4", "m4v", "mkv", "webm", "3gp", "mov", "ts")

    fun isVideoName(name: String): Boolean = name.substringAfterLast('.', "").lowercase() in VIDEO_EXT

    /** A display name made safe as a file name in one folder: no path, no odd characters, a video extension. */
    fun safeName(raw: String?): String {
        val base = (raw ?: "").substringAfterLast('/').substringAfterLast('\\').trim()
        var s = base.replace(Regex("[^A-Za-z0-9._ -]"), "_").replace(Regex("\\s+"), "_").trim('.', '_', '-')
        if (s.isEmpty()) s = "clip"
        if (!isVideoName(s)) s += ".mp4"
        return s.take(120)
    }

    /** [name], or name-2.mp4, name-3.mp4 ... for the first one [exists] says is free. */
    fun uniqueName(name: String, exists: (String) -> Boolean): String {
        if (!exists(name)) return name
        val dot = name.lastIndexOf('.')
        val stem = if (dot > 0) name.substring(0, dot) else name
        val ext = if (dot > 0) name.substring(dot) else ""
        var n = 2
        while (exists("$stem-$n$ext")) n++
        return "$stem-$n$ext"
    }

    fun formatSize(bytes: Long): String = when {
        bytes < 0 -> "?"
        bytes < 1024 -> "$bytes B"
        bytes < 1024L * 1024 -> String.format(Locale.US, "%.0f KB", bytes / 1024.0)
        bytes < 1024L * 1024 * 1024 -> String.format(Locale.US, "%.1f MB", bytes / (1024.0 * 1024))
        else -> String.format(Locale.US, "%.2f GB", bytes / (1024.0 * 1024 * 1024))
    }

    fun formatDuration(ms: Long): String {
        val s = (ms / 1000).coerceAtLeast(0)
        return if (s >= 3600) String.format(Locale.US, "%d:%02d:%02d", s / 3600, s / 60 % 60, s % 60)
        else String.format(Locale.US, "%d:%02d", s / 60, s % 60)
    }

    /** Same folder DetectorService reads the video setting from. */
    fun dir(ctx: Context): File = ctx.getExternalFilesDir(null) ?: ctx.filesDir

    /** The file the video setting names: a name in [dir], or an absolute path (as DetectorService resolves it). */
    fun resolve(ctx: Context, video: String): File = if (video.startsWith("/")) File(video) else File(dir(ctx), video)

    /** Video files in the folder, newest first. */
    fun list(ctx: Context): List<File> =
        (dir(ctx).listFiles() ?: emptyArray()).filter { it.isFile && isVideoName(it.name) }.sortedByDescending { it.lastModified() }

    class Info(val durationMs: Long?, val width: Int?, val height: Int?, val thumb: Bitmap?)

    private val infoCache = LruCache<String, Info>(64)

    fun cachedInfo(f: File): Info? = infoCache.get(cacheKey(f))

    private fun cacheKey(f: File) = "${f.absolutePath}:${f.length()}:${f.lastModified()}"

    /** Length, size and a small thumbnail; slow (opens the file), call it off the main thread. */
    fun info(f: File): Info {
        infoCache.get(cacheKey(f))?.let { return it }
        val r = MediaMetadataRetriever()
        val info = try {
            r.setDataSource(f.absolutePath)
            val dur = r.extractMetadata(MediaMetadataRetriever.METADATA_KEY_DURATION)?.toLongOrNull()
            val w = r.extractMetadata(MediaMetadataRetriever.METADATA_KEY_VIDEO_WIDTH)?.toIntOrNull()
            val h = r.extractMetadata(MediaMetadataRetriever.METADATA_KEY_VIDEO_HEIGHT)?.toIntOrNull()
            val at = ((dur ?: 0L) / 10).coerceAtMost(3000L) * 1000   // a frame a little way in, not the black first one
            val thumb = try {
                r.getScaledFrameAtTime(at, MediaMetadataRetriever.OPTION_CLOSEST_SYNC, 224, 126)
            } catch (e: Exception) {
                null
            }
            Info(dur, w, h, thumb)
        } catch (e: Exception) {
            Log.w(TAG, "clip info failed for ${f.name}: ${e.message}")
            Info(null, null, null, null)
        } finally {
            try { r.release() } catch (_: Exception) {}
        }
        infoCache.put(cacheKey(f), info)
        return info
    }

    /** "24.9 MB  ·  0:15  ·  1548×1278"; [twoLines] puts the picture size on a line of its own. */
    fun describe(f: File, info: Info?, twoLines: Boolean = false): String {
        val first = listOfNotNull(formatSize(f.length()), info?.durationMs?.let { formatDuration(it) }).joinToString("  ·  ")
        val size = if (info?.width != null && info.height != null) "${info.width}×${info.height}" else return first
        return if (twoLines) "$first\n$size" else "$first  ·  $size"
    }

    /** A copy into the clips folder; the UI polls it, so it survives the activity being recreated. */
    class Import(val name: String, val total: Long) {
        @Volatile var copied = 0L
        @Volatile var done = false
        @Volatile var error: String? = null
        @Volatile var cancelled = false
        /** The same file was already in the folder: nothing was copied. */
        @Volatile var reused = false
        /** The result has been shown to the user. */
        @Volatile var delivered = false
    }

    @Volatile var currentImport: Import? = null
        private set

    /**
     * Copies [uri] into the folder on a background thread (a file with the same name and size is reused). The
     * video source reads plain files, so a picked clip is copied once rather than streamed from the provider.
     */
    fun startImport(ctx: Context, uri: Uri): Import {
        currentImport?.let { if (!it.done) return it }
        val app = ctx.applicationContext
        val cr = app.contentResolver
        var display: String? = null
        var size = -1L
        try {
            cr.query(uri, arrayOf(OpenableColumns.DISPLAY_NAME, OpenableColumns.SIZE), null, null, null)?.use { c ->
                if (c.moveToFirst()) {
                    val ni = c.getColumnIndex(OpenableColumns.DISPLAY_NAME)
                    val si = c.getColumnIndex(OpenableColumns.SIZE)
                    if (ni >= 0 && !c.isNull(ni)) display = c.getString(ni)
                    if (si >= 0 && !c.isNull(si)) size = c.getLong(si)
                }
            }
        } catch (e: Exception) {
            Log.w(TAG, "clip import: no name for $uri: ${e.message}")
        }
        if (display == null && uri.scheme == "file") {
            display = uri.lastPathSegment
            size = uri.path?.let { File(it).length() } ?: -1L
        }
        val folder = dir(app)
        val wanted = safeName(display)
        val existing = File(folder, wanted)
        if (existing.isFile && size > 0 && existing.length() == size) {
            return Import(wanted, size).also { it.reused = true; it.copied = size; it.done = true; currentImport = it }
        }
        val name = uniqueName(wanted) { File(folder, it).exists() }
        val job = Import(name, size)
        currentImport = job
        if (size > 0 && folder.usableSpace < size + 64L * 1024 * 1024) {
            job.error = "Not enough free space: the clip needs ${formatSize(size)}, ${formatSize(folder.usableSpace)} is free"
            job.done = true
            return job
        }
        Thread({
            val part = File(folder, "$name.part")
            try {
                val input = cr.openInputStream(uri) ?: throw IllegalStateException("the clip could not be opened")
                input.use { inp ->
                    part.outputStream().use { out ->
                        val buf = ByteArray(1 shl 20)
                        while (!job.cancelled) {
                            val n = inp.read(buf)
                            if (n < 0) break
                            out.write(buf, 0, n)
                            job.copied += n
                        }
                    }
                }
                if (job.cancelled) {
                    part.delete()
                } else if (!part.renameTo(File(folder, name))) {
                    throw IllegalStateException("could not save $name")
                } else {
                    Log.i(TAG, "clip import: $name, ${formatSize(job.copied)}")
                }
            } catch (e: Exception) {
                Log.e(TAG, "clip import failed", e)
                part.delete()
                job.error = e.message ?: e.javaClass.simpleName
            }
            job.done = true
        }, "clip-import").start()
        return job
    }
}
