package com.wazeomatic.detector

import android.net.Uri
import android.os.Handler
import android.os.Looper
import android.view.LayoutInflater
import android.view.View
import android.view.ViewGroup
import android.widget.ImageView
import android.widget.TextView
import android.widget.Toast
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AlertDialog
import androidx.appcompat.app.AppCompatActivity
import androidx.lifecycle.DefaultLifecycleObserver
import androidx.lifecycle.LifecycleOwner
import androidx.recyclerview.widget.LinearLayoutManager
import androidx.recyclerview.widget.RecyclerView
import com.google.android.material.bottomsheet.BottomSheetBehavior
import com.google.android.material.bottomsheet.BottomSheetDialog
import com.google.android.material.button.MaterialButton
import com.google.android.material.dialog.MaterialAlertDialogBuilder
import com.google.android.material.progressindicator.LinearProgressIndicator
import java.io.File
import java.util.concurrent.Executors

/**
 * Choosing the video clip: a bottom sheet with the clips already in the app's folder, and "add a clip" through the
 * system document picker. A picked clip is copied into the folder (with a progress dialog) and then selected.
 * Create it while the activity is being constructed: the picker launcher has to be registered before onStart.
 */
class ClipPicker(private val act: AppCompatActivity, private val onChosen: (String) -> Unit) {
    private val main = Handler(Looper.getMainLooper())
    private var sheet: BottomSheetDialog? = null
    private var progress: AlertDialog? = null
    private var watching: Clips.Import? = null

    private val openDocument = act.registerForActivityResult(ActivityResultContracts.OpenDocument()) { uri: Uri? ->
        if (uri != null) import(uri)
    }

    init {
        act.lifecycle.addObserver(object : DefaultLifecycleObserver {
            override fun onResume(owner: LifecycleOwner) {
                // an import started before the activity was recreated (rotation): show its progress again
                Clips.currentImport?.let { if (!it.delivered && watching !== it) watch(it) }
            }

            override fun onDestroy(owner: LifecycleOwner) {
                main.removeCallbacksAndMessages(null)
                sheet?.dismiss()
                progress?.dismiss()
            }
        })
    }

    fun pickFromPhone() {
        try {
            openDocument.launch(arrayOf("video/*"))
        } catch (e: Exception) {
            Toast.makeText(act, "No file picker on this phone: ${e.message}", Toast.LENGTH_LONG).show()
        }
    }

    /** Copies [uri] into the clips folder, then selects it. */
    fun import(uri: Uri) {
        sheet?.dismiss()
        watch(Clips.startImport(act, uri))
    }

    private fun watch(job: Clips.Import) {
        watching = job
        if (!job.done && progress == null) {
            val v = LayoutInflater.from(act).inflate(R.layout.dialog_import, null)
            progress = MaterialAlertDialogBuilder(act)
                .setTitle("Adding clip")
                .setView(v)
                .setCancelable(false)
                .setNegativeButton("Cancel") { _, _ -> job.cancelled = true }
                .show()
        }
        tick()
    }

    private fun tick() {
        val job = watching ?: return
        val d = progress
        if (d != null) {
            val bar = d.findViewById<LinearProgressIndicator>(R.id.importProgress)
            val text = d.findViewById<TextView>(R.id.importText)
            if (job.total > 0) {
                bar?.isIndeterminate = false
                bar?.setProgressCompat((job.copied * 1000 / job.total).toInt().coerceIn(0, 1000), true)
                text?.text = "${job.name}\n${Clips.formatSize(job.copied)} of ${Clips.formatSize(job.total)}"
            } else {
                bar?.isIndeterminate = true
                text?.text = "${job.name}\n${Clips.formatSize(job.copied)}"
            }
        }
        if (!job.done) {
            main.postDelayed({ tick() }, 200)
            return
        }
        progress?.dismiss()
        progress = null
        watching = null
        if (job.delivered) return
        job.delivered = true
        when {
            job.cancelled -> Toast.makeText(act, "Clip not added", Toast.LENGTH_SHORT).show()
            job.error != null -> MaterialAlertDialogBuilder(act)
                .setTitle("Could not add the clip")
                .setMessage(job.error)
                .setPositiveButton("OK", null)
                .show()
            else -> {
                Toast.makeText(act, if (job.reused) "${job.name} was already here" else "Added ${job.name}", Toast.LENGTH_SHORT).show()
                onChosen(job.name)
            }
        }
    }

    /** The sheet with the clips in the folder; [selected] is the current video setting. */
    fun show(selected: String) {
        val v = LayoutInflater.from(act).inflate(R.layout.sheet_clips, null)
        val dialog = BottomSheetDialog(act)
        dialog.setContentView(v)
        dialog.behavior.state = BottomSheetBehavior.STATE_EXPANDED
        dialog.behavior.skipCollapsed = true
        v.findViewById<TextView>(R.id.clipsFolder).text = "Clips in ${Clips.dir(act).absolutePath}"
        v.findViewById<MaterialButton>(R.id.btnImport).setOnClickListener {
            dialog.dismiss()
            pickFromPhone()
        }
        val list = v.findViewById<RecyclerView>(R.id.clipsList)
        val empty = v.findViewById<TextView>(R.id.clipsEmpty)
        val adapter = ClipAdapter(selected, onPick = { f ->
            dialog.dismiss()
            onChosen(f.name)
        }, onDelete = { f, refresh -> confirmDelete(f, refresh) })
        list.layoutManager = LinearLayoutManager(act)
        list.adapter = adapter
        fun refresh() {
            val files = Clips.list(act)
            adapter.submit(files)
            empty.visibility = if (files.isEmpty()) View.VISIBLE else View.GONE
        }
        adapter.refresh = { refresh() }
        refresh()
        dialog.setOnDismissListener { adapter.close(); if (sheet === dialog) sheet = null }
        sheet = dialog
        dialog.show()
    }

    private fun confirmDelete(f: File, refresh: () -> Unit) {
        MaterialAlertDialogBuilder(act)
            .setTitle("Delete ${f.name}?")
            .setMessage("This removes the ${Clips.formatSize(f.length())} file from the app's folder. It cannot be undone.")
            .setNegativeButton("Keep", null)
            .setPositiveButton("Delete") { _, _ ->
                if (!f.delete()) Toast.makeText(act, "Could not delete ${f.name}", Toast.LENGTH_SHORT).show()
                refresh()
            }
            .show()
    }

    private class ClipAdapter(
        private val selected: String,
        private val onPick: (File) -> Unit,
        private val onDelete: (File, () -> Unit) -> Unit,
    ) : RecyclerView.Adapter<ClipAdapter.Holder>() {
        private var files: List<File> = emptyList()
        var refresh: () -> Unit = {}
        private val loader = Executors.newSingleThreadExecutor()
        private val main = Handler(Looper.getMainLooper())

        class Holder(v: View) : RecyclerView.ViewHolder(v) {
            val thumb: ImageView = v.findViewById(R.id.clipThumb)
            val name: TextView = v.findViewById(R.id.clipItemName)
            val info: TextView = v.findViewById(R.id.clipItemInfo)
            val check: ImageView = v.findViewById(R.id.clipItemCheck)
            val delete: MaterialButton = v.findViewById(R.id.clipItemDelete)
        }

        fun submit(list: List<File>) {
            files = list
            @Suppress("NotifyDataSetChanged")
            notifyDataSetChanged()
        }

        fun close() = loader.shutdownNow()

        override fun getItemCount() = files.size

        override fun onCreateViewHolder(parent: ViewGroup, viewType: Int) =
            Holder(LayoutInflater.from(parent.context).inflate(R.layout.item_clip, parent, false))

        override fun onBindViewHolder(h: Holder, position: Int) {
            val f = files[position]
            h.name.text = f.name
            h.check.visibility = if (f.name == selected || f.absolutePath == selected) View.VISIBLE else View.INVISIBLE
            h.itemView.setOnClickListener { onPick(f) }
            h.delete.setOnClickListener { onDelete(f) { refresh() } }
            val cached = Clips.cachedInfo(f)
            bindInfo(h, f, cached)
            if (cached == null) {
                h.itemView.tag = f.absolutePath
                if (!loader.isShutdown) loader.execute {
                    val info = Clips.info(f)
                    main.post { if (h.itemView.tag == f.absolutePath) bindInfo(h, f, info) }
                }
            }
        }

        private fun bindInfo(h: Holder, f: File, info: Clips.Info?) {
            h.info.text = Clips.describe(f, info)
            if (info?.thumb != null) h.thumb.setImageBitmap(info.thumb) else h.thumb.setImageResource(R.drawable.ic_movie)
            h.thumb.scaleType = if (info?.thumb != null) ImageView.ScaleType.CENTER_CROP else ImageView.ScaleType.CENTER_INSIDE
        }
    }
}
