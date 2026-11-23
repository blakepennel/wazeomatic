package com.wazeomatic.detector

import android.content.res.ColorStateList
import android.graphics.Color
import android.view.LayoutInflater
import android.view.View
import android.view.ViewGroup
import android.widget.ImageView
import android.widget.TextView
import androidx.recyclerview.widget.RecyclerView
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale
import kotlin.math.roundToInt

/** How a report reads on screen: "Hazard › Pothole", "87% · pothole · 12:03:44". */
object ReportFormat {
    private val timeFmt = SimpleDateFormat("HH:mm:ss", Locale.US)

    fun title(ev: HazardEvent): String {
        val t = Schema.HAZARD_TYPES[ev.hazardType] ?: return ev.hazardType
        return if (t.wazeSubcategory.isEmpty()) t.wazeCategory else "${t.wazeCategory} › ${t.wazeSubcategory}"
    }

    fun time(ev: HazardEvent): String = synchronized(timeFmt) { timeFmt.format(Date((ev.ts * 1000).toLong())) }

    fun detail(ev: HazardEvent, withTime: Boolean): String = listOfNotNull(
        "${(ev.confidence * 100).roundToInt()}%",
        ev.label?.replace('_', ' '),
        if (ev.source != "phone") "from ${ev.source}" else null,
        if (withTime) time(ev) else null,
    ).joinToString("  ·  ")

    /** Badge colour per Waze category. */
    fun color(ev: HazardEvent): Int = when (Schema.HAZARD_TYPES[ev.hazardType]?.wazeCategory) {
        "Police" -> Color.rgb(64, 120, 230)
        "Crash" -> Color.rgb(214, 64, 64)
        "Traffic" -> Color.rgb(196, 70, 140)
        "Bad weather" -> Color.rgb(70, 140, 190)
        "Blocked lane" -> Color.rgb(140, 100, 210)
        else -> Color.rgb(222, 135, 20)
    }

    fun icon(ev: HazardEvent): Int =
        if (Schema.HAZARD_TYPES[ev.hazardType]?.wazeCategory == "Police") R.drawable.ic_shield else R.drawable.ic_warning
}

/** The recent reports (PipelineBus.recentEvents, newest first). */
class ReportsAdapter : RecyclerView.Adapter<ReportsAdapter.Holder>() {
    private var items: List<Pair<HazardEvent, Boolean>> = emptyList()

    class Holder(v: View) : RecyclerView.ViewHolder(v) {
        val badge: View = v.findViewById(R.id.reportBadge)
        val icon: ImageView = v.findViewById(R.id.reportIcon)
        val title: TextView = v.findViewById(R.id.reportTitle)
        val sub: TextView = v.findViewById(R.id.reportSub)
        val tag: TextView = v.findViewById(R.id.reportTag)
    }

    fun submit(list: List<Pair<HazardEvent, Boolean>>) {
        items = list
        @Suppress("NotifyDataSetChanged")
        notifyDataSetChanged()
    }

    override fun getItemCount() = items.size

    override fun onCreateViewHolder(parent: ViewGroup, viewType: Int) =
        Holder(LayoutInflater.from(parent.context).inflate(R.layout.item_report, parent, false))

    override fun onBindViewHolder(h: Holder, position: Int) {
        val (ev, dry) = items[position]
        h.badge.backgroundTintList = ColorStateList.valueOf(ReportFormat.color(ev))
        h.icon.setImageResource(ReportFormat.icon(ev))
        h.title.text = ReportFormat.title(ev)
        h.sub.text = ReportFormat.detail(ev, withTime = true)
        val ctx = h.itemView.context
        if (dry) {
            h.tag.text = "DRY"
            h.tag.setTextColor(ctx.getColor(R.color.state_warn))
        } else {
            h.tag.text = "SENT"
            h.tag.setTextColor(ctx.getColor(R.color.state_error))
        }
    }
}
