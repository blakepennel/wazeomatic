package com.wazeomatic.detector

import android.content.Intent
import android.os.Bundle
import android.text.InputType
import android.view.LayoutInflater
import android.view.View
import android.view.ViewGroup
import android.view.inputmethod.EditorInfo
import android.widget.ArrayAdapter
import android.widget.LinearLayout
import android.widget.TextView
import androidx.activity.OnBackPressedCallback
import androidx.appcompat.app.AppCompatActivity
import androidx.core.view.ViewCompat
import androidx.core.view.WindowCompat
import androidx.core.view.WindowInsetsCompat
import com.google.android.material.appbar.MaterialToolbar
import com.google.android.material.card.MaterialCardView
import com.google.android.material.chip.Chip
import com.google.android.material.chip.ChipGroup
import com.google.android.material.dialog.MaterialAlertDialogBuilder
import com.google.android.material.materialswitch.MaterialSwitch
import com.google.android.material.slider.Slider
import com.google.android.material.textfield.MaterialAutoCompleteTextView
import com.google.android.material.textfield.TextInputEditText
import com.google.android.material.textfield.TextInputLayout

/**
 * Grouped settings (source, reporting, detection, scene, advanced) built from [SettingsSpec]. Every change is
 * written straight away through [Settings.withExtras], the path the launch extras take; the main screen restarts a
 * running detector when it gets the result back.
 */
class SettingsActivity : AppCompatActivity() {
    private lateinit var container: LinearLayout
    private var changed = false
    /** key -> refresh the row's enabled state (rows whose dependsOn names that key). */
    private val dependents = HashMap<String, MutableList<View>>()

    private val clipPicker = ClipPicker(this) { name ->
        write("video", name)
        rebuild()
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        WindowCompat.setDecorFitsSystemWindows(window, false)
        setContentView(R.layout.activity_settings)
        changed = savedInstanceState?.getBoolean(STATE_CHANGED) ?: false
        container = findViewById(R.id.settingsContainer)
        ViewCompat.setOnApplyWindowInsetsListener(findViewById(R.id.settingsRoot)) { v, insets ->
            val b = insets.getInsets(WindowInsetsCompat.Type.systemBars() or WindowInsetsCompat.Type.displayCutout() or WindowInsetsCompat.Type.ime())
            v.setPadding(b.left, b.top, b.right, b.bottom)
            WindowInsetsCompat.CONSUMED
        }
        val bar = findViewById<MaterialToolbar>(R.id.settingsToolbar)
        bar.setNavigationOnClickListener { finishWithResult() }
        bar.menu.add("Reset").setIcon(R.drawable.ic_restore).setShowAsAction(android.view.MenuItem.SHOW_AS_ACTION_IF_ROOM)
        bar.menu.getItem(0).iconTintList = android.content.res.ColorStateList.valueOf(getColor(R.color.md_on_surface))
        bar.setOnMenuItemClickListener { confirmReset(); true }
        onBackPressedDispatcher.addCallback(this, object : OnBackPressedCallback(true) {
            override fun handleOnBackPressed() = finishWithResult()
        })
        // make sure every key is saved, so each row can read its current value by key
        Settings.load(this).save(this)
        rebuild()
    }

    override fun onSaveInstanceState(outState: Bundle) {
        super.onSaveInstanceState(outState)
        outState.putBoolean(STATE_CHANGED, changed)
    }

    private fun finishWithResult() {
        currentFocus?.clearFocus()          // commits a text field still being edited
        setResult(RESULT_OK, Intent().putExtra(EXTRA_CHANGED, changed))
        finish()
    }

    private fun confirmReset() {
        MaterialAlertDialogBuilder(this)
            .setTitle("Reset all settings?")
            .setMessage("Every setting goes back to its default (dry run on). Clips in the app's folder are kept.")
            .setNegativeButton("Cancel", null)
            .setPositiveButton("Reset") { _, _ ->
                Settings.reset(this)
                Settings().save(this)
                changed = true
                rebuild()
            }
            .show()
    }

    // ---- values by key ----

    private fun prefs() = getSharedPreferences("settings", MODE_PRIVATE)   // the file Settings.save writes

    private fun value(key: String): String = prefs().all[key]?.toString() ?: ""

    private fun bool(key: String) = value(key).let { it.equals("true", true) || it == "1" }

    /** Writes through the launch-extras path, so ranges and parsing are exactly what `am start --es` gets. */
    private fun write(key: String, v: String) {
        if (value(key) == v) return
        Settings.load(this).withExtras(Bundle().apply { putString(key, v) }).save(this)
        changed = true
        dependents[key]?.forEach { setEnabledDeep(it, bool(key)) }
    }

    // ---- building the rows ----

    private fun rebuild() {
        container.removeAllViews()
        dependents.clear()
        val inf = LayoutInflater.from(this)
        for (group in SettingsSpec.visible()) {
            val header = TextView(this).apply {
                text = group.title
                setTextAppearance(R.style.Text_Wazeomatic_Label)
                setTextColor(getColor(R.color.md_primary))
                setPadding(dp(16), dp(20), dp(16), dp(8))
            }
            container.addView(header)
            val card = MaterialCardView(this, null, com.google.android.material.R.attr.materialCardViewFilledStyle).apply {
                setCardBackgroundColor(getColor(R.color.md_surface_container))
                radius = dp(20).toFloat()
            }
            val list = LinearLayout(this).apply { orientation = LinearLayout.VERTICAL; setPadding(0, dp(4), 0, dp(4)) }
            card.addView(list)
            container.addView(card)
            for (spec in group.items) {
                val row = when (spec) {
                    is SettingSpec.Toggle -> toggleRow(inf, list, spec)
                    is SettingSpec.Range -> rangeRow(inf, list, spec)
                    is SettingSpec.Choice -> choiceRow(inf, list, spec)
                    is SettingSpec.Text -> textRow(inf, list, spec)
                    is SettingSpec.Multi -> multiRow(inf, list, spec)
                    is SettingSpec.Geometry -> geometryRow(inf, list, spec)
                    is SettingSpec.Clip -> clipRow(inf, list, spec)
                }
                list.addView(row)
                spec.dependsOn?.let { dep ->
                    dependents.getOrPut(dep) { ArrayList() }.add(row)
                    setEnabledDeep(row, bool(dep))
                }
            }
        }
        container.addView(TextView(this).apply {
            text = "Launch extras (adb shell am start --es key value) set the same keys; see the README."
            setTextAppearance(R.style.Text_Wazeomatic_Small)
            setTextColor(getColor(R.color.md_on_surface_variant))
            setPadding(dp(16), dp(20), dp(16), 0)
        })
    }

    private fun dp(v: Int) = (v * resources.displayMetrics.density).toInt()

    /** Greys out a row whose switch it depends on is off (alpha on the row only, enabled on every child). */
    private fun setEnabledDeep(v: View, on: Boolean) {
        v.alpha = if (on) 1f else 0.45f
        fun walk(x: View) {
            x.isEnabled = on
            if (x is ViewGroup) for (i in 0 until x.childCount) walk(x.getChildAt(i))
        }
        walk(v)
    }

    private fun bindText(row: View, spec: SettingSpec) {
        row.findViewById<TextView>(R.id.settingTitle).text = spec.title
        row.findViewById<TextView>(R.id.settingHelp).text = spec.help
    }

    private fun toggleRow(inf: LayoutInflater, parent: ViewGroup, spec: SettingSpec.Toggle): View {
        val row = inf.inflate(R.layout.item_setting_switch, parent, false)
        bindText(row, spec)
        val sw = row.findViewById<MaterialSwitch>(R.id.settingSwitch)
        sw.isChecked = bool(spec.key)
        sw.contentDescription = spec.title
        sw.setOnCheckedChangeListener { b, on ->
            if (spec.key == "dry_run" && !on && !liveConfirmed) {
                // going live files real Waze reports: ask first
                b.isChecked = true
                confirmLive { sw.isChecked = false; write(spec.key, "false") }
                return@setOnCheckedChangeListener
            }
            write(spec.key, on.toString())
        }
        row.setOnClickListener { sw.toggle() }
        return row
    }

    private var confirmingLive = false

    private fun confirmLive(onYes: () -> Unit) {
        if (confirmingLive) return
        confirmingLive = true
        MaterialAlertDialogBuilder(this)
            .setTitle("Send real reports to Waze?")
            .setMessage("With dry run off, the macro submits every report in Waze for real, where other drivers see it.")
            .setNegativeButton("Keep dry run", null)
            .setPositiveButton("Go live") { _, _ -> confirmingLive = false; liveConfirmed = true; onYes(); liveConfirmed = false }
            .setOnDismissListener { confirmingLive = false }
            .show()
    }

    private var liveConfirmed = false

    private fun rangeRow(inf: LayoutInflater, parent: ViewGroup, spec: SettingSpec.Range): View {
        val row = inf.inflate(R.layout.item_setting_slider, parent, false)
        bindText(row, spec)
        val label = row.findViewById<TextView>(R.id.settingValue)
        val slider = row.findViewById<Slider>(R.id.settingSlider)
        val current = value(spec.key).toFloatOrNull() ?: spec.from
        slider.valueFrom = spec.from
        slider.valueTo = spec.to
        slider.stepSize = spec.step
        slider.value = spec.snap(current)
        slider.setLabelFormatter { spec.format(it) }
        slider.contentDescription = spec.title
        // the label shows the saved value, even one outside the slider's range (set over adb)
        label.text = spec.format(current)
        slider.addOnChangeListener { _, v, fromUser ->
            if (!fromUser) return@addOnChangeListener
            label.text = spec.format(v)
            write(spec.key, spec.encode(v))
        }
        return row
    }

    private fun choiceRow(inf: LayoutInflater, parent: ViewGroup, spec: SettingSpec.Choice): View {
        val row = inf.inflate(R.layout.item_setting_choice, parent, false)
        bindText(row, spec)
        val current = value(spec.key)
        val options = spec.options.toMutableList()
        if (options.none { it.first == current }) options.add(current to spec.labelOf(current))
        val drop = row.findViewById<MaterialAutoCompleteTextView>(R.id.settingDropdown)
        drop.setAdapter(ArrayAdapter(this, android.R.layout.simple_list_item_1, options.map { it.second }))
        drop.setText(options.first { it.first == current }.second, false)
        drop.contentDescription = spec.title
        drop.setOnItemClickListener { _, _, pos, _ ->
            val v = options[pos].first
            write(spec.key, v)
            if (spec.key == "source" && v == "video" && !Clips.resolve(this, value("video")).isFile) clipPicker.show(value("video"))
        }
        return row
    }

    private fun textRow(inf: LayoutInflater, parent: ViewGroup, spec: SettingSpec.Text): View {
        val row = inf.inflate(R.layout.item_setting_text, parent, false)
        bindText(row, spec)
        val layout = row.findViewById<TextInputLayout>(R.id.settingInputLayout)
        val edit = row.findViewById<TextInputEditText>(R.id.settingEdit)
        layout.placeholderText = spec.hint
        edit.setText(value(spec.key))
        edit.contentDescription = spec.title
        if (spec.numeric) edit.inputType = InputType.TYPE_CLASS_NUMBER
        fun commit() {
            val v = edit.text?.toString()?.trim() ?: ""
            val err = spec.validate(v)
            layout.error = err
            if (err == null) write(spec.key, v)
        }
        edit.setOnEditorActionListener { _, action, _ ->
            if (action == EditorInfo.IME_ACTION_DONE) { commit(); edit.clearFocus() }
            false
        }
        edit.setOnFocusChangeListener { _, has -> if (!has) commit() }
        return row
    }

    private fun multiRow(inf: LayoutInflater, parent: ViewGroup, spec: SettingSpec.Multi): View {
        val row = inf.inflate(R.layout.item_setting_chips, parent, false)
        bindText(row, spec)
        val group = row.findViewById<ChipGroup>(R.id.settingChips)
        val picked = spec.parse(value(spec.key)).toMutableSet()
        for ((v, label) in spec.options) {
            val chip = (inf.inflate(R.layout.item_chip_filter, group, false) as Chip).apply {
                text = label
                isChecked = v in picked
                setOnCheckedChangeListener { _, on ->
                    if (on) picked.add(v) else picked.remove(v)
                    write(spec.key, spec.encode(picked))
                }
            }
            group.addView(chip)
        }
        return row
    }

    private fun geometryRow(inf: LayoutInflater, parent: ViewGroup, spec: SettingSpec.Geometry): View {
        val box = LinearLayout(this).apply { orientation = LinearLayout.VERTICAL }
        box.addView(TextView(this).apply {
            text = spec.title
            setTextAppearance(R.style.Text_Wazeomatic_Body)
            setPadding(dp(16), dp(12), dp(16), 0)
        })
        box.addView(TextView(this).apply {
            text = spec.help
            setTextAppearance(R.style.Text_Wazeomatic_Small)
            setTextColor(getColor(R.color.md_on_surface_variant))
            setPadding(dp(16), dp(2), dp(16), 0)
        })
        val g = Settings.load(this).geometry
        val values = mutableListOf(g[0].toFloat(), g[1].toFloat(), g[2].toFloat())
        for ((i, part) in spec.parts.withIndex()) {
            val row = inf.inflate(R.layout.item_setting_slider, box, false)
            row.findViewById<TextView>(R.id.settingTitle).text = part.title
            row.findViewById<TextView>(R.id.settingHelp).visibility = View.GONE
            val label = row.findViewById<TextView>(R.id.settingValue)
            val slider = row.findViewById<Slider>(R.id.settingSlider)
            slider.valueFrom = part.from
            slider.valueTo = part.to
            slider.stepSize = part.step
            slider.value = part.snap(values[i])
            slider.setLabelFormatter { part.format(it) }
            slider.contentDescription = part.title
            label.text = part.format(values[i])
            slider.addOnChangeListener { _, v, fromUser ->
                if (!fromUser) return@addOnChangeListener
                values[i] = v
                label.text = part.format(v)
                write(spec.key, spec.encode(values))
            }
            box.addView(row)
        }
        return box
    }

    private fun clipRow(inf: LayoutInflater, parent: ViewGroup, spec: SettingSpec.Clip): View {
        val row = inf.inflate(R.layout.item_setting_action, parent, false)
        bindText(row, spec)
        val name = value(spec.key)
        val v = row.findViewById<TextView>(R.id.settingValue)
        v.text = if (Clips.resolve(this, name).isFile) name else "$name (not in the folder)"
        row.setOnClickListener { clipPicker.show(value(spec.key)) }
        return row
    }

    companion object {
        const val EXTRA_CHANGED = "settings_changed"
        private const val STATE_CHANGED = "changed"
    }
}
