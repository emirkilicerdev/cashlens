package com.cashlens

import android.Manifest
import android.content.Context
import android.content.pm.PackageManager
import android.content.res.Configuration
import android.os.Bundle
import android.speech.tts.TextToSpeech
import android.util.Log
import android.view.View
import android.widget.ImageButton
import android.widget.LinearLayout
import android.widget.ProgressBar
import android.widget.TextView
import androidx.appcompat.app.AppCompatActivity
import androidx.camera.core.*
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.camera.view.PreviewView
import androidx.core.app.ActivityCompat
import androidx.core.content.ContextCompat
import com.google.android.material.chip.Chip
import com.google.android.material.chip.ChipGroup
import java.util.*
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors

class MainActivity : AppCompatActivity(), TextToSpeech.OnInitListener {

    private lateinit var previewView: PreviewView
    private lateinit var flipButton: ImageButton
    private lateinit var flashButton: ImageButton
    private lateinit var ttsButton: ImageButton
    private lateinit var langButton: TextView
    private lateinit var currencyFlag: TextView
    private lateinit var currencyName: TextView
    private lateinit var faceText: TextView
    private lateinit var denominationText: TextView
    private lateinit var confidenceLabel: TextView
    private lateinit var confidenceBar: ProgressBar
    private lateinit var conversionResult: TextView
    private lateinit var conversionRate: TextView
    private lateinit var conversionResultSection: LinearLayout
    private lateinit var conversionCollapsibleSection: LinearLayout
    private lateinit var conversionToggle: TextView
    private lateinit var chipGroup: ChipGroup

    private lateinit var cameraExecutor: ExecutorService
    private lateinit var recognizer: CurrencyRecognizer
    private lateinit var exchangeService: ExchangeRateService
    private var tts: TextToSpeech? = null
    private var ttsReady = false
    private var camera: Camera? = null

    private var isBackCamera = true
    private var isFlashOn = false
    private var isConversionVisible = true
    private var ttsEnabled = true
    private var selectedTargetCurrency = "TRY"
    private var currentLang = "tr"
    private var lastResult: RecognitionResult? = null
    private var pendingLabel = ""
    private var pendingCount = 0
    private val TTS_CONFIRM_COUNT = 3
    private var exchangeRates: Map<String, Double> = emptyMap()
    private var lastInferenceMs = 0L
    private val CONFIDENCE_THRESHOLD = 0.82f
    private val UNCERTAIN_THRESHOLD = 0.55f   // bu ile 0.82 arası şüpheli
    private val UNCERTAIN_WARN_COUNT = 3      // 3 kare şüpheli → uyar
    private val NO_MONEY_WARN_COUNT = 5       // 5 kare çok düşük → "para değil"
    private var pendingUncertainCount = 0
    private var pendingNoMoneyCount = 0
    private var lastWarningState = ""         // "uncertain" / "no_money" / "" (yeniden seslendirmeyi engeller)

    private val prefs by lazy { getSharedPreferences("cashlens_ui", MODE_PRIVATE) }

    private val chipCurrencyMap = mapOf(
        R.id.chipTRY to "TRY", R.id.chipUSD to "USD", R.id.chipEUR to "EUR",
        R.id.chipGBP to "GBP", R.id.chipJPY to "JPY", R.id.chipINR to "INR",
        R.id.chipAUD to "AUD", R.id.chipCAD to "CAD", R.id.chipBRL to "BRL",
        R.id.chipMXN to "MXN", R.id.chipSGD to "SGD", R.id.chipNZD to "NZD",
        R.id.chipMYR to "MYR", R.id.chipIDR to "IDR", R.id.chipPHP to "PHP",
        R.id.chipCHF to "CHF"
    )

    override fun attachBaseContext(newBase: Context) {
        val lang = newBase.getSharedPreferences("cashlens_ui", MODE_PRIVATE)
            .getString("app_lang", "tr") ?: "tr"
        super.attachBaseContext(applyLocale(newBase, lang))
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        currentLang       = prefs.getString("app_lang", "tr") ?: "tr"
        ttsEnabled        = prefs.getBoolean("tts_enabled", true)
        isConversionVisible = prefs.getBoolean("conversion_visible", true)

        bindViews()
        setupButtons()
        setupChips()

        cameraExecutor  = Executors.newSingleThreadExecutor()
        recognizer      = CurrencyRecognizer(this)
        exchangeService = ExchangeRateService(this)
        tts             = TextToSpeech(this, this)

        exchangeService.fetchRates { rates, fromCache ->
            exchangeRates = rates
            runOnUiThread {
                conversionRate.text = if (fromCache) getString(R.string.label_offline) else ""
                lastResult?.let { updateConversionUI(it) }
            }
        }

        if (hasCameraPermission()) startCamera()
        else ActivityCompat.requestPermissions(this, arrayOf(Manifest.permission.CAMERA), 100)
    }

    private fun applyLocale(context: Context, lang: String): Context {
        val locale = Locale(lang)
        Locale.setDefault(locale)
        val config = Configuration(context.resources.configuration)
        config.setLocale(locale)
        return context.createConfigurationContext(config)
    }

    private fun bindViews() {
        previewView           = findViewById(R.id.previewView)
        flipButton            = findViewById(R.id.flipButton)
        flashButton           = findViewById(R.id.flashButton)
        ttsButton             = findViewById(R.id.ttsButton)
        langButton            = findViewById(R.id.langButton)
        currencyFlag          = findViewById(R.id.currencyFlag)
        currencyName          = findViewById(R.id.currencyName)
        faceText              = findViewById(R.id.faceText)
        denominationText      = findViewById(R.id.denominationText)
        confidenceLabel       = findViewById(R.id.confidenceLabel)
        confidenceBar         = findViewById(R.id.confidenceBar)
        conversionResult      = findViewById(R.id.conversionResult)
        conversionRate        = findViewById(R.id.conversionRate)
        conversionResultSection = findViewById(R.id.conversionResultSection)
        conversionCollapsibleSection = findViewById(R.id.conversionCollapsibleSection)
        conversionToggle      = findViewById(R.id.conversionToggle)
        chipGroup             = findViewById(R.id.currencyChipGroup)

        langButton.text  = currentLang.uppercase()
        ttsButton.alpha  = if (ttsEnabled) 1.0f else 0.35f

        // Çevrim sonucu başlangıç görünürlüğü
        applyConversionVisibility()
    }

    private fun applyConversionVisibility() {
        conversionCollapsibleSection.visibility = if (isConversionVisible) View.VISIBLE else View.GONE
        conversionToggle.text = if (isConversionVisible) "▾" else "▸"
    }

    private fun setupButtons() {
        flipButton.setOnClickListener {
            isBackCamera = !isBackCamera
            isFlashOn = false
            flashButton.alpha = 0.35f
            camera?.cameraControl?.enableTorch(false)
            if (hasCameraPermission()) startCamera()
        }

        flashButton.setOnClickListener {
            isFlashOn = !isFlashOn
            camera?.cameraControl?.enableTorch(isFlashOn)
            flashButton.alpha = if (isFlashOn) 1.0f else 0.35f
        }

        ttsButton.setOnClickListener {
            ttsEnabled = !ttsEnabled
            ttsButton.alpha = if (ttsEnabled) 1.0f else 0.35f
            if (!ttsEnabled) tts?.stop()
            prefs.edit().putBoolean("tts_enabled", ttsEnabled).apply()
        }

        langButton.setOnClickListener {
            currentLang = if (currentLang == "tr") "en" else "tr"
            prefs.edit().putString("app_lang", currentLang).apply()
            recreate()
        }

        // Tüm "Çevrim ... ▾" satırı tıklanabilir
        findViewById<View>(R.id.conversionHeader).setOnClickListener {
            isConversionVisible = !isConversionVisible
            prefs.edit().putBoolean("conversion_visible", isConversionVisible).apply()
            applyConversionVisibility()
        }
    }

    private fun setupChips() {
        selectedTargetCurrency = prefs.getString("selected_currency", "TRY") ?: "TRY"
        val savedChipId = chipCurrencyMap.entries.firstOrNull { it.value == selectedTargetCurrency }?.key
        savedChipId?.let { chipGroup.check(it) }
        updateChipColors(savedChipId ?: R.id.chipTRY)

        chipGroup.setOnCheckedStateChangeListener { _, checkedIds ->
            val id = checkedIds.firstOrNull() ?: return@setOnCheckedStateChangeListener
            selectedTargetCurrency = chipCurrencyMap[id] ?: "TRY"
            prefs.edit().putString("selected_currency", selectedTargetCurrency).apply()
            updateChipColors(id)
            lastResult?.let { updateConversionUI(it) }
        }
    }

    private fun updateChipColors(selectedId: Int) {
        chipCurrencyMap.keys.forEach { chipId ->
            val chip = chipGroup.findViewById<Chip>(chipId) ?: return@forEach
            if (chipId == selectedId) {
                chip.setChipBackgroundColorResource(R.color.chip_selected)
                chip.setTextColor(0xFFFFFFFF.toInt())
            } else {
                chip.setChipBackgroundColorResource(R.color.chip_unselected)
                chip.setTextColor(0xFF555555.toInt())
            }
        }
    }

    override fun onInit(status: Int) {
        if (status == TextToSpeech.SUCCESS) {
            val activeLocale = if (currentLang == "tr") Locale("tr", "TR") else Locale.ENGLISH
            val result = tts?.setLanguage(activeLocale) ?: -1
            if (result < 0) tts?.setLanguage(Locale.ENGLISH)
            tts?.setSpeechRate(0.9f)
            ttsReady = true
            Log.d("TTS", "TTS hazır, dil: $activeLocale")

            // TTS hazır olana kadar ekranda bekleyen sonuç varsa hemen oku
            lastResult?.let { r ->
                if (ttsEnabled) {
                    val denomValue = r.denomination.toDoubleOrNull()
                    val converted = if (denomValue != null && exchangeRates.isNotEmpty())
                        exchangeService.convert(denomValue, r.currency, selectedTargetCurrency, exchangeRates)
                    else null
                    val speech = buildNaturalSpeech(r, converted, activeLocale)
                    tts?.speak(speech, TextToSpeech.QUEUE_FLUSH, null, "cashlens")
                }
            }
        } else {
            Log.e("TTS", "TTS başlatılamadı: $status")
        }
    }

    private fun hasCameraPermission() =
        ContextCompat.checkSelfPermission(this, Manifest.permission.CAMERA) == PackageManager.PERMISSION_GRANTED

    override fun onRequestPermissionsResult(code: Int, perms: Array<String>, results: IntArray) {
        super.onRequestPermissionsResult(code, perms, results)
        if (results.firstOrNull() == PackageManager.PERMISSION_GRANTED) startCamera()
    }

    private fun startCamera() {
        val future = ProcessCameraProvider.getInstance(this)
        future.addListener({
            val provider = future.get()
            val preview = Preview.Builder().build()
                .also { it.setSurfaceProvider(previewView.surfaceProvider) }
            val analyzer = ImageAnalysis.Builder()
                .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                .build()
                .also { it.setAnalyzer(cameraExecutor, ::analyzeFrame) }

            val preferred = if (isBackCamera) CameraSelector.DEFAULT_BACK_CAMERA
                            else CameraSelector.DEFAULT_FRONT_CAMERA
            val fallback  = if (isBackCamera) CameraSelector.DEFAULT_FRONT_CAMERA
                            else CameraSelector.DEFAULT_BACK_CAMERA

            provider.unbindAll()
            camera = try {
                provider.bindToLifecycle(this, preferred, preview, analyzer)
            } catch (e: Exception) {
                Log.w("CashLens", "Preferred camera unavailable (${e.message}), trying fallback")
                try {
                    provider.bindToLifecycle(this, fallback, preview, analyzer)
                } catch (e2: Exception) {
                    Log.e("CashLens", "No camera available: ${e2.message}")
                    runOnUiThread {
                        android.widget.Toast.makeText(
                            this,
                            "Kamera başlatılamadı. Lütfen gerçek cihazda deneyin.",
                            android.widget.Toast.LENGTH_LONG
                        ).show()
                    }
                    null
                }
            }
        }, ContextCompat.getMainExecutor(this))
    }

    private fun analyzeFrame(imageProxy: ImageProxy) {
        val now = System.currentTimeMillis()
        if (now - lastInferenceMs < 1500L) { imageProxy.close(); return }
        lastInferenceMs = now
        try {
            val bitmap = imageProxy.toBitmap()
            val result = recognizer.recognize(bitmap)
            runOnUiThread {
                when {
                    result.confidence >= CONFIDENCE_THRESHOLD -> {
                        // Yüksek güven — normal tanıma
                        pendingUncertainCount = 0
                        pendingNoMoneyCount = 0
                        lastWarningState = ""
                        lastResult = result
                        updateMainUI(result)
                        updateConversionUI(result)
                        speakResult(result)
                    }
                    result.confidence >= UNCERTAIN_THRESHOLD -> {
                        // Orta güven — şüpheli banknot
                        pendingNoMoneyCount = 0
                        pendingUncertainCount++
                        if (pendingUncertainCount >= UNCERTAIN_WARN_COUNT) {
                            showUncertaintyWarning()
                            pendingUncertainCount = 0
                        }
                    }
                    else -> {
                        // Düşük güven — büyük ihtimal para değil
                        pendingUncertainCount = 0
                        pendingNoMoneyCount++
                        if (pendingNoMoneyCount >= NO_MONEY_WARN_COUNT) {
                            showNoMoneyWarning()
                            pendingNoMoneyCount = 0
                        }
                    }
                }
            }
        } catch (e: Exception) {
            Log.e("CashLens", "Inference error", e)
        } finally {
            imageProxy.close()
        }
    }

    private fun showUncertaintyWarning() {
        if (lastWarningState == "uncertain") return  // aynı uyarı tekrar gösterilmesin
        lastWarningState = "uncertain"
        lastResult = null

        val orange = 0xFFFF9800.toInt()
        currencyFlag.text = "!"
        currencyFlag.setTextColor(orange)
        currencyName.text = getString(R.string.warning_uncertain)
        currencyName.setTextColor(orange)
        faceText.text = ""
        denominationText.text = "?"
        denominationText.setTextColor(orange)
        confidenceLabel.text = ""
        confidenceBar.progress = 0
        conversionResult.text = "—"
        conversionRate.text = ""

        if (ttsEnabled && ttsReady && tts?.isSpeaking != true) {
            tts?.speak(getString(R.string.tts_warning_uncertain), TextToSpeech.QUEUE_FLUSH, null, "cashlens-warn")
        }
    }

    private fun showNoMoneyWarning() {
        if (lastWarningState == "no_money") return
        lastWarningState = "no_money"
        lastResult = null

        val gray = 0xFF888888.toInt()
        currencyFlag.text = "?"
        currencyFlag.setTextColor(gray)
        currencyName.text = getString(R.string.warning_no_money)
        currencyName.setTextColor(gray)
        faceText.text = ""
        denominationText.text = ""
        denominationText.setTextColor(gray)
        confidenceLabel.text = ""
        confidenceBar.progress = 0
        conversionResult.text = "—"
        conversionRate.text = ""

        if (ttsEnabled && ttsReady && tts?.isSpeaking != true) {
            tts?.speak(getString(R.string.tts_warning_no_money), TextToSpeech.QUEUE_FLUSH, null, "cashlens-warn")
        }
    }

    private fun updateMainUI(result: RecognitionResult) {
        // Uyarı durumundan döndüysek renkleri sıfırla
        currencyFlag.setTextColor(0xFF1B873F.toInt())
        currencyName.setTextColor(0xFF111111.toInt())
        denominationText.setTextColor(0xFF111111.toInt())

        currencyFlag.text     = result.currency
        currencyName.text     = "${currencyFullName(result.currency)} (${result.currency})"
        faceText.text         = if (result.face == "Ön") getString(R.string.label_face_front)
                                else getString(R.string.label_face_back)
        denominationText.text = result.denomination

        val confPct = (result.confidence * 100).toInt()
        confidenceLabel.text  = "${getString(R.string.label_confidence)}: %$confPct"
        confidenceBar.progress = confPct
        confidenceBar.progressTintList = android.content.res.ColorStateList.valueOf(
            when {
                confPct >= 85 -> 0xFF1B873F.toInt()
                confPct >= 65 -> 0xFFFF9800.toInt()
                else          -> 0xFFE53935.toInt()
            }
        )
    }

    private fun updateConversionUI(result: RecognitionResult) {
        if (exchangeRates.isEmpty()) { conversionResult.text = "—"; return }
        val denomValue = result.denomination.toDoubleOrNull() ?: return
        if (result.currency == selectedTargetCurrency) {
            conversionResult.text = "${currencySymbol(selectedTargetCurrency)} ${result.denomination}"
            conversionRate.text = getString(R.string.label_already_same)
            return
        }
        val converted = exchangeService.convert(denomValue, result.currency, selectedTargetCurrency, exchangeRates)
        if (converted != null) {
            conversionResult.text = "${currencySymbol(selectedTargetCurrency)} ${formatAmount(converted, selectedTargetCurrency)}"
            val unitRate = exchangeService.convert(1.0, result.currency, selectedTargetCurrency, exchangeRates)
            if (unitRate != null)
                conversionRate.text = "1 ${result.currency} = ${formatAmount(unitRate, selectedTargetCurrency)} $selectedTargetCurrency"
        }
    }

    private fun speakResult(result: RecognitionResult) {
        if (!ttsEnabled || !ttsReady) return

        // Henüz konuşuyorsa tekrar başlatma
        if (tts?.isSpeaking == true) return

        if (result.label == pendingLabel) pendingCount++
        else { pendingLabel = result.label; pendingCount = 1 }

        if (pendingCount < TTS_CONFIRM_COUNT) return
        pendingCount = 0

        val denomValue = result.denomination.toDoubleOrNull()
        val converted = if (denomValue != null && exchangeRates.isNotEmpty())
            exchangeService.convert(denomValue, result.currency, selectedTargetCurrency, exchangeRates) else null

        val activeLocale = if (currentLang == "tr") Locale("tr", "TR") else Locale.ENGLISH
        val langResult = tts?.setLanguage(activeLocale) ?: -1
        if (langResult < 0) tts?.setLanguage(Locale.ENGLISH)

        val speech = buildNaturalSpeech(result, converted, activeLocale)
        Log.d("TTS", "[$activeLocale] $speech")
        tts?.speak(speech, TextToSpeech.QUEUE_FLUSH, null, "cashlens")
    }

    private fun buildNaturalSpeech(
        result: RecognitionResult,
        converted: Double?,
        locale: Locale
    ): String {
        val isTurkish = locale.language == "tr"
        val targetName = currencyFullName(selectedTargetCurrency, isTurkish)
        val sourceName = currencyFullName(result.currency, isTurkish)

        return buildString {
            append("${result.denomination} $sourceName. ")
            // Çevrim gizliyse sadece kaynak para birimini söyle, çevrim cümlesini ekleme
            if (isConversionVisible && converted != null && result.currency != selectedTargetCurrency) {
                val formattedAmount = formatAmount(converted, selectedTargetCurrency)
                if (isTurkish)
                    append("$targetName karşılığı yaklaşık $formattedAmount.")
                else
                    append("That's approximately $formattedAmount $targetName.")
            }
        }
    }

    private fun formatAmount(amount: Double, currency: String) =
        if (currency in listOf("JPY", "IDR", "PKR")) String.format("%,.0f", amount)
        else String.format("%,.2f", amount)

    private fun currencySymbol(code: String) = when (code) {
        "TRY" -> "₺"; "USD" -> "$"; "EUR" -> "€"; "GBP" -> "£"
        "JPY" -> "¥"; "INR" -> "₹"; "AUD" -> "A$"; "CAD" -> "C$"
        "BRL" -> "R$"; "MXN" -> "$"; "SGD" -> "S$"; "NZD" -> "NZ$"
        "MYR" -> "RM"; "IDR" -> "Rp"; "PHP" -> "₱"; "CHF" -> "Fr"
        else -> code
    }

    private fun currencyFullName(code: String, turkish: Boolean = (currentLang == "tr")) = when (code) {
        "TRY" -> if (turkish) "Türk Lirası"        else "Turkish Lira"
        "USD" -> if (turkish) "Amerikan Doları"     else "US Dollar"
        "EUR" -> if (turkish) "Euro"                else "Euro"
        "GBP" -> if (turkish) "İngiliz Sterlini"    else "British Pound"
        "JPY" -> if (turkish) "Japon Yeni"          else "Japanese Yen"
        "INR" -> if (turkish) "Hint Rupisi"         else "Indian Rupee"
        "BRL" -> if (turkish) "Brezilya Reali"      else "Brazilian Real"
        "CAD" -> if (turkish) "Kanada Doları"       else "Canadian Dollar"
        "AUD" -> if (turkish) "Avustralya Doları"   else "Australian Dollar"
        "SGD" -> if (turkish) "Singapur Doları"     else "Singapore Dollar"
        "MXN" -> if (turkish) "Meksika Pesosu"      else "Mexican Peso"
        "MYR" -> if (turkish) "Malezya Ringgiti"    else "Malaysian Ringgit"
        "IDR" -> if (turkish) "Endonezya Rupisi"    else "Indonesian Rupiah"
        "PHP" -> if (turkish) "Filipin Pesosu"      else "Philippine Peso"
        "NZD" -> if (turkish) "Yeni Zelanda Doları" else "New Zealand Dollar"
        "CHF" -> if (turkish) "İsviçre Frangı"      else "Swiss Franc"
        "PKR" -> if (turkish) "Pakistan Rupisi"     else "Pakistani Rupee"
        else  -> code
    }

    override fun onDestroy() {
        super.onDestroy()
        tts?.stop(); tts?.shutdown()
        cameraExecutor.shutdown()
        recognizer.close()
    }
}
