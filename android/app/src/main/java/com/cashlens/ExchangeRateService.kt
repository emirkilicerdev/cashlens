package com.cashlens

import android.content.Context
import android.util.Log
import okhttp3.*
import org.json.JSONObject
import java.io.IOException

class ExchangeRateService(private val context: Context) {

    private val prefs = context.getSharedPreferences("cashlens_prefs", Context.MODE_PRIVATE)
    private val client = OkHttpClient()

    // Frankfurter API — ücretsiz, key gerektirmez
    // EUR bazlı kurlar döner: { "rates": { "TRY": 35.2, "USD": 1.08, ... } }
    private val API_URL = "https://api.frankfurter.app/latest"

    fun fetchRates(onResult: (rates: Map<String, Double>, fromCache: Boolean) -> Unit) {
        val request = Request.Builder().url(API_URL).build()

        client.newCall(request).enqueue(object : Callback {
            override fun onFailure(call: Call, e: IOException) {
                Log.w("ExchangeRate", "API failed, using cache: ${e.message}")
                val cached = loadCached()
                onResult(cached, true)
            }

            override fun onResponse(call: Call, response: Response) {
                try {
                    val body = response.body?.string() ?: throw IOException("Empty response")
                    val json = JSONObject(body)
                    val ratesJson = json.getJSONObject("rates")
                    val rates = mutableMapOf<String, Double>()

                    // EUR'un kendisi de dahil
                    rates["EUR"] = 1.0
                    ratesJson.keys().forEach { key ->
                        rates[key] = ratesJson.getDouble(key)
                    }

                    saveToCache(rates)
                    onResult(rates, false)
                } catch (e: Exception) {
                    Log.e("ExchangeRate", "Parse error", e)
                    onResult(loadCached(), true)
                }
            }
        })
    }

    private fun saveToCache(rates: Map<String, Double>) {
        val json = JSONObject(rates as Map<*, *>).toString()
        prefs.edit().putString("exchange_rates", json).putLong("rates_time", System.currentTimeMillis()).apply()
    }

    private fun loadCached(): Map<String, Double> {
        val json = prefs.getString("exchange_rates", null) ?: return emptyMap()
        return try {
            val obj = JSONObject(json)
            val map = mutableMapOf<String, Double>()
            obj.keys().forEach { map[it] = obj.getDouble(it) }
            map
        } catch (e: Exception) {
            emptyMap()
        }
    }

    fun hasCached() = prefs.contains("exchange_rates")

    fun getCachedTime(): Long = prefs.getLong("rates_time", 0L)

    /** EUR bazlı kurlar üzerinden çeviri yapar */
    fun convert(amount: Double, fromCurrency: String, toCurrency: String, rates: Map<String, Double>): Double? {
        val fromRate = rates[fromCurrency] ?: return null
        val toRate = rates[toCurrency] ?: return null
        // amount (fromCurrency) -> EUR -> toCurrency
        return (amount / fromRate) * toRate
    }
}
