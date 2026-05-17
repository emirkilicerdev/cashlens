# CashLens 💸

**CashLens**, görme engelli bireyler için tasarlanmış, yapay zeka destekli bir para tanıma uygulamasıdır. Kamerayı bir banknotun üzerine tutmanız yeterli — uygulama para birimini ve kupürü sesli olarak okur, anlık döviz kuruna göre çeviri yapar.

---

## 📱 Ekran Görüntüleri

> *(Uygulamayı çalıştırıp ekran görüntüsü ekleyebilirsiniz)*

---

## ✨ Özellikler

- 🔍 **Gerçek Zamanlı Para Tanıma** — Kamera görüntüsünden anlık banknot tespiti
- 🌍 **17 Para Birimi Desteği** — TRY, USD, EUR, GBP, JPY, INR, AUD, CAD, BRL, MXN, SGD, NZD, MYR, IDR, PHP, CHF ve daha fazlası
- 🔊 **Sesli Okuma (TTS)** — Tanınan para birimini ve döviz karşılığını doğal cümlelerle seslendirir
- 💱 **Anlık Döviz Çevirisi** — Frankfurter API ile güncel kurlar, internet yoksa önbellek kullanır
- 🌐 **TR / EN Dil Desteği** — Arayüz ve sesli okuma Türkçe veya İngilizce
- 🔦 **Flaş Desteği** — Karanlık ortamlarda flaş açma/kapama
- 📷 **Kamera Döndürme** — Ön/arka kamera geçişi
- 💾 **Kalıcı Ayarlar** — Seçili para birimi, dil tercihi ve ses durumu uygulama yeniden açılsa bile korunur
- 📴 **Offline Mod** — İnternet bağlantısı olmadan son kaydedilen döviz kurlarıyla çalışır

---

## 🧠 Model Mimarisi

Uygulama iki aşamalı bir TFLite pipeline kullanır:

```
Kamera Görüntüsü (224×224)
        │
        ▼
┌─────────────────────┐
│  banknote_encoder   │  ← MobileNetV2 tabanlı encoder (Microsoft BankNote-Net)
│  (16 MB, float32)   │  → 256 boyutlu embedding vektörü
└─────────────────────┘
        │
        ▼
┌──────────────────────────┐
│  all_currencies_         │  ← Özel eğitilmiş sınıflandırıcı
│  classifier (501 KB)     │  → 224 sınıf (17 para birimi × kupür × yüz)
└──────────────────────────┘
        │
        ▼
  RecognitionResult
  (currency, denomination, face, confidence)
```

### Desteklenen Sınıflar
Her para birimi için **ön yüz** ve **arka yüz** ayrı ayrı tanınır.  
Örnek etiket formatı: `TRY_100_1` → Türk Lirası, 100 TL, ön yüz

### Eğitim
- **Veri seti:** [Microsoft BankNote-Net](https://github.com/microsoft/banknote-net) (24.826 görsel)
- **Eğitim scripti:** `banknote-net-main/src/train_all_currencies.py`
- **Doğruluk:** ~%94 validation accuracy
- **Framework:** TensorFlow 2.x + tf_keras

---

## 🏗️ Proje Yapısı

```
cashlens/
├── android/                          # Android uygulaması
│   └── app/
│       └── src/main/
│           ├── java/com/cashlens/
│           │   ├── MainActivity.kt       # Ana aktivite
│           │   ├── SplashActivity.kt     # Açılış ekranı
│           │   ├── CurrencyRecognizer.kt # TFLite inference motoru
│           │   └── ExchangeRateService.kt# Döviz kuru servisi
│           ├── assets/
│           │   ├── banknote_encoder.tflite
│           │   ├── all_currencies_classifier.tflite
│           │   └── labels.json
│           └── res/
│               ├── layout/
│               │   ├── activity_main.xml
│               │   └── activity_splash.xml
│               ├── values/             # İngilizce string kaynakları
│               └── values-tr/          # Türkçe string kaynakları
│
└── banknote-net-main/                # Model eğitim altyapısı
    └── src/
        ├── train_all_currencies.py   # Tüm para birimleri için eğitim scripti
        ├── predict_custom.py         # Test/tahmin scripti
        └── trained_models/           # Eğitilmiş model dosyaları
```

---

## 🚀 Kurulum ve Çalıştırma

### Gereksinimler
- Android Studio Hedgehog veya üzeri
- JDK 17
- Android API 24+ (Android 7.0)
- Gradle 8.4

### Adımlar

1. **Repoyu klonla:**
   ```bash
   git clone https://github.com/emirkilicerdev/cashlens.git
   cd cashlens/android
   ```

2. **Android Studio'da aç:**
   - `File → Open` → `cashlens/android` klasörünü seç

3. **Sync:**
   - Gradle sync otomatik başlar, tamamlanmasını bekle

4. **Çalıştır:**
   - Cihaz veya emülatör seç → ▶ Run

> **Not:** TFLite model dosyaları (`*.tflite`) `android/app/src/main/assets/` altında bulunur ve repoya dahildir.

---

## 📡 Döviz Kuru API

[Frankfurter API](https://www.frankfurter.app/) kullanılmaktadır:

| Özellik | Detay |
|---|---|
| Ücretsiz | Evet, API key gerekmez |
| Kaynak | Avrupa Merkez Bankası |
| Güncelleme | Günlük (hafta içi) |
| Limit | Sınırsız istek |
| Önbellek | Son kurlar SharedPreferences'a kaydedilir |

---

## 🛠️ Kullanılan Teknolojiler

| Teknoloji | Amaç |
|---|---|
| Kotlin | Ana uygulama dili |
| CameraX 1.3.1 | Kamera önizleme ve görüntü analizi |
| TensorFlow Lite 2.16.1 | Model çıkarımı |
| OkHttp 4.12.0 | HTTP istekleri |
| Material Components 1.12.0 | UI bileşenleri |
| Android TTS | Sesli okuma |
| SharedPreferences | Kalıcı ayarlar |

---

## ⚙️ Konfigürasyon

`MainActivity.kt` içindeki sabitler:

```kotlin
// Güven eşiği — altındaki sonuçlar gösterilmez
private val CONFIDENCE_THRESHOLD = 0.82f

// Sesli okuma için üst üste kaç kez aynı sonuç gelmeli
private val TTS_CONFIRM_COUNT = 3
```

---

## 🎯 Model Yeniden Eğitme

```bash
cd banknote-net-main
pip install -r requirements.txt   # veya: conda env create -f env.yaml
python src/train_all_currencies.py
```

Eğitim tamamlandıktan sonra üretilen `.tflite` dosyalarını `android/app/src/main/assets/` altına kopyalayın.

---

## 📄 Lisans

Bu proje [MIT Lisansı](LICENSE) ile lisanslanmıştır.

Model eğitiminde kullanılan BankNote-Net veri seti Microsoft'a aittir:  
[github.com/microsoft/banknote-net](https://github.com/microsoft/banknote-net)

---

## 👤 Geliştirici

**Emir Kılıç**  
GitHub: [@emirkilicerdev](https://github.com/emirkilicerdev)
