# CashLens: Yapay Zeka Ar-Ge Raporu

## Para Tanıma Modelinin Gelişim Süreci ve Mühendislik Kararları

---

## Giriş

CashLens, görme engelli bireylerin ellerindeki banknotları anında tanımasını sağlayan bir mobil yapay zeka uygulamasıdır. Kullanıcı sadece kamerayı paranın üzerine tutar; uygulama para birimini, kupürü ve döviz karşılığını sesli olarak okur.

Bu rapor, uygulamanın arkasındaki para tanıma modelinin nasıl geliştirildiğini, hangi yöntemlerin denendiğini, neden bazılarının yetersiz kaldığını ve en sonunda neden Microsoft'un BankNote-Net modelinin tercih edildiğini teknik ama anlaşılır bir dille açıklamaktadır.

---

## Temel Kavramlar

Yöntemlere geçmeden önce, bu rapor boyunca sıkça kullanılacak birkaç temel kavramı açıklamak gerekir.

### Sinir Ağı (Neural Network) Nedir?

İnsan beynindeki nöronlardan ilham alan yapay sinir ağları, veriden örüntü öğrenen matematiksel sistemlerdir. Bir sinir ağına binlerce fotoğraf gösterip her birinin ne olduğunu söylediğinizde, ağ zamanla kendi içinde "100 dolar böyle görünür" gibi kurallar üretmeyi öğrenir.

### CNN (Evrişimsel Sinir Ağı) Nedir?

Görüntü işleme için özelleştirilmiş sinir ağı mimarisidir. Normal sinir ağları bir fotoğrafı piksel piksel düz bir liste olarak işlerken, CNN fotoğrafı küçük pencereler halinde tarayarak kenar, köşe, doku gibi yerel örüntüleri hiyerarşik olarak öğrenir. Gözün retinasının çalışma prensibine benzer: önce basit çizgiler, sonra şekiller, sonra nesneler.

### MobileNetV2 Nedir?

Google tarafından geliştirilen MobileNetV2, akıllı telefon gibi kısıtlı donanımda çalışmak üzere optimize edilmiş hafif bir CNN mimarisidir. ImageNet yarışmasında milyonlarca fotoğrafla önceden eğitilmiştir; bu sayede sıfırdan başlamak yerine, genel görsel anlama yeteneğini transfer ederek daha az veriyle daha hızlı öğrenebilir. CashLens'in tüm aşamalarında görüntü işleme omurgası olarak kullanılmıştır.

### Embedding (Gömme Vektörü) Nedir?

Bir fotoğrafı sinir ağına verdiğinizde, ağ onu yüzlerce veya binlerce sayıdan oluşan bir vektöre dönüştürebilir. Bu vektöre "embedding" denir. Embedding, fotoğrafın matematiksel parmak izidir. İyi eğitilmiş bir modelde birbirine benzeyen fotoğrafların embedding'leri de birbirine yakın, farklı fotoğraflarınki ise birbirinden uzak olur. 100 dolarlık iki farklı banknotun embedding'leri uzayda yan yana dururken, 100 dolarlık ve 50 euroluk banknotun embedding'leri birbirinden uzak durur.

### Encoder Nedir?

Bir fotoğrafı alıp embedding'e dönüştüren sinir ağı modülüdür. Encoder'ı bir "öz çıkarıcı" olarak düşünebilirsiniz: fotoğrafın arka planını, aydınlatmasını ve gürültüsünü süzüp yalnızca nesnenin özünü temsil eden kompakt bir matematiksel yapı üretir. CashLens'te encoder, 224×224 piksellik bir görüntüyü 256 sayıdan oluşan bir vektöre sıkıştırır.

### Classifier (Sınıflandırıcı) Nedir?

Encoder'ın ürettigi embedding'i alıp "bu hangi sınıfa ait?" sorusunu cevaplayan ikinci bir modüldür. CashLens'te sınıflandırıcı, 256 boyutlu embedding'i 224 farklı banknot sınıfından (17 para birimi × çeşitli kupür × ön/arka yüz) birine atar.

---

## Aşama 1 — Klasik Sınıflandırma (Baseline CNN)

### Yöntem

İlk yaklaşım en doğrudan olanıdır: fotoğrafı al, sınıfı tahmin et. MobileNetV2 omurgası üzerine bir sınıflandırma katmanı eklenerek model eğitildi. Her fotoğraf için model doğrudan "bu 100 Türk Lirası ön yüzü" gibi bir cevap üretmeye çalışır.

```
Fotoğraf → MobileNetV2 → Düzleştirme → Tam Bağlantılı Katman → Sınıf Tahmini
```

### Neden Yetersiz Kaldı?

Doğrulama veri setinde %90'ın üzerinde doğruluk elde edildi. Ancak gerçek kamerada ciddi sorunlar yaşandı:

**Arka plan ezberleme (Overfitting):** Model banknotun kendisini değil, eğitim fotoğraflarındaki arka planı (masa rengi, el dokusu, gölge yönü) ezberledi. Yeni bir ortamda aynı banknotu gösterdiğinde tanıyamadı.

**Işık ve açı hassasiyeti:** Eğitim veri setindeki fotoğraflar belirli ışık koşullarında çekilmişti. Farklı bir aydınlatmada veya hafif eğik tutulduğunda güven puanı dramatik biçimde düştü.

**Az veriyle sınırlı genelleme:** ~33.000 fotoğraflık veri setiyle 8 para birimi ve onlarca kupür öğretilmeye çalışıldı. Her sınıfa düşen örnek sayısı gerçek dünyanın sonsuz çeşitliliğini temsil etmek için yetersiz kaldı.

---

## Aşama 2 — Kontrastif Öğrenme: Siamese Network ve Triplet Loss

### Yöntem

Klasik sınıflandırmanın yetersizliği, tamamen farklı bir felsefeye geçişi zorunlu kıldı. Sorun şuydu: model "bu hangi sınıf?" yerine "bu iki fotoğraf birbirine benziyor mu?" sorusunu öğrenebilir mi?

**Siamese (Siyam) Ağı:** Aynı ağırlıkları paylaşan ikiz iki encoder kullanılır. İki farklı fotoğraf aynı anda bu ikiz encoder'lardan geçirilir ve üretilen embedding'lerin mesafesi ölçülür. Aynı banknotsa mesafe küçük, farklı banknotsa mesafe büyük olmalıdır.

**Triplet Loss (Üçlü Kayıp Fonksiyonu):** Eğitim sırasında her seferinde üç fotoğraf birlikte kullanılır:
- **Anchor (Çıpa):** Referans fotoğraf (örn. 100 TL ön yüz)
- **Positive (Pozitif):** Aynı sınıftan farklı bir fotoğraf (başka bir 100 TL ön yüz)
- **Negative (Negatif):** Farklı bir sınıftan fotoğraf (örn. 50 TL veya 100 USD)

Model şu kuralı öğrenir: *Anchor ile Positive arasındaki mesafe, Anchor ile Negative arasındaki mesafeden küçük olmalıdır.* Bu kural tekrar tekrar uygulanarak embedding uzayı şekillendirilir.

```
Anchor  → Encoder ─┐
Positive → Encoder ─┼→ Mesafe Hesapla → Triplet Loss
Negative → Encoder ─┘
```

### Neden Daha İyiydi?

Model artık "bu 100 TL mi?" yerine "banknotun matematiksel özü nedir?" öğreniyor. Arka plan, aydınlatma ve gölge gürültü olarak filtreleniyor çünkü model iki fotoğrafın birbirine benzerliğini öğreniyor, belirli bir piksel örüntüsünü değil.

### Neden Yine de Yetersiz Kaldı?

Triplet loss mimarisinin kritik bir kısıtı vardır: her adımda yalnızca **bir** negatif örnek iterilir. Oysa bir banknotun gerçek dünyada ayırt edilmesi gereken yüzlerce farklı sınıf vardır. Modelin öğrenme verimliliği sınırlı kaldı. Özellikle birbirine görsel olarak yakın banknotlarda (örn. GBP 10 ve GBP 20 arka yüzü) kararsızlık devam etti.

---

## Aşama 3 — Supervised Contrastive Learning (SupCon)

### Yöntem

Triplet loss'un tek-negatif kısıtını aşmak için Supervised Contrastive Learning'e (SupCon) geçildi. 2020 yılında Google Brain tarafından yayımlanan bu yöntem, aynı anda bir grup fotoğrafı (batch) karşılaştırarak öğrenir.

**Batch-wise Karşılaştırma:** 64 fotoğraflık bir grup alınır. Her fotoğrafın 2 farklı şekilde manipüle edilmiş kopyası oluşturulur (toplam 128 görüntü). Bunlar aynı anda encoder'dan geçirilir.

**Ağır Veri Artırma (Data Augmentation):** Her fotoğrafa kasıtlı bozulmalar uygulanır:
- Rastgele bulanıklaştırma
- Kontrast ve parlaklık değişimi
- Renk bozumu ve gölge ekleme
- Rastgele kırpma ve döndürme

Bu sayede model, "bu iki görüntü birbirinden çok farklı görünse de aynı banknottur" ilişkisini öğrenir.

**SupCon Loss:** Bir batch içinde aynı sınıfa ait tüm çiftler birbirine çekilirken, farklı sınıftaki tüm çiftler aynı anda itilir. Triplet loss'un tek-negatif kısıtının aksine, her adımda onlarca negatif aynı anda kullanılır.

```
Batch (64 fotoğraf × 2 kopya = 128 görüntü)
        ↓
    Encoder (MobileNetV2 + Projection Head)
        ↓
   256-boyutlu Embedding'ler
        ↓
   SupCon Loss: Aynı sınıf → Yaklaştır | Farklı sınıf → Uzaklaştır
```

### Test-Time Augmentation (TTA)

Canlı kamerada ek bir güvenilirlik katmanı olarak TTA uygulandı. Tek bir kamera karesi 3 farklı kontrast ve parlaklık ayarında işlenip tahminlerin ortalaması alındı. Bu, modelin tek bir "kötü an"dan hatalı sonuç üretmesini engelledi.

### Veri Seti ve Kalite Sorunları

Ar-Ge sürecinde internet üzerindeki çeşitli açık kaynaklardan ~33.000 görüntü derlendi. Ancak sayısal büyüklük yanıltıcıdır; asıl sorun **veri kalitesiydi.**

**Tutarsız kaynak kalitesi:** Fotoğraflar Roboflow, Kaggle ve çeşitli akademik arşivlerden toplandı. Her kaynak farklı bir ekip tarafından farklı koşullarda oluşturulmuştu. Bir kısım tarayıcıyla düz zemin üzerinde çekilmiş temiz görüntülerden, bir kısım ise telefon kamerasıyla tutarsız ışık altında çekilmiş bulanık karelerden oluşuyordu. Model bu heterojen yapıyı bir standart olarak öğrenmekte zorlandı.

**Gerçekçi olmayan çekim koşulları:** İnternetten toplanan banknot fotoğraflarının büyük çoğunluğu düz bir yüzey üzerine yatırılmış, sabit ışık altında, elle tutulmadan çekilmiş görüntülerdi. Oysa gerçek kullanım senaryosunda banknot titreyen bir elde tutulur, arka planda parmaklar ve avuç içi görünür, ışık açısı sürekli değişir. Model "laboratuvar banknotunu" tanımayı öğrendi, "gerçek hayat banknotunu" değil.

**Etiket hataları:** Toplu indirilen veri setlerinde yanlış etiketlenmiş görüntüler mevcuttu. 50 INR olarak etiketlenmiş görseller arasında 100 INR bulunabiliyordu. Model bu hataları da "doğru bilgi" olarak öğrendi.

**Sınıflar arası dengesizlik:** INR için ~14.000 görüntü varken EUR için yalnızca ~1.278 vardı. Bu dengesizlik modelin INR'yi "aşırı" öğrenirken EUR'yu yüzeysel öğrenmesine yol açtı.

**Sonuç olarak:** Doğrulama setinde %99 doğruluk elde edilmesi yanıltıcıydı; zira doğrulama seti de aynı kalitesiz kaynaktan geliyordu. Gerçek kamerada performans çok daha düşük kaldı.

### Sonuçlar

- **%99.0 doğrulama doğruluğu** — ancak bu oran kalitesiz veri seti üzerinde ölçüldüğünden gerçek dünya performansını yansıtmıyordu
- Arka plan gürültüsüne kısmi bağışıklık kazanıldı
- Triplet loss ve klasik CNN'e kıyasla daha kararlı canlı kamera performansı

### Neden Yine de Mükemmel Değildi?

Mimari ne kadar iyi olursa olsun, kalitesiz veriyle beslenen bir model kalitesiz sonuç üretir. Yapay zekada buna **"garbage in, garbage out"** (çöp girer, çöp çıkar) ilkesi denir. Üç aşamanın tamamında bu duvarla karşılaşıldı ve aşılması kendi imkânlarımızla mümkün olmadı.

---

## Nihai Mühendislik Kararı: Microsoft BankNote-Net

### Neden Transfer Learning?

Üç aşamalı Ar-Ge sürecinde kontrastif öğrenmenin klasik CNN'den çok üstün olduğunu lokal donanımda doğruladık. Ancak sınırı veri seti büyüklüğüydü; bu durum kendi modelimizle aşılabilecek bir engel değildi.

Microsoft Research, aynı kontrastif öğrenme felsefesini (SupCon benzeri mimari) **24.816 yüksek kaliteli, profesyonelce etiketlenmiş banknot görüntüsüyle** uygulamış ve BankNote-Net'i üretmiştir. Sayı olarak bizim veri setimizden daha az görünse de asıl fark veri kalitesindedir: gerçek sahada, kontrollü koşullarda ve uzman gözetiminde toplanmış görüntüler, internetten derlenmiş binlerce tutarsız görüntüden çok daha güçlü bir model üretir.

### Mimari Entegrasyon

Microsoft'un encoder'ı donuk (frozen) tutularak üzerine kendi eğittiğimiz hafif bir sınıflandırıcı eklendi:

```
Kamera (224×224 piksel)
        ↓
Microsoft BankNote-Net Encoder   ← Dondurulmuş, değiştirilmiyor
(MobileNetV2 tabanlı, 16 MB)
        ↓
256 boyutlu Embedding
        ↓
Özel Eğitilmiş Sınıflandırıcı   ← Bizim eğittiğimiz, 501 KB
(17 para birimi, 224 sınıf)
        ↓
Para Birimi + Kupür + Yüz + Güven Puanı
```

Bu yaklaşım modern yapay zeka mühendisliğinde **Transfer Learning** olarak adlandırılır ve endüstri standardıdır.

### Neden Doğru Karardı?

| Kriter | Kendi SupCon Modelimiz | Microsoft BankNote-Net |
|---|---|---|
| Eğitim verisi | ~32.000 fotoğraf (8 para birimi) | 24.816 fotoğraf, profesyonel kalite (17 para birimi) |
| Para birimi kapsamı | TRY, USD, EUR, INR, SAR, AUD, JPY, KRW | 17 para birimi (SAR, KRW hariç) |
| Işık varyasyonu | Sınırlı | Küresel çeşitlilik |
| Yıpranmış banknot | Zayıf | Güçlü |
| Model boyutu | Benzer | 16 MB (optimize) |
| Pil tüketimi | Yüksek | Optimize |
| Geliştirme süresi | Haftalarca GPU eğitimi | Transfer + ince ayar |

---

## Sonuç

CashLens'in model geliştirme süreci, yapay zeka mühendisliğinin gerçek dünya dinamiklerini yansıtmaktadır:

1. **Klasik CNN** ile hızlı prototip oluşturuldu; gerçek kamerada yetersizliği görüldü.
2. **Triplet Loss ile Siamese Network** mimari anlayışı değiştirdi; gürültü bağışıklığı kazanıldı ancak öğrenme kapasitesi sınırlı kaldı.
3. **Supervised Contrastive Learning** ile teorik tavan zorlandı; %99 doğruluk elde edildi ancak veri seti duvarına çarpıldı.
4. **Microsoft BankNote-Net** ile bu duvar aşıldı; milyonlarca görüntüyle eğitilmiş, küresel ölçekte genelleştirilmiş bir encoder, hafif ve özel eğitilmiş bir sınıflandırıcıyla birleştirildi.

En iyi mühendislik kararı her zaman sıfırdan en karmaşığı inşa etmek değildir. Akademik ve endüstriyel kaliteyi kanıtlamış bir temel üzerine, probleme özgü uzmanlığı eklemek; hem kaynak verimliliği hem de son kullanıcı deneyimi açısından üstün sonuçlar doğurur.

---

*CashLens — Görme Engelli Bireyler için Yapay Zeka Destekli Para Tanıma*
