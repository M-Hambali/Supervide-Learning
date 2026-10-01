# =============================================================================
# Analisis Regresi Komparatif: OLS, WLS, GLS, dan Ridge Regression
# Pengujian asumsi klasik, pemodelan alternatif, dan evaluasi RMSE & AIC
# Penulis : Muhamad Hambali
# =============================================================================
# Alur script:
#   0. Persiapan (library & data)
#   1. Eksplorasi variabel respon Y
#   A. Uji asumsi klasik pada OLS (normalitas, multikolinearitas,
#      heteroskedastisitas, autokorelasi)
#   B. Pemodelan: OLS, WLS, GLS, Ridge
#   C. Perbandingan model (RMSE & AIC)
# =============================================================================


# =============================================================================
# 0. PERSIAPAN
# =============================================================================

# Install paket yang belum ada (jalankan sekali saja, hilangkan tanda # jika perlu)
# install.packages(c("data.table", "dplyr", "ggplot2", "gridExtra", "corrplot",
#                    "car", "lmtest", "tseries", "nlme", "glmnet"))

library(data.table)  # fread(): membaca file besar dengan cepat
library(dplyr)       # manipulasi data (arrange, dll.)
library(ggplot2)     # visualisasi
library(gridExtra)   # menggabungkan beberapa grafik ggplot dalam satu kanvas
library(corrplot)    # visualisasi matriks korelasi
library(car)         # vif() untuk uji multikolinearitas
library(lmtest)      # bptest() (heteroskedastisitas) & dwtest() (autokorelasi)
library(tseries)     # jarque.bera.test() (normalitas)
library(nlme)        # gls() untuk Generalized Least Squares
library(glmnet)      # Ridge Regression (cv.glmnet & glmnet)

# --- Membaca data ------------------------------------------------------------
# Hanya 2.500 baris pertama yang dipakai karena:
#  (1) gls() membentuk matriks kovarians N x N -> N = 1,5 juta tidak muat di RAM.
#  (2) Semua model harus dilatih pada data yang SAMA agar AIC/RMSE sebanding.
#  (3) Baris diambil berurutan (bukan acak) agar urutan waktu dan struktur
#      autokorelasi AR(1) tetap terjaga.

n_sample  <- 2500
file_path <- "raw_data_simulasi.txt"   # ubah sesuai lokasi file Anda

df <- as.data.frame(fread(file_path, nrows = n_sample))

# Cek dimensi data dan jumlah missing value
cat("Dimensi data :", nrow(df), "baris dan", ncol(df), "kolom\n")
cat("Missing value:", sum(is.na(df)), "\n")

# Cuplikan 5 baris pertama (beberapa kolom saja)
print(round(head(df[, c("X1", "X2", "X3", "X4", "X5", "X28", "X29", "X30", "Y")], 5), 4))


# =============================================================================
# 1. STATISTIK DESKRIPTIF VARIABEL RESPON (Y)
# =============================================================================

# Ringkasan lima angka + mean, serta simpangan baku
print(summary(df$Y))
cat("Standar deviasi Y:", round(sd(df$Y), 4), "\n")

# Histogram + kurva densitas: melihat bentuk distribusi (simetris / menceng)
p_dens_y <- ggplot(df, aes(x = Y)) +
  geom_histogram(aes(y = after_stat(density)), bins = 35,
                 fill = "#2c3e50", color = "white", alpha = 0.8) +
  geom_density(color = "#e74c3c", linewidth = 1.2) +
  theme_minimal(base_size = 12) +
  labs(title = "Distribusi & Densitas Respon Y", x = "Nilai Y", y = "Densitas")

# Boxplot: titik merah = pencilan (di luar 1,5 x IQR)
p_box_y <- ggplot(df, aes(y = Y)) +
  geom_boxplot(fill = "#3498db", color = "#2c3e50", alpha = 0.8,
               outlier.color = "#e74c3c", outlier.size = 1.8) +
  theme_minimal(base_size = 12) +
  labs(title = "Boxplot Respon Y", y = "Nilai Y")

grid.arrange(p_dens_y, p_box_y, ncol = 2)


# =============================================================================
# BAGIAN A: UJI ASUMSI KLASIK PADA MODEL OLS
# =============================================================================

# Model OLS dasar: Y diregresikan terhadap semua X (tanda "." = semua kolom lain)
m_ols <- lm(Y ~ ., data = df)

# Sisaan (residual) dan nilai prediksi (fitted) -> bahan semua uji asumsi
res_ols <- residuals(m_ols)
fit_ols <- fitted(m_ols)

# -----------------------------------------------------------------------------
# A.1 UJI NORMALITAS SISAAN
# H0: sisaan berdistribusi normal | H1: tidak normal | tolak H0 jika p < 0,05
# -----------------------------------------------------------------------------

# Histogram sisaan vs kurva normal teoritis (garis putus-putus merah)
p_hist_res <- ggplot(data.frame(res = res_ols), aes(x = res)) +
  geom_histogram(aes(y = after_stat(density)), bins = 40,
                 fill = "#34495e", color = "white", alpha = 0.8) +
  stat_function(fun = dnorm,
                args = list(mean = mean(res_ols), sd = sd(res_ols)),
                color = "#e74c3c", linewidth = 1.1, linetype = "dashed") +
  theme_minimal() +
  labs(title = "Histogram Sisaan OLS vs Kurva Normal", x = "Sisaan", y = "Densitas")

# Q-Q plot: titik yang menempel garis diagonal = normal; menyimpang di ekor = heavy tails
p_qq_res <- ggplot(data.frame(res = res_ols), aes(sample = res)) +
  stat_qq(color = "#2980b9", alpha = 0.6) +
  stat_qq_line(color = "#e74c3c", linewidth = 1.1) +
  theme_minimal() +
  labs(title = "Normal Q-Q Plot Sisaan OLS",
       x = "Kuantil Teoritis", y = "Kuantil Sampel")

grid.arrange(p_hist_res, p_qq_res, ncol = 2)

# Tiga uji formal normalitas
jb_test <- jarque.bera.test(res_ols)   # berbasis skewness & kurtosis
sw_test <- shapiro.test(res_ols)       # Shapiro-Wilk
ks_test <- ks.test(res_ols, "pnorm", mean = mean(res_ols), sd = sd(res_ols))  # Kolmogorov-Smirnov

p_norm <- c(jb_test$p.value, sw_test$p.value, ks_test$p.value)
normality_table <- data.frame(
  Metode_Uji    = c("Jarque-Bera", "Shapiro-Wilk", "Kolmogorov-Smirnov"),
  Statistik_Uji = c(jb_test$statistic, sw_test$statistic, ks_test$statistic),
  p_value       = p_norm,
  Keputusan     = ifelse(p_norm < 0.05, "Tolak H0 (Tidak Normal)", "Gagal Tolak H0 (Normal)")
)
print(normality_table, digits = 5)

# -----------------------------------------------------------------------------
# A.2 UJI MULTIKOLINEARITAS (VIF)
# VIF < 5 aman | 5-10 sedang | > 10 parah
# -----------------------------------------------------------------------------

# VIF tiap prediktor, diurutkan dari yang terbesar
vif_vals <- car::vif(m_ols)
vif_df <- data.frame(Variabel = names(vif_vals), VIF = unname(vif_vals)) %>%
  arrange(desc(VIF))

print(head(vif_df, 12), digits = 4)    # 12 VIF tertinggi
cat("Jumlah variabel dengan VIF > 10:", sum(vif_df$VIF > 10), "dari 30\n")
cat("VIF maksimum                   :", round(max(vif_df$VIF), 2), "\n")

# Matriks korelasi antar prediktor: sel gelap = korelasi kuat (sumber kolinearitas)
X_matrix <- as.matrix(df[, paste0("X", 1:30)])
cor_mat  <- cor(X_matrix)
corrplot(cor_mat, method = "color", type = "upper",
         tl.col = "black", tl.cex = 0.65,
         title = "Matriks Korelasi Antar Prediktor (X1 - X30)",
         mar = c(0, 0, 2, 0))

# Barplot VIF; garis merah = batas kritis VIF = 10
ggplot(vif_df, aes(x = reorder(Variabel, VIF), y = VIF, fill = VIF > 10)) +
  geom_bar(stat = "identity", width = 0.7) +
  geom_hline(yintercept = 10, linetype = "dashed", color = "red", linewidth = 1) +
  scale_fill_manual(values = c("steelblue", "#e74c3c"),
                    labels = c("VIF <= 10 (Aman)", "VIF > 10 (Parah)")) +
  coord_flip() +
  theme_minimal() +
  labs(title = "Nilai VIF untuk Setiap Prediktor",
       x = "Variabel", y = "VIF", fill = "Kategori")

# -----------------------------------------------------------------------------
# A.3 UJI HETEROSKEDASTISITAS (Breusch-Pagan)
# H0: ragam sisaan homogen | H1: heteroskedastis | tolak H0 jika p < 0,05
# -----------------------------------------------------------------------------

# Plot residual vs fitted: pola corong (melebar/menyempit) = heteroskedastis
p_het <- ggplot(data.frame(fit = fit_ols, res = res_ols), aes(x = fit, y = res)) +
  geom_point(color = "#2c3e50", alpha = 0.5, size = 1.5) +
  geom_hline(yintercept = 0, color = "#e74c3c", linetype = "dashed", linewidth = 1) +
  geom_smooth(method = "loess", formula = y ~ x, color = "#2980b9", se = FALSE, linewidth = 1) +
  theme_minimal() +
  labs(title = "Residuals vs Fitted Values",
       x = "Fitted Values", y = "Residuals")
print(p_het)

bp_test <- bptest(m_ols)
bp_table <- data.frame(
  Metode_Uji    = "Studentized Breusch-Pagan",
  Statistik_BP  = bp_test$statistic,
  Derajat_Bebas = bp_test$parameter,
  p_value       = bp_test$p.value,
  Keputusan     = ifelse(bp_test$p.value < 0.05,
                         "Tolak H0 (Heteroskedastisitas)",
                         "Gagal Tolak H0 (Homoskedastis)")
)
print(bp_table, digits = 5, row.names = FALSE)

# -----------------------------------------------------------------------------
# A.4 UJI AUTOKORELASI (Durbin-Watson)
# H0: tidak ada autokorelasi (rho = 0) | H1: autokorelasi positif (rho > 0)
# DW ~ 2 -> bebas autokorelasi; DW << 2 -> autokorelasi positif
# -----------------------------------------------------------------------------

par(mfrow = c(1, 2))   # dua grafik berdampingan

# ACF: batang di lag 1 yang melewati garis biru putus-putus = autokorelasi signifikan
acf(res_ols, main = "ACF Sisaan OLS", col = "#2980b9", lwd = 2)

# Runutan 200 sisaan pertama: pola bergelombang/berkelompok = tidak acak (ada AR)
plot(res_ols[1:200], type = "o", pch = 16, col = "#2c3e50", cex = 0.7,
     main = "Runutan Sisaan (200 Obs. Pertama)",
     xlab = "Indeks Observasi (t)", ylab = "Sisaan (e_t)")
abline(h = 0, col = "red", lty = 2, lwd = 1.5)

par(mfrow = c(1, 1))   # kembalikan layout normal

dw_result   <- dwtest(m_ols, alternative = "greater")
rho_empiris <- 1 - (dw_result$statistic / 2)   # hubungan hampiran: DW ~ 2(1 - rho)

dw_table <- data.frame(
  Metode_Uji        = "Durbin-Watson",
  Statistik_DW      = dw_result$statistic,
  Estimasi_Rho_Lag1 = rho_empiris,
  p_value           = dw_result$p.value,
  Keputusan         = ifelse(dw_result$p.value < 0.05,
                             "Tolak H0 (Autokorelasi Positif)",
                             "Gagal Tolak H0 (Bebas Autokorelasi)")
)
print(dw_table, digits = 5, row.names = FALSE)


# =============================================================================
# BAGIAN B: PEMODELAN REGRESI
# =============================================================================
# Pelanggaran yang ditemukan -> solusi:
#   Multikolinearitas -> Ridge | Autokorelasi AR(1) -> GLS
#   Heteroskedastisitas (tidak signifikan) -> WLS tetap dibuat sebagai pembanding

# -----------------------------------------------------------------------------
# Model 1: OLS
# -----------------------------------------------------------------------------
m_ols <- lm(Y ~ ., data = df)
summary_ols <- summary(m_ols)

cat("\n--- Model OLS ---\n")
cat("R-squared     :", round(summary_ols$r.squared, 5), "\n")
cat("Adj R-squared :", round(summary_ols$adj.r.squared, 5), "\n")
cat("RSE           :", round(summary_ols$sigma, 5), "\n")
cat("df sisaan     :", summary_ols$df[2], "\n")

# -----------------------------------------------------------------------------
# Model 2: WLS (bobot = 1 / estimasi ragam sisaan)
# -----------------------------------------------------------------------------
# Ragam sebenarnya tidak diketahui, jadi diestimasi lewat regresi pembantu:
#   ln(e^2) = a0 + a1 * fitted  ->  ragam duga = exp(nilai prediksi)
res_sq      <- residuals(m_ols)^2
aux_model   <- lm(log(res_sq) ~ fitted(m_ols))
var_fitted  <- exp(fitted(aux_model))
weights_wls <- 1 / var_fitted        # observasi dengan ragam besar diberi bobot kecil

m_wls <- lm(Y ~ ., data = df, weights = weights_wls)
summary_wls <- summary(m_wls)

cat("\n--- Model WLS ---\n")
cat("R-squared     :", round(summary_wls$r.squared, 5), "\n")
cat("Adj R-squared :", round(summary_wls$adj.r.squared, 5), "\n")
cat("RSE           :", round(summary_wls$sigma, 5), "\n")
# Catatan: hasil mirip OLS karena data homoskedastis (bobot nyaris seragam)

# -----------------------------------------------------------------------------
# Model 3: GLS dengan struktur korelasi AR(1)
# -----------------------------------------------------------------------------
# corAR1() membuat gls() mengestimasi koefisien autokorelasi (phi) sekaligus
# koefisien regresi. Data harus berurutan waktu (sudah terpenuhi).
m_gls <- gls(Y ~ ., data = df, correlation = corAR1())
summary_gls <- summary(m_gls)

# Ambil estimasi phi dari struktur korelasi model
phi_estimated <- coef(m_gls$modelStruct$corStruct, unconstrained = FALSE)

cat("\n--- Model GLS (AR1) ---\n")
cat("Phi (AR1)           :", round(phi_estimated, 5), "\n")
cat("Residual Std. Error :", round(m_gls$sigma, 5), "\n")
cat("Log-Lik GLS         :", round(logLik(m_gls), 2), "\n")
cat("Log-Lik OLS         :", round(logLik(m_ols), 2), "\n")
# Log-likelihood GLS jauh lebih tinggi -> model lebih sesuai dengan data

# -----------------------------------------------------------------------------
# Model 4: Ridge Regression (penalti L2)
# -----------------------------------------------------------------------------
# glmnet butuh input berupa matriks X dan vektor Y
X_mat <- as.matrix(df[, paste0("X", 1:30)])
Y_vec <- df$Y

# Cari lambda (kekuatan penalti) terbaik dengan 10-fold cross-validation.
# alpha = 0 -> Ridge (alpha = 1 akan menjadi Lasso)
set.seed(42)   # agar hasil CV dapat direproduksi
cv_ridge <- cv.glmnet(X_mat, Y_vec, alpha = 0, nfolds = 10)

best_lambda <- cv_ridge$lambda.min   # lambda dengan error CV terkecil
lambda_1se  <- cv_ridge$lambda.1se   # lambda paling sederhana dalam 1 SE dari minimum

cat("\n--- Ridge Regression ---\n")
cat("lambda.min :", round(best_lambda, 5), "\n")
cat("lambda.1se :", round(lambda_1se, 5), "\n")

# Kurva CV error: garis putus-putus kiri = lambda.min, kanan = lambda.1se
plot(cv_ridge)
title("Kurva 10-Fold CV Error Ridge Regression", line = 2.5)

# Model Ridge final memakai lambda.min
m_ridge <- glmnet(X_mat, Y_vec, alpha = 0, lambda = best_lambda)

# -----------------------------------------------------------------------------
# Efek shrinkage: koefisien OLS vs Ridge pada variabel yang sangat kolinear
# -----------------------------------------------------------------------------
coef_ols   <- coef(m_ols)
coef_ridge <- as.vector(coef(m_ridge))
names(coef_ridge) <- rownames(coef(m_ridge))

collinear_vars <- c("X1", "X2", "X6", "X7", "X26", "X27", "X28", "X29")

coef_comp_df <- data.frame(
  Variabel        = collinear_vars,
  Nilai_VIF       = vif_vals[collinear_vars],
  Koefisien_OLS   = coef_ols[collinear_vars],
  Koefisien_Ridge = coef_ridge[collinear_vars]
)
print(coef_comp_df, digits = 4, row.names = FALSE)
# Koefisien OLS "liar"/berlawanan tanda; Ridge menyusutkannya jadi lebih stabil


# =============================================================================
# BAGIAN C: PERBANDINGAN MODEL (RMSE & AIC)
# =============================================================================

# 1. Nilai prediksi tiap model (data yang sama -> perbandingan adil)
pred_ols   <- predict(m_ols)
pred_wls   <- predict(m_wls)
pred_gls   <- predict(m_gls)
pred_ridge <- as.numeric(predict(m_ridge, newx = X_mat, s = best_lambda))

# 2. Fungsi RMSE = akar dari rata-rata kuadrat selisih nilai aktual & prediksi
calc_rmse <- function(actual, predicted) {
  sqrt(mean((actual - predicted)^2))
}

rmse_ols   <- calc_rmse(Y_vec, pred_ols)
rmse_wls   <- calc_rmse(Y_vec, pred_wls)
rmse_gls   <- calc_rmse(Y_vec, pred_gls)
rmse_ridge <- calc_rmse(Y_vec, pred_ridge)

# 3. AIC: OLS, WLS, GLS memakai fungsi AIC() bawaan R
aic_ols <- AIC(m_ols)
aic_wls <- AIC(m_wls)
aic_gls <- AIC(m_gls)

# AIC Ridge dihitung manual karena glmnet tidak menyediakannya, memakai
# Effective Degrees of Freedom (EDF) melalui dekomposisi SVD:
#   EDF = sum( d^2 / (d^2 + lambda) ) + 1 (intersep)
# Catatan: ini hampiran; skala lambda glmnet (yang menstandardisasi X secara
# internal) tidak persis sama dengan skala pada SVD di bawah, sehingga AIC
# Ridge sebaiknya dibaca sebagai perkiraan kasar.
n_obs     <- length(Y_vec)
X_scaled  <- scale(X_mat)
svd_d     <- svd(X_scaled)$d
edf_ridge <- sum(svd_d^2 / (svd_d^2 + best_lambda)) + 1
rss_ridge <- sum((Y_vec - pred_ridge)^2)
aic_ridge <- n_obs * log(2 * pi) + n_obs * log(rss_ridge / n_obs) + n_obs +
  2 * (edf_ridge + 1)

# 4. Tabel perbandingan (peringkat 1 = terbaik, nilai terkecil)
comparison_table <- data.frame(
  Model_Regresi  = c("OLS", "WLS", "GLS", "Ridge"),
  RMSE           = c(rmse_ols, rmse_wls, rmse_gls, rmse_ridge),
  Peringkat_RMSE = rank(c(rmse_ols, rmse_wls, rmse_gls, rmse_ridge)),
  AIC            = c(aic_ols, aic_wls, aic_gls, aic_ridge),
  Peringkat_AIC  = rank(c(aic_ols, aic_wls, aic_gls, aic_ridge))
)
cat("\n=== Perbandingan Kinerja Model ===\n")
print(comparison_table, digits = 6, row.names = FALSE)

# 5. Grafik perbandingan RMSE dan AIC
p_rmse <- ggplot(comparison_table,
                 aes(x = reorder(Model_Regresi, RMSE), y = RMSE, fill = Model_Regresi)) +
  geom_bar(stat = "identity", width = 0.6, show.legend = FALSE) +
  geom_text(aes(label = round(RMSE, 4)), vjust = -0.4, size = 4, fontface = "bold") +
  # potong sumbu Y agar selisih kecil antar model terlihat
  coord_cartesian(ylim = c(min(comparison_table$RMSE) * 0.95,
                           max(comparison_table$RMSE) * 1.05)) +
  scale_fill_brewer(palette = "Blues") +
  theme_minimal() +
  labs(title = "Perbandingan RMSE", subtitle = "Semakin kecil semakin baik",
       x = "Model", y = "RMSE")

p_aic <- ggplot(comparison_table,
                aes(x = reorder(Model_Regresi, AIC), y = AIC, fill = Model_Regresi)) +
  geom_bar(stat = "identity", width = 0.6, show.legend = FALSE) +
  geom_text(aes(label = round(AIC, 1)), vjust = -0.4, size = 4, fontface = "bold") +
  coord_cartesian(ylim = c(min(comparison_table$AIC) * 0.9,
                           max(comparison_table$AIC) * 1.05)) +
  scale_fill_brewer(palette = "Oranges") +
  theme_minimal() +
  labs(title = "Perbandingan AIC", subtitle = "Semakin kecil semakin baik",
       x = "Model", y = "AIC")

grid.arrange(p_rmse, p_aic, ncol = 2)

# =============================================================================
# CATATAN INTERPRETASI (ringkas)
# - OLS punya RMSE in-sample terendah karena memang meminimalkan RSS tak berbobot.
# - WLS ~ OLS karena data homoskedastis (Breusch-Pagan tidak signifikan).
# - GLS punya AIC terbaik karena memodelkan autokorelasi AR(1) yang kuat.
# - Ridge menstabilkan koefisien yang kolinear (VIF tinggi), dengan RMSE sedikit
#   lebih besar akibat bias penalti.
# =============================================================================
