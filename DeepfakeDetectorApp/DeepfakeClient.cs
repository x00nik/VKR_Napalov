using System;
using System.IO;
using System.Net.Http;
using System.Net.Http.Headers;
using System.Text.Json;
using System.Text.Json.Serialization;
using System.Threading.Tasks;

namespace DeepfakeDetectorApp
{
    /// <summary>
    /// Клиент для взаимодействия с сервером детекции дипфейков
    /// </summary>
    public class DeepfakeClient : IDisposable
    {
        private readonly HttpClient _httpClient;
        private string _baseUrl;

        public DeepfakeClient(string? baseUrl = null)
        {
            // Загрузка из конфигурации если URL не указан
            if (string.IsNullOrEmpty(baseUrl))
            {
                var config = ConfigManager.LoadConfig();
                _baseUrl = config.ServerUrl;
            }
            else
            {
                _baseUrl = baseUrl;
            }

            _httpClient = new HttpClient
            {
                Timeout = TimeSpan.FromMinutes(5)
            };
        }

        /// <summary>
        /// Получить текущий URL сервера
        /// </summary>
        public string GetServerUrl() => _baseUrl;

        /// <summary>
        /// Изменить URL сервера
        /// </summary>
        public void SetServerUrl(string newUrl)
        {
            _baseUrl = newUrl;
        }

        /// <summary>
        /// Проверка состояния сервера
        /// </summary>
        public async Task<bool> CheckHealthAsync()
        {
            try
            {
                var response = await _httpClient.GetAsync($"{_baseUrl}/health");
                return response.IsSuccessStatusCode;
            }
            catch
            {
                return false;
            }
        }

        /// <summary>
        /// Проверка аудио файла
        /// </summary>
        /// <param name="audioFilePath">Путь к аудио файлу</param>
        /// <param name="quickMode">true = быстрая проверка CNN, false = полный анализ ансамблем</param>
        public async Task<DeepfakeResult> CheckAudioAsync(string audioFilePath, bool quickMode = true)
        {
            if (!File.Exists(audioFilePath))
                throw new FileNotFoundException("Аудио файл не найден", audioFilePath);

            try
            {
                using var content = new MultipartFormDataContent();
                using var fileStream = File.OpenRead(audioFilePath);
                using var streamContent = new StreamContent(fileStream);

                streamContent.Headers.ContentType = new MediaTypeHeaderValue("audio/wav");
                content.Add(streamContent, "file", Path.GetFileName(audioFilePath));

                // Выбираем endpoint в зависимости от режима
                string endpoint = quickMode ? "/quick_check" : "/full_analysis";
                var response = await _httpClient.PostAsync($"{_baseUrl}{endpoint}", content);
                response.EnsureSuccessStatusCode();

                var json = await response.Content.ReadAsStringAsync();
                var options = new JsonSerializerOptions
                {
                    PropertyNameCaseInsensitive = true
                };

                if (quickMode)
                {
                    var quickResponse = JsonSerializer.Deserialize<QuickCheckApiResponse>(json, options);
                    return ConvertQuickCheckToResult(quickResponse!);
                }
                else
                {
                    var fullResponse = JsonSerializer.Deserialize<FullAnalysisApiResponse>(json, options);
                    return ConvertFullAnalysisToResult(fullResponse!);
                }
            }
            catch (Exception ex)
            {
                throw new Exception($"Ошибка при проверке аудио: {ex.Message}", ex);
            }
        }

        private DeepfakeResult ConvertQuickCheckToResult(QuickCheckApiResponse response)
        {
            return new DeepfakeResult
            {
                IsDeepfake = response.IsSuspicious,
                Confidence = response.Confidence,
                ProcessingTime = response.ProcessingTimeSeconds,
                Models = new ModelResults
                {
                    Cnn = new ModelPrediction
                    {
                        Prediction = response.IsSuspicious ? "deepfake" : "real",
                        Confidence = response.DeepfakeProbability
                    },
                    Lstm = new ModelPrediction { Prediction = "n/a", Confidence = 0 },
                    Wav2Vec2 = new ModelPrediction { Prediction = "n/a", Confidence = 0 }
                }
            };
        }

        private DeepfakeResult ConvertFullAnalysisToResult(FullAnalysisApiResponse response)
        {
            return new DeepfakeResult
            {
                IsDeepfake = response.IsDeepfake,
                Confidence = response.Confidence,
                ProcessingTime = response.ProcessingTimeSeconds,
                Models = new ModelResults
                {
                    Cnn = new ModelPrediction
                    {
                        Prediction = response.ModelPredictions?.Cnn == 1 ? "deepfake" : "real",
                        Confidence = response.IndividualProbabilities?.Cnn ?? 0
                    },
                    Lstm = new ModelPrediction
                    {
                        Prediction = response.ModelPredictions?.Lstm == 1 ? "deepfake" : "real",
                        Confidence = response.IndividualProbabilities?.Lstm ?? 0
                    },
                    Wav2Vec2 = new ModelPrediction
                    {
                        Prediction = response.ModelPredictions?.Wav2vec2 == 1 ? "deepfake" : "real",
                        Confidence = response.IndividualProbabilities?.Wav2vec2 ?? 0
                    }
                }
            };
        }

        public void Dispose()
        {
            _httpClient?.Dispose();
        }
    }

    #region API Response Models

    public class QuickCheckApiResponse
    {
        [JsonPropertyName("status")]
        public string Status { get; set; } = "";

        [JsonPropertyName("is_suspicious")]
        public bool IsSuspicious { get; set; }

        [JsonPropertyName("deepfake_probability")]
        public double DeepfakeProbability { get; set; }

        [JsonPropertyName("confidence")]
        public double Confidence { get; set; }

        [JsonPropertyName("processing_time_seconds")]
        public double ProcessingTimeSeconds { get; set; }
    }

    public class FullAnalysisApiResponse
    {
        [JsonPropertyName("status")]
        public string Status { get; set; } = "";

        [JsonPropertyName("is_deepfake")]
        public bool IsDeepfake { get; set; }

        [JsonPropertyName("deepfake_probability")]
        public double DeepfakeProbability { get; set; }

        [JsonPropertyName("confidence")]
        public double Confidence { get; set; }

        [JsonPropertyName("processing_time_seconds")]
        public double ProcessingTimeSeconds { get; set; }

        [JsonPropertyName("model_predictions")]
        public ModelPredictionsApi? ModelPredictions { get; set; }

        [JsonPropertyName("individual_probabilities")]
        public IndividualProbabilitiesApi? IndividualProbabilities { get; set; }
    }

    public class ModelPredictionsApi
    {
        [JsonPropertyName("cnn")]
        public int Cnn { get; set; }

        [JsonPropertyName("lstm")]
        public int Lstm { get; set; }

        [JsonPropertyName("wav2vec2")]
        public int Wav2vec2 { get; set; }
    }

    public class IndividualProbabilitiesApi
    {
        [JsonPropertyName("cnn")]
        public double Cnn { get; set; }

        [JsonPropertyName("lstm")]
        public double Lstm { get; set; }

        [JsonPropertyName("wav2vec2")]
        public double Wav2vec2 { get; set; }
    }

    #endregion

    #region Application Models

    /// <summary>
    /// Результат анализа дипфейка
    /// </summary>
    public class DeepfakeResult
    {
        public bool IsDeepfake { get; set; }
        public double Confidence { get; set; }
        public double ProcessingTime { get; set; }
        public ModelResults Models { get; set; } = new();
    }

    public class ModelResults
    {
        public ModelPrediction Cnn { get; set; } = new();
        public ModelPrediction Lstm { get; set; } = new();
        public ModelPrediction Wav2Vec2 { get; set; } = new();
    }

    public class ModelPrediction
    {
        public string Prediction { get; set; } = "";
        public double Confidence { get; set; }
    }

    #endregion
}

