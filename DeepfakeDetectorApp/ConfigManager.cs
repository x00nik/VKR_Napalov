using System;
using System.IO;
using System.Text.Json;

namespace DeepfakeDetectorApp
{
    /// <summary>
    /// Менеджер конфигурации приложения
    /// </summary>
    public class ConfigManager
    {
        private const string CONFIG_FILE = "config.json";
        private static AppConfig? _config;

        /// <summary>
        /// Загрузка конфигурации из файла
        /// </summary>
        public static AppConfig LoadConfig()
        {
            if (_config != null)
                return _config;

            try
            {
                if (File.Exists(CONFIG_FILE))
                {
                    string json = File.ReadAllText(CONFIG_FILE);
                    _config = JsonSerializer.Deserialize<AppConfig>(json);
                    
                    if (_config != null)
                        return _config;
                }
            }
            catch (Exception ex)
            {
                System.Diagnostics.Debug.WriteLine($"Ошибка загрузки конфигурации: {ex.Message}");
            }

            // Конфигурация по умолчанию
            _config = new AppConfig
            {
                ServerUrl = "http://localhost:5000",
                ConnectionTimeout = 300
            };

            return _config;
        }

        /// <summary>
        /// Сохранение конфигурации в файл
        /// </summary>
        public static void SaveConfig(AppConfig config)
        {
            try
            {
                var options = new JsonSerializerOptions
                {
                    WriteIndented = true
                };

                string json = JsonSerializer.Serialize(config, options);
                File.WriteAllText(CONFIG_FILE, json);
                
                _config = config;
            }
            catch (Exception ex)
            {
                throw new Exception($"Ошибка сохранения конфигурации: {ex.Message}", ex);
            }
        }

        /// <summary>
        /// Обновление URL сервера
        /// </summary>
        public static void UpdateServerUrl(string serverUrl)
        {
            var config = LoadConfig();
            config.ServerUrl = serverUrl;
            SaveConfig(config);
        }
    }

    /// <summary>
    /// Модель конфигурации приложения
    /// </summary>
    public class AppConfig
    {
        /// <summary>
        /// URL Python сервера
        /// </summary>
        public string ServerUrl { get; set; } = "http://localhost:5000";

        /// <summary>
        /// Таймаут подключения в секундах
        /// </summary>
        public int ConnectionTimeout { get; set; } = 300;
    }
}

