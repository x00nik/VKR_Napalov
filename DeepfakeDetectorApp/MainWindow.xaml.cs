using System;
using System.IO;
using System.Windows;
using System.Windows.Media;
using System.Windows.Threading;
using System.Windows.Controls;
using NAudio.Wave;

namespace DeepfakeDetectorApp
{
    /// <summary>
    /// Главное окно приложения детекции дипфейков
    /// Использует двухэтапный подход:
    /// 1. Быстрая проверка 5 сек на CNN
    /// 2. Полный анализ ансамблем при обнаружении признаков дипфейка
    /// </summary>
    public partial class MainWindow : Window
    {
        private WaveInEvent? waveIn;
        private WaveFileWriter? writer;
        private DispatcherTimer? timer;
        private DateTime startTime;
        private DeepfakeClient? client;
        
        private string quickCheckFile = "quick_check.wav";
        private string fullRecordingFile = "full_recording.wav";
        
        private double fullDuration = 10;
        private const double QUICK_CHECK_DURATION = 5;
        
        private bool isQuickCheck = true;
        private bool isRecording = false;

        public MainWindow()
        {
            InitializeComponent();
            
            // Загрузка конфигурации и создание клиента
            var config = ConfigManager.LoadConfig();
            client = new DeepfakeClient();
            
            // Отображение текущего URL сервера
            ServerUrlBox.Text = config.ServerUrl;
            
            LoadDevices();
            CheckServerStatus();
        }

        private async void CheckServerStatus()
        {
            try
            {
                bool isOnline = await client!.CheckHealthAsync();
                
                if (isOnline)
                {
                    ServerStatusText.Text = "✅ Сервер подключен";
                    ServerStatusText.Foreground = new SolidColorBrush((Color)ColorConverter.ConvertFromString("#4CAF50"));
                }
                else
                {
                    ServerStatusText.Text = "⚠️ Сервер недоступен";
                    ServerStatusText.Foreground = new SolidColorBrush((Color)ColorConverter.ConvertFromString("#FF9800"));
                }
            }
            catch
            {
                ServerStatusText.Text = "❌ Ошибка подключения к серверу";
                ServerStatusText.Foreground = new SolidColorBrush((Color)ColorConverter.ConvertFromString("#F44336"));
            }
        }

        private void LoadDevices()
        {
            MicrophoneCombo.Items.Clear();
            
            try
            {
                for (int i = 0; i < WaveInEvent.DeviceCount; i++)
                {
                    var caps = WaveInEvent.GetCapabilities(i);
                    string name = caps.ProductName;
                    
                    if (name.Contains("Stereo", StringComparison.OrdinalIgnoreCase) || 
                        name.Contains("Микшер", StringComparison.OrdinalIgnoreCase) ||
                        name.Contains("Mix", StringComparison.OrdinalIgnoreCase))
                    {
                        name = "🔊 " + name + " (рекомендуется)";
                    }
                    
                    MicrophoneCombo.Items.Add(new AudioDevice(name, i));
                }
                
                if (MicrophoneCombo.Items.Count > 0)
                    MicrophoneCombo.SelectedIndex = 0;
                else
                    MessageBox.Show("Устройства записи не найдены!", "Ошибка", MessageBoxButton.OK, MessageBoxImage.Warning);
            }
            catch (Exception ex)
            {
                MessageBox.Show($"Ошибка при загрузке устройств: {ex.Message}", "Ошибка", MessageBoxButton.OK, MessageBoxImage.Error);
            }
        }

        private void RefreshBtn_Click(object sender, RoutedEventArgs e)
        {
            LoadDevices();
            CheckServerStatus();
        }

        private void UpdateServerBtn_Click(object sender, RoutedEventArgs e)
        {
            string newUrl = ServerUrlBox.Text.Trim();
            
            // Валидация URL
            if (string.IsNullOrEmpty(newUrl))
            {
                MessageBox.Show("Введите URL сервера!", "Ошибка", MessageBoxButton.OK, MessageBoxImage.Warning);
                return;
            }

            if (!newUrl.StartsWith("http://") && !newUrl.StartsWith("https://"))
            {
                MessageBox.Show("URL должен начинаться с http:// или https://", "Ошибка", MessageBoxButton.OK, MessageBoxImage.Warning);
                return;
            }

            try
            {
                // Сохранение новой конфигурации
                ConfigManager.UpdateServerUrl(newUrl);
                
                // Обновление клиента
                client?.Dispose();
                client = new DeepfakeClient(newUrl);
                
                // Проверка подключения
                CheckServerStatus();
                
                MessageBox.Show($"URL сервера обновлен:\n{newUrl}", "Успешно", MessageBoxButton.OK, MessageBoxImage.Information);
            }
            catch (Exception ex)
            {
                MessageBox.Show($"Ошибка обновления URL:\n{ex.Message}", "Ошибка", MessageBoxButton.OK, MessageBoxImage.Error);
            }
        }

        private async void RecordBtn_Click(object sender, RoutedEventArgs e)
        {
            if (isRecording)
            {
                StopRecording();
                return;
            }

            if (MicrophoneCombo.SelectedItem is not AudioDevice device)
            {
                MessageBox.Show("Выберите устройство!", "Ошибка", MessageBoxButton.OK, MessageBoxImage.Warning);
                return;
            }

            if (!double.TryParse(DurationBox.Text, out fullDuration) || fullDuration < 5)
            {
                MessageBox.Show("Длительность должна быть не менее 5 секунд!", "Ошибка", MessageBoxButton.OK, MessageBoxImage.Warning);
                return;
            }

            try
            {
                ResultCard.Visibility = Visibility.Collapsed;
                
                isQuickCheck = true;
                UpdateUI("🎙️ Запись для быстрой проверки...", 
                        $"Записывается 5 секунд для анализа CNN", 
                        Colors.Orange);
                
                StartRecording(device.DeviceNumber, quickCheckFile, QUICK_CHECK_DURATION);
            }
            catch (Exception ex)
            {
                MessageBox.Show($"Ошибка: {ex.Message}", "Ошибка", MessageBoxButton.OK, MessageBoxImage.Error);
            }
        }

        private void StartRecording(int deviceNumber, string fileName, double duration)
        {
            StopRecording();

            try
            {
                waveIn = new WaveInEvent
                {
                    DeviceNumber = deviceNumber,
                    WaveFormat = new WaveFormat(16000, 1),
                    BufferMilliseconds = 50
                };

                string path = Path.GetFullPath(fileName);
                writer = new WaveFileWriter(path, waveIn.WaveFormat);

                waveIn.DataAvailable += (s, e) =>
                {
                    try
                    {
                        if (writer != null && e.BytesRecorded > 0)
                        {
                            writer.Write(e.Buffer, 0, e.BytesRecorded);
                            writer.Flush();
                        }
                    }
                    catch { }
                };

                waveIn.RecordingStopped += OnRecordingStopped;

                waveIn.StartRecording();
                isRecording = true;
                startTime = DateTime.Now;

                RecordBtn.Content = "⏹️ ОСТАНОВИТЬ";
                RecordBtn.Background = new SolidColorBrush((Color)ColorConverter.ConvertFromString("#F44336"));

                ProgressBar.Maximum = duration;
                ProgressBar.Value = 0;

                timer = new DispatcherTimer { Interval = TimeSpan.FromMilliseconds(100) };
                timer.Tick += Timer_Tick;
                timer.Start();
            }
            catch (Exception ex)
            {
                MessageBox.Show(
                    $"Ошибка при запуске записи:\n{ex.Message}\n\n" +
                    "Попробуйте:\n" +
                    "1. Выбрать другое устройство\n" +
                    "2. Включить Stereo Mix (если есть)\n" +
                    "3. Перезапустить приложение",
                    "Ошибка записи",
                    MessageBoxButton.OK,
                    MessageBoxImage.Error
                );
                
                isRecording = false;
                UpdateUI("❌ Ошибка записи", "Выберите другое устройство", Colors.Red);
            }
        }

        private void Timer_Tick(object? sender, EventArgs e)
        {
            if (!isRecording) return;

            double elapsed = (DateTime.Now - startTime).TotalSeconds;
            ProgressBar.Value = Math.Min(elapsed, ProgressBar.Maximum);

            if (isQuickCheck)
            {
                UpdateUI("🎙️ Запись...", 
                        $"{elapsed:F1} / {QUICK_CHECK_DURATION:F0} сек (быстрая проверка)", 
                        Colors.Orange);
            }
            else
            {
                UpdateUI("🎙️ Полная запись...", 
                        $"{elapsed:F1} / {fullDuration:F0} сек", 
                        Colors.Blue);
            }

            if (elapsed >= ProgressBar.Maximum)
            {
                StopRecording();
            }
        }

        private async void OnRecordingStopped(object? sender, StoppedEventArgs e)
        {
            await Dispatcher.InvokeAsync(async () =>
            {
                isRecording = false;
                RecordBtn.Content = "🎙️ НАЧАТЬ ЗАПИСЬ";
                RecordBtn.Background = new SolidColorBrush((Color)ColorConverter.ConvertFromString("#2196F3"));
                
                timer?.Stop();
                timer = null;

                try
                {
                    if (writer != null)
                    {
                        writer.Flush();
                        writer.Dispose();
                        writer = null;
                    }
                }
                catch { }

                try
                {
                    if (waveIn != null)
                    {
                        waveIn.Dispose();
                        waveIn = null;
                    }
                }
                catch { }

                await System.Threading.Tasks.Task.Delay(500);
            });

            await System.Threading.Tasks.Task.Delay(200);

            if (isQuickCheck)
            {
                await ProcessQuickCheck();
            }
            else
            {
                await ProcessFullCheck();
            }
        }

        private async System.Threading.Tasks.Task ProcessQuickCheck()
        {
            Dispatcher.Invoke(() =>
            {
                UpdateUI("🔍 Быстрая проверка CNN...", "Анализируется 5-секундный фрагмент", Colors.Blue);
                ProgressBar.IsIndeterminate = true;
            });

            await System.Threading.Tasks.Task.Delay(500);

            if (!File.Exists(quickCheckFile))
            {
                Dispatcher.Invoke(() =>
                {
                    ProgressBar.IsIndeterminate = false;
                    UpdateUI("❌ Ошибка", "Файл записи не найден", Colors.Red);
                    MessageBox.Show("Файл записи не был создан. Попробуйте еще раз.", "Ошибка", MessageBoxButton.OK, MessageBoxImage.Error);
                });
                return;
            }

            try
            {
                var result = await client!.CheckAudioAsync(quickCheckFile, quickMode: true);

                Dispatcher.Invoke(async () =>
                {
                    ProgressBar.IsIndeterminate = false;

                    if (result.IsDeepfake)
                    {
                        UpdateUI("⚠️ Подозрение на дипфейк!", 
                                $"CNN уверенность: {result.Confidence:P0}. Начинается полная запись...", 
                                Colors.Orange);

                        await System.Threading.Tasks.Task.Delay(2000);

                        if (MicrophoneCombo.SelectedItem is AudioDevice device)
                        {
                            isQuickCheck = false;
                            UpdateUI("🎙️ Полная запись для детального анализа...", 
                                    $"Записывается {fullDuration:F0} секунд", 
                                    Colors.Blue);
                            
                            StartRecording(device.DeviceNumber, fullRecordingFile, fullDuration);
                        }
                    }
                    else
                    {
                        UpdateUI("✓ Проверка завершена", "CNN не обнаружил признаков дипфейка", Colors.Green);
                        ShowResult(result, isQuick: true);
                    }
                });
            }
            catch (Exception ex)
            {
                Dispatcher.Invoke(() =>
                {
                    ProgressBar.IsIndeterminate = false;
                    UpdateUI("❌ Ошибка проверки", ex.Message, Colors.Red);
                    MessageBox.Show($"Ошибка при проверке:\n{ex.Message}\n\nПроверьте что Python сервер запущен.", 
                                  "Ошибка", MessageBoxButton.OK, MessageBoxImage.Error);
                });
            }
        }

        private async System.Threading.Tasks.Task ProcessFullCheck()
        {
            Dispatcher.Invoke(() =>
            {
                UpdateUI("🔍 Полная проверка ансамблем...", "CNN + LSTM + Wav2Vec2", Colors.Blue);
                ProgressBar.IsIndeterminate = true;
            });

            await System.Threading.Tasks.Task.Delay(500);

            if (!File.Exists(fullRecordingFile))
            {
                Dispatcher.Invoke(() =>
                {
                    ProgressBar.IsIndeterminate = false;
                    UpdateUI("❌ Ошибка", "Файл записи не найден", Colors.Red);
                    MessageBox.Show("Файл записи не был создан. Попробуйте еще раз.", "Ошибка", MessageBoxButton.OK, MessageBoxImage.Error);
                });
                return;
            }

            try
            {
                var result = await client!.CheckAudioAsync(fullRecordingFile, quickMode: false);

                Dispatcher.Invoke(() =>
                {
                    ProgressBar.IsIndeterminate = false;
                    ShowResult(result, isQuick: false);
                });
            }
            catch (Exception ex)
            {
                Dispatcher.Invoke(() =>
                {
                    ProgressBar.IsIndeterminate = false;
                    UpdateUI("❌ Ошибка проверки", ex.Message, Colors.Red);
                    MessageBox.Show($"Ошибка при полной проверке:\n{ex.Message}", 
                                  "Ошибка", MessageBoxButton.OK, MessageBoxImage.Error);
                });
            }
        }

        private void StopRecording()
        {
            if (waveIn != null && isRecording)
            {
                waveIn.StopRecording();
            }
        }

        private void UpdateUI(string status, string subStatus, Color color)
        {
            StatusText.Text = status;
            SubStatusText.Text = subStatus;
            StatusIndicator.Background = new SolidColorBrush(color);
        }

        private void ShowResult(DeepfakeResult result, bool isQuick)
        {
            ResultCard.Visibility = Visibility.Visible;

            if (result.IsDeepfake)
            {
                ResultText.Text = "⚠️ ОБНАРУЖЕН ДИПФЕЙК";
                ResultIndicator.Background = new SolidColorBrush((Color)ColorConverter.ConvertFromString("#F44336"));
                UpdateUI("⚠️ Дипфейк обнаружен", 
                        isQuick ? "Быстрая проверка CNN" : "Полная проверка ансамблем", 
                        Colors.Red);
            }
            else
            {
                ResultText.Text = "✓ НАСТОЯЩИЙ ГОЛОС";
                ResultIndicator.Background = new SolidColorBrush((Color)ColorConverter.ConvertFromString("#4CAF50"));
                UpdateUI("✓ Аудио подлинное", 
                        isQuick ? "Быстрая проверка CNN" : "Полная проверка ансамблем", 
                        Colors.Green);
            }

            ConfidenceText.Text = $"Уверенность: {result.Confidence:P0}";

            ModelDetailsPanel.Children.Clear();

            if (isQuick)
            {
                AddModelDetail("CNN", result.Models.Cnn.Prediction, result.Models.Cnn.Confidence);
                
                var note = new TextBlock
                {
                    Text = "ℹ️ Выполнена быстрая проверка. Для детального анализа используется только CNN.",
                    FontSize = 12,
                    Foreground = new SolidColorBrush((Color)ColorConverter.ConvertFromString("#757575")),
                    TextWrapping = TextWrapping.Wrap,
                    FontStyle = FontStyles.Italic,
                    Margin = new Thickness(0, 10, 0, 0)
                };
                ModelDetailsPanel.Children.Add(note);
            }
            else
            {
                AddModelDetail("CNN", result.Models.Cnn.Prediction, result.Models.Cnn.Confidence);
                AddModelDetail("LSTM", result.Models.Lstm.Prediction, result.Models.Lstm.Confidence);
                AddModelDetail("Wav2Vec2", result.Models.Wav2Vec2.Prediction, result.Models.Wav2Vec2.Confidence);
            }

            ProcessingTimeText.Text = $"Время обработки: {result.ProcessingTime:F2}с";
        }

        private void AddModelDetail(string modelName, string prediction, double confidence)
        {
            var border = new Border
            {
                Background = new SolidColorBrush((Color)ColorConverter.ConvertFromString("#F5F5F5")),
                CornerRadius = new CornerRadius(4),
                Padding = new Thickness(12, 8, 12, 8),
                Margin = new Thickness(0, 0, 0, 8)
            };

            var grid = new Grid();
            grid.ColumnDefinitions.Add(new ColumnDefinition { Width = new GridLength(1, GridUnitType.Star) });
            grid.ColumnDefinitions.Add(new ColumnDefinition { Width = GridLength.Auto });

            var nameText = new TextBlock
            {
                Text = modelName,
                FontSize = 13,
                FontWeight = FontWeights.Medium,
                Foreground = new SolidColorBrush((Color)ColorConverter.ConvertFromString("#212121")),
                VerticalAlignment = VerticalAlignment.Center
            };
            Grid.SetColumn(nameText, 0);

            var resultStack = new StackPanel
            {
                Orientation = Orientation.Horizontal,
                HorizontalAlignment = HorizontalAlignment.Right
            };

            var predictionText = new TextBlock
            {
                Text = prediction == "deepfake" ? "Дипфейк" : "Настоящий",
                FontSize = 13,
                FontWeight = FontWeights.SemiBold,
                Foreground = new SolidColorBrush(prediction == "deepfake" ? 
                    (Color)ColorConverter.ConvertFromString("#F44336") : 
                    (Color)ColorConverter.ConvertFromString("#4CAF50")),
                Margin = new Thickness(0, 0, 10, 0)
            };

            var confidenceText = new TextBlock
            {
                Text = $"{confidence:P0}",
                FontSize = 13,
                Foreground = new SolidColorBrush((Color)ColorConverter.ConvertFromString("#757575"))
            };

            resultStack.Children.Add(predictionText);
            resultStack.Children.Add(confidenceText);
            Grid.SetColumn(resultStack, 1);

            grid.Children.Add(nameText);
            grid.Children.Add(resultStack);
            border.Child = grid;

            ModelDetailsPanel.Children.Add(border);
        }

        private class AudioDevice
        {
            public string Name { get; }
            public int DeviceNumber { get; }

            public AudioDevice(string name, int deviceNumber)
            {
                Name = name;
                DeviceNumber = deviceNumber;
            }

            public override string ToString() => Name;
        }
    }
}

