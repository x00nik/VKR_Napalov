"""
Сервер для потоковой детекции дипфейков аудио.
Использует ResNet50 модель из проекта Mardvey-UMA.
"""
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import logging
import numpy as np
from datetime import datetime
from typing import Optional

from deepfake_detector import DeepfakeDetector, SAMPLE_RATE

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Создание приложения FastAPI
app = FastAPI(
    title="Audio Deepfake Detection Streaming API",
    description="API для потоковой детекции дипфейков в аудио (ResNet50)",
    version="4.0.0",
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Глобальный детектор
detector: Optional[DeepfakeDetector] = None

# Константы для потоковой обработки
CHUNK_DURATION = 2.0  # секунды на чанк для анализа
ANALYSIS_BUFFER_SIZE = 5.0  # буфер в секундах для накопления перед ансамблевым анализом


class AudioStreamProcessor:
    """Обработчик потокового аудио с ResNet50 моделью"""
    
    def __init__(self, detector: DeepfakeDetector):
        self.detector = detector
        self.audio_buffer = []
        self.sample_rate = SAMPLE_RATE
        self.last_analysis_time = 0
        self.analysis_count = 0
        
    def add_audio_chunk(self, audio_bytes: bytes) -> Optional[dict]:
        """
        Добавляет чанк аудио в буфер и анализирует при необходимости
        
        Args:
            audio_bytes: сырые PCM данные (16-bit, mono, 16kHz)
        
        Returns:
            Результат анализа если буфер заполнен, иначе None
        """
        # Преобразуем bytes в numpy array
        audio_data = np.frombuffer(audio_bytes, dtype=np.int16)
        
        # Нормализация в [-1, 1]
        audio_float = audio_data.astype(np.float32) / 32768.0
        
        # Добавляем в буфер
        self.audio_buffer.extend(audio_float.tolist())
        
        # Проверяем размер буфера
        buffer_duration = len(self.audio_buffer) / self.sample_rate
        
        if buffer_duration >= ANALYSIS_BUFFER_SIZE:
            return self.analyze_buffer()
        
        return None
    
    def analyze_buffer(self) -> dict:
        """Анализ накопленного буфера ResNet50 моделью"""
        try:
            self.analysis_count += 1
            logger.info(
                f"[ResNet50] Анализ #{self.analysis_count}, буфер: {len(self.audio_buffer) / self.sample_rate:.1f} сек"
            )
            
            # Преобразуем буфер в numpy
            waveform = np.asarray(self.audio_buffer, dtype=np.float32)
            
            # Используем метод analyze_waveform из детектора
            result = self.detector.analyze_waveform(waveform)
            
            # Добавляем дополнительные поля для потокового API
            result['timestamp'] = datetime.now().isoformat()
            result['analysis_number'] = self.analysis_count
            result['buffer_duration'] = len(self.audio_buffer) / self.sample_rate
            result['alert_level'] = self._get_alert_level(result.get('deepfake_probability', 0.0))
            
            # Очищаем буфер, но оставляем немного для overlap
            overlap_samples = int(1.0 * self.sample_rate)  # 1 секунда overlap
            self.audio_buffer = self.audio_buffer[-overlap_samples:]
            
            return result
            
        except Exception as e:
            logger.error(f"Ошибка при анализе буфера: {e}")
            import traceback
            traceback.print_exc()
            return {
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
    
    def _get_alert_level(self, probability: float) -> str:
        """Определение уровня тревоги"""
        if probability < 0.3:
            return "safe"
        elif probability < 0.5:
            return "low"
        elif probability < 0.7:
            return "medium"
        elif probability < 0.85:
            return "high"
        else:
            return "critical"
    
    def reset(self):
        """Сброс буфера"""
        self.audio_buffer = []
        self.analysis_count = 0
        logger.info("Буфер сброшен")


@app.on_event("startup")
async def startup_event():
    """Инициализация при запуске сервера"""
    global detector
    logger.info("Запуск сервера потоковой детекции дипфейков...")
    
    try:
        detector = DeepfakeDetector()
        logger.info("✓ ResNet50 детектор инициализирован")
    except Exception as e:
        logger.error(f"Не удалось загрузить ResNet50 детектор: {e}")
        detector = None
    
    logger.info("Сервер готов к приёму потокового аудио!")


@app.get("/")
async def root():
    """Информация о сервере"""
    return {
        "service": "Audio Deepfake Detection Streaming API",
        "version": "4.0.0",
        "status": "running",
        "description": "Потоковая система детекции дипфейков ResNet50 в реальном времени",
        "models": {
            "resnet50": "Mel-Spectrogram based ResNet50 (transfer learning)"
        },
        "endpoints": {
            "stream": {
                "path": "/ws/stream",
                "protocol": "WebSocket",
                "description": "Потоковая передача аудио и получение результатов ResNet50"
            },
            "health": {
                "path": "/health",
                "method": "GET",
                "description": "Проверка состояния сервера"
            }
        },
        "audio_format": {
            "sample_rate": SAMPLE_RATE,
            "channels": 1,
            "format": "PCM 16-bit signed",
            "analysis_buffer": f"{ANALYSIS_BUFFER_SIZE} seconds"
        },
        "note": "Все анализы выполняются одной ResNet50 моделью"
    }


@app.get("/health")
async def health_check():
    """Проверка состояния сервера"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "detector_loaded": detector is not None,
        "detector_type": type(detector).__name__
    }


@app.websocket("/ws/stream")
async def websocket_stream(websocket: WebSocket):
    """
    WebSocket endpoint для потоковой передачи аудио
    
    Протокол:
    1. Клиент подключается
    2. Клиент отправляет бинарные данные (PCM 16-bit, mono, 16kHz)
    3. Сервер накапливает в буфер
    4. При заполнении буфера сервер анализирует и отправляет результат
    5. Цикл продолжается до отключения
    """
    await websocket.accept()
    logger.info(f"WebSocket подключение установлено: {websocket.client}")
    
    if detector is None:
        await websocket.send_json({
            "error": "Детектор не инициализирован",
            "timestamp": datetime.now().isoformat()
        })
        await websocket.close()
        return
    
    # Создаем процессор для этого соединения
    processor = AudioStreamProcessor(detector)
    
    try:
        # Отправляем приветственное сообщение
        await websocket.send_json({
            "status": "connected",
            "message": "Готов к приёму аудио потока",
            "config": {
                "sample_rate": SAMPLE_RATE,
                "analysis_buffer_seconds": ANALYSIS_BUFFER_SIZE,
                "format": "PCM 16-bit mono"
            },
            "timestamp": datetime.now().isoformat()
        })
        
        while True:
            # Получаем данные от клиента
            data = await websocket.receive()
            
            if "bytes" in data:
                # Обработка аудио данных
                audio_bytes = data["bytes"]
                
                # Добавляем в буфер и анализируем
                result = processor.add_audio_chunk(audio_bytes)
                
                if result is not None:
                    # Отправляем результат анализа
                    await websocket.send_json({
                        "type": "analysis_result",
                        **result
                    })
                
            elif "text" in data:
                # Обработка текстовых команд
                try:
                    import json
                    message = json.loads(data["text"])
                    
                    if message.get("command") == "reset":
                        processor.reset()
                        await websocket.send_json({
                            "type": "command_response",
                            "command": "reset",
                            "status": "success",
                            "timestamp": datetime.now().isoformat()
                        })
                    elif message.get("command") == "ping":
                        await websocket.send_json({
                            "type": "pong",
                            "timestamp": datetime.now().isoformat()
                        })
                    else:
                        await websocket.send_json({
                            "type": "error",
                            "message": f"Неизвестная команда: {message.get('command')}",
                            "timestamp": datetime.now().isoformat()
                        })
                except:
                    pass
    
    except WebSocketDisconnect:
        logger.info(f"WebSocket отключение: {websocket.client}")
    except Exception as e:
        logger.error(f"Ошибка в WebSocket соединении: {e}")
        import traceback
        traceback.print_exc()
        try:
            await websocket.send_json({
                "type": "error",
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            })
        except:
            pass
    finally:
        try:
            await websocket.close()
        except:
            pass
        logger.info("WebSocket соединение закрыто")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Сервер потоковой детекции дипфейков аудио")
    parser.add_argument("--host", type=str, default="127.0.0.1", 
                        help="IP адрес сервера (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=5000, 
                        help="Порт сервера (default: 5000)")
    parser.add_argument("--reload", action="store_true", 
                        help="Включить auto-reload при изменении кода")
    
    args = parser.parse_args()
    
    logger.info("=" * 60)
    logger.info("Запуск сервера потоковой детекции дипфейков аудио")
    logger.info("=" * 60)
    logger.info(f"Хост: {args.host}")
    logger.info(f"Порт: {args.port}")
    logger.info(f"URL: http://{args.host}:{args.port}")
    logger.info(f"WebSocket: ws://{args.host}:{args.port}/ws/stream")
    logger.info("=" * 60)
    
    uvicorn.run(
        "server_streaming:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        log_level="info"
    )


