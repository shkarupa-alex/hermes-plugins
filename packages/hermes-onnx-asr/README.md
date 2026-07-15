# hermes-onnx-asr

Независимый плагин распознавания речи для Hermes Agent. Он подключается один
раз как общий STT-provider и поэтому одинаково обслуживает голосовые сообщения
из VK, Telegram и любой другой платформы, которая использует стандартный
транскрипционный pipeline Hermes.

По умолчанию используется `gigaam-v3-e2e-rnnt`, `int8`, только
`CPUExecutionProvider` — в том числе на macOS. Silero VAD загружается один раз,
переиспользует ту же GigaAM-модель и применяется к записям от 20 секунд.

## Установка

```bash
pip install hermes-onnx-asr
hermes plugins enable onnx-asr
hermes onnx-asr setup
hermes onnx-asr doctor
```

Мастер записывает настройки в активный профиль Hermes. API-ключи не нужны.
Список мастер получает из `onnx_asr.loader.AsrNames` — того же typed-registry,
который использует CLI установленной версии `onnx-asr`. Для каждой модели
показываются поддерживаемые `int8`/`fp32` и статус `certified` либо
`pending smoke`. Выбрать в мастере и рекламировать через Hermes можно только
модель, уже прошедшую обязательный CPU/codec/transcription release smoke;
остальные upstream-модели остаются видимыми, но не выдаются за проверенные.
Скачивание выполняется только после явного подтверждения. Модель может занимать
несколько гигабайт; нужен `ffmpeg` для OGG/Opus, MP3, M4A/AAC, FLAC и WebM.

Проверить discovery можно командами `hermes plugins list` и
`hermes onnx-asr list-models`.

## Конфигурация

```yaml
stt:
  provider: onnx_asr
  onnx_asr:
    model: gigaam-v3-e2e-rnnt
    quantization: int8
    model_dir: ~/.hermes/models/onnx-asr
    allow_runtime_download: false
    language: null
    vad:
      min_audio_seconds: 20
      engine: silero
      threshold: 0.5
      negative_threshold: 0.35
      min_speech_duration_ms: 250
      max_speech_duration_s: 20
      min_silence_duration_ms: 100
      speech_pad_ms: 30
    audio:
      max_duration_seconds: null
      temp_safety_margin_bytes: 1073741824
    runtime:
      concurrency: 1
      queue_depth: 4
      ffmpeg_timeout_seconds: 3600
      transcription_timeout_seconds: 21600
      intra_op_num_threads: 0
      inter_op_num_threads: 0
```

Семантика `vad.min_audio_seconds`:

- `20` — VAD для записей длительностью 20 секунд и больше;
- `0` — VAD для каждой записи;
- `null` — полностью отключить VAD и не загружать Silero.

Переменные окружения с префиксом `HERMES_ONNX_ASR__` могут переопределять
настройки через синтаксис pydantic-settings, например
`HERMES_ONNX_ASR__VAD__MIN_AUDIO_SECONDS=0`. Список ONNX providers намеренно не
является настройкой и всегда равен `CPUExecutionProvider`.

## Модели и офлайн-режим

```bash
hermes onnx-asr list-models
hermes onnx-asr fetch gigaam-v3-e2e-rnnt --quantization int8
hermes onnx-asr fetch-vad silero
hermes onnx-asr warmup
```

Пользовательский список синхронизируется с upstream `AsrNames`, а каталог внутри
wheel обязан содержать ровно тот же набор и фиксирует для него Hugging Face
repository и immutable commit. Несовпадение останавливает setup/doctor вместо
показа модели, которую плагин не сможет безопасно загрузить.
Загрузка идёт в приватный staging-каталог под межпроцессным lock. После загрузки
плагин вычисляет SHA-256 каждого файла, записывает `manifest.json`, проверяет его
и только затем атомарно публикует bundle. Повреждённый существующий bundle
перемещается рядом с суффиксом `invalid-*`, а не перезаписывается.

При `allow_runtime_download: false` вызов `transcribe()` никогда не скачивает
файлы. Сначала выполните `fetch` и `fetch-vad`. При `true` тот же pinned fetch
может быть запущен первым запросом; автоматического выбора другой модели нет.

## Форматы и длинные записи

Совместимый PCM WAV передаётся прямо в `onnx-asr` с `channel="mean"`; отдельный
NumPy-пайплайн не строится. Остальные контейнеры преобразуются `ffmpeg` во
временный mono PCM16 WAV 16 кГц. Включён protocol allowlist, поэтому плейлист или
контейнер не может заставить ffmpeg сходить по HTTP. Временный каталог живёт до
фактического завершения worker job — даже если ожидающий вызов уже получил
timeout — и удаляется самим worker.

Плагин не задаёт лимит длительности по умолчанию и CLI может обрабатывать
многочасовые файлы. `onnx-asr` читает PCM в память; ориентировочный объём
нормализованного WAV — `duration_seconds * 16000 * 2` байт. Перед конвертацией
проверяется свободное место с запасом 1 GiB.

Сам Hermes 0.18.2 до вызова provider отклоняет gateway-файлы больше 25 MiB.
Обойти этот общий лимит внешний плагин не может. Для больших файлов используйте:

```bash
hermes onnx-asr transcribe /path/to/recording.m4a
```

## Очередь и таймауты

Одна inference-задача выполняется одновременно, ещё четыре ждут FIFO. Шестой
одновременный запрос получает `asr_queue_full` до копирования входного файла.
Слот резервируется первым; затем source получает worker-owned hardlink либо
копию и только после этого публикуется worker. При cross-filesystem copy заранее
проверяется размер файла плюс `temp_safety_margin_bytes`, а `ENOSPC` возвращает
`insufficient_temp_space`. Ошибка staging освобождает резерв очереди.

ONNX Runtime inference нельзя
безопасно отменить: после `asr_timeout` worker продолжает владеть моделью и
временными файлами, пока вычисление действительно не завершится; следующий
запрос не стартует поверх него.

## Диагностика

`hermes onnx-asr doctor` проверяет версию Hermes, минимальные поддерживаемые
версии `onnx-asr` и ONNX Runtime, конфигурацию, ffmpeg, manifest модели и VAD, загрузку pipeline,
identity общего ASR/resampler у VAD wrapper и provider каждого обнаруженного
ONNX session. Warm-up сначала проходит 8 kHz resampler, затем распознаёт
встроенный синтетический русский речевой fixture и отдельно прогоняет его через
Silero VAD. После этого точный role manifest аудируется повторно. Любой CUDA,
CoreML, неизвестная/дублированная session или иной provider считается фатальной
ошибкой.

Провайдер возвращает стандартный Hermes envelope и стабильный `error_code`.
Пользовательские ошибки не содержат пути, URL, исходное аудио или внутренний
текст исключений.

## Ограничения v1

- один активный Hermes profile на процесс; смена `HERMES_HOME` отклоняется;
- runtime concurrency фиксирована в 1;
- русский quality-gate corpus и кроссплатформенная release-матрица выполняются
  в release CI и не входят в wheel;
- `warmup` содержит короткий русский speech/VAD smoke; отдельный 30-клиповый
  20-минутный corpus с human-verified reference, лицензиями и SHA-256 должен
  находиться в checkout под `release-corpus/` (`manifest.jsonl`,
  `baseline.json` и указанные в manifest аудиофайлы). Он не входит в wheel, но
  является обязательным versioned release artifact: без него tag/publish gate
  намеренно останавливается.
