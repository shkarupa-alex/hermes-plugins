# Hermes plugins

Монорепозиторий содержит два плагина для Hermes Agent:

- `hermes-onnx-asr` — локальное CPU-распознавание голосовых сообщений;
- `hermes-vk-community` — адаптер сообщений сообщества VK.

## Установка из каталогов GitHub

Рекомендуемый вариант установки обоих плагинов напрямую из этого
монорепозитория:

```bash
hermes plugins install shkarupa-alex/hermes-plugins/packages/hermes-onnx-asr --enable
hermes plugins install shkarupa-alex/hermes-plugins/packages/hermes-vk-community --enable
hermes gateway restart
```

Затем запустите мастера настройки и диагностику:

```bash
hermes onnx-asr setup
hermes onnx-asr doctor
hermes vk setup
hermes vk doctor
hermes gateway restart
```

Мастер VK принимает ссылку на сообщество и одну или несколько ссылок на
разрешённые профили, включая буквенные адреса вроде
`https://vk.com/shkarupa.alex`. Числовые ID он определяет через VK API. Токен
хранится только как `VK_COMMUNITY_TOKEN` в `.env` активного профиля Hermes, а
`group_id` и `allowed_user_ids` — в несекретном блоке `platforms.vk` файла
`config.yaml`.

Подробности: [ONNX ASR](packages/hermes-onnx-asr/README.md) и
[VK Community](packages/hermes-vk-community/README.md).
