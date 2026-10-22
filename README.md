# claude-skills

Общие скилы и скрипты для Claude (Hadat).

## Структура
- `hadat-label-lock/` — скил проверки и восстановления этикеток на сгенерированных кадрах.
  - `SKILL.md` — инструкция для Claude.
  - `scripts/` — рабочие скрипты (`label_rewrap.py`, `label_finish.py`, `bottle_lock.py`, `label_restore.py`, `label_flat.py`, `label_cyl.py`, `label_quad.py`, `white_images_label_lock.py`) и `README_scripts.md` с описанием.

## Правила работы
- Скрипты — единственный источник. Изменения вносятся сюда, а не в локальные копии.
- Старые версии не хранить в основной ветке; при необходимости вынести в `_archive/`.
- Перед изменением скрипта — короткое описание, что меняется и зачем.
