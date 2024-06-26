# MLOps. Практическое задание №4 (vo_HW)

## Содержание
- [Описание](#описание)
- [Технологии](#технологии)
- [Команда проекта](#команда-проекта)

## Описание
*main.py*
- Скрипт создающий и модифицирующий датасет catboost Titanic.
- Этап #1: создаётся датасет catboost Titanic, делится на train и test подсеты и записывается в файлы titanic_train.scv и titanic_test.scv в директории *datasets\\*
- Этап #2: из файлов titanic_train.scv и titanic_test.scv считываются датасеты, заполняютс None значения, обновлённые датасеты записываются в те же файлы.
- Этап #3: из файлов titanic_train.scv и titanic_test.scv считываются датасеты, признак Sex кодируется в 'male' > '0' 'female' > '1', обновлённые датасеты записываются в те же файлы.
- Этап #4: из файлов titanic_train.scv и titanic_test.scv считываются датасеты, cоздаётся новый признак с использованием one-hot-encoding для строкового признака 'Sex', обновлённые датасеты записываются в те же файлы.
- На каждый запуск *main.py* создаётся дирекория в *outputs\\* c информацией и логами запуска.

*datasets*
- Директория хранения датасетов.
- Версионирование исуществляется посредством DVC с сохранением датасетов в Google Disk.

*config*
- Директория хранения config файлов Hydra.

## Технологии
- [Python](https://www.python.org/)
- [DVC](https://dvc.org/)
- [GitHub](https://github.com/)
- [Hydra](https://hydra.cc/)

## Команда проекта
Контакты и инструкции, как связаться с командой разработки.

- [Денис С.](tg://abc) — Developer, Орг. вопросы, оформление
