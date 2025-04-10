### О проекте
В данном проекте я принял участие в [соревновании](https://www.kaggle.com/competitions/adengi-internship)

Всего здесь 3 ноутбука:
- v1.0 (Score: 0.43) - Первая попытка участия, повторение общего хода подготовки данных, моделей и т.д. 
- v2.0 (Score: 0.51) - Вторая попытка, эксперименты с различными методами ML.
- v3.0  (Score: 0.90) - Работа над ошибками, анализ методов, использованных коллегами, улучшение собственных коментенеций.

#### Description
В этом соревновании вам нужно будет натренировать модель оттока, которая лучше всех сможет генерализовать общую зависимость оттока клиентов.

#### Evaluation
Ответы модели будут оценены с помощью F1 метрики между предсказанными ответами и реальными.

#### Первые мысли
По сути стандартная задача бинарной классификаци.

Необходимо предсказать отток клиентов.

В зависимости от кол-ва данных, будем выбирать модель.

Предварительно, как многим известно, в подобных задачах себя очень хорошо зарекомендовали бустинги.
Я взглянул на файлик train.csv, он весит 2.2Гб, данных много, я думаю лучшим решением будут именно бустинги.

Но, т.к. я практикуюсь, мне не важно какой скор я выбью, лично мне важно попробовать разные подходы и поэкспериментировать.

Разумеется в рамках разумных сроков(решаю эту задачу только по выходным).

#### О данных
- monthly_income - среднемесячный заработок клиента (зарплата)
- payment_frequency - частота получения зарплаты (month - 1 раз в месяц, 2 weeks - раз в две недели, и тд)
- status - статус клиента (самозанятый, рабочий, и тд)
- work_experience - кол-во лет стажа клиента
- client_type - тип клиента (новый, повторный)
- settlement - город клиента
- requested_sum - запрашиваемая сумма клиента для займа, если interface - alfa
- region - регион клиента (область, округ, и тд)
- loan_id - уникальный идентификатор займа
- client_id - уникальный идентификатор клиента
- main_agreement_amount - основная одобренная сумма клиенту по займу (может быть больше, чем approved_amount)
- main_agreement_term - основной одобренный срок по займу
- requested_period_days - запрашиваемый срок по займу
- requested_amount - запрашиваемая сумма клиента по займу
- req_app_amount - разница между запрашиваемой суммой займа и одобренной
- approved_amount - одобренная сумма по займу
- source - канал привлечения клиента
- first_source - первый канал привлечения клиента
- period_days - период страховки по займу
- interface - интерфейс, откуда пришла заявка - (site, mobile)
- created_at - дата открытия займа
- type - тип займа (тип продукта)
- closed_at - дата закрытия займа
- days_finish_loan - время в днях, затраченное на закрытие займа (closed_at - created_at)
- gender - пол клиента
- ag - возраст клиента
- repayment_type - Тип комиссии по займу (с 2.5% - with_comission, 5% - with_big_comission, 0% - no_comission)
- loan_order - порядковый номер займа
- have_extension - имеется ли пролонгация по данному займу
- cnt_ext - кол-во пролонгаций по займу
- start_dt - дата начала (список) пролонгаций по займу
- term - срок пролонгации (список)
- price - цена пролонгации (список)
- elecs_sum - штрафы, пени
- recurents_sum - штрафы, пени (там вроде как различия в этапах их начисления)
- tamount - общий кэшфлоу клиента (общая сумма которая была на аккаунте клиента)
- issues - сумма просрочек, штрафов и пени
- principal - сумма основного долга
- interest - прибыль с клиента
- overdue_interest - прибыль с клиента, если есть прослойка
- overdue_fee - штрафы в просрочек
- contact_cases - кол-во обращений клиента с коллекшн
- nbki_score - скор клиента от рисков
- churn - колонка таргета
