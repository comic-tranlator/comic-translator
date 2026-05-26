#set align(center)
#set par(leading: 0.1em)
*График выполнения проекта *(работы)

#table(
    stroke: 0.5pt,
    columns: (65%, auto, auto),
    align: center + top,
    table.header(
        [Раздел проекта (работы)],
        [Календарный срок выполнения],
        [Отметка о выполнении],
    ),
    table.cell(align: left)[Подбор материала, его анализ и обобщение],
    [03.04.2025],
    [],
    table.cell(align: left)[
        Изучение доступных библиотеки и инструментов, документаций. Анализ и
        проектирование системы технического зрения
    ],
    [17.04.2025],
    [],

    table.cell(align: left)[Представление раздела «Теоретическая часть»],
    [24.04.2025],
    [],

    table.cell(align: left)[Проверка дипломного проекта, составление отзыва],
    [28.05.2025],
    [],
)

#grid(
    columns: (40%, 1fr, 1fr),
    rows: auto,
    column-gutter: 1em,
    row-gutter: 0.5em,
    [
        #set align(left)
        Дата выдачи задания
    ],
    grid.cell(stroke: (bottom: 0.5pt + black))[],
    grid.cell(stroke: (bottom: 0.5pt + black))[_-_],
    [],
    text(size: 8pt)[(подпись руководителя, дата)],
    text(size: 8pt)[(И. О. Ф., руководителя)],

    [
        #set align(left)
        *_С заданием ознакомлен(-а)_*
    ],
    grid.cell(stroke: (bottom: 0.5pt + black))[],
    grid.cell(stroke: (bottom: 0.5pt + black))[_И. А. Скребцов_],
    [],
    text(size: 8pt)[(подпись студента, дата)],
    text(size: 8pt)[(И. О. Ф., студента)],
)
